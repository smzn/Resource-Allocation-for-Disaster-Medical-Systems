#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VirtualOpenBCMPSimulation.py

仮想環境対応版OpenBCMPシミュレーション
- 統合災害パラメータ生成結果を読み込み
- 重力モデル適用後の遷移確率行列（P_global.csv）に対応
- 地理的ノードIDとルーティングノードIDのマッピング処理
- 動的なノード・クラス構成に対応
"""

import os, math, json, heapq, time, gzip, csv, argparse
from dataclasses import dataclass, field
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# SciPy があれば推奨の台形則を使う（無ければ NumPy にフォールバック）
try:
    from scipy.integrate import trapezoid as _trapz
except Exception:  # noqa
    _trapz = np.trapz


# -------------------- Event --------------------
@dataclass(order=True)
class Event:
    time: float
    seq: int
    kind: str = field(compare=False)   # 'EXT_ARR','INT_ARR','DEPT'
    node: int = field(compare=False, default=None)
    cls:  int = field(compare=False, default=None)
    ver:  int = field(compare=False, default=0)  # 予約無効化用（PS/LCFS-PR）


# ----------------- Virtual Environment Aware Simulator --------------
class VirtualOpenBCMPSimulation:
    def __init__(self,
                 param_dir="./parameters_disaster_integrated",
                 virtual_area_dir="./out_virtual_area",
                 theory_dir="./outputs",
                 out_dir="./sim_outputs_virtual",
                 seed=42,
                 T_end=50_000.0,
                 warmup=10_000.0,
                 snapshot_dt=10.0,
                 log_every=50_000,
                 max_events=10_000_000,
                 rolling_window=None,
                 rolling_window_frac=0.10,
                 save_excel=True,
                 verbose=True,
                 use_gravity_routing=True):  # 重力モデル適用済み遷移確率を使うか
        
        self.param_dir = param_dir
        self.virtual_area_dir = virtual_area_dir
        self.theory_dir = theory_dir
        self.out_dir = out_dir
        self.seed = seed
        self.T_end = float(T_end)
        self.warmup = float(warmup)
        self.snapshot_dt = float(snapshot_dt)
        self.log_every = int(log_every)
        self.max_events = int(max_events)
        self.rolling_window = rolling_window
        self.rolling_window_frac = float(rolling_window_frac)
        self.save_excel = save_excel
        self.verbose = verbose
        self.use_gravity_routing = use_gravity_routing

        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "plots"), exist_ok=True)

        # RNG / PQ
        self.rng = np.random.default_rng(self.seed)
        self._q = []
        self._seq = 0

        # model / params
        self.routing = None
        self.external = None
        self.service = None
        self.geo_mapping = None  # 地理的ID → ルーティングID
        self.nodes, self.classes = [], []
        self.ir2pos = {}
        self.node_type = {}
        self.fcfs_mu, self.fcfs_m = {}, {}
        self.mu_ir = {}

        # state
        self.now = 0.0
        self.events = 0
        self.last_event_kind = None
        self._last_event_node = None
        self._last_event_class = None
        self._last_t = 0.0
        self._accum_warm = defaultdict(float)

        # populations
        self.n = defaultdict(lambda: defaultdict(int))
        # FCFS
        self.busy = defaultdict(int)
        self.queue = defaultdict(deque)
        # PS
        self.ps_ver = defaultdict(int)
        self.ps_has_timer = defaultdict(bool)
        # LCFS-PR (m=1)
        self.lcfs_ver = defaultdict(int)
        self.lcfs_stack = defaultdict(list)

        # rolling（スナップショット窓）
        self._roll_N = None
        self._roll_buf = deque()
        self._roll_sum = defaultdict(float)

        # logs
        self.rmse_series = []
        self.Kir_series = []
        self._rmse_curve_times = []
        self._rmse_curve_vals = []

        # event log writer
        self._evtlog_fp = None
        self._evtlog_writer = None
        self._evtlog_header_written = False

        # snapshot control
        self.next_snapshot = 0.0

    def load_virtual_environment_mapping(self):
        """仮想環境のマッピング情報を読み込み"""
        mapping_path = Path(self.param_dir) / "geo_to_routing_mapping.csv"
        
        if mapping_path.exists():
            df = pd.read_csv(mapping_path)
            self.geo_mapping = dict(zip(df['geo_id'], df['routing_id']))
            if self.verbose:
                print(f"Loaded geographic mapping: {len(self.geo_mapping)} entries")
        else:
            if self.verbose:
                print("No geographic mapping found, using direct routing IDs")
            self.geo_mapping = {}

    def load_parameters(self):
        """パラメータファイル群を読み込み"""
        param_path = Path(self.param_dir)
        
        # 重力モデル適用済みか従来のルーティング行列かを選択
        if self.use_gravity_routing and (param_path / "P_global.csv").exists():
            routing_file = "P_global.csv"
            if self.verbose:
                print("Using gravity-applied routing matrix: P_global.csv")
        else:
            routing_file = "routing_labels.csv"
            if self.verbose:
                print("Using standard routing matrix: routing_labels.csv")
        
        # ファイル読み込み
        self.routing = pd.read_csv(param_path / routing_file, index_col=0)
        self.external = pd.read_csv(param_path / "external.csv")
        self.service = pd.read_csv(param_path / "service.csv")

        # ルーティング行列の検証
        labels = list(self.routing.columns)
        if labels[-1] != "outside":
            raise ValueError("routing の最後の列は 'outside' 必須です。")
        if self.routing.index.tolist() != labels:
            raise ValueError("routing の行ラベルと列ラベルは順序も含め完全一致が必要です。")
        
        self.int_labels = labels[:-1]

        # (i,r) 取り出し - ラベル形式に柔軟対応
        ir_list = []
        for lab in self.int_labels:
            try:
                # 従来形式: "(101,1)" 
                if lab.startswith('(') and lab.endswith(')'):
                    i, r = map(int, lab[1:-1].split(","))
                    ir_list.append((i, r))
                else:
                    # その他の形式（将来拡張用）
                    raise ValueError(f"Unsupported label format: {lab}")
            except Exception as e:
                raise ValueError(f"Cannot parse label '{lab}': {e}")

        self.ir2pos = {ir: k for k, ir in enumerate(ir_list)}
        self.nodes = sorted({i for (i, r) in ir_list})
        self.classes = sorted({r for (i, r) in ir_list})

        if self.verbose:
            print(f"Detected {len(self.nodes)} nodes, {len(self.classes)} classes")
            print(f"Nodes: {self.nodes}")
            print(f"Classes: {self.classes}")

        # サービス仕様の解析
        self._parse_service_specifications()

    def _parse_service_specifications(self):
        """サービス仕様を解析してノード種別を決定"""
        self.node_type = {i: None for i in self.nodes}
        
        for _, row in self.service.iterrows():
            typ = str(row["service_type"]).upper()
            i = int(row["node_id"])
            mu = float(row["mu"])
            m = row.get("m", None)
            m = int(m) if pd.notna(m) else None
            
            if typ == "FCFS":
                self.node_type[i] = "FCFS"
                self.fcfs_mu[i] = mu
                self.fcfs_m[i] = m if m is not None else 1
            elif typ in {"IS", "PS", "LCFS-PR"}:
                self.node_type[i] = typ
                # クラス指定の処理
                cls_spec = row["class"]
                if str(cls_spec) == "*":
                    # 全クラス共通
                    for r in self.classes:
                        self.mu_ir[(i, r)] = mu
                else:
                    # 特定クラス
                    r = int(cls_spec)
                    self.mu_ir[(i, r)] = mu
            else:
                raise ValueError(f"Unknown service_type: {typ}")

        if self.verbose:
            print("== Node Service Specifications ==")
            for i in self.nodes:
                t = self.node_type.get(i)
                if t == "FCFS":
                    print(f" node {i}: FCFS, m={self.fcfs_m[i]}, mu={self.fcfs_mu[i]}")
                else:
                    mus = {r: self.mu_ir.get((i, r), np.nan) for r in self.classes}
                    print(f" node {i}: {t}, mu_ir={mus}")

    def load_theory(self, class_node_metrics_csv="class_node_metrics.csv"):
        """理論値を読み込み"""
        theory_path = Path(self.theory_dir) / class_node_metrics_csv
        if theory_path.exists():
            df = pd.read_csv(theory_path)
            self._K_th = {(int(row["node"]), int(row["cls"])): float(row["K_ir"])
                          for _, row in df.iterrows()}
            if self.verbose:
                print(f"Loaded {len(self._K_th)} theoretical K_ir values")
        else:
            self._K_th = {}
            if self.verbose:
                print("No theoretical values found - RMSE calculation will be skipped")

    # --------------- helpers ---------------
    def _push(self, ev): 
        heapq.heappush(self._q, ev)
    
    def _pop(self): 
        return heapq.heappop(self._q) if self._q else None
    
    def _new_seq(self): 
        self._seq += 1
        return self._seq
    
    def _exp(self, rate): 
        return self.rng.exponential(1.0/rate) if rate > 0 else math.inf

    def _init_state(self):
        """シミュレーション状態を初期化"""
        # reset
        self._q.clear()
        self._seq = 0
        self.now = 0.0
        self.events = 0
        self.last_event_kind = None
        self._last_event_node = None
        self._last_event_class = None
        self._last_t = 0.0
        self._accum_warm.clear()
        self._rmse_curve_times.clear()
        self._rmse_curve_vals.clear()
        self.rmse_series.clear()
        self.Kir_series.clear()
        self.n.clear()
        self.busy.clear()
        self.queue.clear()
        self.ps_ver.clear()
        self.ps_has_timer.clear()
        self.lcfs_ver.clear()
        self.lcfs_stack.clear()
        self._roll_buf.clear()
        self._roll_sum.clear()
        self.next_snapshot = 0.0

        # イベントログ準備
        evtlog_path = os.path.join(self.out_dir, "events.csv.gz")
        self._evtlog_fp = gzip.open(evtlog_path, "wt", newline="")
        self._evtlog_writer = csv.writer(self._evtlog_fp)
        self._evtlog_header_written = False

        # 外部到着の初回スケジュール
        lam0 = defaultdict(float)
        for _, row in self.external.iterrows():
            i, r = int(row["node"]), int(row["class"])
            if (i, r) in self.ir2pos:
                lam0[(i, r)] += float(row["lambda0"])
        
        for (i, r), rate in lam0.items():
            if rate > 0:
                self._push(Event(self._exp(rate), self._new_seq(), "EXT_ARR", i, r))

        # ローリング窓設定
        win = (self.rolling_window if self.rolling_window is not None 
               else (self.rolling_window_frac * self.T_end))
        self._roll_N = max(1, int(round(win / self.snapshot_dt)))

        if self.verbose:
            print(f"[Start] T_end={self.T_end}, warmup={self.warmup}, seed={self.seed}")
            print(f"        snapshot_dt={self.snapshot_dt}, roll_N={self._roll_N}")
            print(f"        Initial arrivals scheduled: {len(lam0)} streams")

    # -------------- routing ----------------
    def _route_from(self, j, s):
        """(j,s)からの次の遷移先を決定"""
        lab_from = f"({j},{s})"
        if lab_from not in self.routing.index:
            # デバッグ情報
            if self.verbose:
                print(f"Warning: Label '{lab_from}' not found in routing matrix")
            return ("outside", None, None)
        
        row = self.routing.loc[lab_from]
        # 確率の正規化チェック
        total_prob = row.sum()
        if abs(total_prob - 1.0) > 1e-6:
            if self.verbose:
                print(f"Warning: Row '{lab_from}' probabilities sum to {total_prob}, normalizing")
            row = row / total_prob
        
        try:
            idx = self.rng.choice(len(row.values), p=row.values)
            lab_to = self.routing.columns[idx]
        except ValueError as e:
            # 確率に負の値がある場合など
            if self.verbose:
                print(f"Error in routing from '{lab_from}': {e}")
            return ("outside", None, None)
        
        if lab_to == "outside":
            return ("outside", None, None)
        
        try:
            i, r = map(int, lab_to[1:-1].split(","))
            return ("internal", i, r)
        except:
            if self.verbose:
                print(f"Warning: Cannot parse destination label '{lab_to}'")
            return ("outside", None, None)

    # -------- PS helpers --------
    def _ps_total_rate(self, i):
        """PSノードの全体サービス率を計算"""
        n_i = sum(self.n[i][r] for r in self.classes)
        if n_i <= 0:
            return 0.0
        total = 0.0
        for r in self.classes:
            n_ir = self.n[i][r]
            if n_ir > 0:
                mu = self.mu_ir.get((i, r), np.nan)
                if np.isfinite(mu) and mu > 0:
                    total += n_ir * mu
        return total / n_i

    def _ps_schedule_next(self, i):
        """PS次回退去をスケジュール"""
        rate = self._ps_total_rate(i)
        if rate <= 0:
            self.ps_has_timer[i] = False
            return
        self.ps_ver[i] += 1
        ver = self.ps_ver[i]
        self._push(Event(self.now + self._exp(rate), self._new_seq(), "DEPT", i, None, ver=ver))
        self.ps_has_timer[i] = True

    def _ps_pick_departing_class(self, i):
        """PS退去クラスを選択"""
        weights, rs = [], []
        for r in self.classes:
            n_ir = self.n[i][r]
            if n_ir > 0:
                mu = self.mu_ir.get((i, r), np.nan)
                if np.isfinite(mu) and mu > 0:
                    rs.append(r)
                    weights.append(n_ir * mu)
        if not weights:
            return None
        w = np.array(weights, dtype=float)
        w /= w.sum()
        return int(self.rng.choice(rs, p=w))

    # -------- LCFS-PR helpers --------
    def _lcfs_current_class(self, i):
        return self.lcfs_stack[i][-1] if self.lcfs_stack[i] else None

    def _lcfs_schedule(self, i):
        r_top = self._lcfs_current_class(i)
        if r_top is None:
            return
        mu = self.mu_ir.get((i, r_top), np.nan)
        if not (np.isfinite(mu) and mu > 0):
            return
        self.lcfs_ver[i] += 1
        ver = self.lcfs_ver[i]
        self._push(Event(self.now + self._exp(mu), self._new_seq(), "DEPT", i, r_top, ver=ver))

    # -------------- accumulators --------------
    def _accumulate(self, new_t):
        """時間積分用の累積計算"""
        dt = new_t - self._last_t
        if dt <= 0:
            return
        if new_t > self.warmup:
            for i in self.nodes:
                for r in self.classes:
                    self._accum_warm[(i, r)] += self.n[i][r] * dt
        self._last_t = new_t

    # -------------- RMSE update --------------
    def _update_rmse(self):
        """RMSE計算・更新"""
        if self.now <= self.warmup or not self._K_th:
            self.rmse_series.append((self.now, float('nan'), float('nan'),
                                     self.last_event_kind, self._last_event_node, self._last_event_class))
            return

        # 累積RMSE
        T = self.now - self.warmup
        errs_c = []
        for i in self.nodes:
            for r in self.classes:
                Kth = self._K_th.get((i, r), float('nan'))
                if not math.isfinite(Kth):
                    continue
                Kbar_c = self._accum_warm[(i, r)] / max(T, 1e-12)
                errs_c.append((Kbar_c - Kth) ** 2)
        rmse_c = math.sqrt(sum(errs_c) / len(errs_c)) if errs_c else float('nan')

        # ローリングRMSE
        errs_r = []
        denom = max(1, len(self._roll_buf))
        for i in self.nodes:
            for r in self.classes:
                Kth = self._K_th.get((i, r), float('nan'))
                if not math.isfinite(Kth):
                    continue
                Kbar_r = self._roll_sum[(i, r)] / denom
                errs_r.append((Kbar_r - Kth) ** 2)
        rmse_r = math.sqrt(sum(errs_r) / len(errs_r)) if errs_r else float('nan')

        self.rmse_series.append((self.now, rmse_c, rmse_r,
                                 self.last_event_kind, self._last_event_node, self._last_event_class))

        # AUC用
        self._rmse_curve_times.append(self.now)
        self._rmse_curve_vals.append(rmse_c if math.isfinite(rmse_c) else 0.0)

    # -------------- snapshot --------------
    def _snapshot(self):
        """状態のスナップショットを記録"""
        # 現在の状態を記録
        K_now = {}
        for i in self.nodes:
            for r in self.classes:
                k = self.n[i][r]
                K_now[(i, r)] = k
                self.Kir_series.append((self.now, i, r, k))
        
        # ローリング窓更新
        self._roll_buf.append(K_now)
        for key, k in K_now.items():
            self._roll_sum[key] += k
        if len(self._roll_buf) > self._roll_N:
            old = self._roll_buf.popleft()
            for key, k_old in old.items():
                self._roll_sum[key] -= k_old
        
        # RMSE更新
        self._update_rmse()
        self.next_snapshot += self.snapshot_dt

    # -------------- event log --------------
    def _log_event(self, ev):
        """イベントログの記録"""
        if not self._evtlog_header_written:
            header = ["time", "kind", "node", "class"]
            for i in self.nodes:
                for r in self.classes:
                    header.append(f"K_({i},{r})")
            self._evtlog_writer.writerow(header)
            self._evtlog_header_written = True

        row = [f"{self.now:.6f}", ev.kind, ev.node, ev.cls]
        for i in self.nodes:
            for r in self.classes:
                row.append(self.n[i][r])
        self._evtlog_writer.writerow(row)

    # -------------- arrivals / departures --------------
    def _arrival_to_node(self, i, r):
        """ノードへの到着処理"""
        typ = (self.node_type.get(i) or "").upper()
        if typ == "FCFS":
            m = self.fcfs_m[i]
            mu = self.fcfs_mu[i]
            if self.busy[i] < m:
                self.busy[i] += 1
                self._push(Event(self.now + self._exp(mu), self._new_seq(), "DEPT", i, r))
            else:
                self.queue[i].append(r)
        elif typ == "IS":
            mu = self.mu_ir.get((i, r), np.nan)
            if np.isfinite(mu) and mu > 0:
                self._push(Event(self.now + self._exp(mu), self._new_seq(), "DEPT", i, r))
        elif typ == "PS":
            self._ps_schedule_next(i)
        elif typ == "LCFS-PR":
            self.lcfs_stack[i].append(r)
            self._lcfs_schedule(i)
        else:
            if self.verbose:
                print(f"Warning: Unsupported node type {typ} at node {i}")

    def _depart_from_node_after_service(self, i, r_served):
        """サービス完了後の処理"""
        typ = (self.node_type.get(i) or "").upper()
        if typ == "FCFS":
            m = self.fcfs_m[i]
            mu = self.fcfs_mu[i]
            if self.queue[i]:
                r_next = self.queue[i].popleft()
                self._push(Event(self.now + self._exp(mu), self._new_seq(), "DEPT", i, r_next))
            else:
                self.busy[i] = max(0, self.busy[i] - 1)
        elif typ == "IS":
            pass  # ISは特に後処理なし

    # -------------- event handlers --------------
    def _handle_external_arrival(self, ev):
        """外部到着の処理"""
        # 次回外部到着をスケジュール
        matching_rows = self.external[
            (self.external["node"] == ev.node) & 
            (self.external["class"] == ev.cls)
        ]
        rate = float(matching_rows["lambda0"].sum()) if len(matching_rows) > 0 else 0.0
        
        if rate > 0:
            self._push(Event(self.now + self._exp(rate), self._new_seq(), "EXT_ARR", ev.node, ev.cls))
        
        # 実際の到着処理
        self.n[ev.node][ev.cls] += 1
        self._arrival_to_node(ev.node, ev.cls)

    def _handle_internal_arrival(self, ev):
        """内部到着の処理"""
        self.n[ev.node][ev.cls] += 1
        self._arrival_to_node(ev.node, ev.cls)

    def _handle_departure(self, ev):
        """退去の処理"""
        i = ev.node
        typ = (self.node_type.get(i) or "").upper()

        if typ == "PS":
            if ev.ver != self.ps_ver[i]:
                return  # 古い予約
            r_dep = self._ps_pick_departing_class(i)
            if r_dep is None:
                self.ps_has_timer[i] = False
                return
            self.n[i][r_dep] = max(0, self.n[i][r_dep] - 1)
            self._ps_schedule_next(i)
            # ルーティング
            kind, i2, r2 = self._route_from(i, r_dep)
            if kind == "internal":
                self._push(Event(self.now, self._new_seq(), "INT_ARR", i2, r2))
            return

        if typ == "LCFS-PR":
            if ev.ver != self.lcfs_ver[i]:
                return  # 古い予約
            if not self.lcfs_stack[i]:
                return
            r_top = self.lcfs_stack[i].pop()
            self.n[i][r_top] = max(0, self.n[i][r_top] - 1)
            self._lcfs_schedule(i)
            kind, i2, r2 = self._route_from(i, r_top)
            if kind == "internal":
                self._push(Event(self.now, self._new_seq(), "INT_ARR", i2, r2))
            return

        # FCFS / IS
        if ev.cls is not None:
            self.n[i][ev.cls] = max(0, self.n[i][ev.cls] - 1)
            self._depart_from_node_after_service(i, ev.cls)
            kind, i2, r2 = self._route_from(i, ev.cls)
            if kind == "internal":
                self._push(Event(self.now, self._new_seq(), "INT_ARR", i2, r2))

    # -------------- main simulation loop --------------
    def run(self):
        """メインシミュレーションループ"""
        self._init_state()
        
        start_time = time.time()
        while self._q and self.events < self.max_events:
            ev = self._pop()
            if ev is None:
                break
            if ev.time > self.T_end:
                self._accumulate(self.T_end)
                self.now = self.T_end
                break

            # 時間積分
            self._accumulate(ev.time)
            self.now = ev.time
            self.events += 1

            # イベント情報記録
            self.last_event_kind = ev.kind
            self._last_event_node = ev.node
            self._last_event_class = ev.cls

            # イベント処理
            try:
                if ev.kind == "EXT_ARR":
                    self._handle_external_arrival(ev)
                elif ev.kind == "INT_ARR":
                    self._handle_internal_arrival(ev)
                elif ev.kind == "DEPT":
                    self._handle_departure(ev)
                else:
                    if self.verbose:
                        print(f"Warning: Unknown event kind: {ev.kind}")
            except Exception as e:
                if self.verbose:
                    print(f"Error processing event {ev}: {e}")
                continue

            # イベントログ記録
            self._log_event(ev)

            # スナップショット
            while self.now >= self.next_snapshot:
                self._snapshot()

            # 進捗ログ
            if self.verbose and (self.events % self.log_every == 0):
                last_rmse = self.rmse_series[-1][1] if self.rmse_series else float('nan')
                elapsed = time.time() - start_time
                print(f"[t={self.now:9.1f}] events={self.events:,}  RMSE(cum)={last_rmse:.4f}  elapsed={elapsed:.1f}s")

        # ループ終了後の処理
        while self.now >= self.next_snapshot:
            self._snapshot()

        if self.verbose and self.rmse_series:
            last_c = self.rmse_series[-1][1]
            last_r = self.rmse_series[-1][2]
            elapsed = time.time() - start_time
            print(f"[FINAL] t={self.now}, events={self.events:,}, RMSE(cum)={last_c:.4f}, RMSE(roll)={last_r:.4f}")
            print(f"        Total elapsed time: {elapsed:.1f}s")

        # イベントログを閉じる
        if self._evtlog_fp is not None:
            self._evtlog_fp.close()
            self._evtlog_fp = None
            self._evtlog_writer = None

    # -------------- results & analysis --------------
    def _compute_auc_rmse(self):
        """RMSE曲線下面積を計算"""
        ts = np.array(self._rmse_curve_times, dtype=float)
        ys = np.array(self._rmse_curve_vals, dtype=float)
        if ts.size < 2:
            return float('nan')
        area = _trapz(ys, ts)
        return area / (ts[-1] - ts[0])

    def save_results(self):
        """結果の保存"""
        # 基本的な時系列データ
        df_rmse = pd.DataFrame(self.rmse_series,
                               columns=["time", "rmse_cumulative", "rmse_rolling",
                                        "last_event", "last_node", "last_class"])
        df_K = pd.DataFrame(self.Kir_series, columns=["time", "node", "cls", "K"])

        df_rmse.to_csv(os.path.join(self.out_dir, "timeseries_RMSE.csv"), index=False)
        df_K.to_csv(os.path.join(self.out_dir, "timeseries_Kir.csv"), index=False)

        # 要約統計
        summary = df_K.groupby(["node", "cls"])["K"].agg(["mean", "std", "min", "max"]).reset_index()
        summary["range"] = summary["max"] - summary["min"]
        summary.to_csv(os.path.join(self.out_dir, "summary_node_class.csv"), index=False)

        # 理論値との比較（ウォームアップ後）
        if self._K_th:
            dfK_w = df_K[df_K["time"] >= self.warmup].copy()
            if len(dfK_w) > 0:
                mean_after_warm = (dfK_w.groupby(["node", "cls"])["K"]
                                .mean().reset_index().rename(columns={"K": "K_mean"}))

                df_th = pd.DataFrame(
                    [(i, r, k) for (i, r), k in self._K_th.items()],
                    columns=["node", "cls", "K_theory"]
                )

                cmp_df = (mean_after_warm.merge(df_th, on=["node", "cls"], how="outer")
                        .sort_values(["node", "cls"]))
                cmp_df["diff"] = cmp_df["K_mean"] - cmp_df["K_theory"]
                cmp_df["rel_error_%"] = 100.0 * cmp_df["diff"] / cmp_df["K_theory"]

                cmp_path = os.path.join(self.out_dir, "compare_node_class_vs_theory.csv")
                cmp_df.to_csv(cmp_path, index=False)

                if self.verbose:
                    print(f"Theory comparison saved to: {cmp_path}")

        # メタデータ
        auc_rmse = self._compute_auc_rmse()
        final_cum = (float(df_rmse["rmse_cumulative"].dropna().iloc[-1]) 
                     if len(df_rmse["rmse_cumulative"].dropna()) > 0 else float('nan'))
        final_roll = (float(df_rmse["rmse_rolling"].dropna().iloc[-1]) 
                      if len(df_rmse["rmse_rolling"].dropna()) > 0 else float('nan'))

        meta = {
            "simulation_type": "virtual_environment_bcmp",
            "param_dir": self.param_dir,
            "virtual_area_dir": self.virtual_area_dir,
            "use_gravity_routing": self.use_gravity_routing,
            "seed": self.seed,
            "T_end": self.T_end,
            "warmup": self.warmup,
            "snapshot_dt": self.snapshot_dt,
            "log_every": self.log_every,
            "max_events": self.max_events,
            "rolling_window": self.rolling_window,
            "rolling_window_frac": self.rolling_window_frac,
            "nodes_count": len(self.nodes),
            "classes_count": len(self.classes),
            "rmse_final_cumulative": final_cum,
            "rmse_final_rolling": final_roll,
            "rmse_auc": auc_rmse,
            "events_processed": self.events,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(os.path.join(self.out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # Excel出力
        if self.save_excel:
            try:
                xls = os.path.join(self.out_dir, "sim_results.xlsx")
                with pd.ExcelWriter(xls, engine="xlsxwriter") as w:
                    df_rmse.to_excel(w, sheet_name="timeseries_RMSE", index=False)
                    df_K.to_excel(w, sheet_name="timeseries_Kir", index=False)
                    summary.to_excel(w, sheet_name="summary_node_class", index=False)
                    if self._K_th and 'cmp_df' in locals():
                        cmp_df.to_excel(w, sheet_name="compare_vs_theory", index=False)
                if self.verbose:
                    print(f"Excel file saved: {xls}")
            except Exception as e:
                if self.verbose:
                    print(f"[WARN] Excel export failed: {e}")

    def plot_results(self):
        """結果の可視化"""
        plots_dir = os.path.join(self.out_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # RMSE時系列プロット
        rmse_path = os.path.join(self.out_dir, "timeseries_RMSE.csv")
        if os.path.exists(rmse_path):
            df = pd.read_csv(rmse_path)
            tmin = self.warmup + 5 * self.snapshot_dt
            dfp = df[df["time"] >= tmin].copy()

            # 統合RMSE図
            fig, ax = plt.subplots(figsize=(12, 6))
            if "rmse_cumulative" in dfp and dfp["rmse_cumulative"].notna().any():
                ax.plot(dfp["time"], dfp["rmse_cumulative"], 
                       label="Cumulative RMSE", linewidth=1.5)
            if "rmse_rolling" in dfp and dfp["rmse_rolling"].notna().any():
                ax.plot(dfp["time"], dfp["rmse_rolling"], 
                       label="Rolling RMSE", linewidth=1.5, alpha=0.8)
            
            ax.set_xlabel("Simulation Time")
            ax.set_ylabel("RMSE")
            ax.set_title("RMSE Evolution (Virtual Environment BCMP)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.savefig(os.path.join(plots_dir, "rmse_over_time.png"), 
                       dpi=150, bbox_inches="tight")
            plt.close(fig)

            # 個別RMSE図
            if "rmse_cumulative" in dfp and dfp["rmse_cumulative"].notna().any():
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(dfp["time"], dfp["rmse_cumulative"], linewidth=2)
                final_val = dfp["rmse_cumulative"].dropna().iloc[-1]
                ax.axhline(final_val, linestyle="--", alpha=0.6, 
                          label=f"Final = {final_val:.4f}")
                ax.set_title("Cumulative RMSE")
                ax.set_xlabel("Time")
                ax.set_ylabel("RMSE")
                ax.grid(True, alpha=0.3)
                ax.legend()
                fig.savefig(os.path.join(plots_dir, "rmse_cumulative.png"), 
                           dpi=150, bbox_inches="tight")
                plt.close(fig)

        # K_ir時系列プロット
        tsK_path = os.path.join(self.out_dir, "timeseries_Kir.csv")
        if os.path.exists(tsK_path):
            dfK = pd.read_csv(tsK_path)
            
            '''
            # ノード別にプロット
            for i in sorted(dfK["node"].unique()):
                fig, ax = plt.subplots(figsize=(12, 6))
                for r in sorted(dfK["cls"].unique()):
                    sub = dfK[(dfK["node"] == i) & (dfK["cls"] == r)]
                    if len(sub) > 0:
                        ax.plot(sub["time"], sub["K"], label=f"Class {r}", linewidth=1.2)
                
                ax.set_title(f"Node {i}: Population K_ir(t)")
                ax.set_xlabel("Simulation Time")
                ax.set_ylabel("Number in System")
                ax.grid(True, alpha=0.3)
                if len(self.classes) <= 8:
                    ax.legend()
                fig.savefig(os.path.join(plots_dir, f"kir_node{i}.png"), 
                           dpi=150, bbox_inches="tight")
                plt.close(fig)

            
            # 箱ひげ図（ウォームアップ後）
            dfKw = dfK[dfK["time"] >= self.warmup].copy()
            if len(dfKw) > 0:
                dfKw["label"] = dfKw.apply(
                    lambda x: f"({int(x['node'])},{int(x['cls'])})", axis=1)
                
                labels = sorted(dfKw["label"].unique(),
                               key=lambda s: (int(s[s.find("(")+1:s.find(",")]),
                                             int(s[s.find(",")+1:s.find(")")])))
                
                data = [dfKw[dfKw["label"] == lab]["K"].values for lab in labels]
                
                if data:
                    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.8), 6))
                    box_plot = ax.boxplot(data, tick_labels=labels, showfliers=False)
                    ax.set_xlabel("(Node, Class)")
                    ax.set_ylabel("Population K")
                    ax.set_title("Population Distribution (After Warmup)")
                    ax.grid(axis="y", alpha=0.3)
                    plt.xticks(rotation=45)
                    fig.savefig(os.path.join(plots_dir, "boxplot_Kir.png"), 
                               dpi=150, bbox_inches="tight")
                    plt.close(fig)
            '''

        if self.verbose:
            print(f"Plots saved to: {plots_dir}")

    def generate_simulation_report(self):
        """シミュレーション結果の要約レポートを生成"""
        report_path = os.path.join(self.out_dir, "simulation_report.txt")
        
        with open(report_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("Virtual Environment BCMP Simulation Report\n")
            f.write("=" * 60 + "\n\n")
            
            # 基本情報
            f.write("SIMULATION CONFIGURATION:\n")
            f.write(f"  Parameter Directory: {self.param_dir}\n")
            f.write(f"  Virtual Area Directory: {self.virtual_area_dir}\n")
            f.write(f"  Use Gravity Routing: {self.use_gravity_routing}\n")
            f.write(f"  Simulation Time: {self.T_end}\n")
            f.write(f"  Warmup Period: {self.warmup}\n")
            f.write(f"  Random Seed: {self.seed}\n")
            f.write(f"  Events Processed: {self.events:,}\n\n")
            
            # ネットワーク構造
            f.write("NETWORK STRUCTURE:\n")
            f.write(f"  Nodes: {len(self.nodes)} ({self.nodes})\n")
            f.write(f"  Classes: {len(self.classes)} ({self.classes})\n")
            f.write(f"  States: {len(self.ir2pos)}\n\n")
            
            # ノード種別
            f.write("NODE TYPES:\n")
            for i in self.nodes:
                typ = self.node_type.get(i, "Unknown")
                f.write(f"  Node {i}: {typ}\n")
            f.write("\n")
            
            # RMSE結果
            if self.rmse_series and self._K_th:
                final_cum = self.rmse_series[-1][1]
                final_roll = self.rmse_series[-1][2]
                auc_rmse = self._compute_auc_rmse()
                
                f.write("RMSE RESULTS:\n")
                f.write(f"  Final Cumulative RMSE: {final_cum:.6f}\n")
                f.write(f"  Final Rolling RMSE: {final_roll:.6f}\n")
                f.write(f"  AUC RMSE: {auc_rmse:.6f}\n\n")
            
            # 理論値比較
            compare_path = os.path.join(self.out_dir, "compare_node_class_vs_theory.csv")
            if os.path.exists(compare_path):
                df_cmp = pd.read_csv(compare_path)
                f.write("THEORY COMPARISON (Top 10 largest absolute errors):\n")
                df_sorted = df_cmp.reindex(df_cmp['diff'].abs().sort_values(ascending=False).index)
                for _, row in df_sorted.head(10).iterrows():
                    f.write(f"  Node {int(row['node'])}, Class {int(row['cls'])}: "
                           f"Sim={row['K_mean']:.3f}, Theory={row['K_theory']:.3f}, "
                           f"Error={row['rel_error_%']:.1f}%\n")
                f.write("\n")
            
            f.write("=" * 60 + "\n")
            f.write("Report generated at: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")

        if self.verbose:
            print(f"Simulation report saved: {report_path}")


# ---------------------- CLI & Main ----------------------
def main():
    parser = argparse.ArgumentParser(description="Virtual Environment BCMP Simulation")
    
    # ディレクトリ設定
    parser.add_argument("--param-dir", default="./parameters_disaster_integrated",
                       help="Parameter directory")
    parser.add_argument("--virtual-area-dir", default="./out_virtual_area",
                       help="Virtual area directory")
    parser.add_argument("--theory-dir", default="./outputs",
                       help="Theory results directory")
    parser.add_argument("--out-dir", default="./sim_outputs_virtual",
                       help="Simulation output directory")
    
    # シミュレーション設定
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--T-end", type=float, default=50000.0,
                       help="Simulation end time")
    parser.add_argument("--warmup", type=float, default=10000.0,
                       help="Warmup period")
    parser.add_argument("--snapshot-dt", type=float, default=10.0,
                       help="Snapshot interval")
    parser.add_argument("--log-every", type=int, default=50000,
                       help="Progress log interval")
    parser.add_argument("--max-events", type=int, default=10000000,
                       help="Maximum events")
    
    # ローリング窓設定
    parser.add_argument("--rolling-window", type=float, default=None,
                       help="Rolling window size (time units)")
    parser.add_argument("--rolling-window-frac", type=float, default=0.10,
                       help="Rolling window as fraction of T_end")
    
    # 出力設定
    parser.add_argument("--save-excel", action="store_true", default=True,
                       help="Save Excel output")
    parser.add_argument("--no-excel", dest="save_excel", action="store_false",
                       help="Don't save Excel output")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Verbose output")
    parser.add_argument("--quiet", dest="verbose", action="store_false",
                       help="Quiet mode")
    
    # 重力モデル使用設定
    parser.add_argument("--use-gravity", action="store_true", default=True,
                       help="Use gravity-applied routing (P_global.csv)")
    parser.add_argument("--no-gravity", dest="use_gravity_routing", action="store_false",
                       help="Use standard routing (routing_labels.csv)")
    
    args = parser.parse_args()
    
    # シミュレーター初期化
    sim = VirtualOpenBCMPSimulation(
        param_dir=args.param_dir,
        virtual_area_dir=args.virtual_area_dir,
        theory_dir=args.theory_dir,
        out_dir=args.out_dir,
        seed=args.seed,
        T_end=args.T_end,
        warmup=args.warmup,
        snapshot_dt=args.snapshot_dt,
        log_every=args.log_every,
        max_events=args.max_events,
        rolling_window=args.rolling_window,
        rolling_window_frac=args.rolling_window_frac,
        save_excel=args.save_excel,
        verbose=args.verbose,
        use_gravity_routing=args.use_gravity_routing
    )
    
    try:
        # データ読み込み
        if sim.verbose:
            print("Loading virtual environment mapping...")
        sim.load_virtual_environment_mapping()
        
        if sim.verbose:
            print("Loading parameters...")
        sim.load_parameters()
        
        if sim.verbose:
            print("Loading theoretical values...")
        sim.load_theory()
        
        # シミュレーション実行
        if sim.verbose:
            print("Starting simulation...")
        sim.run()
        
        # 結果保存・可視化
        if sim.verbose:
            print("Saving results...")
        sim.save_results()
        
        if sim.verbose:
            print("Generating plots...")
        sim.plot_results()
        
        # レポート生成
        sim.generate_simulation_report()
        
        if sim.verbose:
            print(f"✓ Simulation completed successfully!")
            print(f"  Results saved in: {sim.out_dir}")
            
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        if sim.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
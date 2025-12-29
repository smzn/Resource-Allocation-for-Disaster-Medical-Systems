#!/usr/bin/env python3
# bcmp_sa_optimizer.py
# SA(焼きなまし)でBCMPネットワークの窓口数を最適化（段階1でW*、段階2はB=ratio*W*で平準化）
# - 第1段階: 不安定=0を達成する最小総窓口数 W* を探索（ラフ→二分探索）
# - 第2段階: 総上限 B = ceil(ratio * W*) で CV^2 を最小化（等式/不等式はオプション）

import os
import sys
import math
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import time
import random
import logging
from typing import Dict, Tuple, List, Optional

# --- stdoutを逐次出力（バッファ抑止） ---
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# OpenBCMPクラス（同ディレクトリorPYTHONPATHにあること）
from OpenBCMP_v6_3 import OpenBCMP


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


class BCMPSAOptimizer:
    """
    SA(焼きなまし)でBCMPネットワークの窓口数を最適化するクラス
    - FCFSノード（service_type=FCFS かつ列 m を持つ行）を決定変数とする
    - compute_objective_components() で目的の内訳を算出（不安定罰則 + CV^2 + 変更コスト）
    - lexicographic_optimize_sa(): 段階1(W*)→段階2(B=ratio*W*) を自動実行
    """

    def __init__(self,
                 param_dir,
                 out_dir,
                 alpha_aid=1.0,
                 alpha_amb=1.0,
                 alpha_hosp=1.0,
                 total_windows_limit=None,
                 weight_unstable=1000.0,
                 weight_window_cost=0.2,
                 stability_eps=1e-6,               # ρ>=1-ε を不安定と判定
                 individual_ub_mult=2.0,           # 個別上限: m_i ≤ ub_mult * m_i^(0)
                 individual_lb_div=2.0,            # 個別下限: m_i ≥ max(1, floor(m_i^(0)/div))
                 verbose=True,
                 log_file: Optional[str] = None):
        self.param_dir = Path(param_dir)
        self.out_dir = Path(out_dir)
        _ensure_dir(self.out_dir)

        # --- logger 設定 ---
        self.logger = logging.getLogger("BCMPSA")
        self.logger.setLevel(logging.INFO)
        # ハンドラを重複追加しない
        if not self.logger.handlers:
            fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s",
                                    datefmt="%Y-%m-%d %H:%M:%S")
            if log_file:
                fh = logging.FileHandler(log_file, encoding="utf-8")
                fh.setLevel(logging.INFO)
                fh.setFormatter(fmt)
                self.logger.addHandler(fh)
            # コンソールにも（verboseで見える）
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            ch.setFormatter(fmt)
            self.logger.addHandler(ch)

        self.alpha = {"Aid": alpha_aid, "Amb": alpha_amb, "Hosp": alpha_hosp}
        self.total_windows_limit: Optional[int] = total_windows_limit
        self.weight_unstable = float(weight_unstable)
        self.weight_window_cost = float(weight_window_cost)
        self.stability_eps = float(stability_eps)
        self.individual_ub_mult = float(individual_ub_mult)
        self.individual_lb_div = float(individual_lb_div)
        self.verbose = bool(verbose)

        # 必要ファイル
        self.external = self._read_csv("external.csv")
        self.service = self._read_csv("service.csv")
        self.P = self._read_csv("P_global.csv", index_col=0)

        # 初期化
        self.bcmp = self._initialize_bcmp()
        self.initial_windows = self._get_initial_windows()  # {node_id: m}
        self.fcfs_nodes = self._get_fcfs_nodes()            # [node_id]
        self.objective_cache: Dict[Tuple[Tuple[int, int], ...], float] = {}

        if self.verbose:
            print("BCMP SA Optimizer initialized:", flush=True)
            print(f"  param_dir: {self.param_dir}", flush=True)
            print(f"  out_dir  : {self.out_dir}", flush=True)
            print(f"  alpha    : {self.alpha}", flush=True)
            print(f"  total_windows_limit: {self.total_windows_limit if self.total_windows_limit is not None else 'None'}", flush=True)
            print("  weights  : ",
                  f"unstable={self.weight_unstable}, window_cost={self.weight_window_cost}", flush=True)
            print(f"  stability_eps: {self.stability_eps}", flush=True)
            print(f"  bounds: lb=initial/ {self.individual_lb_div}, ub=initial* {self.individual_ub_mult}", flush=True)
            print(f"  FCFS nodes: {len(self.fcfs_nodes)}", flush=True)
            print(f"  initial total windows: {sum(self.initial_windows.values())}", flush=True)

        # メタ情報保存
        self._dump_run_manifest()

    # ---------------- I/O ----------------
    def _read_csv(self, name, index_col=None):
        path = self.param_dir / name
        if not path.exists():
            raise FileNotFoundError(f"必要なファイルが見つかりません: {path}")
        return pd.read_csv(path, index_col=index_col)

    def _initialize_bcmp(self):
        return OpenBCMP(routing_labels=self.P, external=self.external, service=self.service, verbose=False)

    def _dump_run_manifest(self):
        """実行設定のスナップショットを JSON 保存。"""
        manifest = {
            "param_dir": str(self.param_dir),
            "out_dir": str(self.out_dir),
            "alpha": self.alpha,
            "weights": {"unstable": self.weight_unstable, "window_cost": self.weight_window_cost},
            "stability_eps": self.stability_eps,
            "bounds": {"lb_div": self.individual_lb_div, "ub_mult": self.individual_ub_mult},
            "fcfs_nodes": len(self.fcfs_nodes),
            "initial_total_windows": int(sum(self.initial_windows.values()))
        }
        with open(self.out_dir / "run_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    # --------------- Helpers ---------------
    def _find_node_column(self, df: pd.DataFrame) -> str:
        cand = ['node', 'node_id', 'node_ID', 'id', 'ID', 'nodeid']
        cols = df.columns.tolist()
        for c in cand:
            if c in cols:
                return c
        for c in cols:
            low = c.lower()
            if 'node' in low or 'id' in low:
                return c
        raise ValueError(f"ノードID列が見つかりません: {cols}")

    def _get_node_type(self, node_id: int) -> str:
        if   100001 <= node_id <= 199999: return "Region"
        elif 200001 <= node_id <= 299999: return "Movement"
        elif 300001 <= node_id <= 399999: return "Shelter"
        elif 400001 <= node_id <= 499999: return "Ambulance"
        elif 500001 <= node_id <= 599999: return "EmergencyHospital"
        elif 600001 <= node_id <= 699999: return "DisasterHospital"
        elif 700001 <= node_id <= 799999: return "TransferAmbulance"
        elif 900001 <= node_id <= 999999: return "SystemExit"
        else: return "Other"

    def _get_fcfs_nodes(self) -> List[int]:
        df = self.service.copy()
        node_col = self._find_node_column(df)
        svc_col = None
        for c in df.columns:
            if 'service' in c.lower() and 'type' in c.lower():
                svc_col = c; break
        if svc_col is None:
            raise ValueError("サービスタイプ列が見つかりません")
        return df[df[svc_col].str.upper() == 'FCFS'][node_col].astype(int).unique().tolist()

    def _get_initial_windows(self) -> Dict[int, int]:
        df = self.service.copy()
        node_col = self._find_node_column(df)
        svc_col = None
        for c in df.columns:
            if 'service' in c.lower() and 'type' in c.lower():
                svc_col = c; break
        if svc_col is None:
            raise ValueError("サービスタイプ列が見つかりません")
        fc = df[df[svc_col].str.upper() == 'FCFS']
        res: Dict[int, int] = {}
        for _, row in fc.iterrows():
            nid = int(row[node_col])
            if 'm' in df.columns and not pd.isna(row['m']):
                m = int(row['m'])
            else:
                m = 1
            res[nid] = max(1, m)
        return res

    # -------------- Objectives --------------
    def _compose_service_df(self, window_params: Dict[int, int]) -> pd.DataFrame:
        """service.csvを複製し、指定mを反映"""
        df = self.service.copy()
        node_col = self._find_node_column(df)
        for nid, m in window_params.items():
            mask = (df[node_col].astype(int) == int(nid))
            if 'm' in df.columns:
                df.loc[mask, 'm'] = int(m)
        return df

    def compute_objective_components(self, window_params: Dict[int, int]) -> Dict[str, float]:
        """
        目的関数の内訳と合計値を返す。
        keys:
          obj, n_unstable, obj_unstable, obj_variance, window_costs, total_windows

        変更点:
          - 不安定ノード判定の対象を「FCFSノード（self.fcfs_nodes）のみ」に限定
          - それ以外（PS 等）は n_unstable にはカウントしない
          - 層内CV^2の計算ロジックは従来どおり全ノード対象
        """
        # 変更コスト（|Δm|×10）
        window_costs = 0.0
        for nid, m in window_params.items():
            base = self.initial_windows.get(nid, 1)
            window_costs += abs(int(m) - int(base)) * 10.0

        # BCMP理論値
        service_df = self._compose_service_df(window_params)
        bcmp = OpenBCMP(routing_labels=self.P, external=self.external, service=service_df, verbose=False)
        bcmp.solve_lambda()
        bcmp.compute_metrics()
        _, _, node_metrics, _, _ = bcmp.build_dataframes()

        node_col = self._find_node_column(node_metrics)
        node_df = node_metrics.copy()
        # ★ node_id を明示しておく
        node_df['node_id'] = node_df[node_col].astype(int)
        node_df['node_type'] = node_df['node_id'].apply(self._get_node_type)
        node_df['L'] = pd.to_numeric(node_df['L'], errors='coerce')
        node_df['rho_i'] = pd.to_numeric(node_df['rho_i'], errors='coerce')

        # ★ FCFSノード集合
        fcfs_set = set(int(n) for n in self.fcfs_nodes)
        node_df['is_fcfs'] = node_df['node_id'].isin(fcfs_set)

        # 不安定判定: 「FCFSノード かつ (ρ >= 1 - eps または L が非有限)」
        node_df['is_unstable'] = node_df['is_fcfs'] & (
            (node_df['rho_i'] >= 1.0 - self.stability_eps) | ~np.isfinite(node_df['L'])
        )
        n_unstable = int(node_df['is_unstable'].sum())

        # 層内CV^2の加重和（ここは従来どおり全ノード対象）
        obj_variance = 0.0
        for ntype, g in node_df.groupby('node_type'):
            finite = g[np.isfinite(g['L'])]
            if len(finite) >= 2:
                Lvals = finite['L'].to_numpy(dtype=float)
                mu = float(np.mean(Lvals))
                var = float(np.mean((Lvals - mu) ** 2))
                std = float(np.sqrt(var))
                weight = (self.alpha["Aid"] if ntype == "Shelter"
                          else self.alpha["Amb"] if ntype in ["Ambulance", "TransferAmbulance"]
                          else self.alpha["Hosp"] if ntype in ["EmergencyHospital", "DisasterHospital"]
                          else 1.0)
                cv2 = (std / mu) ** 2 if mu > 0 else var
                obj_variance += weight * cv2

        total_windows = int(sum(window_params.values()))
        obj_unstable = float(n_unstable)
        obj = self.weight_unstable * obj_unstable + obj_variance + self.weight_window_cost * window_costs

        return {
            "obj": float(obj),
            "n_unstable": n_unstable,
            "obj_unstable": obj_unstable,
            "obj_variance": float(obj_variance),
            "window_costs": float(window_costs),
            "total_windows": total_windows
        }


    def compute_objective(self, window_params: Dict[int, int]) -> float:
        key = tuple(sorted((int(k), int(v)) for k, v in window_params.items()))
        if key in self.objective_cache:
            return self.objective_cache[key]
        comp = self.compute_objective_components(window_params)
        self.objective_cache[key] = comp["obj"]
        return comp["obj"]

    # -------------- Unstable list --------------
    def list_unstable_nodes(self, window_params: Dict[int, int],
                            save_path: Optional[str] = None,
                            top: Optional[int] = 20) -> pd.DataFrame:
        """
        現在の配分 window_params に対して不安定なノード（rho>=1-eps または L が非有限）を一覧化。
        - 対象は「FCFSノード」のみ（self.fcfs_nodes ベース）
        - 画面に上位(top)件を表示（verbose=Trueのとき）
        - save_path を与えれば CSV 保存
        戻り値: 不安定ノードだけを並べた DataFrame
                [node_id, node_type, m_initial, m, rho_i, L] を含む
        """
        service_df = self._compose_service_df(window_params)
        bcmp = OpenBCMP(routing_labels=self.P, external=self.external, service=service_df, verbose=False)
        bcmp.solve_lambda()
        bcmp.compute_metrics()
        _, _, node_metrics, _, _ = bcmp.build_dataframes()

        node_col = self._find_node_column(node_metrics)
        df = node_metrics.copy()
        df['node_id'] = df[node_col].astype(int)
        df['node_type'] = df['node_id'].apply(self._get_node_type)
        df['L'] = pd.to_numeric(df['L'], errors='coerce')
        df['rho_i'] = pd.to_numeric(df['rho_i'], errors='coerce')

        # ★ FCFSノードに限定
        fcfs_set = set(int(n) for n in self.fcfs_nodes)
        df['is_fcfs'] = df['node_id'].isin(fcfs_set)

        # 不安定判定: 「FCFSノード かつ (ρ >= 1 - eps または L が非有限)」
        df['is_unstable'] = df['is_fcfs'] & (
            (df['rho_i'] >= 1.0 - self.stability_eps) | ~np.isfinite(df['L'])
        )
        df = df[df['is_unstable']].copy()

        df['m'] = df['node_id'].map(lambda nid: int(window_params.get(nid, self.initial_windows.get(nid, 1))))
        df['m_initial'] = df['node_id'].map(lambda nid: int(self.initial_windows.get(nid, 1)))

        df = df.sort_values(['node_type', 'rho_i'], ascending=[True, False])

        cols = ['node_id', 'node_type', 'm_initial', 'm', 'rho_i', 'L']
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            df[cols].to_csv(save_path, index=False)

        if self.verbose:
            n = len(df)
            print(f"\n[unstable] count={n}" + (f" (saved: {save_path})" if save_path else ""), flush=True)
            if n > 0:
                show = df.head(top) if top else df
                for _, r in show.iterrows():
                    print(f"  {r['node_type']:>18}  node {int(r['node_id'])}: "
                          f"m {int(r['m_initial'])}->{int(r['m'])}, "
                          f"rho={float(r['rho_i']):.4f}, L={r['L']}", flush=True)
                if top and n > top:
                    print(f"  ... and {n - top} more (see CSV).", flush=True)
        return df


    # -------------- Reporting --------------
    def show_initial_distribution(self):
        window_by_type: Dict[str, int] = {}
        total_windows = 0
        for nid, m in self.initial_windows.items():
            total_windows += int(m)
            ntype = self._get_node_type(int(nid))
            window_by_type[ntype] = window_by_type.get(ntype, 0) + int(m)

        print("\n=== 初期窓口数分布 ===", flush=True)
        print(f"総窓口数: {total_windows}", flush=True)
        for ntype, cnt in sorted(window_by_type.items()):
            print(f"{ntype}: {cnt} 窓口", flush=True)
        print(f"\nFCFSノード数: {len(self.fcfs_nodes)}", flush=True)
        return total_windows, window_by_type
    
    def show_decision_variables(self, save_csv: Optional[str] = None) -> pd.DataFrame:
        """
        「窓口数 m が最適化の決定変数になっているノード」の一覧を表示する。

        - 対象: self.fcfs_nodes （service_type=FCFS かつ m 列を持つノード）
        - 併せて、初期値 m_initial, 個別下限 lb, 個別上限 ub も示す
        """
        rows = []
        for nid in sorted(self.fcfs_nodes):
            nid = int(nid)
            ntype = self._get_node_type(nid)
            m0 = int(self.initial_windows.get(nid, 1))
            lb = max(1, int(m0 // self.individual_lb_div))
            ub = max(2, int(m0 * self.individual_ub_mult))
            rows.append({
                "node_id": nid,
                "node_type": ntype,
                "m_initial": m0,
                "lb": lb,
                "ub": ub,
            })

        df = pd.DataFrame(rows).sort_values(["node_type", "node_id"])

        print("\n=== 決定変数（窓口数 m）となるノード一覧 ===", flush=True)
        print(f"FCFSノード数: {len(df)}", flush=True)
        # コンソールにきれいに表示（ノード数が多い場合は先頭だけでもよい）
        with pd.option_context("display.max_rows", 200, "display.max_columns", None):
            print(df.to_string(index=False), flush=True)

        if save_csv is not None:
            Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_csv, index=False)
            print(f"\n決定変数一覧を保存しました: {save_csv}", flush=True)

        return df


    def create_result_report(self, solution: Dict[int, int], obj_value: float, output_file: Optional[str] = None):
        initial_obj = self.compute_objective(self.initial_windows)
        comp = self.compute_objective_components(solution)

        lines = []
        lines.append("=== BCMPネットワーク窓口数最適化結果 ===")
        lines.append(f"日時: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append(f"初期解の目的関数値: {initial_obj:.6f}")
        lines.append(f"最適解の目的関数値: {obj_value:.6f}")
        if initial_obj > 0:
            lines.append(f"改善率: {(initial_obj - obj_value) / initial_obj * 100:.2f}%\n")
        else:
            lines.append("改善率: N/A\n")

        lines.append("=== 目的関数の構成要素（最終） ===")
        lines.append(f"不安定ノード数: {comp['n_unstable']}")
        lines.append(f"層内CV^2項   : {comp['obj_variance']:.6f}")
        lines.append(f"変更コスト   : {comp['window_costs']:.6f}")
        lines.append(f"総窓口数     : {comp['total_windows']}\n")

        type_stats: Dict[str, Dict[str, int]] = {}
        total_initial = 0
        total_final = 0
        for nid, m_final in solution.items():
            m_init = self.initial_windows.get(nid, 1)
            total_initial += m_init
            total_final += m_final
            ntype = self._get_node_type(int(nid))
            if ntype not in type_stats:
                type_stats[ntype] = {"初期窓口数": 0, "最終窓口数": 0, "増加": 0, "減少": 0, "変更なし": 0, "変更ノード": []}
            type_stats[ntype]["初期窓口数"] += m_init
            type_stats[ntype]["最終窓口数"] += m_final
            diff = m_final - m_init
            if diff > 0:
                type_stats[ntype]["増加"] += 1
                if abs(diff) >= 2:
                    type_stats[ntype]["変更ノード"].append(f"ノード{nid}: {m_init} → {m_final} (+{diff})")
            elif diff < 0:
                type_stats[ntype]["減少"] += 1
                if abs(diff) >= 2:
                    type_stats[ntype]["変更ノード"].append(f"ノード{nid}: {m_init} → {m_final} ({diff})")
            else:
                type_stats[ntype]["変更なし"] += 1

        for ntype, st in sorted(type_stats.items()):
            lines.append(f"\n{ntype}:")
            lines.append(f"  初期窓口数: {st['初期窓口数']}")
            lines.append(f"  最終窓口数: {st['最終窓口数']}")
            lines.append(f"  変更: 増加={st['増加']}, 減少={st['減少']}, 変更なし={st['変更なし']}")
            if st["変更ノード"]:
                lines.append("  主な変更:")
                for s in sorted(st["変更ノード"])[:5]:
                    lines.append(f"    {s}")

        # （任意）不安定ノードの上位をレポートに添付
        if comp["n_unstable"] > 0:
            df_un = self.list_unstable_nodes(solution, save_path=str(self.out_dir / "unstable_nodes_in_report.csv"), top=10)
            lines.append("\n=== 不安定ノード（上位10件、詳細は unstable_nodes_in_report.csv） ===")
            for _, r in df_un.head(10).iterrows():
                lines.append(f"  {r['node_type']:>18}  node {int(r['node_id'])}: "
                             f"m {int(r['m_initial'])}->{int(r['m'])}, "
                             f"rho={float(r['rho_i']):.4f}, L={r['L']}")

        lines.append("\n=== 総窓口数 ===")
        lines.append(f"初期: {total_initial}")
        lines.append(f"最終: {total_final}")
        lines.append(f"差分: {total_final - total_initial}")

        text = "\n".join(lines)
        print(text, flush=True)
        if output_file is not None:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"\nレポートを保存しました: {output_file}", flush=True)

        # 追加: 解の詳細CSV（rho,L）も保存
        self._save_solution_detail(solution, self.out_dir / "optimal_windows_detail.csv")

        return text

    def _save_solution_detail(self, solution: Dict[int, int], path: Path):
        """解を適用した状態の各ノードの m, rho, L を保存。"""
        svc = self._compose_service_df(solution)
        bcmp = OpenBCMP(routing_labels=self.P, external=self.external, service=svc, verbose=False)
        bcmp.solve_lambda()
        bcmp.compute_metrics()
        _, _, node_metrics, _, _ = bcmp.build_dataframes()
        node_col = self._find_node_column(node_metrics)
        df = node_metrics.copy()
        df["node_id"] = df[node_col].astype(int)
        df["node_type"] = df["node_id"].apply(self._get_node_type)
        df["m_initial"] = df["node_id"].map(lambda n: self.initial_windows.get(int(n), np.nan))
        df["m_opt"] = df["node_id"].map(lambda n: solution.get(int(n), np.nan))
        cols = ["node_id", "node_type", "m_initial", "m_opt", "rho_i", "L", "queue_type"]
        cols = [c for c in cols if c in df.columns]
        df = df[cols].sort_values(["node_type", "node_id"])
        df.to_csv(path, index=False)

    def _save_wstar_artifacts(self, sol_star: Dict[int, int], W_star: int):
        # 詳細（rho, L など）を保存
        self._save_solution_detail(sol_star, self.out_dir / "stage1_Wstar_detail.csv")

        # 窓口数一覧（初期との差分つき）
        rows = []
        for nid, m in sol_star.items():
            nid = int(nid)
            rows.append({
                "node_id": nid,
                "node_type": self._get_node_type(nid),
                "m_initial": int(self.initial_windows.get(nid, 1)),
                "m_wstar": int(m),
                "diff": int(m - self.initial_windows.get(nid, 1))
            })
        df = pd.DataFrame(rows).sort_values(["node_type", "node_id"])
        df.to_csv(self.out_dir / "stage1_Wstar_windows.csv", index=False)

        # 要約テキスト
        with open(self.out_dir / "stage1_Wstar_report.txt", "w", encoding="utf-8") as f:
            f.write(f"W* = {W_star}\n")
            f.write("Saved files:\n")
            f.write("  - stage1_Wstar_detail.csv\n")
            f.write("  - stage1_Wstar_windows.csv\n")


    # --------------- SA core ---------------
    def _enforce_total_exact(self, sol: Dict[int, int], limit: int):
        """総窓口数をlimitにぴったり合わせる（必要時）。
        ★修正: total_windows_limitが設定されている場合、それを超えないように制限
        """
        total = sum(sol.values())
        if total == limit:
            return
        rnd = random.Random()
        nodes = list(self.fcfs_nodes)
        # 個別下限
        min_dict = {n: max(1, int(self.initial_windows[n] // self.individual_lb_div)) for n in nodes}
        # 個別上限
        max_dict = {n: max(2, int(self.initial_windows[n] * self.individual_ub_mult)) for n in nodes}

        if total < limit:
            need = limit - total
            attempts = 0
            max_attempts = need * len(nodes) * 10  # 無限ループ防止
            while need > 0 and attempts < max_attempts:
                nid = rnd.choice(nodes)
                # ★追加: total_windows_limitがある場合は、それを超えないようチェック
                if self.total_windows_limit is not None:
                    current_total = sum(sol.values())
                    if current_total >= self.total_windows_limit:
                        break  # これ以上増やせない
                if sol[nid] < max_dict[nid]:
                    sol[nid] += 1
                    need -= 1
                attempts += 1
        else:
            over = total - limit
            while over > 0:
                nid = rnd.choice(nodes)
                if sol[nid] > min_dict[nid]:
                    sol[nid] -= 1
                    over -= 1

    def optimize_with_simulated_annealing(self,
                                        max_iterations=1000,
                                        initial_temp=90.0,
                                        cooling_rate=0.98,
                                        neighbor_max_step=3,
                                        nodes_per_move_low=1,
                                        nodes_per_move_high=3,
                                        seed: Optional[int] = None,
                                        trace_filename: Optional[str] = "sa_trace.csv",
                                        trace_mode: str = "accept",   # "accept" | "best" | "all"
                                        enforce_total_exact: bool = False,
                                        progress_interval: int = 20,
                                        save_every: int = 100,
                                        # ---- ここから追加（互換性維持のため末尾に）----
                                        start_solution: Optional[Dict[int, int]] = None,
                                        require_stable: bool = False,
                                        max_no_improv: int = 50):
        """
        SAで最適化。必要に応じて総窓口数を等式制約で維持可能。
        progress_interval: 進捗行の出力間隔（イテレーション）
        save_every: ベスト解のチェックポイント保存間隔（0で無効）
        max_no_improv: 改善なしで許容する最大イテレーション数

        追加オプション:
            start_solution: 初期解を明示指定（Noneなら従来通り initial_windows を使用）
            require_stable: True のとき、n_unstable>0 の近傍は棄却（第2段階での安定維持に有用）
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # 全体計測開始
        t_total = time.monotonic()

        if self.verbose:
            print("\n=== シミュレーテッドアニーリングによる最適化 ===", flush=True)
            print(f"最大イテレーション: {max_iterations}", flush=True)
            print(f"初期温度: {initial_temp}", flush=True)
            print(f"冷却率: {cooling_rate}", flush=True)
            if self.total_windows_limit is not None:
                print(f"総窓口数上限: {self.total_windows_limit} (enforce_exact={enforce_total_exact})", flush=True)
            if start_solution is not None:
                print("初期解: start_solution を使用", flush=True)

        # ---- 初期解（必要なら上限内に調整）----
        # 追加: start_solution が指定されていればそれを採用（互換性維持のためNone時は従来通り）
        current_solution = (start_solution or self.initial_windows).copy()

        # 既存ロジック: 上限を超えていれば大きいノードから削減
        if (self.total_windows_limit is not None) and (sum(current_solution.values()) > self.total_windows_limit):
            excess = sum(current_solution.values()) - self.total_windows_limit
            for nid in sorted(self.fcfs_nodes, key=lambda n: current_solution[n], reverse=True):
                if excess <= 0:
                    break
                lb = max(1, int(self.initial_windows[nid] // self.individual_lb_div))
                reducible = current_solution[nid] - lb
                if reducible > 0:
                    dec = min(reducible, excess)
                    current_solution[nid] -= dec
                    excess -= dec

        # 既存ロジック: 等式制約の場合はぴったり合わせ
        if enforce_total_exact and (self.total_windows_limit is not None):
            self._enforce_total_exact(current_solution, self.total_windows_limit)

        current_obj = self.compute_objective(current_solution)
        best_solution = current_solution.copy()
        best_obj = current_obj

        temp = float(initial_temp)
        no_improv = 0
        trace_rows = []
        t0 = time.monotonic()  # ステップ計測（進捗行の elapsed 用）
        accepted = 0
        tried = 0
        last_improve_iter = 0

        def maybe_trace(tag: str, sol: Dict[int, int], obj: float, it: int, T: float,
                        comp_cache: Optional[Dict[str, float]] = None):
            nonlocal accepted, tried, best_obj
            if comp_cache is None:
                comp_cache = self.compute_objective_components(sol)
            trace_rows.append({
                "iter": it, "temp": T, "tag": tag,
                "obj": comp_cache["obj"],
                "n_unstable": comp_cache["n_unstable"],
                "obj_unstable": comp_cache["obj_unstable"],
                "obj_variance": comp_cache["obj_variance"],
                "window_costs": comp_cache["window_costs"],
                "total_windows": comp_cache["total_windows"],
                "best_obj": best_obj,
                "accepted": accepted,
                "tried": tried,
                "accept_rate": (accepted / tried) if tried else 0.0,
                "elapsed_s": (time.monotonic() - t0)  # ループ内の経過
            })

        for it in range(1, max_iterations + 1):
            if temp < 0.1 or no_improv > max_no_improv:
                break

            neighbor = current_solution.copy()
            k = random.randint(nodes_per_move_low, nodes_per_move_high)
            nodes = random.sample(self.fcfs_nodes, k)

            for nid in nodes:
                step = random.randint(-neighbor_max_step, neighbor_max_step)
                if step == 0:
                    step = 1 if random.random() < 0.5 else -1
                new_m = neighbor[nid] + step

                lb = max(1, int(self.initial_windows[nid] // self.individual_lb_div))
                ub = max(2, int(self.initial_windows[nid] * self.individual_ub_mult))

                # ★追加: total_windows_limitがある場合、実質的な個別上限を計算
                if self.total_windows_limit is not None:
                    # 他のノードが最小値の場合に、このノードに割り当てられる最大値
                    other_nodes_min = sum(max(1, int(self.initial_windows[n] // self.individual_lb_div)) 
                                        for n in self.fcfs_nodes if n != nid)
                    effective_ub = self.total_windows_limit - other_nodes_min
                    ub = min(ub, max(lb, effective_ub))

                new_m = min(max(new_m, lb), ub)
                neighbor[nid] = new_m

            # 既存ロジック: 総窓口数制約
            if self.total_windows_limit is not None:
                total = sum(neighbor.values())
                if enforce_total_exact:
                    self._enforce_total_exact(neighbor, self.total_windows_limit)
                else:
                    if total > self.total_windows_limit:
                        excess = total - self.total_windows_limit
                        for nid in sorted(self.fcfs_nodes, key=lambda n: neighbor[n], reverse=True):
                            if excess <= 0:
                                break
                            lb = max(1, int(self.initial_windows[nid] // self.individual_lb_div))
                            reducible = neighbor[nid] - lb
                            if reducible > 0:
                                dec = min(reducible, excess)
                                neighbor[nid] -= dec
                                excess -= dec

            # ---- 追加: 安定必須オプション ----
            # 不安定な近傍を受理候補から除外（段階2の安定維持に有用）
            tried += 1
            comp_tmp = None
            if require_stable:
                comp_tmp = self.compute_objective_components(neighbor)
                if comp_tmp["n_unstable"] > 0:
                    if trace_mode == "all":
                        # rejectを明示したい場合は neighbor 側でトレースしてもよいが、
                        # 現仕様にならい現解のトレースに統一
                        maybe_trace("reject", current_solution, current_obj, it, temp)
                    no_improv += 1
                    temp *= cooling_rate
                    continue
            # ---- 追加ここまで ----

            # 既存ロジック: 評価と受理
            if comp_tmp is not None:
                neighbor_obj = comp_tmp["obj"]  # 二度評価を避ける
            else:
                neighbor_obj = self.compute_objective(neighbor)

            delta = neighbor_obj - current_obj
            accept = (delta < 0) or (random.random() < np.exp(-delta / max(temp, 1e-12)))

            if accept:
                accepted += 1
                current_solution = neighbor
                current_obj = neighbor_obj
                if trace_mode in ("accept", "all"):
                    maybe_trace("accept", current_solution, current_obj, it, temp, comp_cache=comp_tmp)

                if current_obj < best_obj:
                    best_solution = current_solution.copy()
                    best_obj = current_obj
                    last_improve_iter = it
                    no_improv = 0
                    if self.verbose:
                        print(f"iter {it}: best {best_obj:.6f}, T={temp:.4f}", flush=True)
                    if trace_mode in ("best", "all"):
                        maybe_trace("best", best_solution, best_obj, it, temp, comp_cache=comp_tmp)
                else:
                    no_improv += 1
            else:
                if trace_mode == "all":
                    maybe_trace("reject", current_solution, current_obj, it, temp)
                no_improv += 1

            # 既存ロジック: 周期的な進捗行
            if self.verbose and (it % max(1, progress_interval) == 0):
                comp_now = self.compute_objective_components(current_solution)
                acc_rate = accepted / tried if tried else 0.0
                elapsed = time.monotonic() - t0
                print(
                    f"[{it:5d}/{max_iterations}] T={temp:6.3f} | "
                    f"curr={comp_now['obj']:.3f} (best={best_obj:.3f}) | "
                    f"unstable={comp_now['n_unstable']} | "
                    f"cv2={comp_now['obj_variance']:.3f} | "
                    f"m={comp_now['total_windows']} | "
                    f"acc={acc_rate*100:5.1f}% | "
                    f"no_improv={no_improv:3d} | "
                    f"last_improv={last_improve_iter:5d} | "
                    f"elapsed={elapsed:6.1f}s",
                    flush=True
                )

            # 既存ロジック: 周期的にベスト解のチェックポイントを保存
            if save_every and (it % save_every == 0):
                cp_path = self.out_dir / f"checkpoint_best_iter{it}.csv"
                self._save_solution_detail(best_solution, cp_path)

            temp *= cooling_rate

        # 全体計測終了
        total_elapsed = time.monotonic() - t_total

        if self.verbose:
            print(f"\n=== SA完了（{it}イテレーション） ===", flush=True)
            print(f"初期目的: {self.compute_objective((start_solution or self.initial_windows)):.6f}", flush=True)
            print(f"最終目的: {best_obj:.6f}", flush=True)
            print(f"受理率: {(accepted / tried * 100) if tried else 0:.1f}% | "
                f"最終温度: {temp:.3f} | last_improv_iter={last_improve_iter}", flush=True)
            print(f"[time] Total elapsed: {total_elapsed:.2f}s", flush=True)

        # 既存ロジック: トレース保存（詳細）
        if trace_filename:
            df_trace = pd.DataFrame(trace_rows)
            df_trace.to_csv(self.out_dir / trace_filename, index=False)
            if self.verbose:
                print(f"[trace] saved: {self.out_dir / trace_filename}", flush=True)

        return best_solution, best_obj


    # --------- Lexicographic（二段階）最適化 ---------
    def lexicographic_optimize_sa(self,
                                base_limit: Optional[int] = None,
                                step_ratio: float = 0.05,
                                max_mult: float = 2.0,
                                sa_iters_warm=1200,
                                sa_iters_fine=2000,
                                sa_temp=90.0,
                                sa_cool=0.98,
                                seed: Optional[int] = None,
                                budget_ratio: float = 1.2,
                                budget_absolute: Optional[int] = None,
                                use_full_budget: bool = False,
                                progress_interval: int = 20,
                                save_every: int = 100,
                                require_stable_stage2: bool = False
                                ) -> Tuple[Dict[int, int], float, Dict[str, float], int, int]:
        """
        改良版:
        - Stage1(ラフ探索/二分探索)は「前回解を初期解に引き継ぐ」
        - Stage1は limit を常に等式 (=limit) で使い切る（enforce_total_exact=True）
        - これにより W* の探索が単調に積み上がり、必要窓口が素直に増える
        戻り値: (sol_final, obj_final, comp_final, W_star, B)
        """
        # 内部ユーティリティ: limit 固定で SA を1回回す（start_solution を引き継げる）
        def run_sa(limit: int, enforce_exact: bool, iters: int, tname: str,
                start: Optional[Dict[int, int]] = None,
                require_stable: bool = False):
            old_limit = self.total_windows_limit
            try:
                self.total_windows_limit = int(limit)
                sol, obj = self.optimize_with_simulated_annealing(
                    max_iterations=iters,
                    initial_temp=sa_temp,
                    cooling_rate=sa_cool,
                    seed=seed,
                    trace_filename=tname,
                    trace_mode="accept",
                    enforce_total_exact=enforce_exact,   # ★ Stage1 では True を渡す
                    progress_interval=progress_interval,
                    save_every=save_every,
                    start_solution=start,                # ★ 直前の解を引き継ぐ
                    require_stable=require_stable
                )
                comp = self.compute_objective_components(sol)
                if comp["n_unstable"] > 0:
                    # 不安定が残っているときは、詳細を落として原因特定しやすくする
                    self.list_unstable_nodes(sol, save_path=str(self.out_dir / f"unstable_nodes_{tname.replace('.csv','')}.csv"), top=50)
                return sol, obj, comp
            finally:
                self.total_windows_limit = old_limit

        # ---------- Stage1: ラフ探索（first-feasible を掴む） ----------
        tried_count = 10000
        W0 = sum(self.initial_windows.values()) if base_limit is None else int(base_limit)

        # 最初の試行（前回解なし）: =W0 を使い切る
        prev_sol = None
        sol, obj, comp = run_sa(W0, enforce_exact=True, iters=sa_iters_warm,
                                tname=f"sa_trace_stage1_L{W0}.csv",
                                start=prev_sol, require_stable=False)

        if comp["n_unstable"] > 0:
            if self.verbose:
                print(f"[stage1] limit={W0}, n_unstable={comp['n_unstable']}, obj={comp['obj']:.3f}", flush=True)

            # Lb: 不安定側の下限（不可） / Ub: 安定を初めて満たした上限（可）
            Lb = W0
            Ub = None
            tried = 0
            prev_sol = sol  # ★ 次回から引き継ぐ

            # 上限を step_ratio ずつ増やしつつ、毎回 前回解を初期解に
            while comp["n_unstable"] != 0 and tried < tried_count:
                U = int(math.ceil((1.0 + step_ratio) * (Lb if Ub is None else Ub)))

                # ★追加: total_windows_limitに到達したら探索を打ち切る
                if self.total_windows_limit is not None and U >= self.total_windows_limit:
                    if self.verbose:
                        print(f"[stage1] limit={U} が total_windows_limit={self.total_windows_limit} に到達。探索を打ち切ります。", flush=True)
                    # 最後に制限値で1回試行
                    sol, obj, comp = run_sa(self.total_windows_limit, enforce_exact=True, iters=sa_iters_warm,
                                            tname=f"sa_trace_stage1_L{self.total_windows_limit}.csv",
                                            start=prev_sol, require_stable=False)
                    if comp["n_unstable"] == 0:
                        Ub = self.total_windows_limit
                    else:
                        Lb = self.total_windows_limit
                    break

                sol, obj, comp = run_sa(U, enforce_exact=True, iters=sa_iters_warm,
                                        tname=f"sa_trace_stage1_L{U}.csv",
                                        start=prev_sol, require_stable=False)
                if self.verbose:
                    print(f"[stage1] limit={U}, n_unstable={comp['n_unstable']}, obj={comp['obj']:.3f}", flush=True)
                tried += 1
                prev_sol = sol

                if comp["n_unstable"] == 0:
                    Ub = U  # 初めて feasible を得た
                    break
                else:
                    Lb = U  # まだ不安定 → 下限を引き上げ

            if Ub is None:
                # 個別上限 or μ/P の設定上、安定に到達できないケース
                print("[warn] 不安定=0に到達できませんでした（個別上限やμ/Pの見直しが必要）", flush=True)
                # ここでは、最後の解と上限を返す
                return sol, obj, comp, Lb, Lb
        else:
            # すでに W0 で安定 → 二分探索でさらに縮める
            Ub = W0
            Lb = 0
            prev_sol = sol

        # ---------- Stage1: 二分探索（W* を厳密化） ----------
        best_sol, best_obj, best_comp, best_limit = sol, obj, comp, Ub
        L, R = Lb, Ub
        while L < R:
            mid = (L + R) // 2
            # ★ 常に =mid を使い切り、直前の解を初期解に与える
            sol_m, obj_m, comp_m = run_sa(mid, enforce_exact=True, iters=sa_iters_warm,
                                        tname=f"sa_trace_stage1_L{mid}.csv",
                                        start=prev_sol, require_stable=False)
            prev_sol = sol_m  # 次の試行に引き継ぐ（単調に積む）
            if self.verbose:
                print(f"[stage1-bin] limit={mid}, n_unstable={comp_m['n_unstable']}, obj={comp_m['obj']:.3f}", flush=True)

            if comp_m["n_unstable"] > 0:
                L = mid + 1            # 不安定 → 下限を引き上げ
            else:
                best_sol, best_obj, best_comp, best_limit = sol_m, obj_m, comp_m, mid
                R = mid                # 安定 → 上限を縮める

        sol_star, obj_star, comp_star, W_star = best_sol, best_obj, best_comp, best_limit
        print(f"[lexi] minimal limit achieving n_unstable=0: W*={W_star}", flush=True)

        # W*成果物を保存
        try:
            self._save_wstar_artifacts(sol_star, W_star)
            print(f"[stage1] W* solution saved: "
                f"{self.out_dir/'stage1_Wstar_detail.csv'}, "
                f"{self.out_dir/'stage1_Wstar_windows.csv'}", flush=True)
        except Exception as e:
            print(f"[stage1] (note) saving W* artifacts skipped: {e}", flush=True)

        # ---------- Stage2: B 設定（従来通り） ----------
        B = budget_absolute if budget_absolute is not None else int(math.ceil(W_star * max(1.0, budget_ratio)))

        # 個別上限から暗黙に決まる総上限を超えないように調整
        UB_total = sum(max(2, int(self.initial_windows[n] * self.individual_ub_mult)) for n in self.fcfs_nodes)
        if B > UB_total:
            if self.verbose:
                print(f"[warn] 予算 B={B} が暗黙の総上限 {UB_total} を超えています。B を {UB_total} に切り詰めます。", flush=True)
            B = UB_total

        print(f"[stage2] budget B={B} (W*={W_star}, ratio={budget_ratio if budget_absolute is None else 'abs'})", flush=True)

        # ---------- Stage2: B 制約下で CV^2 最小化（W*解を初期解に） ----------
        old_wc = self.weight_window_cost
        self.weight_window_cost = 0.01
        self.objective_cache.clear()  # 重み変更に伴いキャッシュ無効化
        self.total_windows_limit = B
        try:
            sol_final, obj_final = self.optimize_with_simulated_annealing(
                max_iterations=sa_iters_fine,
                initial_temp=max(sa_temp * 0.8, 1.0),
                cooling_rate=sa_cool,
                seed=seed,
                trace_filename="sa_trace_stage2.csv",
                trace_mode="accept",
                enforce_total_exact=use_full_budget,    # ≤B か =B
                progress_interval=progress_interval,
                save_every=save_every,
                start_solution=sol_star,                 # ★ Stage2は W* 解から
                require_stable=require_stable_stage2
            )
            comp_final = self.compute_objective_components(sol_final)
            if comp_final["n_unstable"] > 0:
                self.list_unstable_nodes(sol_final, save_path=str(self.out_dir / "unstable_nodes_stage2.csv"), top=50)
        finally:
            self.weight_window_cost = old_wc

        return sol_final, obj_final, comp_final, W_star, B


# 旧名互換（旧スクリプトからのimport対策）
BCMPGurobiOptimizer = BCMPSAOptimizer


def main():
    parser = argparse.ArgumentParser(
        description="SAによるBCMPネットワーク窓口数最適化（W*→Bでの平準化まで自動）"
    )
    # 基本
    parser.add_argument("--param-dir", type=str, default="./param_out")
    parser.add_argument("--out-dir",   type=str, default="./opt")
    # 重み
    parser.add_argument("--alpha-aid",  type=float, default=1.0)
    parser.add_argument("--alpha-amb",  type=float, default=1.0)
    parser.add_argument("--alpha-hosp", type=float, default=1.0)
    parser.add_argument("--weight-unstable",    type=float, default=1000.0)
    parser.add_argument("--weight-window-cost", type=float, default=0.2)
    parser.add_argument("--stability-eps", type=float, default=1e-6)
    # 個別上下限
    parser.add_argument("--ub-mult", type=float, default=2.0, help="個別上限: m_i ≤ ub_mult * initial")
    parser.add_argument("--lb-div",  type=float, default=2.0, help="個別下限: m_i ≥ floor(initial / lb_div)")
    # SAパラメタ（単発／内部で使用）
    parser.add_argument("--iterations",   type=int,   default=1500)
    parser.add_argument("--initial-temp", type=float, default=90.0)
    parser.add_argument("--cooling-rate", type=float, default=0.98)
    parser.add_argument("--neighbor-max-step", type=int, default=3)
    parser.add_argument("--nodes-per-move-low",  type=int, default=1)
    parser.add_argument("--nodes-per-move-high", type=int, default=3)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--trace-mode", choices=["accept","best","all"], default="accept")
    parser.add_argument("--progress-interval", type=int, default=20, help="進捗行の出力間隔（イテレーション）")
    parser.add_argument("--save-every", type=int, default=100, help="チェックポイント保存間隔（イテレーション）0で無効")
    parser.add_argument("--log-file", type=str, default=None, help="詳細ログを保存するファイル（out_dir/sa_debug.log 推奨）")

    # レキシコ（二段階）最適化
    parser.add_argument("--lexi", action="store_true",
                        help="二段階（W*→B）で実行。Stage1: W*探索（n_unstable=0）、Stage2: B≤…でCV^2最小化")
    parser.add_argument("--lexi-step-ratio", type=float, default=0.05,
                        help="Stage1ラフ探索で上限を増やす比率（例: 0.05 で +5%）")
    parser.add_argument("--lexi-max-mult",   type=float, default=2.0,
                        help="（将来用）Stage1ラフ探索の上限倍率（現在はループ回数で制御）")
    parser.add_argument("--lexi-iters-warm", type=int, default=1200,
                        help="Stage1で用いるSAのイテレーション数")
    parser.add_argument("--lexi-iters-fine", type=int, default=2000,
                        help="Stage2で用いるSAのイテレーション数")

    # 第2段階の予算（B）指定
    parser.add_argument("--budget-ratio", type=float, default=1.2,
        help="第2段階の総窓口上限 B を W* の何倍にするか（既定=1.2）")
    parser.add_argument("--budget-absolute", type=int,
        help="第2段階の上限 B を絶対値で指定（指定時は ratio より優先）")

    # ここがポイント：等式/不等式と安定必須
    parser.add_argument(
        "--use-full-budget", action="store_true",
        help="第2段階で総窓口数を上限にぴったり合わせる（等式 =B）。付けない場合は ≤B（既定）。"
    )
    parser.add_argument(
        "--require-stable-stage2", action="store_true",
        help="第2段階の探索中、n_unstable>0 の近傍を棄却（探索を常に安定領域に限定）"
    )

    # 表示/保存
    parser.add_argument("--show-distribution", action="store_true",
                        help="初期窓口数の分布（層別合計）を表示")
    parser.add_argument("--save-report", action="store_true",
                        help="最終結果のテキストレポートを out_dir/optimization_report.txt に保存")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--show-decision-vars", action="store_true",
                        help="最適化の決定変数となるFCFSノード一覧を表示し、CSVに保存する")

    args = parser.parse_args()

    # out_dir を作成して log_file の既定パスを用意
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    default_log = str(Path(args.out_dir) / "sa_debug.log")
    log_file = args.log_file or default_log

    # オプティマイザ生成（既存の引数は全て維持）
    opt = BCMPSAOptimizer(
        param_dir=args.param_dir,
        out_dir=args.out_dir,
        alpha_aid=args.alpha_aid,
        alpha_amb=args.alpha_amb,
        alpha_hosp=args.alpha_hosp,
        total_windows_limit=None,  # レキシコ内で設定
        weight_unstable=args.weight_unstable,
        weight_window_cost=args.weight_window_cost,
        stability_eps=args.stability_eps,
        individual_ub_mult=args.ub_mult,
        individual_lb_div=args.lb_div,
        verbose=args.verbose,
        log_file=log_file
    )

    if args.show_distribution:
        opt.show_initial_distribution()

    if args.show_decision_vars:
        opt.show_decision_variables(
            save_csv=str(Path(args.out_dir) / "decision_variables.csv")
        )


    start = time.time()

    if args.lexi:
        # 二段階（Stage1→Stage2）
        sol, obj, comp, W_star, B = opt.lexicographic_optimize_sa(
            base_limit=sum(opt.initial_windows.values()),
            step_ratio=args.lexi_step_ratio,
            max_mult=args.lexi_max_mult,
            sa_iters_warm=args.lexi_iters_warm,
            sa_iters_fine=args.lexi_iters_fine,
            sa_temp=args.initial_temp,
            sa_cool=args.cooling_rate,
            seed=args.seed,
            budget_ratio=args.budget_ratio,
            budget_absolute=args.budget_absolute,
            use_full_budget=args.use_full_budget,                # ← 付けたときだけ等式 =B
            progress_interval=args.progress_interval,
            save_every=args.save_every,
            require_stable_stage2=args.require_stable_stage2     # ← 任意で安定必須探索
        )
        print(f"\n=== 段階的最適化完了 ===\nW*={W_star}, B={B}, obj={obj:.6f}, "
              f"n_unstable={comp['n_unstable']}, total_windows={comp['total_windows']}", flush=True)
    else:
        # 単発SA（参考運用）：total_windows_limitを任意で設定したい場合は --budget-absolute を使う
        if args.budget_absolute is not None:
            opt.total_windows_limit = int(args.budget_absolute)
            if args.verbose:
                print(f"[single-SA] 総窓口数上限を {opt.total_windows_limit} に設定（単発）", flush=True)

        sol, obj = opt.optimize_with_simulated_annealing(
            max_iterations=args.iterations,
            initial_temp=args.initial_temp,
            cooling_rate=args.cooling_rate,
            neighbor_max_step=args.neighbor_max_step,
            nodes_per_move_low=args.nodes_per_move_low,
            nodes_per_move_high=args.nodes_per_move_high,
            seed=args.seed,
            trace_filename="sa_trace.csv",
            trace_mode=args.trace_mode,
            enforce_total_exact=False,              # 単発の既定は ≤ 上限
            progress_interval=args.progress_interval,
            save_every=args.save_every
        )
        comp = opt.compute_objective_components(sol)
        print(f"\n=== SA最適化完了（単発） ===\nobj={obj:.6f}, "
              f"n_unstable={comp['n_unstable']}, total_windows={comp['total_windows']}", flush=True)

    elapsed = time.time() - start
    print(f"\n=== 最適化完了 (経過時間: {elapsed:.2f}秒) ===", flush=True)

    # ---------- ここだけ最小変更（レポート整合のため） ----------
    old_wc_for_report = opt.weight_window_cost
    opt.weight_window_cost = 0.01   # ★決め打ち（Stage2と同じ重みで再計算されるように）
    note = "Stage2 window_cost_weight=0.01"
    # -----------------------------------------------------------

    # レポート
    if args.save_report:
        report_file = os.path.join(args.out_dir, "optimization_report.txt")
        opt.create_result_report(sol, obj, output_file=report_file)
    else:
        opt.create_result_report(sol, obj)

    # ---------- 復元を忘れない ----------
    opt.weight_window_cost = old_wc_for_report
    # -------------------------------------

    # 解の保存（簡易）
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    solution_file = os.path.join(args.out_dir, "optimal_windows.csv")
    df = pd.DataFrame({
        'node_id': list(sol.keys()),
        'initial_windows': [opt.initial_windows.get(n, 1) for n in sol.keys()],
        'optimal_windows': list(sol.values()),
        'diff': [sol[n] - opt.initial_windows.get(n, 1) for n in sol.keys()]
    })
    df['node_type'] = df['node_id'].apply(opt._get_node_type)
    df = df.sort_values(['node_type', 'node_id'])
    df.to_csv(solution_file, index=False)
    print(f"最適窓口数を保存しました: {solution_file}", flush=True)


if __name__ == "__main__":
    main()

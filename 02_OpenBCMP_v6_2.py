# open_bcmp.py
# Open BCMP (open network, U=1) with optional class switching
# - Parameters are loaded from ./parameters/
# - Results are written to ./outputs/
# - No CLI args; edit variables in __main__ as needed.

import os
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle


class OpenBCMP:
    """
    Open BCMP network with possible class switching (U=1).

    Required CSVs (paths or DataFrames):
      - routing_labels.csv : (NR+1)x(NR+1) matrix with headers/indices; last row/col is 'outside'
                             Rows and columns must have identical labels and order.
                             Internal labels are '(i,r)' strings; the last label must be 'outside'.
      - external.csv       : columns [node, class, lambda0] (external arrival rates λ0_ir)
      - service.csv        : columns [node, class, service_type, mu, m]
                               * Type 'FCFS' must be given once per node with class='*' and a shared mu (and optional m).
                               * Types 'PS','IS','LCFS-PR' are class-dependent; give one row per (node, class).
      - index_map.csv      : optional documentation (columns [idx, node, class, label]); not required by the solver.
    """

    def __init__(self, routing_labels, external, service, index_map=None, verbose=True):
        self.verbose = verbose
        self.routing = self._load_csv(routing_labels, index_col=0)
        self.external = self._load_csv(external)
        # service.csvは列名を正規化して読み込む
        self.service = self._load_csv(service, normalize_cols=True)
        # index_mapは使用しない（将来的に削除可能）
        self.index_map = None
    
        self._parse_index()
        self._validate_inputs()
        self._prepare_service_specs()
        self.class_switching_present = self._detect_class_switching()

        # placeholders for results
        self.lambda_ir_vec = None
        self.alpha_ir_vec  = None
        self.lambda_ir = {}
        self.alpha_ir  = {}
        self.lambda_i  = {}
        self.alpha_i   = {}
        self.rho_ir    = {}
        self.K_ir      = {}
        self.T_ir      = {}
        self.node_rows = []
        self.class_rows = []
        self.diag_rows = []
        self.runinfo = {}

    # ---------- IO helpers ----------
    def _load_csv(self, src, index_col=None, normalize_cols=False):
        """
        CSVファイルまたはDataFrameを読み込む
        
        Args:
            src: ファイルパスまたはDataFrame
            index_col: インデックスとして使用する列
            normalize_cols: 列名を正規化するかどうか
        """
        if src is None:
            return None
        
        if isinstance(src, pd.DataFrame):
            df = src.copy()
        else:
            df = pd.read_csv(src, index_col=index_col)
        
        # 列名の正規化（service.csv用）
        if normalize_cols:
            col_mapping = {
                'node_id': 'node'
            }
            df.rename(columns=col_mapping, inplace=True)
        
        return df

    # ---------- Parse and validate ----------
    def _parse_index(self):
        """
        ルーティング行列のラベルをパースし、ノード番号とクラス番号を抽出
        6桁ノード番号体系に対応:
        - 10万番台: 地域ノード
        - 20万番台: 移動ノード  
        - 30万番台: 救護所ノード
        - 40万番台: 救護所発救急車
        - 50万番台: 救護病院
        - 60万番台: 災害拠点病院
        - 70万番台: 転送専用救急車
        - 90万番台: システムノード(出口)
        """
        # ルーティング行列の列ラベルを取得
        self.labels = list(self.routing.columns)
        if len(self.labels) < 2:
            raise ValueError("routing_labels.csv must include at least one internal state and 'outside'.")
        if self.labels[-1] != "outside":
            raise ValueError("Last column/label in routing must be 'outside'. Found: %s" % self.labels[-1])
        if self.routing.index.tolist() != self.labels:
            raise ValueError("Row and column labels of routing must match exactly and in the same order.")

        self.outside_label = "outside"
        self.M = len(self.labels) - 1  # 内部状態の数

        # "(node_id,class)" -> (node_id, class) のパース関数
        def parse_ir(label):
            """ラベル文字列から(ノード番号, クラス番号)を抽出"""
            if label == "outside":
                return (0, 0)  # outsideは特殊なノード
            
            # "(100001,1)" 形式の検証
            if not (label.startswith("(") and label.endswith(")")):
                raise ValueError(f"Bad state label: {label} (expected '(node_id,class)' or 'outside')")
            
            # カッコを取り除いて中身を取得
            body = label[1:-1]
            parts = [x.strip() for x in body.split(",")]
            
            if len(parts) != 2:
                raise ValueError(f"Bad '(node_id,class)' label: {label}")
            
            try:
                node_id = int(parts[0])
                class_id = int(parts[1])
            except ValueError:
                raise ValueError(f"Cannot parse integers from label: {label}")
            
            return (node_id, class_id)

        # 内部状態のラベルリストとマッピングを作成
        self.int_labels = self.labels[:-1]  # 'outside'を除く
        self.ir_list = [parse_ir(lab) for lab in self.int_labels]   # [(node_id, class),...]
        
        # (node_id, class) -> 位置インデックス のマッピング
        self.ir2pos = {ir: pos for pos, ir in enumerate(self.ir_list)}
        # 位置インデックス -> (node_id, class) のマッピング
        self.pos2ir = {pos: ir for pos, ir in enumerate(self.ir_list)}
        
        # ユニークなノードIDとクラスIDを抽出
        self.nodes = sorted({node_id for (node_id, class_id) in self.ir_list})
        self.classes = sorted({class_id for (node_id, class_id) in self.ir_list})
        
        self.N = len(self.nodes)  # ノード数
        self.R = len(self.classes)  # クラス数

        # ルーティング行列をサブブロックに分割
        P_full = self.routing.values
        self.P_II = P_full[:self.M, :self.M]     # 内部状態間の遷移
        self.P_Io = P_full[:self.M,  self.M]     # 内部状態 -> outside
        self.P_oI = P_full[ self.M, :self.M]     # outside -> 内部状態 (ゼロであるべき)
        self.P_oo = P_full[ self.M,  self.M]     # outside -> outside (1であるべき)

        # 外部到着率ベクトル λ0_ir を構築
        lam0 = np.zeros(self.M, dtype=float)
        for _, row in self.external.iterrows():
            node_id = int(row["node"])
            class_id = int(row["class"])
            val = float(row["lambda0"])
            
            if (node_id, class_id) in self.ir2pos:
                lam0[self.ir2pos[(node_id, class_id)]] += val
        
        self.lambda0 = lam0
        self.Lambda = float(np.sum(lam0))  # 全体の外部到着率

        if self.verbose:
            print(f"[INFO] Parsed {self.M} internal states: {self.N} nodes, {self.R} classes")
            print(f"[INFO] Total external arrival rate Lambda = {self.Lambda:.6f}")

    def _validate_inputs(self):
        # routing: nonnegativity and row sums == 1
        mat = self.routing.values
        if np.any(mat < -1e-12):
            mn = float(mat.min())
            raise ValueError(f"Routing has negative entries (min={mn}).")
        row_sums = mat.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-10):
            raise ValueError(f"Each routing row must sum to 1. Found: {row_sums.round(6).tolist()}")
        # outside row should be absorbing
        if not (np.allclose(self.P_oI, 0.0) and np.isclose(self.P_oo, 1.0)):
            raise ValueError("The 'outside' row must be absorbing: [0...0,1].")

        # service types valid?
        allowed = {"FCFS", "PS", "IS", "LCFS-PR"}
        bad = self.service[~self.service["service_type"].str.upper().isin(allowed)]
        if len(bad) > 0:
            raise ValueError(f"Unknown service_type(s): {bad['service_type'].unique()}")

    def _prepare_service_specs(self):
        """サービス仕様を解析・準備"""
        self.node_type = {}   # i -> {'type': str, 'mu': float, 'm': int}
        self.mu_ir     = {}   # (i,r) -> mu
        
        # ★ 修正: service.csvに登場する全ノードで初期化
        all_service_nodes = self.service["node"].unique()
        for i in all_service_nodes:
            i = int(i)
            if i not in self.node_type:
                self.node_type[i] = {'type': None, 'mu': None, 'm': None}
        
        for _, row in self.service.iterrows():
            typ = str(row["service_type"]).strip().upper()
            i   = int(row["node"])
            mu  = float(row["mu"])
            
            # m列の処理（"inf"文字列と数値の両方に対応）
            mval = row.get("m", "")
            if pd.isna(mval):
                m = None  # NaNの場合
            elif isinstance(mval, str):
                mval_clean = mval.strip().lower()
                if mval_clean == "inf" or mval_clean == "":
                    m = None  # 無限サーバーまたは未指定
                elif mval_clean.replace(".", "", 1).isdigit():
                    m = int(float(mval_clean))
                else:
                    m = None
            elif isinstance(mval, (int, float)):
                if np.isinf(mval):  # 無限大の場合
                    m = None
                else:
                    m = int(mval)
            else:
                m = None
            
            if typ == "FCFS":
                # FCFSはclass="*"で全クラス共通
                self.node_type[i] = {'type': 'FCFS', 'mu': mu, 'm': (m if m is not None else 1)}
                
            elif typ in {"PS", "IS", "LCFS-PR"}:
                # node_typeを設定（初回のみ）
                if self.node_type[i]['type'] is None:
                    self.node_type[i]['type'] = typ
                elif self.node_type[i]['type'] != typ:
                    if self.verbose:
                        print(f"[WARN] Node {i} has mixed service types: {self.node_type[i]['type']} and {typ}")
                
                # class列の処理（重要:必ず文字列化）
                c = str(row.get("class", "*")).strip()
                if c == "*":
                    # ISノード: 全クラスに同じmuを設定
                    for r in self.classes:
                        self.mu_ir[(i, r)] = mu
                else:
                    # PSノード: クラス別にmuを設定
                    try:
                        r = int(c)
                        self.mu_ir[(i, r)] = mu
                    except (ValueError, TypeError):
                        if self.verbose:
                            print(f"[WARN] Cannot parse class value '{c}' at node {i}")
            else:
                raise ValueError(f"Unsupported service_type: {typ}")
        
        # ★ 修正: self.nodesに存在するノードのみチェック
        for i in self.nodes:
            if i not in self.node_type:
                # ルーティングに登場するがservice.csvに無いノード
                if self.verbose:
                    print(f"[WARN] Node {i} appears in routing but not in service.csv")
                continue
                
            if self.node_type[i]['type'] == 'FCFS':
                if self.node_type[i]['mu'] is None:
                    raise ValueError(f"FCFS node {i} needs mu value")
            elif self.node_type[i]['type'] in {'PS', 'IS', 'LCFS-PR'}:
                for r in self.classes:
                    if (i, r) in self.ir2pos and (i, r) not in self.mu_ir:
                        if self.verbose:
                            print(f"[WARN] Node {i}, class {r}: traffic exists but no mu defined")
                        self.mu_ir[(i, r)] = np.nan
            elif self.node_type[i]['type'] is None:
                raise ValueError(f"Node {i} has no service_type defined in service.csv")
        
    def _detect_class_switching(self):
        # If any internal transition changes class (s -> r with r != s), we regard as class switching present
        switching = False
        for from_pos, (j, s) in enumerate(self.ir_list):
            row = self.P_II[from_pos, :]
            nz = np.where(row > 1e-15)[0]
            for col in nz:
                i, r = self.pos2ir[col]
                if r != s:
                    switching = True
                    return switching
        return switching

    # ---------- Solve traffic equations ----------
    def solve_lambda(self):
        """
        Solve row-vector lambda over internal states from:
            lambda = lambda0 + lambda * P_II
        i.e., lambda * (I - P_II) = lambda0.
        We solve (I - P_II)^T x = lambda0^T for x = lambda^T.
        """
        I = np.eye(self.M)
        A = (I - self.P_II).T
        b = self.lambda0.copy()
        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            x = np.linalg.lstsq(A, b, rcond=None)[0]
        self.lambda_ir_vec = x
        self.alpha_ir_vec  = x / (self.Lambda if self.Lambda > 0 else 1.0)
        # unpack to dicts and per-node sums
        self.lambda_ir = {}
        self.alpha_ir  = {}
        self.lambda_i  = {i: 0.0 for i in self.nodes}
        self.alpha_i   = {i: 0.0 for i in self.nodes}
        for pos in range(self.M):
            i, r = self.pos2ir[pos]
            lam = float(self.lambda_ir_vec[pos])
            alp = float(self.alpha_ir_vec[pos])
            self.lambda_ir[(i, r)] = lam
            self.alpha_ir[(i, r)]  = alp
            self.lambda_i[i] += lam
            self.alpha_i[i]  += alp

    # ---------- Node metric kernels ----------
    def _mmm_metrics(self, lam_i, mu_i, m_i):
        """M/M/m metrics: returns dict with rho, pi0, Pwait, Lq, Wq, W, L, a, unstable"""
        if mu_i <= 0 or m_i <= 0:
            return dict(rho=np.inf, pi0=np.nan, Pwait=np.nan, Lq=np.inf, Wq=np.inf, W=np.inf, L=np.inf, a=np.inf, unstable=True)
        a   = lam_i / mu_i
        rho = lam_i / (m_i * mu_i)
        if rho >= 1 - 1e-12:
            return dict(rho=rho, pi0=np.nan, Pwait=np.nan, Lq=np.inf, Wq=np.inf, W=np.inf, L=np.inf, a=a, unstable=True)
        # pi0
        s = 0.0
        for k in range(m_i):
            s += a**k / math.factorial(k)
        tail = a**m_i / (math.factorial(m_i) * (1 - rho))
        pi0 = 1.0 / (s + tail)
        Pwait = a**m_i / (math.factorial(m_i) * (1 - rho)) * pi0
        Lq = (rho / (1 - rho)) * Pwait
        Wq = Lq / lam_i if lam_i > 0 else 0.0
        W  = Wq + 1.0 / mu_i
        L  = m_i * rho + Lq
        return dict(rho=rho, pi0=pi0, Pwait=Pwait, Lq=Lq, Wq=Wq, W=W, L=L, a=a, unstable=False)

    # ---------- Compute metrics ----------
    def compute_metrics(self):
        self.node_rows = []
        self.class_rows = []
        self.rho_ir.clear(); self.K_ir.clear(); self.T_ir.clear()

        for i in self.nodes:
            typ = (self.node_type[i]['type'] or '').upper()
            lam_i = self.lambda_i.get(i, 0.0)

            if typ == "FCFS":
                mu_i = float(self.node_type[i]['mu'])
                m_i  = int(self.node_type[i]['m'] or 1)
                mm = self._mmm_metrics(lam_i, mu_i, m_i)
                # class-wise (same W for all classes; μ is class-independent)
                for r in self.classes:
                    lam_ir = self.lambda_ir.get((i, r), 0.0)
                    self.rho_ir[(i, r)] = lam_ir / mu_i if mu_i > 0 else np.inf
                    self.T_ir[(i, r)]   = mm["W"] if np.isfinite(mm["W"]) else np.inf
                    self.K_ir[(i, r)]   = lam_ir * self.T_ir[(i, r)]
                    self.class_rows.append(dict(node=i, cls=r, lambda_ir=lam_ir, mu_ir=mu_i,
                                                rho_ir=self.rho_ir[(i, r)], K_ir=self.K_ir[(i, r)], T_ir=self.T_ir[(i, r)]))
                self.node_rows.append(dict(node=i, type=typ, m=m_i, lambda_i=lam_i, mu=mu_i,
                                           rho_i=mm["rho"], W=mm["W"], Wq=mm["Wq"], L=mm["L"], Lq=mm["Lq"],
                                           P_wait=mm["Pwait"], pi0=mm["pi0"]))

            elif typ in {"PS", "LCFS-PR"}:
                # compute rho_i
                rho_i = 0.0
                for r in self.classes:
                    lam_ir = self.lambda_ir.get((i, r), 0.0)
                    mu_ir  = self.mu_ir.get((i, r), np.nan)
                    self.rho_ir[(i, r)] = lam_ir / mu_ir if (lam_ir > 0 and np.isfinite(mu_ir) and mu_ir > 0) else 0.0
                    rho_i += self.rho_ir[(i, r)]
                if rho_i >= 1 - 1e-12:
                    W_i = np.inf; L_i = np.inf; Wq_i = np.inf; Lq_i = np.inf
                    for r in self.classes:
                        lam_ir = self.lambda_ir.get((i, r), 0.0)
                        self.T_ir[(i, r)] = np.inf if lam_ir > 0 else 0.0
                        self.K_ir[(i, r)] = np.inf if lam_ir > 0 else 0.0
                else:
                    # class-wise metrics
                    L_i = 0.0
                    for r in self.classes:
                        lam_ir = self.lambda_ir.get((i, r), 0.0)
                        mu_ir  = self.mu_ir.get((i, r), np.nan)
                        if lam_ir > 0 and np.isfinite(mu_ir) and mu_ir > 0:
                            K_ir = self.rho_ir[(i, r)] / (1 - rho_i)
                            T_ir = K_ir / lam_ir
                        else:
                            K_ir = 0.0; T_ir = 0.0
                        self.K_ir[(i, r)] = K_ir
                        self.T_ir[(i, r)] = T_ir
                        L_i += K_ir
                    W_i  = L_i / lam_i if lam_i > 0 else 0.0
                    Wq_i = np.nan   # PS/LCFS-PR では「純粋待ち時間」の分離は定義しない
                    Lq_i = np.nan
                # append rows
                for r in self.classes:
                    lam_ir = self.lambda_ir.get((i, r), 0.0)
                    mu_ir  = self.mu_ir.get((i, r), np.nan)
                    self.class_rows.append(dict(node=i, cls=r, lambda_ir=lam_ir, mu_ir=mu_ir,
                                                rho_ir=self.rho_ir[(i, r)], K_ir=self.K_ir[(i, r)], T_ir=self.T_ir[(i, r)]))
                self.node_rows.append(dict(node=i, type=typ, m="", lambda_i=lam_i, mu="",
                                           rho_i=rho_i, W=W_i, Wq=Wq_i, L=L_i, Lq=Lq_i, P_wait="", pi0=""))

            elif typ == "IS":
                # Infinite server: no waiting; K_ir = λ_ir / μ_ir, T_ir = 1/μ_ir
                L_i = 0.0
                for r in self.classes:
                    lam_ir = self.lambda_ir.get((i, r), 0.0)
                    mu_ir  = self.mu_ir.get((i, r), np.nan)
                    if lam_ir > 0 and np.isfinite(mu_ir) and mu_ir > 0:
                        K_ir = lam_ir / mu_ir
                        T_ir = 1.0 / mu_ir
                    else:
                        K_ir = 0.0; T_ir = 0.0
                    self.rho_ir[(i, r)] = K_ir  # for IS, we record K_ir as 'load'
                    self.K_ir[(i, r)] = K_ir
                    self.T_ir[(i, r)] = T_ir
                    L_i += K_ir
                W_i = (L_i / lam_i) if lam_i > 0 else 0.0
                self.node_rows.append(dict(node=i, type=typ, m="", lambda_i=lam_i, mu="",
                                           rho_i="", W=W_i, Wq=0.0, L=L_i, Lq=0.0, P_wait="", pi0=""))
                for r in self.classes:
                    lam_ir = self.lambda_ir.get((i, r), 0.0)
                    mu_ir  = self.mu_ir.get((i, r), np.nan)
                    self.class_rows.append(dict(node=i, cls=r, lambda_ir=lam_ir, mu_ir=mu_ir,
                                                rho_ir=self.rho_ir[(i, r)], K_ir=self.K_ir[(i, r)], T_ir=self.T_ir[(i, r)]))
            else:
                raise ValueError(f"Node {i} has unknown/unspecified service_type.")

    # ---------- Summaries and saving ----------
    def build_dataframes(self):
        # traffic solution
        traf_rows = []
        for (i, r), lam in self.lambda_ir.items():
            traf_rows.append(dict(node=i, cls=r,
                                  alpha_ir=self.alpha_ir[(i, r)], lambda_ir=lam,
                                  alpha_i=self.alpha_i[i], lambda_i=self.lambda_i[i]))
        df_traf = pd.DataFrame(traf_rows).sort_values(["node", "cls"]).reset_index(drop=True)

        # node metrics and class-node metrics
        df_node  = pd.DataFrame(self.node_rows).sort_values("node").reset_index(drop=True)
        df_cls   = pd.DataFrame(self.class_rows).sort_values(["node", "cls"]).reset_index(drop=True)

        # network totals
        L_net_total = float(df_cls["K_ir"].replace([np.inf, -np.inf], np.nan).sum())
        W_net_overall = L_net_total / self.Lambda if self.Lambda > 0 else np.nan

        # per-class network summaries
        class_summ_rows = []
        # external per class
        lambda0_class = {r: 0.0 for r in self.classes}
        for _, row in self.external.iterrows():
            lambda0_class[int(row["class"])] += float(row["lambda0"])

        for r in self.classes:
            # totals over nodes
            Ki_sum = float(df_cls[df_cls["cls"] == r]["K_ir"].replace([np.inf, -np.inf], np.nan).sum())
            Lam_r  = lambda0_class.get(r, 0.0)
            if (not self.class_switching_present) and Lam_r > 0:
                W_net_r = Ki_sum / Lam_r
            else:
                W_net_r = np.nan  # class switching present or zero arrival; cannot infer per-class R easily
            class_summ_rows.append(dict(cls=r, Lambda_r=Lam_r, L_net=Ki_sum, W_net=W_net_r))

        df_classnet = pd.DataFrame(class_summ_rows).sort_values("cls").reset_index(drop=True)

        # run summary
        self.runinfo = dict(
            N=self.N, R=self.R, Lambda=self.Lambda,
            class_switching_present=bool(self.class_switching_present),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        df_run = pd.DataFrame([self.runinfo])

        return df_run, df_traf, df_node, df_cls, df_classnet

    def diagnostics(self):
        diags = []
        # routing checks already validated; we still report min/max and row sums
        row_sums = self.routing.values.sum(axis=1)
        for idx, rs in enumerate(row_sums):
            diags.append(dict(check="row_sum", row_label=self.labels[idx], value=float(rs),
                              status="pass" if abs(rs - 1.0) < 1e-10 else "fail", notes="routing row sum"))
        # stability (Type 1/2/4)
        for nr in self.node_rows:
            t = str(nr["type"]).upper()
            if t in {"FCFS", "PS", "LCFS-PR"}:
                rho = nr["rho_i"]
                st = "pass" if (isinstance(rho, (int, float)) and rho < 1.0) else "fail"
                diags.append(dict(check="stability", node=int(nr["node"]), value=float(rho) if isinstance(rho,(int,float)) else np.nan,
                                  status=st, notes=f"type={t}"))
        return pd.DataFrame(diags)

    def save_outputs(self, out_dir, make_excel=True):
        os.makedirs(out_dir, exist_ok=True)
        df_run, df_traf, df_node, df_cls, df_classnet = self.build_dataframes()
        df_diag = self.diagnostics()

        df_run.to_csv(os.path.join(out_dir, "run_summary.csv"), index=False)
        df_traf.to_csv(os.path.join(out_dir, "traffic_solution.csv"), index=False)
        df_node.to_csv(os.path.join(out_dir, "node_metrics.csv"), index=False)
        df_cls.to_csv(os.path.join(out_dir, "class_node_metrics.csv"), index=False)
        df_classnet.to_csv(os.path.join(out_dir, "network_class_summary.csv"), index=False)
        df_diag.to_csv(os.path.join(out_dir, "diagnostics.csv"), index=False)

        if make_excel:
            try:
                xls_path = os.path.join(out_dir, "bcmp_results.xlsx")
                with pd.ExcelWriter(xls_path, engine="xlsxwriter") as writer:
                    df_run.to_excel(writer, sheet_name="run_summary", index=False)
                    df_traf.to_excel(writer, sheet_name="traffic_solution", index=False)
                    df_node.to_excel(writer, sheet_name="node_metrics", index=False)
                    df_cls.to_excel(writer, sheet_name="class_node_metrics", index=False)
                    df_classnet.to_excel(writer, sheet_name="network_class_summary", index=False)
                    self.routing.to_excel(writer, sheet_name="routing_labels")
                    if self.index_map is not None:
                        self.index_map.to_excel(writer, sheet_name="index_map", index=False)
                    self.external.to_excel(writer, sheet_name="external", index=False)
                    self.service.to_excel(writer, sheet_name="service", index=False)
                    self.diagnostics().to_excel(writer, sheet_name="diagnostics", index=False)
            except Exception as e:
                if self.verbose:
                    print(f"[WARN] Excel export failed: {e}")

    def print_summary(self, top_k=5):
        df_run, df_traf, df_node, df_cls, df_classnet = self.build_dataframes()
        print("=== BCMP Run Summary ===")
        print(f"N={self.N}, R={self.R}, Lambda={self.Lambda:.6f}, class_switching={self.class_switching_present}")

        # Top busy nodes
        tmp = df_node.copy()
        tmp["rho_sort"] = pd.to_numeric(tmp["rho_i"], errors="coerce")
        tmp = tmp.sort_values("rho_sort", ascending=False).head(top_k)
        print("\n-- Top busy nodes --")
        if len(tmp) > 0:
            print(tmp[["node", "type", "m", "lambda_i", "rho_i"]].to_string(index=False))
        else:
            print("(no nodes)")

        # K_ir table (node x class)
        k_tbl = df_cls.pivot(index="node", columns="cls", values="K_ir").fillna(0.0)
        print("\n-- K_ir (mean number at node i, class r) --")
        # 丸めは見やすさ用。必要なら桁数を変えてください
        print(k_tbl.round(6).to_string())

        # Network totals
        L_net_total = float(df_cls["K_ir"].replace([np.inf, -np.inf], np.nan).sum())
        W_net_overall = L_net_total / self.Lambda if self.Lambda > 0 else float("nan")
        print(f"\nOverall: L_net_total={L_net_total:.6f}, W_net_overall={W_net_overall:.6f}")


    # --- OpenBCMP class: add this helper (exact per-state routing aggregation is not needed here) ---
    def _layout_positions(self, radius=3.4):
        """位置座標を用意：内部状態は円周、Inputは左、Outsideは右。"""
        import numpy as np
        L = len(self.int_labels)
        xs, ys = [], []
        for k in range(max(L, 1)):
            th = 2*np.pi*k/max(L, 1)
            xs.append(radius*np.cos(th))
            ys.append(radius*np.sin(th))
        pos = {lab: (xs[i], ys[i]) for i, lab in enumerate(self.int_labels)}
        # extra nodes
        pos_input   = (-radius - 2.0, 0.0)
        pos_outside = ( radius + 2.0, 0.0)
        return pos, pos_input, pos_outside

    def draw_network_exact(self,
                        out_path=None,
                        prob_threshold=0.0,      # これ未満の枝は描かない
                        figsize=(10, 7),
                        decimals_p=2,            # 確率の表示桁
                        decimals_lam=2,          # 到着率の表示桁
                        node_fontsize=10,
                        edge_fontsize=9):
        """
        パラメタから “正確に” 図を生成：
        - 頂点：各内部状態 '(i,r)'、Input、Outside
        - Edge(Input -> (i,r))：外部到着率 λ0_ir
        - Edge((j,s) -> (i,r))：routing_labels の確率
        - Edge((j,s) -> Outside)：routing_labels の最終列の確率
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, Circle

        # 位置
        pos, pos_input, pos_out = self._layout_positions(radius=3.6)

        # 事前に min/max を集めて表示範囲を決める
        xs_all = [x for (x, y) in pos.values()] + [pos_input[0], pos_out[0]]
        ys_all = [y for (x, y) in pos.values()] + [pos_input[1], pos_out[1]]

        # Figure
        try:
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        except TypeError:
            fig, ax = plt.subplots(figsize=figsize)
            fig.subplots_adjust(left=0.06, right=0.94, top=0.94, bottom=0.06)
        fig.patch.set_facecolor("white")
        ax.set_aspect("equal"); ax.axis("off")

        # --- helper: 曲線の外側にラベルを置く ---
        def place_label(x1, y1, x2, y2, rad, text, color="#333"):
            xm, ym = 0.5*(x1+x2), 0.5*(y1+y2)
            dx, dy = (x2-x1), (y2-y1)
            nx, ny = -dy, dx
            L = np.hypot(nx, ny) or 1.0
            off = 0.28 * (1 if rad >= 0 else -1)
            ax.text(xm + off*nx/L, ym + off*ny/L, text,
                    fontsize=edge_fontsize, color=color,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
                    zorder=3)

        # --- ノード（内部状態） ---
        for lab, (x, y) in pos.items():
            ax.add_patch(Circle((x, y), 0.28, fc="#F7F7F7", ec="#333", zorder=2))
            ax.text(x, y, lab, ha="center", va="center", fontsize=node_fontsize, zorder=3)

        # --- Input / Outside ノード ---
        ax.add_patch(Circle(pos_input,  0.30, fc="#EEE", ec="#333", zorder=2))
        ax.add_patch(Circle(pos_out,    0.30, fc="#EEE", ec="#333", zorder=2))
        ax.text(pos_input[0], pos_input[1], "Input",   ha="center", va="center", fontsize=node_fontsize, zorder=3)
        ax.text(pos_out[0],   pos_out[1],   "Outside", ha="center", va="center", fontsize=node_fontsize, zorder=3)

        # --- Edge: 外部到着 Input -> (i,r)  ---
        # λ0_ir をそのまま描く（太さはλ0の相対量に比例）
        lam0_map = {lab: 0.0 for lab in self.int_labels}
        for _, row in self.external.iterrows():
            lab = f"({int(row['node'])},{int(row['class'])})"
            if lab in lam0_map:
                lam0_map[lab] += float(row["lambda0"])
        lam0_vals = np.array(list(lam0_map.values()), dtype=float)
        max_lam0  = float(lam0_vals.max()) if lam0_vals.size > 0 else 0.0

        for lab, lam0 in lam0_map.items():
            if lam0 <= 0.0:
                continue
            (x1, y1) = pos_input; (x2, y2) = pos[lab]
            lw = 1.2 + (4.8 * (lam0 / max_lam0 if max_lam0 > 0 else 0.0))
            arr = FancyArrowPatch((x1, y1), (x2, y2),
                                connectionstyle="arc3,rad=0.06",
                                arrowstyle="-|>",
                                mutation_scale=20, shrinkA=20, shrinkB=20,
                                lw=lw, color="#2ca02c", zorder=1)   # 緑
            ax.add_patch(arr)
            place_label(x1, y1, x2, y2, 0.06, r"$\lambda_{0}=$"+f"{lam0:.{decimals_lam}f}", color="#2ca02c")

        # --- Edge: (j,s) -> (i,r) と (j,s) -> Outside ---
        # 行列そのまま（確率）
        for fr in self.int_labels:
            x1, y1 = pos[fr]
            # to internal states
            for to in self.int_labels:
                p = float(self.routing.loc[fr, to])
                if p <= prob_threshold: 
                    continue
                x2, y2 = pos[to]
                rad = 0.22 if fr < to else (-0.22 if fr > to else 0.36)  # 自己遷移は大きめカーブ
                arr = FancyArrowPatch((x1, y1), (x2, y2),
                                    connectionstyle=f"arc3,rad={rad}",
                                    arrowstyle="-|>",
                                    mutation_scale=20, shrinkA=20, shrinkB=20,
                                    lw=1.0 + 6.0*p, color="#1f77b4", alpha=0.95, zorder=1)  # 青
                ax.add_patch(arr)
                place_label(x1, y1, x2, y2, rad, f"{p:.{decimals_p}f}", color="#1f77b4")
            # to Outside
            p_out = float(self.routing.loc[fr, "outside"])
            if p_out > prob_threshold:
                x2, y2 = pos_out
                arr = FancyArrowPatch((x1, y1), (x2, y2),
                                    connectionstyle="arc3,rad=0.08",
                                    arrowstyle="-|>",
                                    mutation_scale=20, shrinkA=20, shrinkB=20,
                                    lw=1.0 + 6.0*p_out, color="#d62728", alpha=0.95, zorder=1)  # 赤
                ax.add_patch(arr)
                place_label(x1, y1, x2, y2, 0.08, f"{p_out:.{decimals_p}f}", color="#d62728")

        # 範囲
        ax.set_xlim(min(xs_all)-1.2, max(xs_all)+1.2)
        ax.set_ylim(min(ys_all)-1.2, max(ys_all)+1.2)

        # 保存 / 表示
        if out_path:
            fig.savefig(out_path, dpi=240, bbox_inches="tight", pad_inches=0.35)
            plt.close(fig)
        else:
            plt.show()

    # OpenBCMP クラス内に追加
    def export_state_outflows(self, out_dir="./outputs", add_labels=True):
        """
        各状態 (i,r) から outside へ出る流量をCSVで出力。
        - outputs/state_outflows.csv:
            node, cls, lambda_ir, p_out, flow_out, [label]
        """
        os.makedirs(out_dir, exist_ok=True)

        rows = []
        for (i, r), lam in self.lambda_ir.items():
            lab = f"({i},{r})"
            p_out = float(self.routing.loc[lab, "outside"])
            rows.append(dict(node=i, cls=r, lambda_ir=lam, p_out=p_out, flow_out=lam*p_out))

        df = pd.DataFrame(rows).sort_values(["node","cls"]).reset_index(drop=True)

        # 任意: index_map があれば可読ラベルを付与
        if add_labels and getattr(self, "index_map", None) is not None and "label" in self.index_map.columns:
            im = self.index_map.rename(columns={"class":"cls"})
            lab_unique = im[["node","cls","label"]].drop_duplicates()
            df = df.merge(lab_unique, on=["node","cls"], how="left")

        df.to_csv(os.path.join(out_dir, "state_outflows.csv"), index=False)
        return df

        # --- helper: node id -> layer name ---
    def _layer_info(self, i: int):
        """
        6桁IDをレイヤにマップ
        戻り値: (layer, subgroup)
        """
        if 100001 <= i <= 199999:
            return ("Region", "Region")
        if 300001 <= i <= 399999:
            # triage=奇数, treatment=偶数
            return ("Shelter/Triage", "Shelter") if (i % 2 == 1) else ("Shelter/Treatment", "Shelter")
        if 500001 <= i <= 599999:
            return ("HospitalY/Triage", "HospitalY") if (i % 2 == 1) else ("HospitalY/Treatment", "HospitalY")
        if 600001 <= i <= 699999:
            return ("HospitalD/Triage", "HospitalD") if (i % 2 == 1) else ("HospitalD/Treatment", "HospitalD")
        if 400001 <= i <= 499999:
            return ("Ambulance400", "Ambulance")
        if 700001 <= i <= 799999:
            return ("Ambulance700", "AmbulanceTransfer")
        if 900001 <= i <= 999999:
            return ("Exit", "Exit")
        return ("Other", "Other")

    def congestion_report(self, out_dir="./outputs", top_k=5, warn_on_inf=True, print_report=True):
        """
        ・平均系内人数 L が無限大 (inf) の検出と警告
        ・レイヤ別に L が大きいノード TOP5 を表示
        ・各行に m, μ(FCFS), λ_i, ρ_i、クラス別K_ir上位(見やすい要約)を付与
        ・CSV出力:
            - congestion_top5_by_layer.csv
            - unstable_nodes.csv
            - unstable_states_classlevel.csv
        """
        os.makedirs(out_dir, exist_ok=True)

        # 既存のDFを組み立て
        df_run, df_traf, df_node, df_cls, df_classnet = self.build_dataframes()

        # レイヤ付与
        df_node["layer"] = df_node["node"].apply(lambda x: self._layer_info(int(x))[0])
        df_node["subgroup"] = df_node["node"].apply(lambda x: self._layer_info(int(x))[1])

        # ノードごとのクラス貢献（K_ir 上位3件を文字列で）
        contrib_map = {}
        for node, g in df_cls.groupby("node"):
            g2 = g[["cls", "K_ir", "mu_ir"]].copy()
            g2 = g2.sort_values("K_ir", ascending=False)
            parts = []
            for _, row in g2.head(3).iterrows():
                cls = int(row["cls"])
                Kir = row["K_ir"]
                if np.isfinite(Kir) and Kir > 0:
                    parts.append(f"{cls}:{Kir:.3f}")
            contrib_map[int(node)] = " ".join(parts)
        df_node["K_ir_top"] = df_node["node"].map(contrib_map).fillna("")

        # 不安定/inf 検出
        def _unstable_row(r):
            rho = r["rho_i"]
            return (isinstance(rho, (int, float)) and rho >= 1.0 - 1e-12) or np.isinf(r["L"]) or np.isinf(r["W"])

        df_node["is_unstable"] = df_node.apply(_unstable_row, axis=1)

        unstable_nodes = df_node[df_node["is_unstable"]].copy()
        unstable_states = df_cls[np.isinf(df_cls["K_ir"]) | np.isinf(df_cls["T_ir"])].copy()

        # レイヤ順（表示の見やすさ用）
        layer_order = [
            "Region",
            "Shelter/Triage", "Shelter/Treatment",
            "HospitalY/Triage", "HospitalY/Treatment",
            "HospitalD/Triage", "HospitalD/Treatment",
            "Ambulance400", "Ambulance700",
            "Exit", "Other"
        ]
        df_node["layer_idx"] = df_node["layer"].apply(lambda x: layer_order.index(x) if x in layer_order else 999)

        # レイヤ別TOP5（Lの大きい順）
        rows = []
        for layer in layer_order:
            sub = df_node[df_node["layer"] == layer].copy()
            if sub.empty:
                continue
            sub = sub.sort_values("L", ascending=False).head(top_k)
            for _, r in sub.iterrows():
                # μはFCFSのみノード行に存在（PS/ISは空）。FCFS以外は空欄でOK
                rows.append(dict(
                    layer=layer,
                    node=int(r["node"]),
                    type=str(r["type"]),
                    m=r["m"],
                    mu=r["mu"],
                    lambda_i=r["lambda_i"],
                    L=r["L"],
                    rho_i=r["rho_i"],
                    K_ir_top=r.get("K_ir_top", "")
                ))
        df_top = pd.DataFrame(rows)

        # 保存
        df_top.to_csv(os.path.join(out_dir, "congestion_top5_by_layer.csv"), index=False)
        unstable_nodes.to_csv(os.path.join(out_dir, "unstable_nodes.csv"), index=False)
        unstable_states.to_csv(os.path.join(out_dir, "unstable_states_classlevel.csv"), index=False)

        # 表示
        if print_report:
            print("\n=== Congestion Report (Top-5 by layer, sorted by L) ===")
            if df_top.empty:
                print("(no nodes)")
            else:
                for layer in layer_order:
                    t = df_top[df_top["layer"] == layer]
                    if t.empty: 
                        continue
                    print(f"\n-- {layer} --")
                    # 表示カラムを揃える（μが空でもOK）
                    cols = ["node", "type", "m", "mu", "lambda_i", "rho_i", "L", "K_ir_top"]
                    print(t[cols].to_string(index=False))

            if warn_on_inf and (len(unstable_nodes) > 0 or len(unstable_states) > 0):
                print("\n!! WARNING: Unstable or infinite metrics detected !!")
                if len(unstable_nodes) > 0:
                    show = unstable_nodes.sort_values(["layer_idx", "L"], ascending=[True, False])
                    print("\n-- Unstable nodes (ρ≥1 or L/W=inf) --")
                    print(show[["layer", "node", "type", "m", "mu", "lambda_i", "rho_i", "L", "W"]].to_string(index=False))
                    print(f"(saved: {os.path.join(out_dir, 'unstable_nodes.csv')})")
                if len(unstable_states) > 0:
                    print("\n-- Unstable states at class-level (K_ir or T_ir is inf) --")
                    print(unstable_states[["node", "cls", "lambda_ir", "mu_ir", "K_ir", "T_ir"]].to_string(index=False))
                    print(f"(saved: {os.path.join(out_dir, 'unstable_states_classlevel.csv')})")
                print("\nSuggestions:")
                print("- FCFS: ρ = λ / (m μ) が1未満になるように m（窓口数）や μ を増やす")
                print("- PS/LCFS-PR: Σ_r λ_ir / μ_ir < 1 となるように各クラスの μ_ir を見直す")
                print("- IS: μ_ir が0/NaNでないか（移動μの算出ミス）を確認する")


# ----------------- main -----------------
if __name__ == "__main__":
    # ===== User settings (edit here) =====
    PARAM_DIR = "./param_out"
    OUT_DIR   = "./outputs"

    FILE_ROUTING = os.path.join(PARAM_DIR, "P_global.csv")
    FILE_EXTERNAL= os.path.join(PARAM_DIR, "external.csv")
    FILE_SERVICE = os.path.join(PARAM_DIR, "service.csv")

    # ===== Run =====
    os.makedirs(OUT_DIR, exist_ok=True)

    bcmp = OpenBCMP(
        routing_labels=FILE_ROUTING,
        external=FILE_EXTERNAL,
        service=FILE_SERVICE,
        verbose=True
    )
    bcmp.solve_lambda()
    bcmp.compute_metrics()
    bcmp.print_summary()
    bcmp.save_outputs(OUT_DIR, make_excel=True)
    bcmp.congestion_report(out_dir=OUT_DIR, top_k=5, warn_on_inf=True, print_report=True)

    print(f"\nResults saved under: {OUT_DIR}")

    '''
    bcmp.draw_network_exact( 
        out_path=os.path.join(OUT_DIR, "network_exact.png"),
        prob_threshold=0.0,    # 0にするとゼロ以外は全部描く
        figsize=(11, 7),
        decimals_p=2,
        decimals_lam=2,
        node_fontsize=10,
        edge_fontsize=9,
    )
    '''
    bcmp.export_state_outflows(out_dir=OUT_DIR, add_labels=True) #どの状態から外へ出たか”の一覧を出す（Cure/Death内訳なし）



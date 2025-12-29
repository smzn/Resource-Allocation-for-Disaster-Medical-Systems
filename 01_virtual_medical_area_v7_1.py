#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_virtual_medical_area_v5_6digit.py

- 仮想医療地域の生成（6桁ノードID対応版）
- 地域ノード（Region）、救護所（Shelter）、救護病院（HospitalY）、災害拠点病院（HospitalD）
- 距離行列、魅力度
- 人口を正規分布（平均・標準偏差指定）で生成、CSV出力
- 人口ヒートマップ + 等高線の可視化
- 6桁ノードID体系（10万番台=地域、30万番台=救護所、50万番台=救護病院、60万番台=災害拠点病院）
"""

import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Optional

def load_triage_matrix(csv_path: Optional[str] = None):
    """
    返り値: shape=(5,5) の 1-index 配列（0行0列は未使用）
    行=判定前, 列=判定後（G=1,Y=2,R=3,B=4）。各行和=1
    """
    import csv
    import numpy as np

    T = np.zeros((5,5), dtype=float)
    if csv_path is None:
        # 推奨ベースライン
        baseline = [
            (1,1,0.94),(1,2,0.05),(1,3,0.01),(1,4,0.00),
            (2,1,0.10),(2,2,0.80),(2,3,0.08),(2,4,0.02),
            (3,1,0.00),(3,2,0.20),(3,3,0.70),(3,4,0.10),
            (4,1,0.00),(4,2,0.00),(4,3,0.00),(4,4,1.00),
        ]
        for fr,to,p in baseline: 
            T[fr,to] = p
    else:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fr = int(row["from_class"]); to = int(row["to_class"]); p = float(row["prob"])
                T[fr,to] += p

    for fr in (1,2,3,4):
        s = T[fr,1:5].sum()
        if not (abs(s - 1.0) <= 1e-9):
            raise ValueError("Triage matrix row sum != 1 (row=%d, sum=%g)" % (fr, s))
    return T



# --------------------------------------------------
# 6桁ノードID生成関数
# --------------------------------------------------
def get_node_id(node_type, facility_id=None, service_type=None):
    """6桁ノードIDを生成する関数"""
    if node_type == "region":
        return 100000 + facility_id
    elif node_type == "aid_station":
        service_offset = 1 if service_type == "treatment" else 0
        return 300000 + (facility_id - 1) * 2 + service_offset + 1
    elif node_type == "emergency_hospital":
        service_offset = 1 if service_type == "treatment" else 0
        return 500000 + (facility_id - 1) * 2 + service_offset + 1
    elif node_type == "disaster_hospital":
        service_offset = 1 if service_type == "treatment" else 0
        return 600000 + (facility_id - 1) * 2 + service_offset + 1
    elif node_type == "ambulance_from_shelter":
        return 400000 + facility_id
    elif node_type == "ambulance_transfer":
        return 700000 + facility_id
    elif node_type == "cure":
        return 900001
    elif node_type == "death":
        return 900002
    else:
        raise ValueError(f"Unknown node_type: {node_type}")

def parse_node_id(node_id):
    """ノードIDから情報を抽出する関数"""
    if 100001 <= node_id <= 199999:
        return {"type": "region", "id": node_id - 100000}
    elif 300001 <= node_id <= 399999:
        facility_id = ((node_id - 300001) // 2) + 1
        service = "treatment" if (node_id - 300001) % 2 == 1 else "triage"
        return {"type": "aid_station", "facility_id": facility_id, "service": service}
    elif 500001 <= node_id <= 599999:
        facility_id = ((node_id - 500001) // 2) + 1
        service = "treatment" if (node_id - 500001) % 2 == 1 else "triage"
        return {"type": "emergency_hospital", "facility_id": facility_id, "service": service}
    elif 600001 <= node_id <= 699999:
        facility_id = ((node_id - 600001) // 2) + 1
        service = "treatment" if (node_id - 600001) % 2 == 1 else "triage"
        return {"type": "disaster_hospital", "facility_id": facility_id, "service": service}
    elif 400001 <= node_id <= 499999:
        return {"type": "ambulance_from_shelter", "id": node_id - 400000}
    elif 700001 <= node_id <= 799999:
        return {"type": "ambulance_transfer", "id": node_id - 700000}
    elif node_id == 900001:
        return {"type": "cure"}
    elif node_id == 900002:
        return {"type": "death"}
    else:
        raise ValueError(f"Unknown node_id: {node_id}")

# --------------------------------------------------
# 補助関数
# --------------------------------------------------
def generate_grid(width_km, height_km, rows, cols):
    xs = np.linspace(0, width_km, cols)
    ys = np.linspace(0, height_km, rows)
    grid = [(x,y) for y in ys for x in xs]
    return grid

def euclid_dist(xy1, xy2):
    return np.sqrt((xy1[0]-xy2[0])**2 + (xy1[1]-xy2[1])**2)

def generate_population(region_count, mean, std, seed=1, integer=True, pop_min=1.0):
    rng = np.random.default_rng(seed)
    vals = rng.normal(loc=mean, scale=std, size=region_count)
    vals = np.clip(vals, pop_min, None)
    if integer:
        vals = np.rint(vals).astype(int)
    
    # 6桁地域IDを生成
    region_node_ids = [get_node_id("region", i+1) for i in range(region_count)]
    return pd.DataFrame({"node_id": region_node_ids, "population": vals})

def write_external(out_dir, region_node_ids, df_pop,
                   total_arrival=None,                 # 使わない（互換のため残置）
                   class_ratio=None,
                   arrivals_per_person=None,           # ここに (x/100)/H を渡す
                   arrivals_per_100k=None):            # 使わない（互換のため残置）
    """
    external.csv を出力（人口比で地域配分、クラスは比率で分解）。
    - arrivals_per_person: 1人あたり到着率 [/時間] = (x/100) / H
      例: x=1%, H=12h → 0.01/12
    出力: external.csv (columns=["node","class","lambda0"])
    """
    import os
    import numpy as np
    import pandas as pd

    if class_ratio is None:
        class_ratio = {1:0.60, 2:0.25, 3:0.12, 4:0.03}  # G,Y,R,B

    # 地域人口（region_node_ids の順に整列）
    if "node_id" not in df_pop.columns or "population" not in df_pop.columns:
        raise ValueError("df_pop に 'node_id' と 'population' 列が必要です。")
    pop = (df_pop.set_index("node_id")
                 .reindex(region_node_ids)["population"]
                 .astype(float)
                 .fillna(1.0))

    # 総到着率 total を決定（今回は arrivals_per_person を使う）
    total_pop = float(pop.sum())
    if arrivals_per_person is None:
        raise ValueError("arrivals_per_person を与えてください（(x/100)/H）。")
    total = float(arrivals_per_person) * total_pop  # 全体の到着率 [/時間]

    # 人口比で地域配分
    w = pop.to_numpy()
    if np.any(w < 0) or np.isclose(w.sum(), 0.0):
        w = np.ones_like(w, dtype=float)
    w = w / w.sum()

    rows = []
    for c, rc in sorted(class_ratio.items()):
        lam_c = total * float(rc)  # クラス確率で分解
        for nid, wi in zip(region_node_ids, w):
            rows.append({"node": int(nid), "class": int(c), "lambda0": float(lam_c * wi)})

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "external.csv"), index=False)

from typing import Optional, Dict, Iterable

def write_service_csv(
    out_dir: str,
    region_node_ids: Iterable[int],
    num_shelters: int,
    num_hy: int,
    num_hd: int,
    region_mu_map: Optional[Dict[int, float]] = None,
    classes: Iterable[int] = (1, 2, 3, 4),

    # ★救急車(400xxx)：デフォルトでFCFS（台数mを反映）
    ambulance_as_fcfs: bool = True,
    ambulance_mu_map: Optional[Dict[int, float]] = None,   # {400xxx: μ}
    ambulance_m_map:  Optional[Dict[int, float]] = None,   # {400xxx: m} 未指定は1.0

    # ★転送救急車(700xxx)：デフォルトでFCFS
    transfer_service_type: str = "FCFS",                   # "FCFS" or "IS"
    transfer_m: Optional[float] = None,                    # FCFS時のm。未指定は1.0
    transfer_mu: float = 3.0,                              # フォールバックμ
    transfer_mu_map: Optional[Dict[int, float]] = None,    # {700xxx: μ} があれば優先

    # 施設の既定（必要なら上書き）
    shelter_triage_m: float = 2.0,
    shelter_triage_mu: float = 12.0,
    hy_triage_m: float = 2.0,
    hy_triage_mu: float = 12.0,
    hd_triage_m: float = 3.0,
    hd_triage_mu: float = 14.0,
    shelter_treatment_mu: Optional[Dict[int, float]] = None,  # {1:12,2:6,3:3}
    hy_treatment_mu: Optional[Dict[int, float]] = None,       # {1:15,2:10,3:5}
    hd_treatment_mu: Optional[Dict[int, float]] = None        # {1:20,2:15,3:8}
):
    """
    service.csv を出力（互換維持）。出力列: node_id, class, service_type, m, mu

    - Region(100xxx): IS（region_mu_mapがあればクラス別、無ければ'*'フォールバック）
    - Shelter(300xxx): Triage=FCFS(m,μ)、Treatment=PS（G/Y/Rでμ指定）
    - HY(500xxx):     Triage=FCFS、Treatment=PS
    - HD(600xxx):     Triage=FCFS、Treatment=PS
    - Ambulance(400xxx): 既定FCFS（m=台数, μはmapから／無ければ5.0）
    - Transfer(700xxx):  既定FCFS（μはmapから／無ければtransfer_mu、mは引数 or 1.0）
    """
    import os, numpy as np, pandas as pd

    # 治療μのデフォルト
    if shelter_treatment_mu is None:
        shelter_treatment_mu = {1: 12.0, 2: 6.0, 3: 3.0}
    if hy_treatment_mu is None:
        hy_treatment_mu = {1: 15.0, 2: 10.0, 3: 5.0}
    if hd_treatment_mu is None:
        hd_treatment_mu = {1: 20.0, 2: 15.0, 3: 8.0}

    rows = []
    classes_tuple = tuple(int(c) for c in classes)

    # --- Region (IS) ---
    for nid in region_node_ids:
        nid_int = int(nid)
        if region_mu_map is not None and nid_int in region_mu_map:
            mu_rc = float(region_mu_map[nid_int])
            for c in classes_tuple:
                rows.append({
                    "node_id": nid_int,
                    "class": str(c),
                    "service_type": "IS",
                    "m": "",
                    "mu": mu_rc
                })
        else:
            rows.append({
                "node_id": nid_int,
                "class": "*",
                "service_type": "IS",
                "m": np.inf,
                "mu": 1.0
            })

    # --- Shelters (300xxx) ---
    for i in range(1, num_shelters + 1):
        t_id = 300000 + (i - 1) * 2 + 1
        p_id = t_id + 1
        # triage
        rows.append({
            "node_id": t_id,
            "class": "*",
            "service_type": "FCFS",
            "m": float(shelter_triage_m),
            "mu": float(shelter_triage_mu)
        })
        # treatment (G/Y/R)
        for c in (1, 2, 3):
            rows.append({
                "node_id": p_id,
                "class": str(c),
                "service_type": "PS",
                "m": "",
                "mu": float(shelter_treatment_mu.get(c))
            })

    # --- Emergency Hospitals HY (500xxx) ---
    for j in range(1, num_hy + 1):
        t_id = 500000 + (j - 1) * 2 + 1
        p_id = t_id + 1
        rows.append({
            "node_id": t_id,
            "class": "*",
            "service_type": "FCFS",
            "m": float(hy_triage_m),
            "mu": float(hy_triage_mu)
        })
        for c in (1, 2, 3):
            rows.append({
                "node_id": p_id,
                "class": str(c),
                "service_type": "PS",
                "m": "",
                "mu": float(hy_treatment_mu.get(c))
            })

    # --- Disaster Base Hospitals HD (600xxx) ---
    for k in range(1, num_hd + 1):
        t_id = 600000 + (k - 1) * 2 + 1
        p_id = t_id + 1
        rows.append({
            "node_id": t_id,
            "class": "*",
            "service_type": "FCFS",
            "m": float(hd_triage_m),
            "mu": float(hd_triage_mu)
        })
        for c in (1, 2, 3):
            rows.append({
                "node_id": p_id,
                "class": str(c),
                "service_type": "PS",
                "m": "",
                "mu": float(hd_treatment_mu.get(c))
            })

    # --- Ambulance From Shelter (400xxx) ---
    for i in range(1, num_shelters + 1):
        amb_id = 400000 + i
        if ambulance_as_fcfs:
            mu = float((ambulance_mu_map or {}).get(amb_id, 5.0))
            m  = float((ambulance_m_map  or {}).get(amb_id, 1.0))
            rows.append({
                "node_id": amb_id,
                "class": "*",
                "service_type": "FCFS",
                "m": m,
                "mu": mu
            })
        else:
            mu = float((ambulance_mu_map or {}).get(amb_id, 5.0))
            rows.append({
                "node_id": amb_id,
                "class": "*",
                "service_type": "IS",
                "m": "",
                "mu": mu
            })

    # --- Ambulance Transfer (700xxx) ---
    trans_fcfs = (transfer_service_type or "IS").upper() == "FCFS"
    for j in range(1, num_hy + 1):
        trans_id = 700000 + j
        if trans_fcfs:
            mu_val = None
            if transfer_mu_map is not None and trans_id in transfer_mu_map:
                mu_val = float(transfer_mu_map[trans_id])
            else:
                mu_val = float(transfer_mu)
            m_val = float(transfer_m if transfer_m is not None else 1.0)
            rows.append({
                "node_id": trans_id,
                "class": "*",
                "service_type": "FCFS",
                "m": m_val,
                "mu": mu_val
            })
        else:
            rows.append({
                "node_id": trans_id,
                "class": "*",
                "service_type": "IS",
                "m": "",
                "mu": float(transfer_mu)
            })

    # --- 保存 ---
    import pandas as pd
    df = pd.DataFrame(rows, columns=["node_id", "class", "service_type", "m", "mu"])
    df.to_csv(os.path.join(out_dir, "service.csv"), index=False)


def write_routing_gravity(
    out_dir,
    classes=(1,2,3,4),
    beta_rs=0.15,
    beta_sh=0.08,
    kernel="exp",
    shelter_attr_path=None,
    hospital_attr_path=None,
    # 互換: 旧引数は無視してOK（Amb起点に統一）
    route_classes_from_shelter=(1,2,3,4),
    # 新仕様: Ambulance から病院へ張るクラス（デフォ: Y,R）
    route_classes_from_ambulance=(2,3),
    also_write_matrix=True,
    matrix_filename="P_global.csv",
    use_index_map_if_exists=True,
    limit_states_to_gravity=True
):
    """
    重力モデルの推移確率を出力。
      - Region(100xxx) → ShelterTriage(300xxx) : 従来どおり
      - AmbulanceFromShelter(400xxx) → HospitalTriage(500/600xxx) : ★新仕様
    """
    import os
    import numpy as np
    import pandas as pd

    # --- 入力 ---
    path_rs = os.path.join(out_dir, "dist_region_to_shelter.csv")
    path_ah = os.path.join(out_dir, "dist_ambulance_to_hospital.csv")
    if not os.path.exists(path_rs):
        raise FileNotFoundError("dist_region_to_shelter.csv が見つかりません。")
    if not os.path.exists(path_ah):
        raise FileNotFoundError("dist_ambulance_to_hospital.csv が見つかりません。")

    D_rs = pd.read_csv(path_rs, index_col=0)
    D_rs.index  = D_rs.index.astype(int);   D_rs.columns = D_rs.columns.astype(int)

    D_ah = pd.read_csv(path_ah, index_col=0)
    D_ah.index  = D_ah.index.astype(int);   D_ah.columns = D_ah.columns.astype(int)

    # 魅力度
    if shelter_attr_path is None:
        shelter_attr_path = os.path.join(out_dir, "attractiveness_A.csv")
    A_s = pd.Series(1.0, index=D_rs.columns)
    if os.path.exists(shelter_attr_path):
        dfA = pd.read_csv(shelter_attr_path)
        if {"node_id","A_final"}.issubset(dfA.columns):
            A_s = (dfA.set_index("node_id")["A_final"].astype(float)
                     .reindex(A_s.index).fillna(1.0))

    A_h = pd.Series(1.0, index=D_ah.columns)
    if hospital_attr_path is not None and os.path.exists(hospital_attr_path):
        dfH = pd.read_csv(hospital_attr_path)
        if {"node_id","A_final"}.issubset(dfH.columns):
            A_h = (dfH.set_index("node_id")["A_final"].astype(float)
                     .reindex(A_h.index).fillna(1.0))

    # カーネル
    def ker(d, beta, mode):
        if mode == "exp":   return np.exp(-beta * d)
        elif mode == "power": return 1.0 / np.power(1.0 + d, beta)
        else: raise ValueError("kernel must be 'exp' or 'power'")

    # Region→Shelter
    W_rs = ker(D_rs.to_numpy(), beta_rs, kernel) * A_s.to_numpy()[None, :]
    denom_rs = W_rs.sum(axis=1, keepdims=True); denom_rs[denom_rs == 0.0] = 1.0
    P_rs = W_rs / denom_rs

    # Ambulance→Hospital（★新仕様）
    W_ah = ker(D_ah.to_numpy(), beta_sh, kernel) * A_h.to_numpy()[None, :]
    denom_ah = W_ah.sum(axis=1, keepdims=True); denom_ah[denom_ah == 0.0] = 1.0
    P_ah = W_ah / denom_ah

    # long形式へ
    rows = []

    # Region→Shelter：全クラス共通でコピー
    ridx = D_rs.index.to_list()
    sidx = D_rs.columns.to_list()
    for i, r in enumerate(ridx):
        for j, s in enumerate(sidx):
            p = float(P_rs[i, j])
            if p <= 0.0: continue
            for c in classes:
                rows.append({"from": f"({r},{c})", "to": f"({s},{c})", "prob": p})

    # Ambulance→Hospital：デフォは Y,R のみ
    aidx = D_ah.index.to_list()
    hidx = D_ah.columns.to_list()
    rcset = set(route_classes_from_ambulance)
    for i, a in enumerate(aidx):
        for j, h in enumerate(hidx):
            p = float(P_ah[i, j])
            if p <= 0.0: continue
            for c in rcset:
                rows.append({"from": f"({a},{c})", "to": f"({h},{c})", "prob": p})

    # 保存
    import pandas as pd
    dfR = pd.DataFrame(rows, columns=["from","to","prob"])
    routing_path = os.path.join(out_dir, "routing_labels.csv")
    dfR.to_csv(routing_path, index=False)

    # P_global（outside吸収+行和=1 調整）
    if also_write_matrix:
        states_from = dfR["from"].unique().tolist()
        states_to   = dfR["to"].unique().tolist()
        states_seen = set(states_from) | set(states_to)

        labels = None
        if use_index_map_if_exists and os.path.exists(os.path.join(out_dir, "index_map.csv")):
            idx = pd.read_csv(os.path.join(out_dir, "index_map.csv"))
            labels_all = idx["label"].astype(str).tolist()
            if limit_states_to_gravity:
                labels = [lab for lab in labels_all if lab in states_seen]
            else:
                labels = labels_all[:]
        if labels is None:
            labels = sorted(states_seen)
        if 'outside' not in labels:
            labels.append('outside')

        P = dfR.pivot_table(index="from", columns="to", values="prob",
                            fill_value=0.0, aggfunc="sum")
        P = P.reindex(index=labels, columns=labels, fill_value=0.0)

        P.loc['outside', :] = 0.0
        P.loc['outside', 'outside'] = 1.0

        row_sums = P.sum(axis=1)
        for lab in P.index:
            if lab == 'outside': continue
            rs = row_sums[lab]
            if rs < 1.0 - 1e-10:
                P.loc[lab, 'outside'] += (1.0 - rs)
            elif rs > 1.0 + 1e-10:
                P.loc[lab, :] /= rs

        P.to_csv(os.path.join(out_dir, matrix_filename))


from typing import Optional, Iterable  # まだなら追加

def write_complete_routing_with_facilities(
    out_dir: str,
    num_shelters: int,
    num_hy: int,
    num_hd: int,
    classes: Iterable[int] = (1, 2, 3, 4),
    triage_matrix_csv: Optional[str] = None,
    # ← これを追加（後方互換）
    p_death_at_triage: float = 0.0,
    also_write_matrix: bool = True,
    matrix_filename: str = "P_global.csv",
    apply_T_at_hospital: bool = True
):

    """
    重力モデルで出力済みの routing_labels.csv に、施設内の退去時遷移を追加して上書き保存。
    - Shelter(300xxx): triage退去時にT適用（G→Cure, Y/R→Amb(400), B→outside）
    - HospitalY(500xxx): triage退去時にT適用（1/2→治療, 3→Transfer(700), 4→outside）
    - HospitalD(600xxx): triage退去時にT適用（1/2/3→治療, 4→outside）
    - 各 Treatment は outside へ 1.0
    """
    # 依存：get_node_id, load_triage_matrix が同ファイルにある前提
    if not os.path.exists(os.path.join(out_dir, "routing_labels.csv")):
        raise FileNotFoundError("routing_labels.csv が見つかりません。先に write_routing_gravity() を実行してください。")
    df = pd.read_csv(os.path.join(out_dir, "routing_labels.csv"))
    rows = df.to_dict("records")

    # T行列
    T = load_triage_matrix(triage_matrix_csv)  # shape=(5,5), 1-index, 各行和=1

    # ------------- Shelter（救護所）-------------
    for i in range(1, num_shelters + 1):
        t_id = get_node_id("aid_station", i, "triage")
        p_id = get_node_id("aid_station", i, "treatment")
        amb_id = get_node_id("ambulance_from_shelter", i)

        # 到着時クラス fr → 判定後 to
        for fr in (1, 2, 3, 4):
            for to in (1, 2, 3, 4):
                p = float(T[fr, to])
                if p <= 0.0:
                    continue
                if to == 1:  # G→Cure
                    rows.append({"from": f"({t_id},{fr})", "to": f"({p_id},1)", "prob": p})
                elif to in (2, 3):  # Y/R→Amb(400)
                    rows.append({"from": f"({t_id},{fr})", "to": f"({amb_id},{to})", "prob": p})
                else:  # B→outside
                    rows.append({"from": f"({t_id},{fr})", "to": "outside", "prob": p})

        # Cure(G) → outside = 1.0
        rows.append({"from": f"({p_id},1)", "to": "outside", "prob": 1.0})

    # ------------- HY（救護病院）-------------
    for j in range(1, num_hy + 1):
        t_id = get_node_id("emergency_hospital", j, "triage")
        p_id = get_node_id("emergency_hospital", j, "treatment")
        trans_id = get_node_id("ambulance_transfer", j)  # 700xxx

        if apply_T_at_hospital:
            # triage退去時にT適用
            for fr in (1, 2, 3, 4):
                for to in (1, 2, 3, 4):
                    p = float(T[fr, to])
                    if p <= 0.0:
                        continue
                    if to in (1, 2):   # G/Y → HY治療（toクラスで入る）
                        rows.append({"from": f"({t_id},{fr})", "to": f"({p_id},{to})", "prob": p})
                    elif to == 3:     # R → 転送救急車（Rで出る）
                        rows.append({"from": f"({t_id},{fr})", "to": f"({trans_id},3)", "prob": p})
                    else:             # B → outside
                        rows.append({"from": f"({t_id},{fr})", "to": "outside", "prob": p})
            # HY治療 → outside
            for c in (1, 2):
                rows.append({"from": f"({p_id},{c})", "to": "outside", "prob": 1.0})
        else:
            # （参考）従来の固定遷移：G/Y→治療, R→転送, B→outside（確率1）
            for c in (1, 2):
                rows.append({"from": f"({t_id},{c})", "to": f"({p_id},{c})", "prob": 1.0})
                rows.append({"from": f"({p_id},{c})", "to": "outside", "prob": 1.0})
            rows.append({"from": f"({t_id},3)", "to": f"({trans_id},3)", "prob": 1.0})
            rows.append({"from": f"({t_id},4)", "to": "outside", "prob": 1.0})

    # ------------- HD（災害拠点病院）-------------
    for k in range(1, num_hd + 1):
        t_id = get_node_id("disaster_hospital", k, "triage")
        p_id = get_node_id("disaster_hospital", k, "treatment")

        if apply_T_at_hospital:
            # triage退去時にT適用
            for fr in (1, 2, 3, 4):
                for to in (1, 2, 3, 4):
                    p = float(T[fr, to])
                    if p <= 0.0:
                        continue
                    if to in (1, 2, 3):   # G/Y/R → HD治療
                        rows.append({"from": f"({t_id},{fr})", "to": f"({p_id},{to})", "prob": p})
                    else:                # B → outside
                        rows.append({"from": f"({t_id},{fr})", "to": "outside", "prob": p})
            # HD治療 → outside
            for c in (1, 2, 3):
                rows.append({"from": f"({p_id},{c})", "to": "outside", "prob": 1.0})
        else:
            # （参考）従来：G/Y/R→治療, B→outside
            for c in (1, 2, 3):
                rows.append({"from": f"({t_id},{c})", "to": f"({p_id},{c})", "prob": 1.0})
                rows.append({"from": f"({p_id},{c})", "to": "outside", "prob": 1.0})
            rows.append({"from": f"({t_id},4)", "to": "outside", "prob": 1.0})

    # ---------- 保存（重複は合算） ----------
    df_out = pd.DataFrame(rows).groupby(["from", "to"], as_index=False)["prob"].sum()
    df_out.to_csv(os.path.join(out_dir, "routing_labels.csv"), index=False)

    # ---------- 行列Pも再生成（outside吸収・行和=1） ----------
    if also_write_matrix:
        states_from = set(df_out["from"].unique())
        states_to   = set(df_out["to"].unique())
        labels = sorted((states_from | states_to) - {"outside"}) + ["outside"]

        P = df_out.pivot_table(index="from", columns="to", values="prob",
                               fill_value=0.0, aggfunc="sum")
        P = P.reindex(index=labels, columns=labels, fill_value=0.0)

        # outside を吸収状態に
        P.loc["outside", :] = 0.0
        P.loc["outside", "outside"] = 1.0

        # 各行の合計を1に（不足はoutsideへ、超過は正規化）
        for idx in P.index:
            if idx == "outside":
                continue
            s = P.loc[idx].sum()
            if s < 1.0 - 1e-10:
                P.loc[idx, "outside"] += (1.0 - s)
            elif s > 1.0 + 1e-10:
                P.loc[idx, :] = P.loc[idx, :] / s

        P.to_csv(os.path.join(out_dir, matrix_filename))


def compute_ambulance_mu_from_gravity(
    out_dir,
    beta_sh=0.08,
    kernel="exp",
    hospital_attr_path=None,
    speed_kmph=5.0
):
    """
    AmbulanceFromShelter(400xxx)→Hospital(トリアージ) の重力から
    期待距離 E[d] を出し、μ = v/E[d] を返す。
    return: dict { 400xxx:int -> mu:float }
    """
    import os
    import numpy as np
    import pandas as pd

    path_ah = os.path.join(out_dir, "dist_ambulance_to_hospital.csv")
    if not os.path.exists(path_ah):
        raise FileNotFoundError("dist_ambulance_to_hospital.csv が見つかりません。")

    D = pd.read_csv(path_ah, index_col=0)
    D.index  = D.index.astype(int); D.columns = D.columns.astype(int)

    A_h = pd.Series(1.0, index=D.columns)
    if hospital_attr_path is not None and os.path.exists(hospital_attr_path):
        dfH = pd.read_csv(hospital_attr_path)
        if {"node_id","A_final"}.issubset(dfH.columns):
            A_h = (dfH.set_index("node_id")["A_final"].astype(float)
                     .reindex(A_h.index).fillna(1.0))

    def ker(d, beta, mode):
        if mode == "exp":   return np.exp(-beta * d)
        elif mode == "power": return 1.0 / np.power(1.0 + d, beta)
        else: raise ValueError("kernel must be 'exp' or 'power'")

    W = ker(D.to_numpy(), beta_sh, kernel) * A_h.to_numpy()[None, :]
    denom = W.sum(axis=1, keepdims=True); denom[denom==0.0] = 1.0
    P = W / denom

    E_d = (P * D.to_numpy()).sum(axis=1)  # 期待距離[km]
    v = float(speed_kmph)
    mu_vals = np.where(E_d > 0.0, v / E_d, np.inf)  # 1/h

    return { int(a): float(mu) for a, mu in zip(D.index.tolist(), mu_vals) }


def compute_region_mu_from_gravity(out_dir,
                                   beta_rs=0.15,
                                   kernel="exp",
                                   shelter_attr_path=None,
                                   speed_kmph=5.0):
    """
    地域→救護所（トリアージ）の重力モデル（距離×魅力度）から
    行方向に正規化した確率 P_rs を再計算し、期待移動距離 E[d] を求めて
    μ_region = v / E[d] (1/h) を返す。

    return: dict { region_node_id:int -> mu:float }
    """
    import os
    import numpy as np
    import pandas as pd

    # 距離（地域→救護所T）
    path_rs = os.path.join(out_dir, "dist_region_to_shelter.csv")
    if not os.path.exists(path_rs):
        raise FileNotFoundError("dist_region_to_shelter.csv が見つかりません。")

    D_rs = pd.read_csv(path_rs, index_col=0)
    D_rs.index  = D_rs.index.astype(int)
    D_rs.columns = D_rs.columns.astype(int)

    # 魅力度（救護所T），無ければ1.0
    if shelter_attr_path is None:
        shelter_attr_path = os.path.join(out_dir, "attractiveness_A.csv")
    A_s = pd.Series(1.0, index=D_rs.columns)
    if os.path.exists(shelter_attr_path):
        dfA = pd.read_csv(shelter_attr_path)
        if {"node_id","A_final"}.issubset(dfA.columns):
            A_s = (dfA.set_index("node_id")["A_final"]
                     .astype(float).reindex(A_s.index).fillna(1.0))

    # カーネル
    def ker(d, beta, mode):
        if mode == "exp":
            return np.exp(-beta * d)
        elif mode == "power":
            return 1.0 / np.power(1.0 + d, beta)
        else:
            raise ValueError("kernel must be 'exp' or 'power'")

    # 重み → 行正規化で P_rs
    W_rs = ker(D_rs.to_numpy(), beta_rs, kernel) * A_s.to_numpy()[None, :]
    denom = W_rs.sum(axis=1, keepdims=True)
    denom[denom == 0.0] = 1.0
    P_rs = W_rs / denom  # shape: (num_regions, num_shelters)

    # 期待距離 E[d] と μ=v/E[d]
    E_d = (P_rs * D_rs.to_numpy()).sum(axis=1)  # 各地域の期待距離
    v = float(speed_kmph)
    mu_vals = np.where(E_d > 0.0, v / E_d, np.inf)  # E[d]=0 は μ=∞（遅延なし）

    # dict で返す（indexは地域node_id）
    mu_map = { int(r): float(mu) for r, mu in zip(D_rs.index.tolist(), mu_vals) }
    return mu_map

from typing import Optional, Dict

def compute_ambulance_mu_fcfs(
    out_dir: str,
    beta_sh: float = 0.08,
    kernel: str = "exp",
    hospital_attr_path: Optional[str] = None,
    # ★救急車の既定速度：被災下でも現実的な 25 km/h
    speed_kmph: float = 25.0,
    # 取り扱い時間（分）
    t_load_min: float = 8.0,
    t_unload_min: float = 12.0,
    t_reset_min: float = 5.0,
    # ★往復係数（0:戻らない, 0.5:半分戻る近似, 1.0:必ず戻る）
    alpha_return: float = 0.5,
) -> Dict[int, float]:
    """
    400xxx→病院(500/600xxx) の重力から期待距離 E[d] を計算し、
    FCFSでの平均サイクル時間 E[T] = 積込 + 片道 + 降ろし + α*片道 + 整備
    の μ=1/E[T] を返す。
    return: { 400xxx -> mu_fcfs (1/h) }
    """
    import os, numpy as np, pandas as pd

    path_ah = os.path.join(out_dir, "dist_ambulance_to_hospital.csv")
    if not os.path.exists(path_ah):
        raise FileNotFoundError("dist_ambulance_to_hospital.csv が見つかりません。")

    D = pd.read_csv(path_ah, index_col=0)
    D.index = D.index.astype(int); D.columns = D.columns.astype(int)

    # 魅力度（任意）
    A_h = pd.Series(1.0, index=D.columns)
    if hospital_attr_path is not None and os.path.exists(hospital_attr_path):
        dfH = pd.read_csv(hospital_attr_path)
        if {"node_id","A_final"}.issubset(dfH.columns):
            A_h = (dfH.set_index("node_id")["A_final"].astype(float)
                     .reindex(A_h.index).fillna(1.0))

    def ker(x, beta, mode):
        import numpy as np
        return np.exp(-beta*x) if mode=="exp" else 1.0/np.power(1.0+x, beta)

    # 重力 → 行正規化 → 期待距離
    W = ker(D.to_numpy(), beta_sh, kernel) * A_h.to_numpy()[None, :]
    denom = W.sum(axis=1, keepdims=True); denom[denom==0.0] = 1.0
    P = W / denom
    E_d = (P * D.to_numpy()).sum(axis=1)  # km

    v = float(speed_kmph)
    t_load  = t_load_min  / 60.0
    t_unld  = t_unload_min/ 60.0
    t_reset = t_reset_min / 60.0

    E_T = t_load + (E_d/v) + t_unld + alpha_return*(E_d/v) + t_reset
    mu  = 1.0 / E_T
    return { int(a): float(m) for a, m in zip(D.index.tolist(), mu) }

def compute_transfer_mu_fcfs(
    out_dir: str,
    beta_sh: float = 0.08,
    kernel: str = "exp",
    # 既定：同じく 25 km/h
    speed_kmph: float = 25.0,
    t_load_min: float = 6.0,     # 転送側はやや短め想定でもOK（調整可）
    t_unload_min: float = 10.0,
    t_reset_min: float = 5.0,
    alpha_return: float = 0.5,
) -> Dict[int, float]:
    """
    700xxx→600xxx（HYからHDへの転送）の FCFS 用 μ を返す。
    距離は dist_hospitalY_to_hospitalD.csv（500xxx→600xxx）を使い、
    行ラベルを 700xxx にマッピングして期待距離を作る。
    return: { 700xxx -> mu_fcfs (1/h) }
    """
    import os, numpy as np, pandas as pd

    path_yd = os.path.join(out_dir, "dist_hospitalY_to_hospitalD.csv")
    if not os.path.exists(path_yd):
        raise FileNotFoundError("dist_hospitalY_to_hospitalD.csv が見つかりません。")

    D = pd.read_csv(path_yd, index_col=0)
    D.index = D.index.astype(int); D.columns = D.columns.astype(int)
    if D.shape[0] == 0:
        return {}

    # 500001,500003,... → j=1,2,... → 700000+j に対応
    j = ((D.index - 500001)//2 + 1).astype(int)
    transfer_ids = (700000 + j).to_numpy()

    D_tr = D.copy()
    D_tr.index = transfer_ids  # 700xxx に置換

    def ker(x, beta, mode):
        import numpy as np
        return np.exp(-beta*x) if mode=="exp" else 1.0/np.power(1.0+x, beta)

    W = ker(D_tr.to_numpy(), beta_sh, kernel)
    denom = W.sum(axis=1, keepdims=True); denom[denom==0.0] = 1.0
    P = W / denom
    E_d = (P * D_tr.to_numpy()).sum(axis=1)

    v = float(speed_kmph)
    t_load  = t_load_min  / 60.0
    t_unld  = t_unload_min/ 60.0
    t_reset = t_reset_min / 60.0

    E_T = t_load + (E_d/v) + t_unld + alpha_return*(E_d/v) + t_reset
    mu  = 1.0 / E_T
    return { int(a): float(m) for a, m in zip(D_tr.index.tolist(), mu) }

import os, pandas as pd, numpy as np

def append_transfer_to_hd(out_dir, beta_sh=0.08, kernel="exp", update_matrix=True):
    """(700xxx,3)→(600xxx,3) を重力で追記し、必要なら P_global.csv も再生成。"""
    path_r = os.path.join(out_dir, "routing_labels.csv")
    path_d = os.path.join(out_dir, "dist_hospitalY_to_hospitalD.csv")
    assert os.path.exists(path_r) and os.path.exists(path_d), "routing_labels.csv / dist_hospitalY_to_hospitalD.csv が見つかりません。"

    dfR = pd.read_csv(path_r)
    D = pd.read_csv(path_d, index_col=0)
    D.index = D.index.astype(int); D.columns = D.columns.astype(int)

    # 500001,500003,... → j=1,2,... → 700000+j に対応
    j = ((D.index - 500001)//2 + 1).astype(int)
    D_tr = D.copy()
    D_tr.index = (700000 + j).to_numpy()  # 行ラベルを 700xxx に置換

    def ker(x, beta, mode):
        return np.exp(-beta*x) if mode=="exp" else 1.0/np.power(1.0+x, beta)

    W = ker(D_tr.to_numpy(), beta_sh, kernel)
    denom = W.sum(axis=1, keepdims=True); denom[denom==0.0] = 1.0
    P = W / denom

    add = []
    t_idx = D_tr.index.to_list()
    h_idx = D_tr.columns.to_list()
    for i,a in enumerate(t_idx):
        for j,h in enumerate(h_idx):
            p = float(P[i,j])
            if p>0:
                add.append({"from": f"({a},3)", "to": f"({h},3)", "prob": p})

    # 追記して集約
    dfR = pd.concat([dfR, pd.DataFrame(add)], ignore_index=True)
    dfR = dfR.groupby(["from","to"], as_index=False)["prob"].sum()
    dfR.to_csv(path_r, index=False)
    print("[OK] 700→600 の転送ルーティングを追記しました。")

    if update_matrix:
        rebuild_P_global(out_dir)
        print("[OK] P_global.csv を再生成しました。")


def rebuild_P_global(out_dir):
    """routing_labels.csv から P_global.csv を再生成。outside を吸収状態にし、各行和=1に調整。"""
    path_r = os.path.join(out_dir, "routing_labels.csv")
    path_P = os.path.join(out_dir, "P_global.csv")
    assert os.path.exists(path_r), "routing_labels.csv が見つかりません。"

    df = pd.read_csv(path_r)

    # ラベル集合
    states = set(df["from"].astype(str)) | set(df["to"].astype(str))
    states.discard("outside")
    labels = sorted(states) + ["outside"]

    # ピボット
    P = df.pivot_table(index="from", columns="to", values="prob", fill_value=0.0, aggfunc="sum")
    P = P.reindex(index=labels, columns=labels, fill_value=0.0)

    # outside を吸収状態に
    P.loc["outside", :] = 0.0
    P.loc["outside", "outside"] = 1.0

    # 各行の和を1に（不足はoutsideへ、超過は正規化）
    for idx in P.index:
        if idx == "outside":
            continue
        s = P.loc[idx].sum()
        if s < 1.0 - 1e-12:
            P.loc[idx, "outside"] += (1.0 - s)
        elif s > 1.0 + 1e-12:
            P.loc[idx, :] = P.loc[idx, :] / s

    P.to_csv(path_P)

def summarize_ambulances(service_csv_path, out_csv_path=None):
    import pandas as pd
    df = pd.read_csv(service_csv_path)

    # ★ 列名の差異に対応
    if "node" not in df.columns:
        if "node_id" in df.columns:
            df = df.rename(columns={"node_id": "node"})
        else:
            raise ValueError(f"service.csv に 'node' 列がありません（columns={list(df.columns)}）")

    # m を数値化（空文字/NaNは0）
    df["m"] = pd.to_numeric(df.get("m", 0), errors="coerce").fillna(0)

    cond_sh = df["node"].between(400000, 499999)
    cond_hh = df["node"].between(700000, 799999)

    sh = df.loc[cond_sh, ["node", "m"]].copy()
    hh = df.loc[cond_hh, ["node", "m"]].copy()

    def _blk(name, part):
        n = len(part)
        total = float(part["m"].sum()) if n else 0.0
        mn = float(part["m"].min()) if n else 0.0
        mx = float(part["m"].max()) if n else 0.0
        print(f"[{name}] nodes={n:4d}  total m={total:.1f}  m[min,max]=[{mn:.1f},{mx:.1f}]")

    print("\n==== Ambulance Usage Summary ====")
    _blk("400xxx Shelter→Hospital", sh)
    _blk("700xxx Hospital→Hospital", hh)

    if out_csv_path:
        out = pd.concat(
            [sh.assign(type="Shelter→Hospital"),
             hh.assign(type="Hospital→Hospital")],
            ignore_index=True
        ).sort_values(["type", "node"])
        out.to_csv(out_csv_path, index=False)
        print(f"[ok] summary saved -> {out_csv_path}")

def print_and_save_service_config(
    out_dir,
    args,
    region_speed,
    amb_params,
    trans_params,
    triage_params,      # dict
    treatment_params,   # dict of dict
    region_mu_map,
    amb_mu_fcfs,
    trans_mu_fcfs
):
    import json, numpy as np, os

    def _stats(values):
        if not values: return {"n": 0, "min": None, "median": None, "mean": None, "max": None}
        a = np.array(list(values), dtype=float)
        return {
            "n": int(a.size),
            "min": float(a.min()),
            "median": float(np.median(a)),
            "mean": float(a.mean()),
            "max": float(a.max()),
        }

    cfg = {
        "seed": args.seed,
        "ambulance_m_defaults": {
            "400xxx": float(args.amb_sh_m_default),
            "700xxx": float(args.amb_hh_m_default),
        },
        # ★ 追加：トリアージ m のデフォルト
        "triage_m_defaults": {
            "shelter": float(args.shelter_triage_m_default),
            "hy": float(args.hy_triage_m_default),
            "hd": float(args.hd_triage_m_default),
        },
        "region_IS": {
            "speed_kmph": float(region_speed),
            "mu_formula": "mu = v / E[d]  (E[d]: Region→Shelter 期待距離)",
        },
        "ambulance_400_FCFS": {
            "speed_kmph": float(amb_params["speed"]),
            "t_load_min": float(amb_params["load"]),
            "t_unload_min": float(amb_params["unload"]),
            "t_reset_min": float(amb_params["reset"]),
            "alpha_return": float(amb_params["alpha"]),
            "mu_formula": "mu = 1 / (t_load + d/v + t_unload + alpha*d/v + t_reset)",
        },
        "transfer_700_FCFS": {
            "speed_kmph": float(trans_params["speed"]),
            "t_load_min": float(trans_params["load"]),
            "t_unload_min": float(trans_params["unload"]),
            "t_reset_min": float(trans_params["reset"]),
            "alpha_return": float(trans_params["alpha"]),
            "mu_formula": "mu = 1 / (t_load + d/v + t_unload + alpha*d/v + t_reset)",
        },
        "triage_FCFS_mu": triage_params,          # {"shelter":12, "hy":12, "hd":14}
        "treatment_PS_mu": treatment_params,      # {"shelter":{1:..,2:..,3:..}, ...}
        "computed_mu_summary": {
            "Region(IS)":     _stats(region_mu_map.values()),
            "Ambulance400":   _stats(amb_mu_fcfs.values()),
            "Transfer700":    _stats(trans_mu_fcfs.values()),
        }
    }

    # --- Console print (見やすく) ---
    print("\n==== Service Rate (μ) Settings ====")
    print(f"Seed: {cfg['seed']}")
    print(f"[Region IS] speed={cfg['region_IS']['speed_kmph']} km/h  formula: {cfg['region_IS']['mu_formula']}")
    print(f"[Ambulance 400 FCFS] v={cfg['ambulance_400_FCFS']['speed_kmph']} km/h, "
          f"t_load={cfg['ambulance_400_FCFS']['t_load_min']}m, "
          f"t_unload={cfg['ambulance_400_FCFS']['t_unload_min']}m, "
          f"t_reset={cfg['ambulance_400_FCFS']['t_reset_min']}m, "
          f"alpha={cfg['ambulance_400_FCFS']['alpha_return']}")
    print(f"[Transfer 700 FCFS]  v={cfg['transfer_700_FCFS']['speed_kmph']} km/h, "
          f"t_load={cfg['transfer_700_FCFS']['t_load_min']}m, "
          f"t_unload={cfg['transfer_700_FCFS']['t_unload_min']}m, "
          f"t_reset={cfg['transfer_700_FCFS']['t_reset_min']}m, "
          f"alpha={cfg['transfer_700_FCFS']['alpha_return']}")
    print(f"[Triage FCFS μ] Shelter={triage_params['shelter']}, HY={triage_params['hy']}, HD={triage_params['hd']}")
    print(f"[Treatment PS μ] Shelter={treatment_params['shelter']}, HY={treatment_params['hy']}, HD={treatment_params['hd']}")
    print(f"[m defaults] 400xxx={cfg['ambulance_m_defaults']['400xxx']}, 700xxx={cfg['ambulance_m_defaults']['700xxx']}")
    # ★ 追加表示：トリアージ m デフォルト
    print(f"[Triage m defaults] Shelter={cfg['triage_m_defaults']['shelter']}, "
          f"HY={cfg['triage_m_defaults']['hy']}, HD={cfg['triage_m_defaults']['hd']}")

    # --- Save JSON manifest ---
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "service_config.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"[ok] service_config.json -> {manifest_path}")


# --------------------------------------------------
# メイン
# --------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="./param_out")
    ap.add_argument("--width-km", type=float, default=60)
    ap.add_argument("--height-km", type=float, default=40)
    ap.add_argument("--grid-rows", type=int, default=26)
    ap.add_argument("--grid-cols", type=int, default=40)
    ap.add_argument("--num-regions", type=int, default=None, help="直接地域数を指定（指定時はgrid-rows/colsより優先）")
    ap.add_argument("--num-shelters", type=int, default=29, help="救護所数")
    ap.add_argument("--num-hospitals-y", type=int, default=4, help="救護病院数")
    ap.add_argument("--num-hospitals-d", type=int, default=2, help="災害拠点病院数")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--placement-strategy", choices=["uniform"], default="uniform")

    # population params
    ap.add_argument("--pop-mean", type=float, default=1200.0)
    ap.add_argument("--pop-std", type=float, default=300.0)
    ap.add_argument("--pop-min", type=float, default=1.0)
    ap.add_argument("--pop-integer", action="store_true")
    ap.add_argument("--incidence-percent", type=float, default=0.5,
                help="地域人口の x%% が被災（例: 0.5 = 0.5%%）")
    ap.add_argument("--horizon-hours", type=float, default=72.0,
                help="到着率に平均化する時間幅 [時間]")
    # argparse 追加
    ap.add_argument("--triage-matrix", type=str, default=None,
                help="トリアージクラス変更行列CSV (from_class,to_class,prob). 省略時は内蔵の既定行列。")
    ap.add_argument("--amb-sh-m-default", type=float, default=3.0,
        help="救護所→病院の救急車デフォルト台数（400xxx の m）")
    ap.add_argument("--amb-hh-m-default", type=float, default=2.0,
        help="病院間転送（EH→DH 等）の救急車デフォルト台数（700xxx の m）")
    # ★ 新規: トリアージ m のデフォルト指定
    ap.add_argument("--shelter-triage-m-default", type=float, default=2.0,
        help="救護所トリアージ(300xxx triage) のデフォルト窓口数 m")
    ap.add_argument("--hy-triage-m-default", type=float, default=2.0,
        help="救護病院トリアージ(500xxx triage) のデフォルト窓口数 m")
    ap.add_argument("--hd-triage-m-default", type=float, default=3.0,
        help="災害拠点病院トリアージ(600xxx triage) のデフォルト窓口数 m")

    ap.add_argument("--amb-summary", action="store_true",
        help="service.csv を読み取り、救急車台数のサマリーを標準出力とCSVで出す")
    ap.add_argument("--config-summary", action="store_true",
        help="サービス率(μ)の前提設定と計算結果の要約を表示＆JSON保存")


    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. グリッド生成（地域）- 6桁ID対応
    if args.num_regions is not None:
        # 直接地域数が指定された場合
        region_count = args.num_regions
        # グリッドサイズを自動調整（可能な限り正方形に近く）
        rows = int(np.sqrt(region_count))
        cols = int(np.ceil(region_count / rows))
        print(f"[INFO] Using {region_count} regions with grid {rows}×{cols}")
        grid = generate_grid(args.width_km, args.height_km, rows, cols)
        # 必要な分だけ取得
        grid = grid[:region_count]
        args.grid_rows = rows  # 可視化用に保存
        args.grid_cols = cols
    else:
        # 従来通りグリッドサイズから決定
        grid = generate_grid(args.width_km, args.height_km, args.grid_rows, args.grid_cols)
        region_count = len(grid)
    
    region_node_ids = [get_node_id("region", i+1) for i in range(region_count)]

    df_regions = pd.DataFrame({
        "node_id": region_node_ids,
        "node_type": "Region",
        "x_km": [xy[0] for xy in grid],
        "y_km": [xy[1] for xy in grid],
    })

    # 2. 救護所・病院の配置（6桁ID対応）
    rng = np.random.default_rng(args.seed)
    shelters_xy = rng.uniform([0,0],[args.width_km,args.height_km], size=(args.num_shelters,2))
    hy_xy = rng.uniform([0,0],[args.width_km,args.height_km], size=(args.num_hospitals_y,2))
    hd_xy = rng.uniform([0,0],[args.width_km,args.height_km], size=(args.num_hospitals_d,2))

    # 救護所（トリアージ・治療のペア）
    shelter_nodes = []
    for i in range(args.num_shelters):
        # トリアージノード
        shelter_nodes.append({
            "node_id": get_node_id("aid_station", i+1, "triage"),
            "node_type": "ShelterTriage",
            "facility_id": i+1,
            "service": "triage",
            "x_km": shelters_xy[i,0],
            "y_km": shelters_xy[i,1],
        })
        # 治療ノード
        shelter_nodes.append({
            "node_id": get_node_id("aid_station", i+1, "treatment"),
            "node_type": "ShelterTreatment", 
            "facility_id": i+1,
            "service": "treatment",
            "x_km": shelters_xy[i,0],
            "y_km": shelters_xy[i,1],
        })
    shelters = pd.DataFrame(shelter_nodes)

    # 救護病院（トリアージ・治療のペア）
    hy_nodes = []
    for i in range(args.num_hospitals_y):
        # トリアージノード
        hy_nodes.append({
            "node_id": get_node_id("emergency_hospital", i+1, "triage"),
            "node_type": "EmergencyHospitalTriage",
            "facility_id": i+1,
            "service": "triage",
            "x_km": hy_xy[i,0],
            "y_km": hy_xy[i,1],
        })
        # 治療ノード
        hy_nodes.append({
            "node_id": get_node_id("emergency_hospital", i+1, "treatment"),
            "node_type": "EmergencyHospitalTreatment",
            "facility_id": i+1,
            "service": "treatment",
            "x_km": hy_xy[i,0],
            "y_km": hy_xy[i,1],
        })
    hy = pd.DataFrame(hy_nodes)

    # 災害拠点病院（トリアージ・治療のペア）
    hd_nodes = []
    for i in range(args.num_hospitals_d):
        # トリアージノード
        hd_nodes.append({
            "node_id": get_node_id("disaster_hospital", i+1, "triage"),
            "node_type": "DisasterHospitalTriage",
            "facility_id": i+1,
            "service": "triage",
            "x_km": hd_xy[i,0],
            "y_km": hd_xy[i,1],
        })
        # 治療ノード
        hd_nodes.append({
            "node_id": get_node_id("disaster_hospital", i+1, "treatment"),
            "node_type": "DisasterHospitalTreatment",
            "facility_id": i+1,
            "service": "treatment",
            "x_km": hd_xy[i,0],
            "y_km": hd_xy[i,1],
        })
    hd = pd.DataFrame(hd_nodes)

    # 救急車ノード
    ambulance_shelter = []
    for i in range(args.num_shelters):
        ambulance_shelter.append({
            "node_id": get_node_id("ambulance_from_shelter", i+1),
            "node_type": "AmbulanceFromShelter",
            "facility_id": i+1,
            "x_km": shelters_xy[i,0],
            "y_km": shelters_xy[i,1],
        })
    ambulance_shelter = pd.DataFrame(ambulance_shelter)

    ambulance_transfer = []
    for i in range(args.num_hospitals_y):
        ambulance_transfer.append({
            "node_id": get_node_id("ambulance_transfer", i+1),
            "node_type": "AmbulanceTransfer",
            "facility_id": i+1,
            "x_km": hy_xy[i,0],
            "y_km": hy_xy[i,1],
        })
    ambulance_transfer = pd.DataFrame(ambulance_transfer)

    # 出口ノード
    exit_nodes = pd.DataFrame([
        {"node_id": get_node_id("cure"), "node_type": "Cure", "x_km": args.width_km/2, "y_km": -5},
        {"node_id": get_node_id("death"), "node_type": "Death", "x_km": args.width_km/2, "y_km": -10}
    ])

    # 全ノード統合
    nodes = pd.concat([df_regions, shelters, hy, hd, ambulance_shelter, ambulance_transfer, exit_nodes], ignore_index=True)

    # 3. 地域人口を生成
    df_pop = generate_population(region_count,
                                mean=args.pop_mean,
                                std=args.pop_std,
                                seed=args.seed,
                                integer=args.pop_integer,
                                pop_min=args.pop_min)

    # nodesに人口情報をマージ
    nodes = nodes.merge(df_pop, on="node_id", how="left")
    nodes.to_csv(os.path.join(args.out_dir,"nodes.csv"), index=False)

    # 地域人口も別途保存
    df_pop.to_csv(os.path.join(args.out_dir,"region_population.csv"), index=False)

    incidence = getattr(args, "incidence_percent", 1.0)
    horizon   = getattr(args, "horizon_hours", 12.0)
    write_external(args.out_dir, region_node_ids, df_pop,
                arrivals_per_person=(incidence/100.0)/horizon,
                class_ratio={1:0.60, 2:0.25, 3:0.12, 4:0.03})



    # 4. 距離行列を出力（6桁ID使用）
    # 地域のトリアージノードIDを取得
    triage_shelter_ids = [get_node_id("aid_station", i+1, "triage") for i in range(args.num_shelters)]
    triage_hy_ids = [get_node_id("emergency_hospital", i+1, "triage") for i in range(args.num_hospitals_y)]
    triage_hd_ids = [get_node_id("disaster_hospital", i+1, "triage") for i in range(args.num_hospitals_d)]

    # Region->Shelter（トリアージ）
    dist_rs = pd.DataFrame([[euclid_dist(grid[i], shelters_xy[j])
                             for j in range(args.num_shelters)]
                             for i in range(region_count)],
                             index=region_node_ids,
                             columns=triage_shelter_ids)
    dist_rs.to_csv(os.path.join(args.out_dir,"dist_region_to_shelter.csv"))

    # HospitalY->HospitalD（トリアージ同士）
    dist_yd = pd.DataFrame([[euclid_dist(hy_xy[i], hd_xy[j])
                             for j in range(args.num_hospitals_d)]
                             for i in range(args.num_hospitals_y)],
                             index=triage_hy_ids,
                             columns=triage_hd_ids)
    dist_yd.to_csv(os.path.join(args.out_dir,"dist_hospitalY_to_hospitalD.csv"))

    # 統合: Shelter -> 全Hospital (Y+D)（トリアージ同士）
    all_hosp_triage_ids = triage_hy_ids + triage_hd_ids
    all_hosp_xy = np.vstack([hy_xy, hd_xy])

    dist_sh_all = pd.DataFrame(
        [[euclid_dist(shelters_xy[i], all_hosp_xy[j]) for j in range(len(all_hosp_xy))]
         for i in range(args.num_shelters)],
        index=triage_shelter_ids,
        columns=all_hosp_triage_ids
    )
    dist_sh_all.to_csv(os.path.join(args.out_dir, "dist_shelter_to_hospital.csv"))

    # ★★★ ここから③を追加（Ambulance起点の距離表）★★★
    # Ambulance(400xxx) → 全Hospital(トリアージ) の距離表
    amb_from_ids = [get_node_id("ambulance_from_shelter", i+1) for i in range(args.num_shelters)]
    dist_amb_to_h = dist_sh_all.copy()   # 数値は同じ。行ラベルだけ 400xxx に差し替え
    dist_amb_to_h.index = amb_from_ids
    dist_amb_to_h.to_csv(os.path.join(args.out_dir, "dist_ambulance_to_hospital.csv"))
    # ★★★ ③ここまで ★★★

    # 5. 救護所の魅力度（simple版：6桁node_id=ShelterTriageで出力）
    # 生成順や連番に依存しないよう、実体の shelters から triage の node_id を参照
    triage_ids = shelters.loc[shelters["service"] == "triage", "node_id"].to_numpy()
    A_final = rng.uniform(0.5, 1.5, size=triage_ids.shape[0])
    attract = (
        pd.DataFrame({"node_id": triage_ids, "A_final": A_final})
        .sort_values("node_id")            # 並び固定が不要ならこの行は削除可
        .reset_index(drop=True)
    )
    attract.to_csv(os.path.join(args.out_dir, "attractiveness_A.csv"), index=False)


    # 6. 可視化: 人口ヒートマップ + 等高線
    if args.num_regions is not None:
        # 地域数が直接指定された場合、グリッドサイズに合わせて調整
        effective_rows = args.grid_rows
        effective_cols = args.grid_cols
        # 不足分はゼロで埋める
        pop_array = np.zeros(effective_rows * effective_cols)
        pop_array[:region_count] = df_pop["population"].to_numpy()
        Z = pop_array.reshape(effective_rows, effective_cols)
        
        # グリッド座標も調整
        grid_full = generate_grid(args.width_km, args.height_km, effective_rows, effective_cols)
        X = np.array([xy[0] for xy in grid_full]).reshape(effective_rows, effective_cols)
        Y = np.array([xy[1] for xy in grid_full]).reshape(effective_rows, effective_cols)
    else:
        # 従来通り
        X = np.array([xy[0] for xy in grid]).reshape(args.grid_rows, args.grid_cols)
        Y = np.array([xy[1] for xy in grid]).reshape(args.grid_rows, args.grid_cols)
        Z = df_pop["population"].to_numpy().reshape(args.grid_rows, args.grid_cols)

    fig, ax = plt.subplots(figsize=(10,6))
    im = ax.imshow(Z, origin="lower",
                   extent=(0,args.width_km,0,args.height_km),
                   cmap="Reds", aspect="auto")
    cs = ax.contour(X, Y, Z, colors="black", linewidths=0.5)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%d")
    fig.colorbar(im, ax=ax, label="Population")
    ax.set_title("Region Population Heatmap & Contours")
    ax.set_xlabel("X [km]"); ax.set_ylabel("Y [km]")

    fig.savefig(os.path.join(args.out_dir,"population_heatmap.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 7. 配置図 (virtual_area.png) - 6桁ID版
    fig, ax = plt.subplots(figsize=(10,6))
    ax.scatter(df_regions["x_km"], df_regions["y_km"], s=10, c="lightgray", label="Regions")
    
    # 施設は物理位置で表示（トリアージ・治療は同じ位置）
    ax.scatter(shelters_xy[:,0], shelters_xy[:,1], s=40, c="blue", marker="^", label="Shelters")
    ax.scatter(hy_xy[:,0], hy_xy[:,1], s=60, c="green", marker="s", label="Emergency Hospitals")
    ax.scatter(hd_xy[:,0], hd_xy[:,1], s=80, c="red", marker="*", label="Disaster Hospitals")
    
    ax.set_xlim(0, args.width_km)
    ax.set_ylim(0, args.height_km)
    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [km]")
    ax.set_title("Virtual Medical Area Layout (6-digit Node IDs)")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(os.path.join(args.out_dir, "virtual_area.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- μ計算の前提（表示にも使う） ---
    REGION_SPEED = 5.0  # km/h

    AMB_SPEED   = 25.0  # km/h
    AMB_LOAD    = 8.0   # min
    AMB_UNLOAD  = 12.0  # min
    AMB_RESET   = 5.0   # min
    AMB_ALPHA   = 0.5   # 片道の戻り係数

    TRANS_SPEED  = 25.0 # km/h
    TRANS_LOAD   = 6.0  # min
    TRANS_UNLOAD = 10.0 # min
    TRANS_RESET  = 5.0  # min
    TRANS_ALPHA  = 0.5  # 片道の戻り係数

    region_mu_map = compute_region_mu_from_gravity(
        args.out_dir, beta_rs=0.15, kernel="exp", speed_kmph=REGION_SPEED)

    amb_mu_fcfs = compute_ambulance_mu_fcfs(
        args.out_dir, beta_sh=0.08, kernel="exp",
        speed_kmph=AMB_SPEED, alpha_return=AMB_ALPHA,
        t_load_min=AMB_LOAD, t_unload_min=AMB_UNLOAD, t_reset_min=AMB_RESET)

    trans_mu_fcfs = compute_transfer_mu_fcfs(
        args.out_dir, beta_sh=0.08, kernel="exp",
        speed_kmph=TRANS_SPEED, alpha_return=TRANS_ALPHA,
        t_load_min=TRANS_LOAD, t_unload_min=TRANS_UNLOAD, t_reset_min=TRANS_RESET)

    # 台数 m（施設ごとに変えたければここで）
    amb_m_map = {400000 + i: args.amb_sh_m_default for i in range(1, args.num_shelters + 1)}
    trans_m   = args.amb_hh_m_default  # 700xxx は一律 m（引数で指定）

    write_service_csv(
        args.out_dir, region_node_ids,
        args.num_shelters, args.num_hospitals_y, args.num_hospitals_d,
        region_mu_map=region_mu_map, classes=(1,2,3,4),
        # ★ここでFCFSを有効化
        ambulance_as_fcfs=True,
        ambulance_mu_map=amb_mu_fcfs,
        ambulance_m_map=amb_m_map,
        transfer_service_type="FCFS",
        transfer_m=trans_m,
        transfer_mu_map=trans_mu_fcfs,
        # ★ 新規: トリアージ m のデフォルト値を反映
        shelter_triage_m=args.shelter_triage_m_default,
        hy_triage_m=args.hy_triage_m_default,
        hd_triage_m=args.hd_triage_m_default,
    )

    # write_service_csv(...) の直後あたり
    if args.amb_summary:
        summarize_ambulances(
            os.path.join(args.out_dir, "service.csv"),
            os.path.join(args.out_dir, "ambulance_summary.csv"),
        )

    if args.config_summary:
        print_and_save_service_config(
            args.out_dir,
            args,
            region_speed=REGION_SPEED,
            amb_params = {"speed": AMB_SPEED, "load": AMB_LOAD, "unload": AMB_UNLOAD, "reset": AMB_RESET, "alpha": AMB_ALPHA},
            trans_params={"speed": TRANS_SPEED, "load": TRANS_LOAD, "unload": TRANS_UNLOAD, "reset": TRANS_RESET, "alpha": TRANS_ALPHA},
            triage_params={"shelter": 12.0, "hy": 12.0, "hd": 14.0},
            treatment_params={
                "shelter": {1: 12.0, 2: 6.0, 3: 3.0},
                "hy":      {1: 15.0, 2: 10.0, 3: 5.0},
                "hd":      {1: 20.0, 2: 15.0, 3: 8.0},
            },
            region_mu_map=region_mu_map,
            amb_mu_fcfs=amb_mu_fcfs,
            trans_mu_fcfs=trans_mu_fcfs,
        )



    # 既存の生成フロー
    write_routing_gravity(args.out_dir)  # Region→Shelter, Ambulance(400)→Hospital など
    write_complete_routing_with_facilities(
        args.out_dir,
        num_shelters=args.num_shelters,
        num_hy=args.num_hospitals_y,
        num_hd=args.num_hospitals_d,
        apply_T_at_hospital=True,        # 病院にもT適用ならTrue
        also_write_matrix=True
    )

    # ★ 転送 700→600 を追記しつつ、P_global も再生成
    append_transfer_to_hd(
        out_dir=args.out_dir,
        beta_sh=0.08,     # write_routing_gravity の設定に合わせる
        kernel="exp",
        update_matrix=True
    )



    print(f"[OK] Built virtual area with 6-digit node IDs and saved to {args.out_dir}")
    print(f"Total nodes: {len(nodes)}")
    print(f"- Regions: {region_count}")
    print(f"- Shelter nodes: {len(shelters)}")
    print(f"- Emergency hospital nodes: {len(hy)}")
    print(f"- Disaster hospital nodes: {len(hd)}")
    print(f"- Ambulance nodes: {len(ambulance_shelter) + len(ambulance_transfer)}")
    print(f"- Exit nodes: {len(exit_nodes)}")


if __name__ == "__main__":
    main()

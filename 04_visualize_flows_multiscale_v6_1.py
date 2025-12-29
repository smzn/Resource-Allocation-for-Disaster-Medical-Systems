#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-scale flow visualization.
Level-1: Category x Category heatmap
Level-2: Map with top-K Shelter→Hospital flows
Level-3: Node rankings (in/out/throughput)

state_outflows.csv に from/to が無い場合は、
P_global.csv と掛け合わせて flow(i->j)=lambda_ir(i)*P[i,j] を復元。
"""
import os, re, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List

# ---------- 小物 ----------
def find_col(cols, cand):
    for c in cand:
        if c in cols: return c
    raise KeyError(f"columns {list(cols)} do not contain any of {cand}")

_lbl_pat = re.compile(r"\(([^,]+),\s*([0-9]+)\)")
def parse_label(lbl: str):
    if str(lbl).lower() == "outside": return ("outside", None)
    m = _lbl_pat.match(str(lbl))
    if m: return (m.group(1), int(m.group(2)))
    try:
        nid, cls = str(lbl).strip("()").split(",")
        return (nid.strip(), int(cls))
    except Exception:
        return (str(lbl), None)

def node_group(node_id: str, nodes_df: Optional[pd.DataFrame]) -> str:
    if node_id.lower() == "outside": return "Outside"
    if nodes_df is not None:
        row = nodes_df.loc[nodes_df["node_id"]==node_id]
        if len(row):
            return str(row.iloc[0]["node_type"])
    if node_id.startswith("E"):  return "Region"
    if node_id.startswith("M"):  return "Movement"
    if node_id.startswith("A"):  return "Ambulance"
    if node_id.startswith("T_EH") or node_id.startswith("P_EH"): return "HospitalY"
    if node_id.startswith("T_DH") or node_id.startswith("P_DH"): return "HospitalD"
    if node_id.startswith("T") or node_id.startswith("P"):       return "Shelter"
    if node_id.startswith("HY"): return "HospitalY"
    if node_id.startswith("HD"): return "HospitalD"
    if node_id.startswith("S"):  return "Shelter"
    return "Other"

# ---------- 入力 ----------
def load_nodes(path:str)->Optional[pd.DataFrame]:
    if not os.path.exists(path): return None
    df = pd.read_csv(path)
    df["node_id"]=df["node_id"].astype(str)
    return df

def _read_P_global(P_path: str) -> pd.DataFrame:
    P = pd.read_csv(P_path)
    if 'label' in P.columns:
        rowlab = P['label'].astype(str); P = P.drop(columns=['label']); P.index = rowlab
    else:
        first = P.columns[0]
        if P.iloc[:,0].dtype == object:
            rowlab = P.iloc[:,0].astype(str); P = P.drop(columns=[first]); P.index = rowlab
    P.columns = [str(c) for c in P.columns]
    return P

def build_edges_from_matrix(P_path: str, so_path: str, class_filter=None) -> pd.DataFrame:
    P  = _read_P_global(P_path)
    so = pd.read_csv(so_path)

    # --- 入力チェック：label不要。node, cls, lambda列だけを使う ---
    lamcol = next((c for c in ['lambda_ir','lambda','lambda_i','throughput'] if c in so.columns), None)
    if not ({'node','cls'}.issubset(so.columns) and lamcol):
        raise KeyError(
            f"{so_path} は 'node','cls' と λ列（lambda_ir|lambda|lambda_i|throughput）が必要です（列: {list(so.columns)}）"
        )

    # state_outflows 側の index を (node, cls) タプルに
    so['node'] = so['node'].astype(str).str.strip()
    so['cls']  = so['cls'].astype(int)
    lam_tuple = so.set_index(['node','cls'])[lamcol]  # index: (node, cls)

    # P_global 側の行ラベルを (node, cls) にパースして対応表を作る
    row_labels = P.index.astype(str)
    tuple_from_label = [parse_label(s) for s in row_labels]           # -> (node, cls or None)
    # clsがNone（outside等）は除外対象（λ_i,r は r が定義される状態用）
    valid = [(t[0], t[1]) for t in tuple_from_label if t[1] is not None]
    label_by_tuple = {t: lbl for t, lbl in zip(valid, row_labels[:len(valid)])}

    # クラスフィルタ（タプルの第2要素で判定）
    if class_filter:
        lam_tuple = lam_tuple[lam_tuple.index.map(lambda t: t[1] in class_filter)]

    # 突き合わせ（共通 (node,cls) を抽出）
    common_tuples = lam_tuple.index.intersection(pd.Index(label_by_tuple.keys()))
    if len(common_tuples) == 0:
        raise ValueError("P_global と state_outflows の (node, cls) が一致しません。")

    # Pの行ラベルへマッピング
    idx_labels = [label_by_tuple[t] for t in common_tuples]
    lam = pd.Series([lam_tuple.loc[t] for t in common_tuples], index=idx_labels)

    # 行ごとに λ を掛けてフロー行列へ
    P2 = P.loc[idx_labels]
    flows = P2.mul(lam, axis=0)

    edges = flows.stack().reset_index()
    edges.columns = ['from','to','flow']
    edges = edges[edges['flow'] > 1e-12]
    return edges


def load_stateflows(path: str, P_path: Optional[str]=None, class_filter=None) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 既に from/to/flow があればそのまま使う
    if any(c in df.columns for c in ['from','state_from','i','origin']) and \
       any(c in df.columns for c in ['to','state_to','j','dest']):
        fcol = find_col(df.columns, ["from","state_from","i","origin"])
        tcol = find_col(df.columns, ["to","state_to","j","dest"])
        vcol = find_col(df.columns, ["flow","rate","value","throughput"])
        return df.rename(columns={fcol:"from", tcol:"to", vcol:"flow"})[['from','to','flow']]

    # from/to が無い → P と λ から復元（この場合 label 列は不要）
    if P_path is None:
        raise FileNotFoundError("--P-global で P_global.csv を指定してください")
    return build_edges_from_matrix(P_path, path, class_filter=class_filter)

# ---------- レベル1：カテゴリ×カテゴリ ----------
CAT_ORDER = ["Input","Region","Movement","Shelter","Ambulance","HospitalY","HospitalD","Outside"]
def category_heatmap(stateflows: pd.DataFrame, nodes: Optional[pd.DataFrame],
                     class_filter: Optional[List[int]], out_png: str):
    fr = stateflows["from"].map(lambda s: parse_label(s)[0])
    to = stateflows["to"].map(lambda s: parse_label(s)[0])
    cls = stateflows["from"].map(lambda s: parse_label(s)[1])
    Gf = fr.map(lambda nid: node_group(nid, nodes))
    Gt = to.map(lambda nid: node_group(nid, nodes))
    df = pd.DataFrame({"Gf":Gf,"Gt":Gt,"cls":cls,"flow":stateflows["flow"]})
    if class_filter: df = df[df["cls"].isin(class_filter)]
    df.loc[fr.str.lower()=="input","Gf"]="Input"
    df.loc[to.str.lower()=="outside","Gt"]="Outside"
    piv = df.pivot_table(index="Gf", columns="Gt", values="flow", aggfunc="sum").fillna(0.0)
    idx = [g for g in CAT_ORDER if g in piv.index]
    col = [g for g in CAT_ORDER if g in piv.columns]
    piv = piv.reindex(index=idx, columns=col, fill_value=0.0)

    fig = plt.figure(figsize=(8,6), dpi=160); ax = fig.add_subplot(111)
    im = ax.imshow(piv.values, origin="upper")
    ax.set_xticks(range(len(col))); ax.set_xticklabels(col, rotation=45, ha="right")
    ax.set_yticks(range(len(idx))); ax.set_yticklabels(idx)
    for i in range(len(idx)):
        for j in range(len(col)):
            val = piv.values[i,j]
            if val>0:
                ax.text(j,i,f"{val:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="flow")
    ax.set_title("Category-to-Category Flow (class={})".format(class_filter or "ALL"))
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", dpi=160); plt.close(fig)
    print("[ok] saved:", out_png)

# ---------- レベル2：地図に主要フロー ----------
def top_flows_map(stateflows: pd.DataFrame, nodes: Optional[pd.DataFrame],
                  class_filter: Optional[List[int]], out_png:str, topk:int=20, minshare:float=0.01):
    if nodes is None:
        print("[warn] nodes.csv 無し → 地図出力をスキップ"); return
    node_xy = nodes.set_index("node_id")[["x_km","y_km","node_type"]]

    fr = stateflows["from"].map(parse_label)
    to = stateflows["to"].map(parse_label)
    df = pd.DataFrame({
        "from_node": [a[0] for a in fr],
        "from_cls":  [a[1] for a in fr],
        "to_node":   [b[0] for b in to],
        "to_cls":    [b[1] for b in to],
        "flow": stateflows["flow"],
    })
    if class_filter: df = df[df["from_cls"].isin(class_filter)]
    def _type(nid):
        if nid.lower()=="outside": return "Outside"
        if nid in node_xy.index: return node_xy.loc[nid,"node_type"]
        if nid.startswith("HY"): return "HospitalY"
        if nid.startswith("HD"): return "HospitalD"
        if nid.startswith("S"):  return "Shelter"
        return "Other"
    t_from = df["from_node"].map(_type); t_to = df["to_node"].map(_type)
    df = df[(t_from=="Shelter") & (t_to.isin(["HospitalY","HospitalD"]))].copy()

    outsum = df.groupby("from_node")["flow"].transform("sum")
    share = df["flow"] / outsum.replace(0, np.nan)
    df = df[(share >= minshare)].copy()
    df = df.sort_values("flow", ascending=False).head(topk)

    fig = plt.figure(figsize=(10,6), dpi=160); ax=fig.add_subplot(111)
    xmin,xmax = nodes["x_km"].min(), nodes["x_km"].max()
    ymin,ymax = nodes["y_km"].min(), nodes["y_km"].max()
    ax.set_xlim(xmin,xmax); ax.set_ylim(ymin,ymax); ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("X (km)"); ax.set_ylabel("Y (km)")
    ax.grid(True, alpha=0.2)

    # --- 追加：エイリアスで node_type を正規化してから散布図描画 ---
    alias = {
        "Aid": "Shelter",
        "EH": "HospitalY", "HY": "HospitalY",
        "DH": "HospitalD", "HD": "HospitalD",
    }
    nodes_plot = nodes.copy()
    nodes_plot["node_type"] = nodes_plot["node_type"].replace(alias)

    for nt, m, s in [("Shelter","s",30),("HospitalY","^",50),("HospitalD","*",80)]:
        nn = nodes_plot[nodes_plot["node_type"]==nt]
        if len(nn):
            ax.scatter(nn["x_km"], nn["y_km"], marker=m, s=s, label=nt, alpha=0.9)

    # --- ここまで置換 ---

    if len(df):
        fmin, fmax = float(df["flow"].min()), float(df["flow"].max())
        for _, r in df.iterrows():
            if r["from_node"] not in node_xy.index or r["to_node"] not in node_xy.index: continue
            x1,y1 = node_xy.loc[r["from_node"],["x_km","y_km"]]
            x2,y2 = node_xy.loc[r["to_node"],["x_km","y_km"]]
            lw = 0.5 + 4.5 * ((r["flow"]-fmin) / (fmax-fmin + 1e-9))
            ax.plot([x1,x2],[y1,y2], linewidth=lw, alpha=0.6)

    # 凡例はハンドルがあるときだけ
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right")
    ax.set_title(f"Top-{topk} Shelter→Hospital flows (class={class_filter or 'ALL'}, minshare={minshare})")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", dpi=160); plt.close(fig)
    print("[ok] saved:", out_png)

# ---------- レベル3：ランキング ----------
def rankings(stateflows: pd.DataFrame, out_dir:str, topn:int=15):
    fr = stateflows["from"].map(lambda s: parse_label(s)[0])
    to = stateflows["to"].map(lambda s: parse_label(s)[0])
    df = pd.DataFrame({"f":fr,"t":to,"flow":stateflows["flow"]})
    out = df.groupby("f")["flow"].sum().rename("outflow")
    inc = df.groupby("t")["flow"].sum().rename("inflow")
    tot = (out.to_frame().join(inc, how="outer").fillna(0.0))
    tot["throughput"] = tot["inflow"]

    def _bar(series, title, fname):
        s = series.sort_values(ascending=False).head(topn)
        fig = plt.figure(figsize=(max(8, len(s)*0.45), 4), dpi=160); ax=fig.add_subplot(111)
        ax.bar(s.index, s.values)
        ax.set_title(title); ax.tick_params(axis='x', labelrotation=45)
        for lbl in ax.get_xticklabels(): lbl.set_horizontalalignment('right')
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight", dpi=160); plt.close(fig)
        print("[ok] saved:", os.path.join(out_dir, fname))
    _bar(tot["outflow"],    "Top outflow nodes",    "rank_outflow.png")
    _bar(tot["inflow"],     "Top inflow nodes",     "rank_inflow.png")
    _bar(tot["throughput"], "Top throughput nodes", "rank_throughput.png")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", default="./param_out/nodes.csv")
    ap.add_argument("--stateflows", default="./outputs/state_outflows.csv")
    ap.add_argument("--node_metrics", default="./outputs/node_metrics.csv")
    ap.add_argument("--P-global", default="./param_out/P_global.csv")
    ap.add_argument("--out-dir", default="./param_out/vis_flows")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--minshare", type=float, default=0.01)
    ap.add_argument("--class-filter", nargs="*", type=int, default=None)
    args = ap.parse_args()

    nodes = load_nodes(args.nodes) if os.path.exists(args.nodes) else None
    sf = load_stateflows(args.stateflows, args.P_global, args.class_filter)

    category_heatmap(sf, nodes, args.class_filter, os.path.join(args.out_dir, "flow_cat_heatmap.png"))
    top_flows_map(sf, nodes, args.class_filter, os.path.join(args.out_dir, "flow_map_topk.png"),
                  topk=args.topk, minshare=args.minshare)
    rankings(sf, os.path.join(args.out_dir, "ranks"), topn=15)

if __name__ == "__main__":
    main()

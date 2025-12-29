#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_visualize_congestion.py

可視化まとめ（v4.4）
- 入力:
  results_dir/
    - node_metrics.csv
    - class_node_metrics.csv  ← なくても find_class_metrics_csv() が探索
  nodes/regions:
    - nodes.csv (01の出力; node_id,node_type,x_km,y_km,...)
    - regions_grid.csv (任意; 背景グリッド用)
- 出力: results_dir/vis/ に画像一式

表示方針（統一流儀）:
  [救護所 Aid a]  :  L = Σ_r L(T_a,r) + L(P_a, Gのみ)
                      rho: ρ_T = Σ_r ρ(T_a,r), ρ_P = ρ(P_a,G) を並列棒
  [病院 HospitalY]:  L = Σ_r L(T_EH,r) + Σ_{1..3} L(P_EH,r)
                      rho: ρ_T = Σ_r ρ(T_EH,r), ρ_P = Σ_{1..3} ρ(P_EH,r)
      HospitalD   :  L = Σ_r L(T_DH,r) + Σ_{1..3} L(P_DH,r)
                      rho: 同様

地理的プロット:
  - 集約した HospitalY/HospitalD の値を HY*/HD* の各座標にブロードキャストして
    バブル、等高線、ヒートマップを描画（Aidは棒グラフのみ）

Python 3.8+対応（|型ヒントは使わない）
"""

import os
import glob
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker

# -----------------------------
# IO helpers
# -----------------------------
def safe_read_csv(path):
    if path and os.path.exists(path):
        return pd.read_csv(path)
    raise FileNotFoundError(f"CSV not found: {path}")

def _clean_metric(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

# -----------------------------
# Load geometry
# -----------------------------
def load_nodes(nodes_csv):
    """nodes.csv（01の出力）から物理座標を取得"""
    df = safe_read_csv(nodes_csv)
    need = {"node_id", "node_type", "x_km", "y_km"}
    if not need.issubset(set(df.columns)):
        raise ValueError(f"{nodes_csv} に {need} が必要です（現在: {list(df.columns)}）")
    # 型整形
    df["node_id"] = df["node_id"].astype(str)
    df["node_type"] = df["node_type"].astype(str)
    for c in ["x_km", "y_km"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_regions(regions_csv):
    """背景グリッド用（任意）。無ければ None を返す"""
    if not regions_csv or not os.path.exists(regions_csv):
        return None
    df = pd.read_csv(regions_csv)
    for c in ["x0_km", "y0_km", "x1_km", "y1_km"]:
        if c not in df.columns:
            return None
    return df

# -----------------------------
# Class-metrics auto detection
# -----------------------------
def find_class_metrics_csv(out_dir):
    """
    out_dir 配下から class_node_metrics.csv を探し、
    列名を標準化して返す。
    必須: node, cls, L, rho_i
    """
    import os
    path = os.path.join(out_dir, "class_node_metrics.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} が見つかりません")

    df = pd.read_csv(path)

    # 列名マッピング
    rename_map = {}
    if "K_ir" in df.columns:
        rename_map["K_ir"] = "L"
    if "rho_ir" in df.columns:
        rename_map["rho_ir"] = "rho_i"

    df = df.rename(columns=rename_map)

    # 型整備
    df["node"]  = pd.to_numeric(df["node"], errors="coerce")
    df["cls"]   = df["cls"]
    df["L"]     = _clean_metric(df["L"])
    df["rho_i"] = _clean_metric(df["rho_i"])

    print(f"[info] Using class metrics: {path}")
    print(f"[info] column mapping: {rename_map}")
    return df

# -----------------------------
# Aggregate logic (Aid / Hospital)
# -----------------------------
def aggregate_units_for_bars(class_metrics, nodes_csv="./param_out/nodes.csv"):
    """
    v6対応：nodes.csv の node_type で集計。
      - Aid: Triage は全クラス、Treatment は G（=cls==1 or 'G'）のみ
      - HospitalY/HospitalD: Triage/Treatment とも全クラス合算
    戻り: (df_aid, df_hosp)
    """
    import pandas as pd

    cm = class_metrics.copy()

    # 列名の正規化
    if "L" not in cm.columns and "K_ir" in cm.columns:
        cm = cm.rename(columns={"K_ir": "L"})
    if "rho_i" not in cm.columns and "rho_ir" in cm.columns:
        cm = cm.rename(columns={"rho_ir": "rho_i"})
    for c in ["L", "rho_i"]:
        cm[c] = _clean_metric(cm[c])

    # node_type を付与
    nodes = pd.read_csv(nodes_csv, dtype={"node_id": str, "node_type": str})
    cm["node"] = cm["node"].astype(str)
    cm = cm.merge(
        nodes[["node_id", "node_type"]].rename(columns={"node_id": "node"}),
        on="node",
        how="left"
    )

    # G（軽症）判定：数値1 or 'G'
    cls_up = cm["cls"].astype(str).str.upper()
    is_G = (pd.to_numeric(cm["cls"], errors="coerce") == 1) | (cls_up == "G")

    # タイプ名（あなたの nodes.csv に合わせて必要ならここだけ調整）
    AID_TRI_TYPES = ["ShelterTriage"]
    AID_TRT_TYPES = ["ShelterTreatment"]
    HY_TRI_TYPES  = ["EmergencyHospitalTriage"]
    HY_TRT_TYPES  = ["EmergencyHospitalTreatment"]
    HD_TRI_TYPES  = ["DisasterHospitalTriage"]
    HD_TRT_TYPES  = ["DisasterHospitalTreatment"]

    def sum_by_types(types, only_G=False):
        sub = cm[cm["node_type"].isin(types)].copy()
        if only_G:
            #sub = sub[is_G]
            sub = sub[is_G.reindex(sub.index).fillna(False)]
        return float(sub["L"].sum()), float(sub["rho_i"].sum())

    # Aid（救護所）: triage=全クラス, treatment=Gのみ
    L_T, rho_T = sum_by_types(AID_TRI_TYPES, only_G=False)
    L_P, rho_P = sum_by_types(AID_TRT_TYPES, only_G=True)
    df_aid = pd.DataFrame([{
        "unit": "Aid",
        "L_T": L_T, "L_P": L_P,
        "rho_T": rho_T, "rho_P": rho_P,
        "L_total": L_T + L_P,
        "rho_total": rho_T + rho_P
    }])

    # HospitalY / HospitalD は triage/treatment とも全クラス合算
    def pack(name, tri_types, trt_types):
        LT, rT = sum_by_types(tri_types, only_G=False)
        LP, rP = sum_by_types(trt_types, only_G=False)
        return {
            "unit": name,
            "L_T": LT, "L_P": LP,
            "rho_T": rT, "rho_P": rP,
            "L_total": LT + LP,
            "rho_total": rT + rP
        }

    df_hosp = pd.DataFrame([
        pack("HospitalY", HY_TRI_TYPES, HY_TRT_TYPES),
        pack("HospitalD", HD_TRI_TYPES, HD_TRT_TYPES),
    ])
    return df_aid, df_hosp


# -----------------------------
# Bars
# -----------------------------
def plot_unit_bars(df_units, outdir, prefix):
    os.makedirs(outdir, exist_ok=True)

    # L: triage + treatment stacked
    ax = df_units.set_index("unit")[["L_T", "L_P"]].plot(kind="bar", stacked=True, figsize=(8, 4))
    ax.set_ylabel("L (people)")
    ax.set_title(f"{prefix}: L (triage + treatment)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"bars_L.{prefix}.png"), dpi=150)
    plt.close()

    # rho: triage vs treatment
    ax = df_units.set_index("unit")[["rho_T", "rho_P"]].plot(kind="bar", figsize=(8, 4))
    ax.axhline(1.0, ls="--", lw=1, color="k")
    ax.set_ylabel("rho")
    ax.set_title(f"{prefix}: rho (triage vs treatment)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"bars_rho.{prefix}.png"), dpi=150)
    plt.close()

    # total (参考)
    ax = df_units.set_index("unit")[["L_total"]].plot(kind="bar", legend=False, figsize=(8, 3))
    ax.set_ylabel("L total")
    ax.set_title(f"{prefix}: L total")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"bars_Ltotal.{prefix}.png"), dpi=150)
    plt.close()

# -----------------------------
# Make physical points for map
# -----------------------------
def _load_attr_weights(out_dir, nodes_df):
    """param_out/attractiveness_A.csv があれば [node_id,node_type,A_final] を返す。なければ空DF。"""
    import os
    p = os.path.join(out_dir, "attractiveness_A.csv")
    if not os.path.exists(p):
        return pd.DataFrame(columns=["node_id","node_type","A_final"])
    a = pd.read_csv(p)
    if not {"node_id","node_type","A_final"}.issubset(a.columns):
        return pd.DataFrame(columns=["node_id","node_type","A_final"])
    a = a[["node_id","node_type","A_final"]].copy()
    a["node_id"] = a["node_id"].astype(str)
    a["node_type"] = a["node_type"].astype(str)
    a["A_final"] = pd.to_numeric(a["A_final"], errors="coerce").fillna(0.0)
    # nodes にないIDは弾く
    valid = set(nodes_df["node_id"].astype(str))
    a = a[a["node_id"].isin(valid)]
    return a

def _split_total_to_nodes(nodes_sub, total_value, weights_sub=None):
    """同一 node_type の複数点へ total_value を分配（重みA_finalがあれば比率配分、無ければ均等）。"""
    if len(nodes_sub) == 0:
        return pd.Series(dtype=float)
    if weights_sub is None or "A_final" not in weights_sub or weights_sub["A_final"].sum() <= 1e-12:
        # 均等
        return pd.Series(total_value / float(len(nodes_sub)), index=nodes_sub.index)
    w = weights_sub["A_final"].reindex(nodes_sub.index).fillna(0.0)
    s = w.sum()
    if s <= 1e-12:
        return pd.Series(total_value / float(len(nodes_sub)), index=nodes_sub.index)
    return (w / s) * float(total_value)

def broadcast_hospital_points(nodes_df, df_hosp, df_aid=None, out_dir=None):
    """
    v4.2互換：HospitalだけでなくShelterにも総量を配分して座標点を作る。
    - HospitalY/HospitalD: df_hospの L_total/rho_total を各点へ（均等 or 重み付き）
    - Shelter: df_aid の合計(L_total/rho_total)を各Shelterへ（均等 or 重み付き）
    戻り: DataFrame [node_id,type,x_km,y_km,rho_i,L]
    """
    rows = []
    # 重み
    weights = None
    if out_dir is not None:
        weights_df = _load_attr_weights(out_dir, nodes_df)
        if len(weights_df):
            weights = weights_df.set_index("node_id")

    def _distribute(node_type, total_L, total_rho):
        #sub = nodes_df[nodes_df["node_type"] == node_type].copy()
        sub = _select_points_for(node_type, nodes_df)
        if len(sub) == 0:
            return
        if weights is not None:
            # node_id で A_final を引いて、nodes_sub の index に揃える
            vals = sub["node_id"].map(weights["A_final"]).to_numpy()
            wsub = pd.DataFrame({"A_final": vals}, index=sub.index)
        else:
            wsub = None

        L_each = _split_total_to_nodes(sub.set_index(sub.index), total_L, wsub)
        R_each = _split_total_to_nodes(sub.set_index(sub.index), total_rho, wsub)
        for i, r in sub.reset_index().iterrows():
            rows.append({
                "node_id": r["node_id"], "type": node_type,
                "x_km": float(r["x_km"]), "y_km": float(r["y_km"]),
                "rho_i": float(R_each.iloc[i]),
                "L": float(L_each.iloc[i])
            })

    # HospitalY/HospitalD は個別（df_hosp）
    if len(df_hosp):
        for typ in ["HospitalY","HospitalD"]:
            if typ in set(df_hosp["unit"]):
                rec = df_hosp[df_hosp["unit"] == typ].iloc[0]
                _distribute(typ, float(rec["L_total"]), float(rec["rho_total"]))

    # Shelter は Aid の合計（df_aid 合算）を配分
    if df_aid is not None and len(df_aid):
        total_L  = float(df_aid["L_total"].sum())
        total_R  = float(df_aid["rho_total"].sum())
        _distribute("Shelter", total_L, total_R)

    df_phys = pd.DataFrame(rows)
    for c in ["rho_i","L"]:
        if c in df_phys:
            df_phys[c] = _clean_metric(df_phys[c])

    
    return df_phys

# -----------------------------
# Map plotting utils
# -----------------------------
def draw_background(ax, regions_df=None, nodes_df=None):
    """背景グリッドと目印"""
    if regions_df is not None and len(regions_df):
        # 軸スケールを領域サイズに合わせる
        x_max = float(regions_df["x1_km"].max())
        y_max = float(regions_df["y1_km"].max())
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
        # グリッド線
        for x in sorted(set(regions_df["x0_km"]).union(set(regions_df["x1_km"]))):
            ax.axvline(x, lw=0.4, alpha=0.25)
        for y in sorted(set(regions_df["y0_km"]).union(set(regions_df["y1_km"]))):
            ax.axhline(y, lw=0.4, alpha=0.25)
    else:
        # nodes から外接矩形
        if nodes_df is not None and len(nodes_df):
            x_max = float(nodes_df["x_km"].max()) * 1.02
            y_max = float(nodes_df["y_km"].max()) * 1.02
            ax.set_xlim(0, x_max)
            ax.set_ylim(0, y_max)
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")

# 追加：拠点ラベル読込（node_catalog.csv があれば優先）
def _load_node_labels(catalog_csv):
    """
    node_catalog.csv から node_id→label を作る（列名のゆらぎに対応）。
    優先順: 'label','name','display_name','node_key','node_label'
    見つからなければ None を返す。
    """
    import os
    if not catalog_csv or not os.path.exists(catalog_csv):
        return None
    df = pd.read_csv(catalog_csv)
    if "node_id" not in df.columns:
        return None
    df["node_id"] = df["node_id"].astype(str)
    for c in ["label","name","display_name","node_key","node_label"]:
        if c in df.columns:
            lab = df[["node_id", c]].copy().rename(columns={c: "label"})
            lab["label"] = lab["label"].astype(str)
            return lab
    return None

# 置き換え：ラベル付きバブル
def plot_bubble(df_phys, metric_col, out_path, title,
                catalog_csv=None, label_topn=12, label_all=False):
    """
    バブル図 + 拠点ラベル重ね描き
    - catalog_csv があれば node_id を名称に変換
    - label_topn 上位だけにラベル（label_all=True で全点）
    - Shelter/HY/HD をマーカーで描き分け & 凡例表示
    """
    if len(df_phys) == 0:
        return

    # ラベル辞書（任意）
    labels_df = _load_node_labels(catalog_csv)
    label_map = dict(zip(labels_df["node_id"], labels_df["label"])) if labels_df is not None else {}

    plt.figure(figsize=(10, 6), dpi=160)
    ax = plt.gca()
    draw_background(ax, None, df_phys.rename(columns={"x_km": "x_km", "y_km": "y_km"}))

    # 全体で正規化（サイズは値比でスケーリング）
    v = _clean_metric(df_phys[metric_col])
    vmax = float(max(v.max(), 1e-12))
    size = 2800.0 * (v / vmax)  # 以前より少し抑えめ

    # タイプごとに描き分け（凡例付き）
    markers = {"Shelter": "o", "HospitalY": "^", "HospitalD": "s"}
    for t, sub in df_phys.groupby("type"):
        idx = sub.index
        ax.scatter(sub["x_km"], sub["y_km"],
                   s=size.loc[idx].values, alpha=0.35,
                   marker=markers.get(t, "o"), edgecolor="k", linewidths=0.4,
                   label=t)

    # ラベル付け対象（上位N or 全点）
    if label_all:
        lab_df = df_phys.copy()
    else:
        lab_df = df_phys.copy().sort_values(metric_col, ascending=False).head(label_topn)

    # ラベル文字列（catalog優先 → 型+ID）
    def _fmt_label(row):
        base = label_map.get(str(row["node_id"]), f'{row["type"]}-{row["node_id"]}')
        if metric_col == "rho_i":
            valtxt = f"ρ={row[metric_col]:.2f}"
        else:
            # L は整数寄りなら桁落とし
            valtxt = f"L={row[metric_col]:.0f}" if row[metric_col] >= 10 else f"L={row[metric_col]:.2f}"
        return f"{base}\n{valtxt}"

    # 文字が重なりにくいよう軽いオフセット
    for i, r in lab_df.iterrows():
        ax.annotate(_fmt_label(r),
                    xy=(r["x_km"], r["y_km"]),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.5", alpha=0.85))

    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_title(title)
    ax.margins(0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def idw_grid(x, y, z, nx=240, ny=160, power=2):
    """v4.2互換の簡易IDW（やや高解像度）"""
    if len(x) < 3:
        return None, None, None
    xi = np.linspace(min(x), max(x), nx)
    yi = np.linspace(min(y), max(y), ny)
    XI, YI = np.meshgrid(xi, yi)
    ZI = np.zeros_like(XI, dtype=float)
    WI = np.zeros_like(XI, dtype=float)
    eps = 1e-6
    for (xi0, yi0, zi0) in zip(x, y, z):
        d = np.hypot(XI - xi0, YI - yi0) + eps
        w = 1.0 / np.power(d, power)
        ZI += w * zi0
        WI += w
    ZI /= np.maximum(WI, eps)
    return XI, YI, ZI


def plot_contour(df_phys, metric_col, regions_df, out_path, title):
    if len(df_phys) < 3:
        return
    x = df_phys["x_km"].values
    y = df_phys["y_km"].values
    z = df_phys[metric_col].values

    XI, YI, ZI = idw_grid(x, y, z, nx=240, ny=160, power=2)
    if XI is None:
        return

    plt.figure(figsize=(10, 6), dpi=160)
    ax = plt.gca()
    draw_background(ax, regions_df, df_phys)

    # 等高線（レベル多め）
    cs = ax.contour(XI, YI, ZI, levels=12)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.3g")

    # 位置マーカー（Shelterも含めて）
    markers = {"Shelter":"s","HospitalY":"^","HospitalD":"*"}
    sizes   = {"Shelter":30,"HospitalY":70,"HospitalD":90}
    for t in ["Shelter","HospitalY","HospitalD"]:
        sub = df_phys[df_phys["type"] == t]
        if len(sub):
            ax.scatter(sub["x_km"], sub["y_km"], marker=markers[t], s=sizes[t],
                       linewidths=0.8, label=t)
    ax.legend(loc="upper right")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_heat(df_phys, metric_col, regions_df, out_path, title):
    if len(df_phys) < 3:
        return
    x = df_phys["x_km"].values
    y = df_phys["y_km"].values
    z = df_phys[metric_col].values
    XI, YI, ZI = idw_grid(x, y, z, nx=140, ny=100, power=2)
    if XI is None:
        return
    plt.figure(figsize=(11, 6.5))
    ax = plt.gca()
    draw_background(ax, regions_df, df_phys)
    im = ax.imshow(ZI, origin="lower",
                   extent=[XI.min(), XI.max(), YI.min(), YI.max()],
                   aspect="auto", cmap="viridis", alpha=0.85)
    cb = plt.colorbar(im, ax=ax)
    cb.set_label(metric_col)
    # 病院位置
    for t, m, s in [("HospitalY", "^", 70), ("HospitalD", "*", 90)]:
        sub = df_phys[df_phys["type"] == t]
        if len(sub):
            ax.scatter(sub["x_km"], sub["y_km"], marker=m, s=s, c="w", edgecolor="k", linewidths=0.5, label=t)
    ax.legend(loc="upper right")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_histogram(df_phys, metric_col, out_path):
    if len(df_phys) == 0:
        return
    plt.figure(figsize=(9, 5))
    ax = plt.gca()
    for k, sub in df_phys.groupby("type"):
        v = _clean_metric(sub[metric_col])
        if len(v):
            ax.hist(v, bins=20, alpha=0.5, label=k)
    ax.set_title(f"{metric_col} distribution by type")
    ax.set_xlabel(metric_col)
    ax.set_ylabel("count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def _select_points_for(category, nodes_df):
    """
    category: 'Shelter' | 'HospitalY' | 'HospitalD'
    v6 命名から代表点（Treatment側）を抽出
    """
    if category == "Shelter":
        return nodes_df[nodes_df["node_type"] == "ShelterTreatment"].copy()
    if category == "HospitalY":
        return nodes_df[nodes_df["node_type"] == "EmergencyHospitalTreatment"].copy()
    if category == "HospitalD":
        return nodes_df[nodes_df["node_type"] == "DisasterHospitalTreatment"].copy()
    return nodes_df.iloc[0:0].copy()


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=str, default="./param_out", help="前段の出力ディレクトリ（class_* CSVがある場所）")
    ap.add_argument("--nodes", type=str, default="./param_out/nodes.csv", help="01の nodes.csv")
    ap.add_argument("--regions", type=str, default="./param_out/regions_grid.csv", help="01の regions_grid.csv（任意）")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.results)
    vis_dir = os.path.join(out_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    print("Step 5: Visualizing congestion (rho_i & L)...")

    # 読み込み
    nodes_df = load_nodes(args.nodes)
    regions_df = load_regions(args.regions)

    # class-level metrics を安全探索で読込
    class_metrics = find_class_metrics_csv(out_dir)

    # 集約: Aid & Hospital
    #df_aid, df_hosp = aggregate_units_for_bars(class_metrics)
    df_aid, df_hosp = aggregate_units_for_bars(class_metrics, nodes_csv=args.nodes)
    df_aid.to_csv(os.path.join(vis_dir, "aggregate_Aid.csv"), index=False)
    df_hosp.to_csv(os.path.join(vis_dir, "aggregate_Hospital.csv"), index=False)

    # 棒グラフ（Aid / Hospital）
    plot_unit_bars(df_aid, vis_dir, prefix="Aid")
    plot_unit_bars(df_hosp, vis_dir, prefix="Hospital")

    # 地図用ポイント（Shelter/Hospitalへブロードキャスト）
    df_phys = broadcast_hospital_points(nodes_df, df_hosp, df_aid=df_aid, out_dir=out_dir)


    # バブル
#    plot_bubble(df_phys, "rho_i", os.path.join(vis_dir, "bubble_rho_i.png"), "rho_i bubble map")
#    plot_bubble(df_phys, "L",     os.path.join(vis_dir, "bubble_L.png"),     "L bubble map")
    # 変更後（ラベル付き、上位15件に表示。catalogがあれば名称に変換）
    catalog_csv = os.path.join(out_dir, "node_catalog.csv")  # 無ければ自動で無視されます
    plot_bubble(df_phys, "rho_i", os.path.join(vis_dir, "bubble_rho_i.png"),
                "rho_i bubble map", catalog_csv=catalog_csv, label_topn=15)
    plot_bubble(df_phys, "L",     os.path.join(vis_dir, "bubble_L.png"),
                "L bubble map",   catalog_csv=catalog_csv, label_topn=15)

    # 等高線・ヒート
    plot_contour(df_phys, "rho_i", regions_df, os.path.join(vis_dir, "contour_rho_i.png"),
                 "rho_i over virtual area (contour)")
    plot_contour(df_phys, "L",     regions_df, os.path.join(vis_dir, "contour_L.png"),
                 "L over virtual area (contour)")
    plot_heat(df_phys, "rho_i", regions_df, os.path.join(vis_dir, "heat_rho_i.png"),
              "rho_i over virtual area (heat)")
    plot_heat(df_phys, "L",     regions_df, os.path.join(vis_dir, "heat_L.png"),
              "L over virtual area (heat)")

    # ヒスト
    plot_histogram(df_phys, "rho_i", os.path.join(vis_dir, "hist_rho_i.png"))
    plot_histogram(df_phys, "L",     os.path.join(vis_dir, "hist_L.png"))

    # 目安ダイアログ
    for name, df in [("Aid", df_aid), ("Hospital", df_hosp)]:
        print(f"[diag] {name}:")
        if len(df):
            print(df[["unit", "rho_T", "rho_P", "rho_total", "L_total"]].to_string(index=False))
        else:
            print(" (no rows)")
    print(f"[ok] figures saved under: {vis_dir}")
    

if __name__ == "__main__":
    main()

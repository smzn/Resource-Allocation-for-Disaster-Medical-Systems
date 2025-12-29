#!/usr/bin/env python3
# 07_AfterOptimization_v7_1.py
# SA最適化後の解（optimal_windows.csv）を service に反映し、
# OpenBCMP の標準成果物を ./outputs（または指定先）へ一括保存する後処理スクリプト。
# - 読み取り: param_out/{P_global.csv, external.csv, service.csv}, opt/optimal_windows.csv
# - 書き出し: out_dir (= ./outputs) に CSV/Excel/レポート類
# - 既存の outputs/logs/* は一切変更しません（作成/上書きしません）

import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import time

from OpenBCMP_v6_3 import OpenBCMP  # 同ディレクトリ or PYTHONPATH を想定

def _find_node_column(df: pd.DataFrame) -> str:
    cand = ['node', 'node_id', 'node_ID', 'id', 'ID', 'nodeid']
    for c in cand:
        if c in df.columns:
            return c
    for c in df.columns:
        lc = c.lower()
        if 'node' in lc or 'id' in lc:
            return c
    raise ValueError(f"ノードID列が見つかりません: {df.columns.tolist()}")

def _normalize_service_cols(df: pd.DataFrame) -> pd.DataFrame:
    # service.csv側：node_id を node に正規化
    if 'node_id' in df.columns and 'node' not in df.columns:
        df = df.rename(columns={'node_id': 'node'})
    return df

def main():
    ap = argparse.ArgumentParser(description="最適化結果から OpenBCMP 成果物を別プログラムで保存（後処理）")
    # 入出力ルート
    ap.add_argument("--param-dir", type=str, default="./param_out", help="初期パラメタのディレクトリ")
    ap.add_argument("--opt-dir",   type=str, default="./opt",        help="最適化結果のディレクトリ（optimal_windows.csv 想定）")
    #ap.add_argument("--out-dir",   type=str, default="./outputs",    help="成果物の書き出し先")
    ap.add_argument("--out-dir",   type=str, default="./outputs/afteropt", help="成果物の書き出し先（固定ミラー）")
    ap.add_argument("--solution-csv", type=str, default=None,        help="解CSVの明示指定（既定: <opt-dir>/optimal_windows.csv）")

    # 生成物オプション
    ap.add_argument("--make-excel",   action="store_true", help="bcmp_results.xlsx を出力")
    ap.add_argument("--draw-network", action="store_true", help="network_exact.png を出力（小規模向け）")

    # 運用オプション
    ap.add_argument("--verbose",        action="store_true", help="進捗を表示")
    ap.add_argument("--ensure-unique",  action="store_true", help="出力先をタイムスタンプ付きサブフォルダにして既存上書きを避ける")
    ap.add_argument("--log-file",       type=str, default=None,
                    help="後処理の実行ログを追記するファイルパス（省略時はログを書きません。例: outputs/logs/step3_afteropt.log）")

    args = ap.parse_args()

    # パス解決
    param_dir = Path(args.param_dir)
    opt_dir   = Path(args.opt_dir)

    base_out = Path(args.out_dir)
    #out_dir  = base_out / f"afteropt_{time.strftime('%Y%m%d_%H%M%S')}" if args.ensure_unique else base_out
    out_dir  = base_out
    out_dir.mkdir(parents=True, exist_ok=True)

    # 入力ファイル
    file_P   = param_dir / "P_global.csv"
    file_ext = param_dir / "external.csv"
    file_svc = param_dir / "service.csv"
    solution_csv = Path(args.solution_csv) if args.solution_csv else (opt_dir / "optimal_windows.csv")

    for p in [file_P, file_ext, file_svc, solution_csv]:
        if not Path(p).exists():
            raise FileNotFoundError(f"必要ファイルが見つかりません: {p}")

    # ロガー（任意・追記モード）— outputs/logs は指定しない限り触りません
    def _log(msg: str):
        if args.verbose:
            print(msg, flush=True)
        if args.log_file:
            Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
            with open(args.log_file, "a", encoding="utf-8") as f:
                f.write(msg.rstrip() + "\n")

    _log("=== AfterOptimization: start ===")
    _log(f"param_dir={param_dir} | opt_dir={opt_dir} | out_dir={out_dir}")
    _log(f"solution_csv={solution_csv}")

    # 解を読み込み（node列名は動的に取得）
    sol_df = pd.read_csv(solution_csv)
    node_col_sol = _find_node_column(sol_df)
    if "optimal_windows" not in sol_df.columns:
        raise ValueError(f"{solution_csv} に 'optimal_windows' 列がありません")
    sol_map = {int(r[node_col_sol]): int(r["optimal_windows"]) for _, r in sol_df.iterrows()}

    # service に m を反映（FCFSのみ）
    svc = pd.read_csv(file_svc)
    svc = _normalize_service_cols(svc)
    node_col = _find_node_column(svc)

    svc_type_col = None
    for c in svc.columns:
        if 'service' in c.lower() and 'type' in c.lower():
            svc_type_col = c
            break
    if svc_type_col is None:
        raise ValueError("service.csv に service_type 列が見当たりません")

    if 'm' not in svc.columns:
        svc['m'] = np.nan

    # ★ 修正：ここが途中で切れていた箇所
    mask_fcfs = svc[svc_type_col].astype(str).str.upper() == "FCFS"

    for i in svc[mask_fcfs].index:
        nid = int(svc.loc[i, node_col])
        if nid in sol_map:
            svc.loc[i, 'm'] = int(max(1, sol_map[nid]))

    # 再現用に最終serviceを保存（出力先は out_dir）
    svc_final_path = out_dir / "service_final.csv"
    svc.to_csv(svc_final_path, index=False)
    _log(f"[info] service_final.csv saved -> {svc_final_path}")

    # OpenBCMPで理論値を算出し、out_dir に保存
    bcmp = OpenBCMP(routing_labels=str(file_P), external=str(file_ext),
                    service=str(svc_final_path), verbose=args.verbose)
    bcmp.solve_lambda()
    bcmp.compute_metrics()
    bcmp.save_outputs(str(out_dir), make_excel=args.make_excel)

    # 混雑レポート & 出口フロー
    bcmp.congestion_report(out_dir=str(out_dir), top_k=5, warn_on_inf=True, print_report=False)
    bcmp.export_state_outflows(out_dir=str(out_dir), add_labels=True)

    # 任意：ネットワーク図（規模が大きいと重いので必要時のみ）
    if args.draw_network:
        try:
            bcmp.draw_network_exact(out_path=str(out_dir / "network_exact.png"))
        except Exception as e:
            _log(f"[WARN] draw_network_exact failed: {e}")

    _log("=== AfterOptimization: done ===")
    print("\n[done] 後処理完了。出力先:", out_dir.resolve())
    print("- run_summary.csv / traffic_solution.csv / node_metrics.csv / class_node_metrics.csv")
    print("- network_class_summary.csv / diagnostics.csv")
    if args.make_excel: print("- bcmp_results.xlsx")
    print("- congestion_top5_by_layer.csv / unstable_nodes.csv / unstable_states_classlevel.csv")
    print("- state_outflows.csv / service_final.csv")
    if args.draw_network: print("- network_exact.png")
    if args.log_file: print(f"- ログ追記: {args.log_file}")

if __name__ == "__main__":
    main()

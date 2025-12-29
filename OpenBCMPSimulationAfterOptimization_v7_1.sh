#!/usr/bin/env bash
# run_sim_afteropt.sh
# 最適化後の service_final.csv を service.csv にして一時 param を組み、
# 05_VirtualOpenBCMPSimulation_v4_1.py を実行する。
# - 既存ファイルは上書きしない（出力は時刻付きディレクトリ）
# - 理論値は outputs/afteropt（最適化後の class_node_metrics.csv）を参照

set -euo pipefail
trap 'echo "[ERROR] failed at line $LINENO"; exit 1' ERR

# ========= 設定（必要なら変更） =========
PARAM_DIR="./param_out"
AFTEROPT_DIR="./outputs/afteropt"   # 固定運用
VIRTUAL_DIR="./out_virtual_area"    # あればそのまま指定、無ければ存在チェックで警告のみ
SIM_PARAM_DIR="./param_for_sim_afteropt"
SIM_OUT_DIR="./sim_outputs_afteropt_$(date +%Y%m%d_%H%M%S)"
PY_SIM="05_VirtualOpenBCMPSimulation_v4_1.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# シミュレーション条件（環境変数で上書き可能）
SEED="${SEED:-1104}"
T_END="${T_END:-50000}"
WARMUP="${WARMUP:-10000}"
SNAP_DT="${SNAP_DT:-10.0}"
LOG_EVERY="${LOG_EVERY:-50000}"
MAX_EVENTS="${MAX_EVENTS:-10000000}"

# ========= 前提チェック =========
[[ -f "$PY_SIM" ]] || { echo "[ERROR] not found: $PY_SIM"; exit 1; }
[[ -f "$PARAM_DIR/P_global.csv" ]] || { echo "[ERROR] $PARAM_DIR/P_global.csv not found"; exit 1; }
[[ -f "$PARAM_DIR/external.csv" ]] || { echo "[ERROR] $PARAM_DIR/external.csv not found"; exit 1; }
[[ -f "$AFTEROPT_DIR/service_final.csv" ]] || { echo "[ERROR] $AFTEROPT_DIR/service_final.csv not found (run 07 first)"; exit 1; }

if [[ ! -f "$AFTEROPT_DIR/class_node_metrics.csv" ]]; then
  echo "[warn] $AFTEROPT_DIR/class_node_metrics.csv not found -> RMSE比較はスキップされます"
fi
if [[ ! -d "$VIRTUAL_DIR" ]]; then
  echo "[warn] $VIRTUAL_DIR not found -> メタ情報のみ、実行は継続します"
fi

# ========= 一時 param の構築（上書きなし） =========
mkdir -p "$SIM_PARAM_DIR" "$SIM_OUT_DIR"
cp -f "$PARAM_DIR/P_global.csv" "$SIM_PARAM_DIR/"
cp -f "$PARAM_DIR/external.csv" "$SIM_PARAM_DIR/"

# service_final.csv → service.csv（列名 node→node_id の互換調整）
"$PYTHON_BIN" - <<'PY'
import pandas as pd, sys
src = "./outputs/afteropt/service_final.csv"
dst = "./param_for_sim_afteropt/service.csv"
df = pd.read_csv(src)
if "node_id" not in df.columns and "node" in df.columns:
    df = df.rename(columns={"node": "node_id"})
# safety: m は整数最小1
if "m" in df.columns:
    df["m"] = df["m"].fillna(1).astype(int).clip(lower=1)
df.to_csv(dst, index=False)
print(f"[ok] wrote {dst}")
PY

echo "[info] SIM_PARAM_DIR = $SIM_PARAM_DIR"
echo "[info] SIM_OUT_DIR   = $SIM_OUT_DIR"

# ========= 実行コマンド組み立て =========
CMD=( "$PYTHON_BIN" "$PY_SIM"
  --param-dir "$SIM_PARAM_DIR"
  --virtual-area-dir "$VIRTUAL_DIR"
  --theory-dir "$AFTEROPT_DIR"
  --out-dir "$SIM_OUT_DIR"
  --seed "$SEED"
  --T-end "$T_END"
  --warmup "$WARMUP"
  --snapshot-dt "$SNAP_DT"
  --log-every "$LOG_EVERY"
  --max-events "$MAX_EVENTS"
  --use-gravity
)

printf '[debug] '; printf '%q ' "${CMD[@]}"; echo
"${CMD[@]}"

echo
echo "[OK] Simulation started. Results will be saved under:"
echo "     $SIM_OUT_DIR"

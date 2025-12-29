#!/usr/bin/env bash
# run_07_03_04.sh (fixed)
# 07_AfterOptimization_v7_1.py -> 03_visualize_congestion_v4_4.py -> 04_visualize_flows_multiscale_v6_1.py
# * 07 の成果は固定先 ./outputs/afteropt へ上書き出力
# * 03/04 は ./outputs/afteropt をそのまま参照
# * outputs/logs は維持（07 は追記）

set -euo pipefail
trap 'echo "[ERROR] failed at line $LINENO"; exit 1' ERR

# === paths ===
PARAM_DIR="./param_out"
OPT_DIR="./opt"
OUT_DIR="./outputs"
AFTEROPT_DIR="$OUT_DIR/afteropt"

PY07="07_AfterOptimization_v7_3.py"
PY03="03_visualize_congestion_v4_4.py"
PY04="04_visualize_flows_multiscale_v6_1.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# === 07 のオプション（必要に応じて調整）===
MAKE_EXCEL="--make-excel"   # 不要なら空に
DRAW_NETWORK=""             # 小規模のみ "--draw-network"
VERBOSE="--verbose"
LOG_DIR="$OUT_DIR/logs"
LOG_FILE_STEP3="$LOG_DIR/step3_afteropt.log"

# 前提チェック
for f in "$PY07" "$PY03" "$PY04"; do
  [[ -f "$f" ]] || { echo "[ERROR] not found: $f"; exit 1; }
done
mkdir -p "$OUT_DIR" "$LOG_DIR" "$AFTEROPT_DIR"

echo "== Step 1/3: AfterOptimization (07) =="
# 固定先に出力（タイムスタンプなし・上書き可）
"$PYTHON_BIN" "$PY07" \
  --param-dir "$PARAM_DIR" \
  --opt-dir "$OPT_DIR" \
  --out-dir "$AFTEROPT_DIR" \
  $MAKE_EXCEL $DRAW_NETWORK $VERBOSE \
  --log-file "$LOG_FILE_STEP3"

# 可視化入力は固定の afteropt
LATEST_DIR="$AFTEROPT_DIR"
echo "[info] results dir for visualization: $LATEST_DIR"

# ========== Step 2: 03 可視化 ==========
echo "== Step 2/3: Visualize congestion (03) =="
cmd03=( "$PYTHON_BIN" "$PY03" --results "$LATEST_DIR" )
[[ -f "$PARAM_DIR/nodes.csv" ]]        && cmd03+=( --nodes "$PARAM_DIR/nodes.csv" )
[[ -f "$PARAM_DIR/regions_grid.csv" ]] && cmd03+=( --regions "$PARAM_DIR/regions_grid.csv" )
printf '[debug] '; printf '%q ' "${cmd03[@]}"; echo
"${cmd03[@]}"

# ========== Step 3: 04 フロー多尺度可視化 ==========
echo "== Step 3/3: Visualize flows multi-scale (04) =="
HELP_04="$("$PYTHON_BIN" "$PY04" -h 2>&1 || true)"
if grep -q -- '--stateflows' <<<"$HELP_04"; then
  OUT_VF="$LATEST_DIR/vis_flows"  # afteropt 配下なので既存成果は保持
  cmd04=( "$PYTHON_BIN" "$PY04"
          --nodes "$PARAM_DIR/nodes.csv"
          --stateflows "$LATEST_DIR/state_outflows.csv"
          --P-global "$PARAM_DIR/P_global.csv"
          --out-dir "$OUT_VF"
          --topk 20
          --minshare 0.01
          --class-filter 1 2 3 )
else
  cmd04=( "$PYTHON_BIN" "$PY04" --results "$LATEST_DIR" )
  [[ -f "$PARAM_DIR/nodes.csv" ]] && cmd04+=( --nodes "$PARAM_DIR/nodes.csv" )
fi
printf '[debug] '; printf '%q ' "${cmd04[@]}"; echo
"${cmd04[@]}"

echo "[OK] All done."
echo " - AfterOptimization: $LATEST_DIR"
echo " - Visuals(03):      $LATEST_DIR/vis"
echo " - Visuals(04):      $LATEST_DIR/vis_flows  (または 04 の既定先)"
echo " - Logs:             $OUT_DIR/logs/"

#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# Step 2専用 実行スクリプト（Step1は既に完了している前提）
# =========================================================

# -------- User settings --------
PARAM_OUT="./param_out"          # Step1で生成済みのパラメータディレクトリ
OUTPUTS="./outputs"
OPT_DIR="./opt"

OPT_PY="06_OpenBCMPOptimization_v7_11.py"  # SA最適化エントリポイント
OPT_SEED=1116000                          # 最適化の乱数シード

# SA params
INITIAL_TEMP=90
COOLING_RATE=0.994

# 予算比: 第2段階の最大窓口数 = ceil( ratio × 第1段階W* )
BUDGET_RATIO="1.10"

# Optional
USE_VENV="../myopenbcmp/bin/activate"     # あれば有効化
DEBUG_TRACE=0                              # 1で set -x

# -------- Prep --------
[[ "${DEBUG_TRACE}" -eq 1 ]] && set -x
trap 'echo "[ERROR] Script failed at line $LINENO" >&2' ERR

# 出力系は作成
mkdir -p "${OUTPUTS}" "${OPT_DIR}" "${OUTPUTS}/logs"

# venv（存在すれば）有効化
if [[ -f "${USE_VENV}" ]]; then
  # shellcheck source=/dev/null
  source "${USE_VENV}"
  echo "[INFO] Activated venv: ${USE_VENV}"
else
  echo "[INFO] No venv found at ${USE_VENV}, continue without it."
fi

# -------- Preflight: Step2の必須物チェック --------
if [[ ! -d "${PARAM_OUT}" ]]; then
  echo "[FATAL] ${PARAM_OUT} not found. Point PARAM_OUT to Step1の出力先."
  exit 1
fi

for f in external.csv service.csv P_global.csv; do
  if [[ ! -f "${PARAM_OUT}/${f}" ]]; then
    echo "[FATAL] Missing ${PARAM_OUT}/${f}. Step1の生成物を確認してください。"
    exit 1
  fi
done

if [[ ! -f "./02_OpenBCMP_v6_2.py" ]]; then
  echo "[FATAL] 02_OpenBCMP_v6_2.py が見つかりません（06_* で import されます）。"
  exit 1
fi

if [[ ! -f "${OPT_PY}" ]]; then
  echo "[FATAL] Optimizer entrypoint not found: ${OPT_PY}"
  exit 1
fi

echo "[OK] Preflight passed."

# -------- Step 2: Two-stage SA optimization (lexicographic) --------
echo "===== Step 2: Lexicographic SA optimization (ratio=${BUDGET_RATIO}) ====="

OPT_CMD=(
  python3 "${OPT_PY}"
  --param-dir "${PARAM_OUT}"
  --out-dir "${OPT_DIR}"
  --lexi
  # --use-full-budget           # 付けると =B（通常は ≤B）
  # --lexi-max-mult 5.0
  --ub-mult 500
  --weight-unstable 5000
  --initial-temp "${INITIAL_TEMP}"
  --cooling-rate "${COOLING_RATE}"
  --seed "${OPT_SEED}"
  --budget-ratio "${BUDGET_RATIO}"
  --require-stable-stage2
  --progress-interval 20
  --trace-mode accept           # accept/best/all
  --show-decision-vars
  --verbose
  --save-report
)

echo "[CMD] ${OPT_CMD[*]}"
"${OPT_CMD[@]}" 2>&1 | tee "${OUTPUTS}/logs/step2_opt.log"

echo "===== ALL DONE ====="
echo "Param dir : ${PARAM_OUT}"
echo "Opt dir   : ${OPT_DIR}"
echo "Logs      : ${OUTPUTS}/logs"

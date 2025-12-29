#!/usr/bin/env bash
set -euo pipefail

# =========================
# User settings (edit here)
# =========================
PARAM_OUT="./param_out"
OUTPUTS="./outputs"
OPT_DIR="./opt"

# Python entrypoints
PY01="01_virtual_medical_area_v7_1.py"   # 生成スクリプト
OPT_PY="06_OpenBCMPOptimization_v7_7.py"        # SA最適化（2段階・率指定）

# Seeds
BUILD_SEED=20251104000    # 仮想地域生成の乱数シード
OPT_SEED=1104000               # 最適化の乱数シード

# Ambulance defaults
AMB_SH_M_DEFAULT=1        # 400xxx: 救護所→病院のデフォ台数 m
AMB_HH_M_DEFAULT=1        # 700xxx: 病院間転送のデフォ台数 m
AMB_SUMMARY=1             # 1ならサマリ出力
AMB_M_FILE=""             # 施設ごと台数CSV（使うときだけパス設定）

# Triage window defaults (★追加)
SHELTER_TRIAGE_M_DEFAULT=1  # 300xxx triage m
HY_TRIAGE_M_DEFAULT=1       # 500xxx triage m
HD_TRIAGE_M_DEFAULT=1       # 600xxx triage m

# Arrival model (必要なら調整)
INCIDENCE_PERCENT=0.5     # 地域人口の x% が被災
HORIZON_HOURS=72          # 到着率の平均化時間幅[h]

# SA (Simulated Annealing) params
INITIAL_TEMP=90
COOLING_RATE=0.98

# 率：第2段階の最大窓口数 = ceil( 率 × 第1段階の最小必要窓口数 )
BUDGET_RATIO="1.10"

# Optional toggles
USE_VENV="../myopenbcmp/bin/activate"   # 存在すれば有効化
DEBUG_TRACE=0                           # 1にすると set -x

# =========================
# Prep
# =========================
[[ "${DEBUG_TRACE}" -eq 1 ]] && set -x
trap 'echo "[ERROR] Script failed at line $LINENO" >&2' ERR

mkdir -p "${PARAM_OUT}" "${OUTPUTS}" "${OPT_DIR}" "${OUTPUTS}/logs"

# 仮想環境（存在すれば）有効化
if [[ -f "${USE_VENV}" ]]; then
  # shellcheck source=/dev/null
  source "${USE_VENV}"
  echo "[INFO] Activated venv: ${USE_VENV}"
else
  echo "[INFO] No venv found at ${USE_VENV}, continue without it."
fi

# =========================
# Step 1: Build virtual area & parameters
# =========================
echo "===== Step 1: Build virtual medical area & parameters ====="

BUILD_CMD=(
  python3 "${PY01}"
  --out-dir "${PARAM_OUT}"
  --seed "${BUILD_SEED}"
  --amb-sh-m-default "${AMB_SH_M_DEFAULT}"
  --amb-hh-m-default "${AMB_HH_M_DEFAULT}"
  --shelter-triage-m-default "${SHELTER_TRIAGE_M_DEFAULT}"
  --hy-triage-m-default "${HY_TRIAGE_M_DEFAULT}"
  --hd-triage-m-default "${HD_TRIAGE_M_DEFAULT}"
  --incidence-percent "${INCIDENCE_PERCENT}"
  --horizon-hours "${HORIZON_HOURS}"
  --config-summary
)
# 任意フラグの付与
[[ "${AMB_SUMMARY}" -eq 1 ]] && BUILD_CMD+=( --amb-summary )
[[ -n "${AMB_M_FILE}" ]] && BUILD_CMD+=( --ambulance-m-file "${AMB_M_FILE}" )

echo "[CMD] ${BUILD_CMD[*]}"
"${BUILD_CMD[@]}" 2>&1 | tee "${OUTPUTS}/logs/step1_build.log"

# =========================
# Step 2: Two-stage SA optimization with ratio
# =========================
echo "===== Step 2: Lexicographic SA optimization (ratio=${BUDGET_RATIO}) ====="

OPT_CMD=(
  python3 "${OPT_PY}"
  --param-dir "${PARAM_OUT}"
  --out-dir "${OPT_DIR}"
  --lexi
  # --use-full-budget           # ← 付けると =B（等式）。普段は ≤B にしたいので外す
  # --lexi-max-mult 5.0         # ← 将来用。残してもいいが今は未使用に近い
  --ub-mult 8                   # 初期mの8倍まで許容（Amb系の安定化を容易に）
  --weight-unstable 5000        # 不安定罰則を強化
  --initial-temp "${INITIAL_TEMP}"
  --cooling-rate 0.994          # 温度低下を緩やかに
  --seed "${OPT_SEED}"
  --budget-ratio "${BUDGET_RATIO}"
  --require-stable-stage2       # ★ 第2段階は探索中も n_unstable=0 を強制（推奨）
  --progress-interval 20        # 進捗行の頻度（任意）
  --trace-mode accept           # トレース量（任意: accept/best/all）
  --verbose
  --save-report
)

echo "[CMD] ${OPT_CMD[*]}"
"${OPT_CMD[@]}" 2>&1 | tee "${OUTPUTS}/logs/step2_opt.log"



echo "===== ALL DONE ====="
echo "Param dir : ${PARAM_OUT}"
echo "Opt dir   : ${OPT_DIR}"
echo "Logs      : ${OUTPUTS}/logs"

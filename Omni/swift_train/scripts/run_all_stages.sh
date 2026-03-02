#!/bin/bash
# ================================================================
# 全流程一键训练：Stage 1 → 2 → 3 → 4 → 5
#
# 依次执行五个阶段，每个阶段自动传递 checkpoint 到下一阶段。
# ================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── 全局配置 ──────────────────────────────────────────
export MODEL="${MODEL:-Qwen/Qwen2.5-Omni-7B}"
export NUM_GPUS="${NUM_GPUS:-8}"
BASE_OUTPUT="${BASE_OUTPUT:-${PROJECT_DIR}/output}"

echo "============================================================"
echo "  Omni 全流程训练"
echo "============================================================"
echo "  模型:     ${MODEL}"
echo "  GPU:      ${NUM_GPUS}"
echo "  输出:     ${BASE_OUTPUT}"
echo "============================================================"

START_TIME=$(date +%s)

# ── Stage 1: 视觉-语言对齐 ──────────────────────────────
echo ""
echo ">>> [1/5] Stage 1: 视觉-语言对齐..."
export OUTPUT_DIR="${BASE_OUTPUT}/stage1"
bash "${SCRIPT_DIR}/run_stage1.sh"
STAGE1_CKPT=$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -1 || echo "")
echo "Stage 1 checkpoint: ${STAGE1_CKPT:-无}"

# ── Stage 2: 音频-语言对齐 ──────────────────────────────
echo ""
echo ">>> [2/5] Stage 2: 音频-语言对齐..."
export OUTPUT_DIR="${BASE_OUTPUT}/stage2"
export STAGE1_CKPT
bash "${SCRIPT_DIR}/run_stage2.sh"
STAGE2_CKPT=$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -1 || echo "")
echo "Stage 2 checkpoint: ${STAGE2_CKPT:-无}"

# ── Stage 3: 全模态联合 SFT ─────────────────────────────
echo ""
echo ">>> [3/5] Stage 3: 全模态联合 SFT..."
export OUTPUT_DIR="${BASE_OUTPUT}/stage3"
bash "${SCRIPT_DIR}/run_stage3.sh"
STAGE3_CKPT=$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -1 || echo "")
echo "Stage 3 checkpoint: ${STAGE3_CKPT:-无}"

# ── Stage 4: 语音生成训练 ────────────────────────────────
echo ""
echo ">>> [4/5] Stage 4: 语音生成训练..."
export OUTPUT_DIR="${BASE_OUTPUT}/stage4"
export STAGE3_CKPT
export MODE="swift"
bash "${SCRIPT_DIR}/run_stage4.sh"

# ── Stage 5: DPO 偏好对齐 ───────────────────────────────
echo ""
echo ">>> [5/5] Stage 5: DPO 偏好对齐..."
export OUTPUT_DIR="${BASE_OUTPUT}/stage5"
export RLHF_TYPE="dpo"
bash "${SCRIPT_DIR}/run_stage5.sh"

# ── 总结 ─────────────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED=$(( (END_TIME - START_TIME) / 3600 ))

echo ""
echo "============================================================"
echo "  全流程训练完成！"
echo "============================================================"
echo "  总耗时: ${ELAPSED} 小时"
echo "  输出目录: ${BASE_OUTPUT}"
echo ""
echo "  各阶段 checkpoint:"
for stage in 1 2 3 4 5; do
    ckpt=$(ls -d "${BASE_OUTPUT}/stage${stage}"/checkpoint-* 2>/dev/null | sort -V | tail -1 || echo "无")
    echo "    Stage ${stage}: ${ckpt}"
done
echo ""
echo "  下一步：运行评测"
echo "    bash eval/run_eval.sh --model ${BASE_OUTPUT}/stage5"
echo "============================================================"

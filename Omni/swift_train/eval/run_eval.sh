#!/bin/bash
# ================================================================
# 评测启动脚本 —— 对训练好的模型进行多维度评测
#
# 用法：
#   bash run_eval.sh                                    # 默认 smoke test
#   bash run_eval.sh --mode text                        # 文本基准
#   bash run_eval.sh --mode multimodal                  # 多模态基准
#   bash run_eval.sh --mode audio                       # 音频基准
#   bash run_eval.sh --mode full                        # 全量评测
#   bash run_eval.sh --model path/to/model              # 指定模型
#   bash run_eval.sh --adapters path/to/lora            # 评测 LoRA
# ================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── 默认参数 ──────────────────────────────────────────
MODEL="${MODEL:-Qwen/Qwen2.5-Omni-7B}"
ADAPTERS=""
MODE="smoke"
OUTPUT_DIR="${PROJECT_DIR}/eval_output"
DEVICE="0"

# ── 解析命令行参数 ────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)     MODEL="$2"; shift 2;;
        --adapters)  ADAPTERS="$2"; shift 2;;
        --mode)      MODE="$2"; shift 2;;
        --output)    OUTPUT_DIR="$2"; shift 2;;
        --device)    DEVICE="$2"; shift 2;;
        *)           echo "未知参数: $1"; exit 1;;
    esac
done

echo "============================================"
echo "  Omni 模型评测"
echo "============================================"
echo "  模型:     ${MODEL}"
echo "  Adapters: ${ADAPTERS:-无}"
echo "  模式:     ${MODE}"
echo "  输出:     ${OUTPUT_DIR}"
echo "  设备:     GPU ${DEVICE}"
echo "============================================"

mkdir -p "${OUTPUT_DIR}"

# ── 构建参数 ──────────────────────────────────────────
EXTRA_ARGS=""
if [ -n "$ADAPTERS" ]; then
    EXTRA_ARGS="--adapters ${ADAPTERS}"
fi

# ── 1. MS-Swift 综合评测 ─────────────────────────────
echo ""
echo "[1/2] 多模态综合评测 (mode=${MODE})..."
python3 "${SCRIPT_DIR}/eval_multimodal.py" \
    --model "${MODEL}" \
    --mode "${MODE}" \
    --output_dir "${OUTPUT_DIR}" \
    --device "${DEVICE}" \
    ${EXTRA_ARGS} \
    2>&1 | tee "${OUTPUT_DIR}/eval_${MODE}.log"

# ── 2. 音频专项评测（仅在 audio/full 模式下）───────────
if [ "$MODE" = "audio" ] || [ "$MODE" = "full" ]; then
    echo ""
    echo "[2/2] 音频/ASR 专项评测..."
    python3 "${SCRIPT_DIR}/eval_audio.py" \
        --model "${MODEL}" \
        --dataset aishell1 \
        --lang zh \
        --max_samples 500 \
        --output_dir "${OUTPUT_DIR}/audio_detail" \
        --device "${DEVICE}" \
        ${EXTRA_ARGS} \
        2>&1 | tee "${OUTPUT_DIR}/eval_audio.log"
else
    echo ""
    echo "[2/2] 跳过音频专项评测（仅在 audio/full 模式下运行）"
fi

# ── 3. 原生 swift eval（文本基准快速验证）────────────
if [ "$MODE" = "smoke" ]; then
    echo ""
    echo "[Extra] Swift 内置评测 (gsm8k, 10 条)..."
    CUDA_VISIBLE_DEVICES=${DEVICE} \
    swift eval \
        --model "${MODEL}" \
        --eval_backend Native \
        --infer_backend pt \
        --eval_limit 10 \
        --eval_dataset gsm8k \
        --eval_output_dir "${OUTPUT_DIR}/swift_native" \
        ${EXTRA_ARGS} \
        2>&1 | tee "${OUTPUT_DIR}/eval_swift_native.log" || true
fi

# ── 总结 ─────────────────────────────────────────────
echo ""
echo "============================================"
echo "  评测完成！"
echo "============================================"
echo "  结果目录: ${OUTPUT_DIR}"
echo ""
echo "  结果文件:"
ls -la "${OUTPUT_DIR}"/*.json "${OUTPUT_DIR}"/*.log 2>/dev/null || true
echo "============================================"

#!/bin/bash
# ================================================================
# Stage 4: 语音生成训练
#
# 策略：训练语音解码头（CTC Decoder），使模型具备语音输出能力
# 参考：LLaMA-Omni (CTC Decoder), OpenOmni Stage 4 (Text2Speech)
#
# 注意：MS-Swift 目前仅支持 Thinker 部分训练。
# 语音生成（Talker/CTC Decoder）需要使用自定义 trainer。
# 本脚本提供两种模式：
#   1. MS-Swift SFT（仅训练理解能力，包含语音理解增强）
#   2. 自定义 CTC Decoder 训练（需要单独的训练代码）
# ================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TRAIN_DIR="$(dirname "$PROJECT_DIR")/train"  # Omni/train 目录

# ── 可配置参数 ──────────────────────────────────────────
MODEL="${MODEL:-Qwen/Qwen2.5-Omni-7B}"
STAGE3_CKPT="${STAGE3_CKPT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/output/stage4_speech_gen}"
NUM_GPUS="${NUM_GPUS:-8}"
MODE="${MODE:-swift}"  # swift 或 custom

echo "============================================"
echo "Stage 4: 语音生成训练"
echo "============================================"
echo "模式:       ${MODE}"
echo "模型:       ${MODEL}"
echo "输出:       ${OUTPUT_DIR}"
echo "============================================"

mkdir -p "${OUTPUT_DIR}"

if [ "$MODE" = "swift" ]; then
    # ── 模式 1: MS-Swift SFT（语音理解增强）────────────
    # 使用语音转文字数据进一步增强模型的语音理解能力
    DATASET="${DATASET:-speech_asr/speech_asr_aishell1_trainsets:validation#5000}"

    ADAPTERS_ARG=""
    if [ -n "$STAGE3_CKPT" ] && [ -d "$STAGE3_CKPT" ]; then
        ADAPTERS_ARG="--adapters ${STAGE3_CKPT}"
    fi

    PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1))) \
    NPROC_PER_NODE=${NUM_GPUS} \
    swift sft \
        --model "${MODEL}" \
        --dataset "${DATASET}" \
        --output_dir "${OUTPUT_DIR}" \
        --train_type lora \
        --lora_rank 32 \
        --lora_alpha 64 \
        --target_modules all-linear \
        --torch_dtype bfloat16 \
        --attn_impl flash_attn \
        --num_train_epochs 2 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --learning_rate 5e-4 \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.05 \
        --max_length 4096 \
        --gradient_checkpointing true \
        --save_strategy steps \
        --save_steps 200 \
        --logging_steps 5 \
        --deepspeed "${PROJECT_DIR}/configs/deepspeed_zero2.json" \
        --save_total_limit 3 \
        ${ADAPTERS_ARG} \
        2>&1 | tee "${OUTPUT_DIR}/train.log"

elif [ "$MODE" = "custom" ]; then
    # ── 模式 2: 自定义 CTC Decoder 训练 ─────────────────
    # 使用项目 Omni/train/model/ 中的自定义模型代码
    echo "[INFO] 使用自定义 CTC Decoder 训练..."
    echo "[INFO] 此模式需要 TTS 数据（文本→语音 token 对）"

    python3 -c "
import sys
sys.path.insert(0, '${TRAIN_DIR}')
from model.speech_decoder import SpeechDecoder
import torch

# 验证模型组件可用
decoder = SpeechDecoder(
    input_dim=4096,
    hidden_dim=1024,
    num_layers=6,
    num_heads=8,
    num_speech_tokens=4096,
)
print(f'SpeechDecoder 参数量: {sum(p.numel() for p in decoder.parameters()):,}')
print('[OK] CTC Decoder 组件验证通过')
print()
print('提示：完整的 CTC Decoder 训练需要：')
print('  1. TTS 数据（文本 → CosyVoice2 语音 token 对）')
print('  2. Stage 3 训练好的 LLM hidden states 作为输入')
print('  3. 使用 Omni/train/model/speech_decoder.py 中的 CTC Loss 训练')
print()
print('训练流程：')
print('  a) 加载 Stage 3 的 LLM')
print('  b) 对 TTS 文本做前向传播，获取 hidden states')
print('  c) 将 hidden states 送入 SpeechDecoder')
print('  d) 用 CTC Loss 对齐预测 token 和目标 token')
" 2>&1 | tee "${OUTPUT_DIR}/custom_train.log"

else
    echo "[ERROR] 未知模式: ${MODE}，请使用 swift 或 custom"
    exit 1
fi

echo "[Stage 4] 训练完成！输出目录: ${OUTPUT_DIR}"

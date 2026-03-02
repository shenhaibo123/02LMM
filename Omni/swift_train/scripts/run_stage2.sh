#!/bin/bash
# ================================================================
# Stage 2: 音频-语言对齐
#
# 策略：冻结音频编码器和 LLM，训练音频投影层
# 参考：Baichuan-Omni Stage I-4, OpenOmni Stage 1
#
# 目标：使音频编码器的输出特征对齐到 LLM 嵌入空间
# 数据：音频-文本对（AISHELL-1, LibriSpeech）
# ================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── 可配置参数 ──────────────────────────────────────────
MODEL="${MODEL:-Qwen/Qwen2.5-Omni-7B}"
# 接续 Stage 1 的 checkpoint（如有）
STAGE1_CKPT="${STAGE1_CKPT:-}"
DATASET="${DATASET:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/output/stage2_audio_align}"
NUM_GPUS="${NUM_GPUS:-8}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
MAX_LENGTH="${MAX_LENGTH:-4096}"

# 默认数据集：AISHELL-1（MS-Swift 内置支持）
if [ -z "$DATASET" ]; then
    DATASET="speech_asr/speech_asr_aishell1_trainsets#50000"
fi

# 如果有 Stage 1 的 checkpoint，从那里恢复
MODEL_ARG="${MODEL}"
RESUME_ARGS=""
if [ -n "$STAGE1_CKPT" ] && [ -d "$STAGE1_CKPT" ]; then
    echo "[INFO] 从 Stage 1 checkpoint 恢复: ${STAGE1_CKPT}"
    RESUME_ARGS="--resume_from_checkpoint ${STAGE1_CKPT}"
fi

echo "============================================"
echo "Stage 2: 音频-语言对齐"
echo "============================================"
echo "模型:       ${MODEL_ARG}"
echo "数据:       ${DATASET}"
echo "输出:       ${OUTPUT_DIR}"
echo "GPU 数量:   ${NUM_GPUS}"
echo "学习率:     ${LEARNING_RATE}"
echo "============================================"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1))) \
NPROC_PER_NODE=${NUM_GPUS} \
swift sft \
    --model "${MODEL_ARG}" \
    --dataset "${DATASET}" \
    --output_dir "${OUTPUT_DIR}" \
    --train_type full \
    --freeze_llm true \
    --freeze_vit true \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LEARNING_RATE} \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --max_length ${MAX_LENGTH} \
    --gradient_checkpointing true \
    --save_strategy steps \
    --save_steps 500 \
    --eval_strategy steps \
    --eval_steps 500 \
    --logging_steps 10 \
    --dataloader_num_workers 4 \
    --deepspeed "${PROJECT_DIR}/configs/deepspeed_zero2.json" \
    --save_total_limit 3 \
    ${RESUME_ARGS} \
    2>&1 | tee "${OUTPUT_DIR}/train.log"

echo "[Stage 2] 训练完成！输出目录: ${OUTPUT_DIR}"

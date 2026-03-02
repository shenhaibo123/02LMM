#!/bin/bash
# ================================================================
# Stage 5: DPO/GRPO 偏好对齐
#
# 策略：使用 DPO 或 GRPO 进行人类偏好对齐
# 参考：OpenOmni 情感 DPO, Qwen2.5-Omni Talker DPO, OmniGAIA OmniDPO
#
# 目标：提升模型输出质量和人类偏好对齐程度
# 数据：chosen/rejected 偏好对
# ================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── 可配置参数 ──────────────────────────────────────────
MODEL="${MODEL:-Qwen/Qwen2.5-Omni-7B}"
STAGE3_CKPT="${STAGE3_CKPT:-}"  # Stage 3 LoRA adapter 路径
RLHF_TYPE="${RLHF_TYPE:-dpo}"   # dpo 或 grpo
DATASET="${DATASET:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/output/stage5_${RLHF_TYPE}}"
NUM_GPUS="${NUM_GPUS:-8}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"

# 默认数据集
if [ -z "$DATASET" ]; then
    DATASET="hjh0119/shareAI-Llama3-DPO-zh-en-emoji#10000"
fi

echo "============================================"
echo "Stage 5: ${RLHF_TYPE^^} 偏好对齐"
echo "============================================"
echo "模型:       ${MODEL}"
echo "RLHF 类型:  ${RLHF_TYPE}"
echo "数据:       ${DATASET}"
echo "输出:       ${OUTPUT_DIR}"
echo "GPU 数量:   ${NUM_GPUS}"
echo "学习率:     ${LEARNING_RATE}"
echo "============================================"

mkdir -p "${OUTPUT_DIR}"

# 构建命令
TRAIN_CMD="swift rlhf \
    --rlhf_type ${RLHF_TYPE} \
    --model ${MODEL} \
    --dataset ${DATASET} \
    --output_dir ${OUTPUT_DIR} \
    --train_type lora \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LEARNING_RATE} \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --max_length 4096 \
    --gradient_checkpointing true \
    --save_strategy steps \
    --save_steps 100 \
    --logging_steps 5 \
    --deepspeed ${PROJECT_DIR}/configs/deepspeed_zero2.json \
    --save_total_limit 3"

# 如果有 Stage 3 的 LoRA adapter
if [ -n "$STAGE3_CKPT" ] && [ -d "$STAGE3_CKPT" ]; then
    TRAIN_CMD="${TRAIN_CMD} --adapters ${STAGE3_CKPT}"
fi

# DPO 特定参数
if [ "$RLHF_TYPE" = "dpo" ]; then
    TRAIN_CMD="${TRAIN_CMD} --loss_type sigmoid"
fi

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1))) \
NPROC_PER_NODE=${NUM_GPUS} \
eval ${TRAIN_CMD} 2>&1 | tee "${OUTPUT_DIR}/train.log"

echo "[Stage 5] ${RLHF_TYPE^^} 训练完成！输出目录: ${OUTPUT_DIR}"

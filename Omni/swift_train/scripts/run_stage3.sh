#!/bin/bash
# ================================================================
# Stage 3: 全模态联合 SFT
#
# 策略：冻结编码器，解冻投影层和 LLM，混合多模态数据联合微调
# 参考：Baichuan-Omni Stage II (600K SFT), OpenOmni Stage 3
#
# 目标：在多模态混合数据上进行指令微调
# 数据：图文 + 音频 + 视频 + 纯文本混合指令数据
# ================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── 可配置参数 ──────────────────────────────────────────
MODEL="${MODEL:-Qwen/Qwen2.5-Omni-7B}"
DATASET="${DATASET:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/output/stage3_multimodal_sft}"
NUM_GPUS="${NUM_GPUS:-8}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
TRAIN_TYPE="${TRAIN_TYPE:-lora}"  # lora 或 full
LORA_RANK="${LORA_RANK:-64}"

# 默认混合数据集（MS-Swift 支持多数据集混合，用 # 指定数量）
if [ -z "$DATASET" ]; then
    DATASET="AI-ModelScope/alpaca-gpt4-data-zh#20000"
    # 可追加更多数据集（用空格分隔）：
    # DATASET="${DATASET} speech_asr/speech_asr_aishell1_trainsets:validation#2000"
fi

echo "============================================"
echo "Stage 3: 全模态联合 SFT"
echo "============================================"
echo "模型:       ${MODEL}"
echo "训练方式:   ${TRAIN_TYPE}"
echo "数据:       ${DATASET}"
echo "输出:       ${OUTPUT_DIR}"
echo "GPU 数量:   ${NUM_GPUS}"
echo "有效批大小: $((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))"
echo "学习率:     ${LEARNING_RATE}"
echo "============================================"

# ── 构建训练命令 ────────────────────────────────────────
TRAIN_CMD="swift sft \
    --model ${MODEL} \
    --dataset ${DATASET} \
    --output_dir ${OUTPUT_DIR} \
    --train_type ${TRAIN_TYPE} \
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
    --save_steps 200 \
    --eval_strategy steps \
    --eval_steps 200 \
    --logging_steps 5 \
    --dataloader_num_workers 4 \
    --deepspeed ${PROJECT_DIR}/configs/deepspeed_zero2.json \
    --save_total_limit 5"

# LoRA 特定参数
if [ "$TRAIN_TYPE" = "lora" ]; then
    TRAIN_CMD="${TRAIN_CMD} \
    --lora_rank ${LORA_RANK} \
    --lora_alpha $((LORA_RANK * 2)) \
    --target_modules all-linear"
fi

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1))) \
NPROC_PER_NODE=${NUM_GPUS} \
eval ${TRAIN_CMD} 2>&1 | tee "${OUTPUT_DIR}/train.log"

echo "[Stage 3] 训练完成！输出目录: ${OUTPUT_DIR}"

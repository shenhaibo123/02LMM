#!/bin/bash
# ================================================================
# Stage 1: 视觉-语言对齐
#
# 策略：冻结视觉编码器和 LLM，仅训练视觉投影层
# 参考：Baichuan-Omni Stage I-1, OpenOmni Stage 2
#
# 目标：使视觉编码器的输出特征对齐到 LLM 嵌入空间
# 数据：图像-文本对（LLaVA-CC3M-Pretrain-595K）
# ================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── 可配置参数 ──────────────────────────────────────────
MODEL="${MODEL:-Qwen/Qwen2.5-Omni-7B}"
DATASET="${DATASET:-}"  # 留空则使用默认数据集
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/output/stage1_vision_align}"
NUM_GPUS="${NUM_GPUS:-8}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
MAX_LENGTH="${MAX_LENGTH:-4096}"

# 默认数据集
if [ -z "$DATASET" ]; then
    DATASET="${PROJECT_DIR}/datasets/stage1/swift_stage1_vision.jsonl"
    if [ ! -f "$DATASET" ]; then
        echo "[INFO] 未找到本地数据集，使用 MS-Swift 内置数据集"
        DATASET="liuhaotian/LLaVA-CC3M-Pretrain-595K"
    fi
fi

echo "============================================"
echo "Stage 1: 视觉-语言对齐"
echo "============================================"
echo "模型:       ${MODEL}"
echo "数据:       ${DATASET}"
echo "输出:       ${OUTPUT_DIR}"
echo "GPU 数量:   ${NUM_GPUS}"
echo "批大小:     ${BATCH_SIZE} × ${GRAD_ACCUM} = $((BATCH_SIZE * GRAD_ACCUM))"
echo "学习率:     ${LEARNING_RATE}"
echo "============================================"

# ── 训练命令 ────────────────────────────────────────────
# 冻结策略：使用 LoRA 仅训练 aligner/projector 层
# MS-Swift 对 Qwen2.5-Omni 会自动识别架构
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1))) \
NPROC_PER_NODE=${NUM_GPUS} \
swift sft \
    --model "${MODEL}" \
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
    2>&1 | tee "${OUTPUT_DIR}/train.log"

echo "[Stage 1] 训练完成！输出目录: ${OUTPUT_DIR}"

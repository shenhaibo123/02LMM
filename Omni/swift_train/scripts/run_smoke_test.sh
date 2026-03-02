#!/bin/bash
# ================================================================
# Smoke Test: 单卡快速验证全流程
#
# 使用少量数据和短训练步数验证每个阶段是否可以正常运行。
# 预计 30-60 分钟完成（单卡 A100-80G）。
# ================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_DIR}/output/smoke_test"

echo "============================================================"
echo "  Omni Smoke Test —— 快速验证全训练流程"
echo "============================================================"

mkdir -p "${OUTPUT_DIR}"

# ── 1. 验证环境 ──────────────────────────────────────────
echo ""
echo "[1/6] 验证环境..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'显存: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')

try:
    import swift
    print(f'MS-Swift: {swift.__version__}')
except ImportError:
    print('MS-Swift: 未安装！请运行 pip install ms-swift -U')
    exit(1)

try:
    import evalscope
    print(f'EvalScope: 已安装')
except ImportError:
    print('EvalScope: 未安装（评测功能不可用）')

print('[OK] 环境验证通过')
"

# ── 2. 准备测试数据 ──────────────────────────────────────
echo ""
echo "[2/6] 准备测试数据..."
python3 "${PROJECT_DIR}/data/prepare_datasets.py" \
    --stages all \
    --data_dir "${OUTPUT_DIR}/datasets" \
    --smoke

# ── 3. Stage 1+2: 模态对齐 Smoke Test ───────────────────
echo ""
echo "[3/6] Stage 1+2: 模态对齐 Smoke Test..."
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen2.5-Omni-7B \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#100' \
    --output_dir "${OUTPUT_DIR}/stage1_2" \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --target_modules all-linear \
    --freeze_llm true \
    --freeze_vit true \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-3 \
    --max_length 512 \
    --gradient_checkpointing true \
    --save_strategy no \
    --logging_steps 1 \
    --max_steps 20 \
    2>&1 | tee "${OUTPUT_DIR}/stage1_2.log"

echo "[OK] Stage 1+2 Smoke Test 通过"

# ── 4. Stage 3: 联合 SFT Smoke Test ─────────────────────
echo ""
echo "[4/6] Stage 3: 联合 SFT Smoke Test..."
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen2.5-Omni-7B \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#100' \
    --output_dir "${OUTPUT_DIR}/stage3" \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --max_length 512 \
    --gradient_checkpointing true \
    --save_strategy no \
    --logging_steps 1 \
    --max_steps 20 \
    2>&1 | tee "${OUTPUT_DIR}/stage3.log"

echo "[OK] Stage 3 Smoke Test 通过"

# ── 5. Stage 4: 自定义组件验证 ──────────────────────────
echo ""
echo "[5/6] Stage 4: 自定义组件验证..."
python3 -c "
import sys, os
sys.path.insert(0, os.path.join('$(dirname "$PROJECT_DIR")', 'train'))
from model.speech_decoder import SpeechDecoder
from model.projectors import MLPProjector, ConvGMLPProjector
import torch

# SpeechDecoder 验证
decoder = SpeechDecoder(input_dim=1024, hidden_dim=256, num_layers=2,
                        num_heads=4, num_speech_tokens=4096)
h = torch.randn(2, 32, 1024)
logits, loss = decoder(h)
print(f'SpeechDecoder: logits={logits.shape}, loss={loss}')

# 带 CTC Loss 训练
target = torch.randint(0, 4096, (2, 10))
lengths_in = torch.full((2,), 32, dtype=torch.long)
lengths_out = torch.full((2,), 10, dtype=torch.long)
logits, loss = decoder(h, target_tokens=target,
                       input_lengths=lengths_in, target_lengths=lengths_out)
print(f'CTC Training: loss={loss.item():.4f}')
loss.backward()
print('[OK] CTC backward 通过')

# Projectors 验证
mlp = MLPProjector(768, 512, 1024)
out = mlp(torch.randn(2, 64, 768))
print(f'MLPProjector: {out.shape}')

conv = ConvGMLPProjector(768, 512, 1024, max_seq_len=16)
out = conv(torch.randn(2, 64, 768))
print(f'ConvGMLPProjector: {out.shape}')

print('[OK] Stage 4 组件验证通过')
"

# ── 6. Stage 5: DPO Smoke Test ──────────────────────────
echo ""
echo "[6/6] Stage 5: DPO Smoke Test..."
CUDA_VISIBLE_DEVICES=0 \
swift rlhf \
    --rlhf_type dpo \
    --model Qwen/Qwen2.5-Omni-7B \
    --dataset 'hjh0119/shareAI-Llama3-DPO-zh-en-emoji#100' \
    --output_dir "${OUTPUT_DIR}/stage5" \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-6 \
    --max_length 512 \
    --gradient_checkpointing true \
    --save_strategy no \
    --logging_steps 1 \
    --max_steps 10 \
    2>&1 | tee "${OUTPUT_DIR}/stage5.log"

echo "[OK] Stage 5 DPO Smoke Test 通过"

# ── 总结 ─────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Smoke Test 全部完成！"
echo "============================================================"
echo "  输出目录: ${OUTPUT_DIR}"
echo ""
echo "  下一步："
echo "    1. 准备完整数据: python data/prepare_datasets.py --stages all"
echo "    2. 逐阶段训练:  bash scripts/run_stage1.sh"
echo "    3. 评测:        bash eval/run_eval.sh"
echo "============================================================"

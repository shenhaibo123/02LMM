#!/bin/bash
# ============================================================
# Omni 训练环境一键安装脚本
# 使用方式: bash setup_env.sh [--cuda CUDA_VERSION]
# 示例:     bash setup_env.sh --cuda 12.4
# ============================================================
set -e

CUDA_VERSION="${CUDA_VERSION:-12.4}"

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda) CUDA_VERSION="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "=========================================="
echo "  Omni 训练环境安装"
echo "  CUDA 版本: ${CUDA_VERSION}"
echo "=========================================="

# 1. 安装 PyTorch (根据 CUDA 版本)
echo "[1/6] 安装 PyTorch..."
if [[ "$CUDA_VERSION" == "12.4" ]]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
elif [[ "$CUDA_VERSION" == "12.1" ]]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
elif [[ "$CUDA_VERSION" == "11.8" ]]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "不支持的 CUDA 版本: ${CUDA_VERSION}，使用默认安装"
    pip install torch torchvision torchaudio
fi

# 2. 安装 ms-swift
echo "[2/6] 安装 ms-swift..."
pip install "ms-swift>=3.12.0" modelscope

# 3. 安装 DeepSpeed
echo "[3/6] 安装 DeepSpeed..."
pip install deepspeed

# 4. 安装 flash-attn（可选但推荐）
echo "[4/6] 安装 flash-attn（如失败可跳过）..."
pip install flash-attn --no-build-isolation 2>/dev/null || echo "flash-attn 安装失败，将使用标准注意力机制"

# 5. 安装项目依赖
echo "[5/6] 安装项目依赖..."
pip install -r "${SCRIPT_DIR}/requirements.txt"

# 6. 安装 CosyVoice2（语音合成，阶段4-5需要）
echo "[6/6] 安装 CosyVoice2..."
if [ ! -d "${SCRIPT_DIR}/third_party/CosyVoice" ]; then
    mkdir -p "${SCRIPT_DIR}/third_party"
    git clone --depth 1 https://github.com/FunAudioLLM/CosyVoice.git "${SCRIPT_DIR}/third_party/CosyVoice"
    cd "${SCRIPT_DIR}/third_party/CosyVoice"
    pip install -e . 2>/dev/null || echo "CosyVoice2 安装失败，阶段 4-5 将不可用"
    cd "${SCRIPT_DIR}"
else
    echo "CosyVoice2 已存在，跳过"
fi

# 验证安装
echo ""
echo "=========================================="
echo "  验证安装结果"
echo "=========================================="
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA:    {torch.cuda.is_available()} ({torch.version.cuda})')
print(f'  GPU数量: {torch.cuda.device_count()}')
try:
    import swift; print(f'  ms-swift: ok')
except: print('  ms-swift: 未安装')
try:
    import deepspeed; print(f'  DeepSpeed: {deepspeed.__version__}')
except: print('  DeepSpeed: 未安装')
try:
    from flash_attn import flash_attn_func; print('  flash-attn: ok')
except: print('  flash-attn: 未安装（可选）')
try:
    import whisper; print('  whisper: ok')
except: print('  whisper: 未安装')
print('  安装完成!')
"

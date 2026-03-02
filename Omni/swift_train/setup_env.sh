#!/bin/bash
# ================================================================
# 环境安装脚本
#
# 安装 Omni MS-Swift 训练所需的全部依赖。
# 支持 CUDA 和 macOS 环境。
#
# 用法：
#   bash setup_env.sh              # 完整安装
#   bash setup_env.sh --minimal    # 最小安装（无 flash-attn）
#   bash setup_env.sh --check      # 仅检查环境
# ================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MINIMAL=false
CHECK_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --minimal) MINIMAL=true; shift;;
        --check)   CHECK_ONLY=true; shift;;
        *)         echo "未知参数: $1"; exit 1;;
    esac
done

echo "============================================"
echo "  Omni 训练环境配置"
echo "============================================"

# ── 环境检查 ──────────────────────────────────────────
check_env() {
    echo ""
    echo "[检查] Python 环境..."
    python3 --version

    echo ""
    echo "[检查] PyTorch..."
    python3 -c "
import torch
print(f'  PyTorch:  {torch.__version__}')
print(f'  CUDA:     {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:      {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
    print(f'  显存:     {mem:.1f} GB')
    print(f'  CUDA 版本: {torch.version.cuda}')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print(f'  MPS:      可用 (Apple Silicon)')
else:
    print(f'  WARNING:  无 GPU 加速')
"

    echo ""
    echo "[检查] 关键依赖..."
    python3 -c "
packages = {
    'ms-swift': 'swift',
    'transformers': 'transformers',
    'modelscope': 'modelscope',
    'datasets': 'datasets',
    'deepspeed': 'deepspeed',
    'peft': 'peft',
    'evalscope': 'evalscope',
    'flash-attn': 'flash_attn',
}

for name, module in packages.items():
    try:
        m = __import__(module)
        ver = getattr(m, '__version__', '已安装')
        print(f'  {name:20s} {ver}')
    except ImportError:
        print(f'  {name:20s} ❌ 未安装')
"
    echo ""
    echo "[检查] 完成！"
}

if [ "$CHECK_ONLY" = true ]; then
    check_env
    exit 0
fi

# ── 安装依赖 ──────────────────────────────────────────
echo ""
echo "[1/4] 安装 MS-Swift..."
pip install ms-swift -U

echo ""
echo "[2/4] 安装核心依赖..."
pip install transformers>=4.45 modelscope>=1.23 datasets>=2.18 peft>=0.12

echo ""
echo "[3/4] 安装训练与评测依赖..."
pip install deepspeed>=0.14 evalscope>=0.6
pip install soundfile librosa Pillow matplotlib tqdm

if [ "$MINIMAL" = false ]; then
    echo ""
    echo "[4/4] 安装 Flash Attention（可选，需要 CUDA）..."
    pip install flash-attn --no-build-isolation 2>/dev/null || \
        echo "  [跳过] flash-attn 安装失败（可能无 CUDA 或编译环境不完整）"
else
    echo ""
    echo "[4/4] 跳过 Flash Attention（最小安装模式）"
fi

# ── 验证安装 ──────────────────────────────────────────
echo ""
echo "============================================"
echo "  安装完成，验证环境..."
echo "============================================"
check_env

echo ""
echo "============================================"
echo "  环境配置完成！"
echo "============================================"
echo ""
echo "  下一步："
echo "    1. 运行 Smoke Test:   bash scripts/run_smoke_test.sh"
echo "    2. 准备数据:          python data/prepare_datasets.py --stages all"
echo "    3. 开始训练:          bash scripts/run_stage1.sh"
echo "============================================"

#!/usr/bin/env bash
# ============================================================================
# Omni 训练数据下载脚本
#
# 用法:
#   bash download_data.sh                    # 下载全部阶段数据
#   bash download_data.sh --stages 1,2       # 仅下载指定阶段
#   bash download_data.sh --smoke            # Smoke test 模式（每个数据集仅取少量样本）
#   bash download_data.sh --data_dir /data   # 指定数据保存目录
# ============================================================================

set -euo pipefail

# ── 默认参数 ─────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/datasets"
STAGES="all"
SMOKE=false
MAX_SAMPLES=""

# ── 参数解析 ─────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data_dir)   DATA_DIR="$2"; shift 2 ;;
        --stages)     STAGES="$2"; shift 2 ;;
        --smoke)      SMOKE=true; shift ;;
        --max_samples) MAX_SAMPLES="$2"; shift 2 ;;
        -h|--help)
            echo "用法: bash download_data.sh [选项]"
            echo "  --data_dir DIR    数据保存目录 (默认: ./datasets)"
            echo "  --stages STAGES   逗号分隔的阶段号或 'all' (默认: all)"
            echo "  --smoke           Smoke test 模式"
            echo "  --max_samples N   每个数据集最大样本数"
            exit 0
            ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# ── 环境检查 ─────────────────────────────────────────────
echo "============================================"
echo "  Omni 训练数据下载"
echo "============================================"
echo "数据目录: ${DATA_DIR}"
echo "阶段:     ${STAGES}"
echo "Smoke:    ${SMOKE}"
echo ""

# 检查 Python 和必要库
python3 -c "import datasets" 2>/dev/null || {
    echo "[WARN] datasets 库未安装，尝试安装..."
    pip install datasets -q
}

python3 -c "import soundfile" 2>/dev/null || {
    echo "[WARN] soundfile 库未安装，尝试安装..."
    pip install soundfile -q
}

mkdir -p "${DATA_DIR}"

# ── 调用 Python 下载脚本 ────────────────────────────────
PREPARE_CMD="python3 ${SCRIPT_DIR}/prepare_data.py --data_dir ${DATA_DIR} --stages ${STAGES}"

if [ "${SMOKE}" = true ]; then
    PREPARE_CMD="${PREPARE_CMD} --smoke"
fi

if [ -n "${MAX_SAMPLES}" ]; then
    PREPARE_CMD="${PREPARE_CMD} --max_samples ${MAX_SAMPLES}"
fi

echo "执行: ${PREPARE_CMD}"
echo ""
eval ${PREPARE_CMD}

# ── 下载后统计 ──────────────────────────────────────────
echo ""
echo "============================================"
echo "  数据统计"
echo "============================================"

if command -v python3 &>/dev/null; then
    python3 "${SCRIPT_DIR}/data_stats.py" --data_dir "${DATA_DIR}" --summary_only
fi

echo ""
echo "============================================"
echo "  下载完成"
echo "============================================"
echo "数据保存在: ${DATA_DIR}"
du -sh "${DATA_DIR}" 2>/dev/null || true
echo ""

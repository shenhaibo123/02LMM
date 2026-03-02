#!/usr/bin/env bash
# ============================================================================
# Omni 全流程训练脚本
#
# 按顺序执行 5 个训练阶段，自动传递 checkpoint。
#
# 用法:
#   bash run_all.sh                     # 正式训练全流程
#   bash run_all.sh --smoke             # Smoke test 全流程
#   bash run_all.sh --start_stage 3     # 从 Stage 3 开始
#   bash run_all.sh --end_stage 3       # 执行到 Stage 3 结束
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

# ── 参数 ──────────────────────────────────────────────────
SMOKE=false
START_STAGE=1
END_STAGE=5
GPUS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke)        SMOKE=true; shift ;;
        --start_stage)  START_STAGE="$2"; shift 2 ;;
        --end_stage)    END_STAGE="$2"; shift 2 ;;
        --gpus)         GPUS="$2"; shift 2 ;;
        -h|--help)
            echo "用法: bash run_all.sh [选项]"
            echo "  --smoke            Smoke test 模式"
            echo "  --start_stage N    起始阶段 (默认 1)"
            echo "  --end_stage N      结束阶段 (默认 5)"
            echo "  --gpus N           GPU 数量"
            exit 0
            ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

TOTAL_START=$(date +%s)

echo "============================================"
echo "  Omni 全流程训练"
echo "  模式: $([ "${SMOKE}" = true ] && echo 'Smoke Test' || echo '正式训练')"
echo "  阶段: ${START_STAGE} → ${END_STAGE}"
echo "  开始: $(date)"
echo "============================================"
echo ""

# ── 预检查 ────────────────────────────────────────────────
if [ "${SMOKE}" = false ]; then
    echo "环境预检查..."
    python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA 不可用，正式训练需要 GPU'
n = torch.cuda.device_count()
print(f'  检测到 {n} 个 GPU')
for i in range(n):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_mem / 1e9
    print(f'    GPU {i}: {name} ({mem:.0f} GB)')
"
    echo ""
fi

# ── 逐阶段执行 ───────────────────────────────────────────
STAGE_STATUS=()

for STAGE in $(seq ${START_STAGE} ${END_STAGE}); do
    STAGE_START=$(date +%s)

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Stage ${STAGE} 开始 ($(date +%H:%M:%S))"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    CMD="bash run_stage.sh --stage ${STAGE}"
    if [ "${SMOKE}" = true ]; then
        CMD="${CMD} --smoke"
    fi
    if [ -n "${GPUS}" ]; then
        CMD="${CMD} --gpus ${GPUS}"
    fi

    if eval ${CMD}; then
        STAGE_END=$(date +%s)
        STAGE_TIME=$((STAGE_END - STAGE_START))
        STAGE_STATUS+=("Stage ${STAGE}: PASS (${STAGE_TIME}s)")
        echo "  Stage ${STAGE} 完成 (${STAGE_TIME}s)"
    else
        STAGE_END=$(date +%s)
        STAGE_TIME=$((STAGE_END - STAGE_START))
        STAGE_STATUS+=("Stage ${STAGE}: FAIL (${STAGE_TIME}s)")
        echo "  [ERROR] Stage ${STAGE} 失败"
        echo "  后续阶段将被跳过"
        break
    fi
    echo ""
done

# ── 汇总 ──────────────────────────────────────────────────
TOTAL_END=$(date +%s)
TOTAL_TIME=$((TOTAL_END - TOTAL_START))

echo ""
echo "============================================"
echo "  全流程训练完成"
echo "============================================"
echo "  总耗时: ${TOTAL_TIME} 秒 ($((TOTAL_TIME / 60)) 分钟)"
echo ""
echo "  各阶段结果:"
for status in "${STAGE_STATUS[@]}"; do
    echo "    ${status}"
done
echo ""

# 列出输出目录
if [ "${SMOKE}" = true ]; then
    ls -la outputs/smoke/ 2>/dev/null && echo "" || echo "  无输出目录"
else
    ls -la outputs/ 2>/dev/null && echo "" || echo "  无输出目录"
fi

echo "============================================"

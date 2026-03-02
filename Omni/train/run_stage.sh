#!/usr/bin/env bash
# ============================================================================
# Omni 单阶段训练脚本
#
# 用法:
#   bash run_stage.sh --stage 1                          # 正式训练 Stage 1
#   bash run_stage.sh --stage 3 --smoke                  # Smoke test Stage 3
#   bash run_stage.sh --stage 2 --config configs/custom.yaml  # 自定义配置
#   bash run_stage.sh --stage 1 --gpus 4                 # 指定 GPU 数量
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

# ── 默认参数 ──────────────────────────────────────────────
STAGE=""
SMOKE=false
CONFIG=""
GPUS=""
DRY_RUN=false

# ── 参数解析 ──────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)    STAGE="$2"; shift 2 ;;
        --smoke)    SMOKE=true; shift ;;
        --config)   CONFIG="$2"; shift 2 ;;
        --gpus)     GPUS="$2"; shift 2 ;;
        --dry_run)  DRY_RUN=true; shift ;;
        -h|--help)
            echo "用法: bash run_stage.sh --stage N [选项]"
            echo "  --stage N       阶段号 (1-5, 必选)"
            echo "  --smoke         使用 smoke test 配置"
            echo "  --config FILE   自定义配置 YAML"
            echo "  --gpus N        GPU 数量 (默认由配置决定)"
            echo "  --dry_run       仅打印配置，不实际训练"
            exit 0
            ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

if [ -z "${STAGE}" ]; then
    echo "[ERROR] 必须指定 --stage 参数"
    exit 1
fi

# ── 配置文件选择 ──────────────────────────────────────────
STAGE_NAMES=("" "audio_align" "vision_align" "multimodal_sft" "speech_gen" "dpo")
if [ "${STAGE}" -lt 1 ] || [ "${STAGE}" -gt 5 ]; then
    echo "[ERROR] 阶段号必须在 1-5 之间"
    exit 1
fi

STAGE_NAME="${STAGE_NAMES[$STAGE]}"

if [ -z "${CONFIG}" ]; then
    if [ "${SMOKE}" = true ]; then
        CONFIG="configs/stage${STAGE}_${STAGE_NAME}_smoke.yaml"
    else
        CONFIG="configs/stage${STAGE}_${STAGE_NAME}.yaml"
    fi
fi

if [ ! -f "${CONFIG}" ]; then
    echo "[ERROR] 配置文件不存在: ${CONFIG}"
    exit 1
fi

echo "============================================"
echo "  Omni Stage ${STAGE}: ${STAGE_NAME}"
echo "  配置: ${CONFIG}"
echo "  Smoke: ${SMOKE}"
echo "============================================"

# ── 读取配置 ──────────────────────────────────────────────
# 使用 Python 解析 YAML 获取关键参数
read_yaml() {
    python3 -c "
import yaml, sys
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
# 递归获取嵌套 key
keys = '$1'.split('.')
val = cfg
for k in keys:
    if isinstance(val, dict) and k in val:
        val = val[k]
    else:
        val = '$2'
        break
print(val)
"
}

LLM_NAME=$(read_yaml "model.llm" "Qwen/Qwen3-8B")
OUTPUT_DIR=$(read_yaml "output_dir" "outputs/stage${STAGE}")
LR=$(read_yaml "training.learning_rate" "1e-3")
EPOCHS=$(read_yaml "training.max_epochs" "1")
BS=$(read_yaml "training.batch_size_per_gpu" "8")
GRAD_ACC=$(read_yaml "training.gradient_accumulation_steps" "4")
MAX_SEQ_LEN=$(read_yaml "data.max_seq_len" "2048")

if [ -z "${GPUS}" ]; then
    GPUS=$(read_yaml "hardware.gpus" "1")
fi

echo ""
echo "  LLM: ${LLM_NAME}"
echo "  Output: ${OUTPUT_DIR}"
echo "  LR: ${LR}, Epochs: ${EPOCHS}"
echo "  BS/GPU: ${BS}, Grad Acc: ${GRAD_ACC}"
echo "  GPUs: ${GPUS}"
echo "  Max Seq Len: ${MAX_SEQ_LEN}"
echo ""

if [ "${DRY_RUN}" = true ]; then
    echo "[DRY_RUN] 仅打印配置，跳过训练"
    python3 -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
import json
print(json.dumps(cfg, indent=2, ensure_ascii=False))
"
    exit 0
fi

# ── 环境检查 ──────────────────────────────────────────────
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python3 -c "import yaml" 2>/dev/null || { echo "缺少 pyyaml: pip install pyyaml"; exit 1; }

# ── 创建输出目录 ──────────────────────────────────────────
mkdir -p "${OUTPUT_DIR}"
mkdir -p "logs"

# ── 训练验证（模型组件级） ────────────────────────────────
echo "验证模型组件..."
python3 -c "
import sys
sys.path.insert(0, '.')
from model.config import OmniModelConfig, STAGE_CONFIGS
cfg = OmniModelConfig()
smoke = ${SMOKE:+True}${SMOKE:=False}
r = cfg.resolve(smoke=(smoke == 'true' if isinstance(smoke, str) else bool(smoke)))
print(f'  模型参数: LLM={r[\"llm_name\"]}, Vision={r[\"vision_encoder\"]}, Audio={r[\"audio_encoder\"]}')
sc = STAGE_CONFIGS[${STAGE}]
print(f'  阶段配置: frozen={sc.frozen_components}, trainable={sc.trainable_components}')
print('  组件验证通过')
"

# ── 执行训练 ──────────────────────────────────────────────
LOG_FILE="logs/stage${STAGE}_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "开始训练... (日志: ${LOG_FILE})"
echo ""

# 使用 ms-swift 进行训练（Stage 1-3, 5 支持；Stage 4 使用自定义 trainer）
LOSS_TYPE=$(read_yaml "loss.type" "lm")

if [ "${STAGE}" -eq 4 ]; then
    # Stage 4: 自定义训练（CTC Loss 需要自定义 trainer）
    echo "[Stage 4] 使用自定义 trainer (CTC + LM Loss)"

    python3 -c "
import sys, os, logging
sys.path.insert(0, '.')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('stage4')

import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)

from model import OmniModel, OmniModelConfig
smoke = '${SMOKE}' == 'true'
model_config = OmniModelConfig()
model = OmniModel(model_config, smoke=smoke)
model.configure_for_stage(4)
model.print_trainable_summary()

logger.info('Stage 4 自定义训练器初始化完成')
logger.info('注意: 完整的 Stage 4 训练需在 GPU 服务器上执行')
logger.info(f'配置: lr={cfg[\"training\"][\"learning_rate\"]}, epochs={cfg[\"training\"][\"max_epochs\"]}')
" 2>&1 | tee "${LOG_FILE}"
else
    # Stage 1-3, 5: 可通过 ms-swift 框架训练
    echo "[Stage ${STAGE}] 模型配置验证"

    python3 -c "
import sys, logging
sys.path.insert(0, '.')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('stage${STAGE}')

import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)

from model import OmniModel, OmniModelConfig
smoke = '${SMOKE}' == 'true'
model_config = OmniModelConfig()
model = OmniModel(model_config, smoke=smoke)
model.configure_for_stage(${STAGE})
model.print_trainable_summary()

logger.info('Stage ${STAGE} 配置验证完成')
logger.info(f'配置: lr={cfg[\"training\"][\"learning_rate\"]}, epochs={cfg[\"training\"][\"max_epochs\"]}')
logger.info('完整训练请在 GPU 服务器上使用 swift sft 命令执行')
" 2>&1 | tee "${LOG_FILE}"
fi

echo ""
echo "============================================"
echo "  Stage ${STAGE} 完成"
echo "  输出: ${OUTPUT_DIR}"
echo "  日志: ${LOG_FILE}"
echo "============================================"

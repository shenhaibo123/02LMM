#!/usr/bin/env bash
# ============================================================================
# Omni 评测脚本
#
# 用法:
#   bash eval.sh --mode smoke              # Smoke test 评测
#   bash eval.sh --mode text               # 纯文本评测
#   bash eval.sh --mode multimodal         # 多模态评测
#   bash eval.sh --mode audio              # 音频/ASR 评测
#   bash eval.sh --mode full               # 完整评测
#   bash eval.sh --mode component          # 组件级验证
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

MODE="smoke"
MODEL_PATH=""
OUTPUT_DIR="eval_results"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)       MODE="$2"; shift 2 ;;
        --model)      MODEL_PATH="$2"; shift 2 ;;
        --output)     OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "用法: bash eval.sh --mode MODE [选项]"
            echo "  --mode MODE     评测模式: smoke|text|multimodal|audio|full|component"
            echo "  --model PATH    模型路径"
            echo "  --output DIR    结果保存目录"
            exit 0
            ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

mkdir -p "${OUTPUT_DIR}"

echo "============================================"
echo "  Omni 模型评测"
echo "  模式: ${MODE}"
echo "  输出: ${OUTPUT_DIR}"
echo "============================================"
echo ""

# ── Component 评测 ────────────────────────────────────────
run_component_eval() {
    echo "=== 组件级评测 ==="

    python3 -c "
import sys, torch, json
sys.path.insert(0, '.')
from model import OmniModel, OmniModelConfig

results = {}

# 1. 模型构建验证
config = OmniModelConfig()
model = OmniModel(config, smoke=True)
total = model.get_total_params()
results['total_params'] = total
print(f'  总参数量: {total:,}')

# 2. 各组件参数统计
stats = model.get_component_params()
for name, s in stats.items():
    results[f'params_{name}'] = s['total']
    print(f'  {name}: {s[\"total\"]:,} params')

# 3. 各阶段冻结/解冻比例
for stage in range(1, 6):
    sc = model.configure_for_stage(stage)
    trainable = model.get_trainable_params()
    ratio = trainable / total * 100
    results[f'stage{stage}_trainable_ratio'] = f'{ratio:.1f}%'
    print(f'  Stage {stage} ({sc.name}): {ratio:.1f}% trainable')

# 4. 评测指标模块
from metrics.eval_metrics import compute_wer, compute_cer, compute_bleu

# WER 测试
wer_result = compute_wer(
    predictions=['hello world', 'good morning everyone'],
    references=['hello world', 'good morning everyone']
)
results['wer_perfect'] = wer_result['wer']
print(f'  WER (perfect match): {wer_result[\"wer\"]:.4f}')

wer_result2 = compute_wer(
    predictions=['hello there', 'good night'],
    references=['hello world', 'good morning everyone']
)
results['wer_mismatch'] = wer_result2['wer']
print(f'  WER (mismatch): {wer_result2[\"wer\"]:.4f}')

# CER 测试
cer_result = compute_cer(
    predictions=['你好世界'],
    references=['你好世界']
)
results['cer_perfect'] = cer_result['cer']
print(f'  CER (perfect match): {cer_result[\"cer\"]:.4f}')

# BLEU 测试
bleu_result = compute_bleu(
    predictions=['the cat sat on the mat'],
    references=['the cat is on the mat']
)
results['bleu'] = bleu_result['bleu']
print(f'  BLEU: {bleu_result[\"bleu\"]:.4f}')

# 保存结果
with open('${OUTPUT_DIR}/component_eval.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f'  结果已保存: ${OUTPUT_DIR}/component_eval.json')
"
}

# ── Smoke 评测 ────────────────────────────────────────────
run_smoke_eval() {
    echo "=== Smoke Test 评测 ==="
    run_component_eval

    echo ""
    echo "=== 数据统计验证 ==="
    if [ -d "data/datasets" ]; then
        python3 data/data_stats.py --data_dir data/datasets --summary_only
    else
        echo "  [SKIP] 无数据目录，跳过数据统计"
    fi
}

# ── Text 评测 ─────────────────────────────────────────────
run_text_eval() {
    echo "=== 文本能力评测 ==="
    echo "  评测集: MMLU, C-Eval, GSM8K"
    echo "  注意: 完整评测需要 GPU 服务器"

    if [ -n "${MODEL_PATH}" ]; then
        echo "  模型: ${MODEL_PATH}"
        # 使用 lm-evaluation-harness
        python3 -c "
try:
    import lm_eval
    print('  lm-eval-harness 已安装')
except ImportError:
    print('  [WARN] 请安装 lm-eval: pip install lm_eval')
"
    else
        echo "  [SKIP] 未指定模型路径，跳过实际评测"
    fi
}

# ── 多模态评测 ────────────────────────────────────────────
run_multimodal_eval() {
    echo "=== 多模态能力评测 ==="
    echo "  评测集: MMBench, MMMU, TextVQA, Video-MME"
    echo "  注意: 需要 evalscope/vlmeval 工具"

    python3 -c "
benchmarks = {
    'MMBench': {'reason': '综合多模态理解，覆盖 20+ 能力维度', 'metric': 'Accuracy'},
    'MMMU': {'reason': '大学水平专业知识多模态问答', 'metric': 'Accuracy'},
    'TextVQA': {'reason': 'OCR + 视觉问答，评估文字理解能力', 'metric': 'Accuracy'},
    'Video-MME': {'reason': '长视频理解，评估时序推理', 'metric': 'Accuracy'},
}
print('  计划评测:')
for name, info in benchmarks.items():
    print(f'    {name}: {info[\"reason\"]} (metric: {info[\"metric\"]})')
"
}

# ── 音频评测 ──────────────────────────────────────────────
run_audio_eval() {
    echo "=== 音频能力评测 ==="
    echo "  评测集: AISHELL-1 test, LibriSpeech test-clean"

    python3 -c "
benchmarks = {
    'AISHELL-1 test': {'reason': '标准中文 ASR 评测，7176 条', 'metric': 'CER', 'baseline': '< 5%'},
    'LibriSpeech test-clean': {'reason': '标准英文 ASR 评测，2620 条', 'metric': 'WER', 'baseline': '< 5%'},
    'AudioCaps test': {'reason': '音频描述生成，评估音频理解', 'metric': 'BLEU-4', 'baseline': '> 20'},
}
print('  计划评测:')
for name, info in benchmarks.items():
    print(f'    {name}: {info[\"reason\"]}')
    print(f'      metric: {info[\"metric\"]}, baseline: {info[\"baseline\"]}')
"
}

# ── 分发执行 ──────────────────────────────────────────────
case "${MODE}" in
    smoke)      run_smoke_eval ;;
    component)  run_component_eval ;;
    text)       run_text_eval ;;
    multimodal) run_multimodal_eval ;;
    audio)      run_audio_eval ;;
    full)
        run_component_eval
        echo ""
        run_text_eval
        echo ""
        run_multimodal_eval
        echo ""
        run_audio_eval
        ;;
    *)
        echo "[ERROR] 未知模式: ${MODE}"
        echo "可选: smoke | component | text | multimodal | audio | full"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "  评测完成"
echo "  结果目录: ${OUTPUT_DIR}"
echo "============================================"

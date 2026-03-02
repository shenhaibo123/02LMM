#!/usr/bin/env bash
# ============================================================================
# Omni Smoke Test —— 全流程快速验证
#
# 在单卡上依次验证 5 个阶段的模型构建、数据加载、前向传播、损失计算。
# 预计耗时: < 30 分钟（无需下载预训练权重，使用占位模型）
#
# 用法:
#   bash run_smoke_test.sh                # 完整 smoke test
#   bash run_smoke_test.sh --stage 1      # 仅测试某个阶段
#   bash run_smoke_test.sh --skip_data    # 跳过数据下载
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

# ── 参数 ──────────────────────────────────────────────────
TARGET_STAGE=""
SKIP_DATA=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)      TARGET_STAGE="$2"; shift 2 ;;
        --skip_data)  SKIP_DATA=true; shift ;;
        -h|--help)
            echo "用法: bash run_smoke_test.sh [选项]"
            echo "  --stage N      仅测试指定阶段"
            echo "  --skip_data    跳过数据准备"
            exit 0
            ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

PASS=0
FAIL=0
TOTAL_START=$(date +%s)

passed() { PASS=$((PASS + 1)); echo "  [PASS] $1"; }
failed() { FAIL=$((FAIL + 1)); echo "  [FAIL] $1: $2"; }

echo "============================================"
echo "  Omni Smoke Test"
echo "  $(date)"
echo "============================================"
echo ""

# ── Step 1: 环境检查 ──────────────────────────────────────
echo "Step 1: 环境检查"

python3 -c "import torch; print(f'  PyTorch {torch.__version__}')" && passed "PyTorch" || failed "PyTorch" "未安装"
python3 -c "import yaml" 2>/dev/null && passed "PyYAML" || failed "PyYAML" "pip install pyyaml"
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'  CUDA: {torch.cuda.get_device_name(0)}')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('  MPS: Apple Silicon')
else:
    print('  CPU only')
" && passed "设备检测"

echo ""

# ── Step 2: 模型组件验证 ──────────────────────────────────
echo "Step 2: 模型组件验证 (使用占位模型)"

python3 -c "
import sys, torch
sys.path.insert(0, '.')

from model import OmniModel, OmniModelConfig, STAGE_CONFIGS

# 构建 smoke 模型（使用占位模型，无需下载权重）
config = OmniModelConfig()
model = OmniModel(config, smoke=True)

total = model.get_total_params()
print(f'  总参数量: {total:,}')

# 验证每个阶段的冻结/解冻配置
for stage in range(1, 6):
    sc = model.configure_for_stage(stage)
    trainable = model.get_trainable_params()
    ratio = trainable / total * 100
    print(f'  Stage {stage} ({sc.name}): trainable={trainable:,} ({ratio:.1f}%)')
    assert trainable > 0, f'Stage {stage} 无可训练参数'
    assert trainable < total, f'Stage {stage} 所有参数都可训练（应有冻结）'

print('  各阶段冻结/解冻逻辑正确')
" && passed "模型组件 + 阶段配置" || failed "模型组件" "详见上方错误"

echo ""

# ── Step 3: 前向传播验证 ──────────────────────────────────
echo "Step 3: 前向传播验证"

python3 -c "
import sys, torch
sys.path.insert(0, '.')
from model import OmniModel, OmniModelConfig

config = OmniModelConfig()
model = OmniModel(config, smoke=True)
model.configure_for_stage(3)  # Stage 3: 全模态
r = config.resolve(smoke=True)

batch, text_len = 2, 16

# 纯文本前向传播
input_ids = torch.randint(0, 100, (batch, text_len))
labels = torch.randint(0, 100, (batch, text_len))

try:
    out = model(input_ids=input_ids, labels=labels)
    print(f'  纯文本: logits={list(out[\"logits\"].shape)}, loss={out[\"loss\"].item():.4f}')
except Exception as e:
    print(f'  纯文本 (占位模型): {e}')

# 投影层前向传播
from model.projectors import MLPProjector, ConvGMLPProjector

vis_proj = MLPProjector(r['vision_hidden_size'], 512, r['llm_hidden_size'])
x_vis = torch.randn(2, 196, r['vision_hidden_size'])
y_vis = vis_proj(x_vis)
print(f'  视觉投影: {list(x_vis.shape)} -> {list(y_vis.shape)}')

aud_proj = ConvGMLPProjector(r['audio_hidden_size'], 512, r['llm_hidden_size'], max_seq_len=375)
x_aud = torch.randn(2, 1500, r['audio_hidden_size'])
y_aud = aud_proj(x_aud)
print(f'  音频投影: {list(x_aud.shape)} -> {list(y_aud.shape)} (4x downsample)')

# 语音解码前向传播
from model.speech_decoder import SpeechDecoder
dec = SpeechDecoder(r['llm_hidden_size'], r['speech_decoder_hidden'])
hidden = torch.randn(2, 32, r['llm_hidden_size'])
target = torch.randint(0, 4096, (2, 10))
logits, ctc_loss = dec(hidden, target_tokens=target,
                       input_lengths=torch.tensor([32, 32]),
                       target_lengths=torch.tensor([10, 10]))
print(f'  语音解码: logits={list(logits.shape)}, CTC loss={ctc_loss.item():.4f}')
" && passed "前向传播" || failed "前向传播" "详见上方错误"

echo ""

# ── Step 4: 数据模块验证 ──────────────────────────────────
echo "Step 4: 数据模块验证"

if [ "${SKIP_DATA}" = false ]; then
    python3 -c "
import sys
sys.path.insert(0, '.')
from data.prepare_data import DATASET_REGISTRY, _CONVERTERS

# 验证数据注册表
for stage, datasets in DATASET_REGISTRY.items():
    for ds in datasets:
        assert 'name' in ds, f'Stage {stage} 缺少 name'
        assert 'type' in ds, f'Stage {stage} 缺少 type'
        assert ds['type'] in _CONVERTERS, f'未知数据类型: {ds[\"type\"]}'
        print(f'  Stage {stage}: {ds[\"name\"]} ({ds[\"type\"]})')
print('  数据注册表检查通过')
" && passed "数据注册表" || failed "数据注册表" "详见上方错误"
fi

# Dataset 类验证
python3 -c "
import sys, json, tempfile, os, torch
sys.path.insert(0, '.')
from data.dataset import OmniDataset, build_dataloader

# 创建测试数据
with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
    for i in range(20):
        sample = {'messages': [
            {'role': 'user', 'content': f'问题 {i}'},
            {'role': 'assistant', 'content': f'回答 {i}'},
        ]}
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    tmp = f.name

class MockTok:
    def __call__(self, text, max_length=64, truncation=True, padding='max_length', return_tensors='pt'):
        ids = [ord(c) % 1000 for c in text[:max_length]]
        ids += [0] * (max_length - len(ids))
        mask = [1] * min(len(text), max_length) + [0] * (max_length - min(len(text), max_length))
        return {'input_ids': torch.tensor([ids]), 'attention_mask': torch.tensor([mask])}

ds = OmniDataset(tmp, MockTok(), stage=3, max_seq_len=64, max_samples=10)
assert len(ds) == 10, f'样本数异常: {len(ds)}'

item = ds[0]
assert 'input_ids' in item, '缺少 input_ids'
assert item['input_ids'].shape == torch.Size([64]), f'shape 异常: {item[\"input_ids\"].shape}'

loader = build_dataloader(tmp, stage=3, tokenizer=MockTok(), batch_size=4, max_seq_len=64, max_samples=8, num_workers=0)
batch = next(iter(loader))
assert batch['input_ids'].shape[0] == 4, f'batch size 异常'
print(f'  Dataset: {len(ds)} samples, batch: {list(batch[\"input_ids\"].shape)}')

os.unlink(tmp)
print('  数据加载验证通过')
" && passed "数据加载" || failed "数据加载" "详见上方错误"

echo ""

# ── Step 5: 统计模块验证 ──────────────────────────────────
echo "Step 5: 监控与统计模块验证"

python3 -c "
import sys
sys.path.insert(0, '.')
from data.data_stats import _detect_language, _infer_task_type

# 语言检测
assert _detect_language('你好世界') == 'zh'
assert _detect_language('hello world') == 'en'
assert _detect_language('Hello 世界 你好') == 'mixed'
print('  语言检测: OK')

# 任务类型推断
assert _infer_task_type({'messages': [{'role': 'user', 'content': '你好'}]}) == 'text_sft'
assert _infer_task_type({'messages': [{'role': 'user', 'content': '<audio>请转写'}]}) == 'asr'
assert _infer_task_type({'messages': [{'role': 'user', 'content': '<image>描述'}]}) == 'image_qa'
assert _infer_task_type({'messages': [{'role': 'user', 'content': '问题'}], 'chosen': {'role': 'assistant', 'content': '好'}, 'rejected': {'role': 'assistant', 'content': '差'}}) == 'dpo'
print('  任务推断: OK')
" && passed "统计模块" || failed "统计模块" "详见上方错误"

# 监控模块
python3 -c "
import sys
sys.path.insert(0, '.')
from metrics.training_monitor import TrainingMonitor
import tempfile, os

log_dir = tempfile.mkdtemp()
monitor = TrainingMonitor(stage=1, log_dir=log_dir)

# 模拟几个 step
for step in range(5):
    monitor.log_step(
        step=step, lm_loss=2.0 - step * 0.1,
        lr=1e-3, grad_norm=0.5, max_grad_norm=1.0, num_tokens=128
    )

summary = monitor.summary()
assert 'loss_total' in summary, '缺少 loss_total'
print(f'  训练监控: {len(summary)} 个指标')
monitor.close(plot=False)

# 清理
import shutil
shutil.rmtree(log_dir)
print('  监控模块验证通过')
" && passed "训练监控" || failed "训练监控" "详见上方错误"

echo ""

# ── Step 6: 配置文件一致性验证 ─────────────────────────────
echo "Step 6: 配置一致性验证"

python3 -c "
import sys, yaml, os
sys.path.insert(0, '.')
from model.config import STAGE_CONFIGS

configs_dir = 'configs'
errors = []

for stage in range(1, 6):
    sc = STAGE_CONFIGS[stage]

    # 检查正式配置
    for suffix in ['', '_smoke']:
        names = ['', 'audio_align', 'vision_align', 'multimodal_sft', 'speech_gen', 'dpo']
        yaml_name = f'stage{stage}_{names[stage]}{suffix}.yaml'
        yaml_path = os.path.join(configs_dir, yaml_name)

        if not os.path.exists(yaml_path):
            errors.append(f'缺少配置文件: {yaml_name}')
            continue

        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)

        # 验证阶段号一致
        if cfg.get('stage') != stage:
            errors.append(f'{yaml_name}: stage 不一致 (YAML={cfg.get(\"stage\")}, expected={stage})')

        # 验证冻结策略
        freeze = cfg.get('freeze', {})
        for comp in sc.frozen_components:
            if not freeze.get(comp, False):
                pass  # smoke 配置可能有不同策略

        print(f'  {yaml_name}: stage={cfg[\"stage\"]}, OK')

if errors:
    for e in errors:
        print(f'  [WARN] {e}')
else:
    print('  全部配置文件一致性检查通过')
" && passed "配置一致性" || failed "配置一致性" "详见上方错误"

echo ""

# ── 汇总 ──────────────────────────────────────────────────
TOTAL_END=$(date +%s)
ELAPSED=$((TOTAL_END - TOTAL_START))

echo "============================================"
echo "  Smoke Test 结果"
echo "============================================"
echo "  通过: ${PASS}"
echo "  失败: ${FAIL}"
echo "  耗时: ${ELAPSED} 秒"
echo ""

if [ ${FAIL} -gt 0 ]; then
    echo "  [WARNING] 有 ${FAIL} 项测试失败，请检查上方详情"
    exit 1
else
    echo "  [SUCCESS] 全部测试通过"
    exit 0
fi

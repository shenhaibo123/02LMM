# Omni 训练代码使用指南

## 概览

本目录包含 Omni 全模态模型的完整训练代码，基于原生 PyTorch 实现，使用 MS-Swift 框架调度。

## 环境要求

- Python >= 3.9
- PyTorch >= 2.1
- ms-swift >= 3.12
- CUDA >= 12.1 (正式训练)

```bash
bash setup_env.sh
```

## 目录说明

```
train/
├── model/                    # 模型定义
│   ├── config.py             # OmniModelConfig + STAGE_CONFIGS
│   ├── omni_model.py         # 主模型类 (455行)
│   ├── projectors.py         # 视觉/音频投影层 (228行)
│   └── speech_decoder.py     # 语音解码头 (234行)
│
├── data/                     # 数据模块
│   ├── download_data.sh      # 数据下载入口
│   ├── prepare_data.py       # 下载+格式转换
│   ├── data_stats.py         # 7 维统计分析
│   └── dataset.py            # PyTorch Dataset + DataLoader
│
├── metrics/                  # 监控评测
│   ├── training_monitor.py   # 训练监控 (JSONL 日志)
│   ├── eval_metrics.py       # WER/CER/BLEU 指标
│   └── visualize.py          # 曲线可视化
│
├── configs/                  # 训练配置
│   ├── stage{1-5}_*.yaml     # 5 个正式 + 5 个 smoke
│   └── deepspeed_zero2.json  # DeepSpeed 配置
│
├── run_stage.sh              # 单阶段训练
├── run_all.sh                # 全流程训练
├── run_smoke_test.sh         # Smoke Test
└── eval.sh                   # 评测
```

## 使用方式

### 1. 数据准备

```bash
# 下载全部数据
bash data/download_data.sh

# 仅下载 Stage 1 数据
bash data/download_data.sh --stages 1

# Smoke test 数据（少量样本）
bash data/download_data.sh --smoke
```

### 2. 数据统计

```bash
python3 data/data_stats.py --data_dir data/datasets
```

### 3. Smoke Test

```bash
bash run_smoke_test.sh
```

验证内容：
- 环境检查 (PyTorch, CUDA, 依赖)
- 模型组件构建与参数统计
- 各阶段冻结/解冻逻辑
- 前向传播 (纯文本 + 投影层 + 语音解码)
- 数据加载与格式转换
- 监控与评测模块

### 4. 单阶段训练

```bash
# 正式训练
bash run_stage.sh --stage 1

# Smoke test
bash run_stage.sh --stage 1 --smoke

# 查看配置
bash run_stage.sh --stage 1 --dry_run
```

### 5. 全流程训练

```bash
# 全部 5 阶段
bash run_all.sh

# 从 Stage 3 开始
bash run_all.sh --start_stage 3

# Smoke 全流程
bash run_all.sh --smoke
```

### 6. 评测

```bash
bash eval.sh --mode smoke       # 组件验证
bash eval.sh --mode component   # 组件详细评测
bash eval.sh --mode text        # 文本评测
bash eval.sh --mode audio       # 音频评测
bash eval.sh --mode full        # 完整评测
```

## 模型配置

`model/config.py` 包含双配置系统：

| 参数 | 正式训练 | Smoke Test |
|------|----------|------------|
| LLM | Qwen3-8B (4096d) | Qwen3-0.5B (1024d) |
| 视觉 | SigLIP-SO400M (1152d, 384px) | CLIP-ViT-B/32 (768d, 224px) |
| 音频 | Whisper-large-v3 (1280d) | Whisper-small (768d) |
| 投影层 | 4096d | 512d |
| 语音解码 | 1024d, 6层 | 256d, 6层 |

使用 `OmniModelConfig().resolve(smoke=True/False)` 自动切换。

## 五阶段冻结策略

| 阶段 | 冻结 | 训练 | 可训练比例 |
|------|------|------|-----------|
| 1 音频对齐 | LLM+编码器+视觉投影+语音解码 | Audio Projector | ~0.4% |
| 2 视觉对齐 | LLM+编码器+音频投影+语音解码 | Visual Projector | ~1.6% |
| 3 联合 SFT | 编码器 | LLM(LoRA)+两投影层 | ~2.7% |
| 4 语音生成 | 编码器+视觉投影 | 语音解码+音频投影+LLM末2层 | ~5.8% |
| 5 DPO 对齐 | 编码器 | 全部投影+LLM(LoRA)+语音解码 | ~6.1% |

## 自测

每个 Python 模块都支持独立运行自测：

```bash
python3 model/config.py           # 配置自测
python3 model/projectors.py       # 投影层自测 (未直接支持, 通过 omni_model.py)
python3 data/dataset.py           # Dataset 自测
```

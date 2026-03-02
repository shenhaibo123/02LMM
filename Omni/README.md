# Omni 全模态大模型训练项目

> 基于 11 篇 Omni 论文方法总结，实现全模态（文本+图像+音频+视频）理解与语音生成的大模型训练方案。

## 项目概述

本项目系统性地：
1. **调研对比** 了 11 篇 Omni 全模态论文的训练方法
2. **深度分析** 了 5 篇重点论文和 4 个开源仓库
3. **设计实现** 了完整的 5 阶段渐进式训练方案
4. **编写验证** 了可执行的模型代码、数据管道、训练脚本

### 模型架构

```
Vision Encoder (SigLIP-SO400M) → MLP Projector    ┐
Audio Encoder  (Whisper-large-v3) → Conv-gMLP Proj ├→ LLM (Qwen3-8B) → Text
Text Embedding                                     ┘         └→ Speech Decoder → Speech Tokens (CosyVoice2)
```

### 五阶段训练流程

| 阶段 | 名称 | 训练组件 | Loss | LR |
|------|------|----------|------|----|
| 1 | 音频-语言对齐 | Audio Projector (~8M) | LM | 1e-3 |
| 2 | 视觉-语言对齐 | Visual Projector (~4M) | LM | 1e-3 |
| 3 | 全模态联合 SFT | LLM (LoRA r=64) + Projectors | LM | 2e-5 |
| 4 | 语音生成训练 | Speech Decoder (~50M) | CTC+LM | 5e-4 |
| 5 | DPO 偏好对齐 | LLM (LoRA r=16) + Decoder | DPO | 5e-6 |

## 目录结构

```
Omni/
├── README.md                      # 本文档
├── OMNI_PROJECT_PROMPT.md         # 总体计划与执行细则
│
├── docs/                          # 调研文档
│   ├── training_comparison.md     # 11 篇论文训练方法对比 (637行)
│   ├── papers/                    # 5 篇重点论文分析报告
│   │   ├── baichuan_omni.md
│   │   ├── mini_omni2.md
│   │   ├── omnigaia.md
│   │   ├── omnivinci.md
│   │   └── openomni.md
│   ├── repos/                     # 4 个开源仓库分析报告
│   │   ├── bc_omni.md
│   │   ├── baichuan_omni_1_5.md
│   │   ├── mini_omni2.md
│   │   └── openomni.md
│   └── *.svg                      # 架构图
│
├── scheme/                        # 训练方案
│   └── training_scheme.md         # 完整训练方案 (选型对比+数据+评测)
│
├── train/                         # 训练代码 (自研)
│   ├── README.md                  # 训练代码使用指南
│   ├── ARCHITECTURE.md            # 架构设计说明
│   ├── model/                     # 模型定义 (~1158行)
│   │   ├── config.py              # 双配置 (正式/smoke)
│   │   ├── omni_model.py          # 主模型类
│   │   ├── projectors.py          # MLP + Conv-gMLP 投影层
│   │   └── speech_decoder.py      # Transformer + CTC 语音解码
│   ├── data/                      # 数据模块
│   │   ├── download_data.sh       # 数据下载脚本
│   │   ├── prepare_data.py        # 下载+转换+占位
│   │   ├── data_stats.py          # 7 维统计分析
│   │   └── dataset.py             # PyTorch Dataset
│   ├── metrics/                   # 监控评测 (~1022行)
│   │   ├── training_monitor.py    # JSONL 日志+GPU 监控
│   │   ├── eval_metrics.py        # WER/CER/BLEU
│   │   └── visualize.py           # 训练曲线可视化
│   ├── configs/                   # 训练配置
│   │   ├── stage{1-5}_*.yaml      # 正式 + smoke 各 5 个
│   │   └── deepspeed_zero2.json
│   ├── run_stage.sh               # 单阶段训练
│   ├── run_all.sh                 # 全流程训练
│   ├── run_smoke_test.sh          # Smoke Test 验证
│   └── eval.sh                    # 评测脚本
│
└── swift_train/                   # MS-Swift 框架训练方案
    ├── README.md
    ├── scripts/                   # 7 个训练脚本
    ├── data/                      # 数据处理
    ├── eval/                      # 评测脚本
    └── configs/
```

## 快速开始

### 环境安装

```bash
cd Omni/train
bash setup_env.sh
```

### Smoke Test（单卡验证，约 30 分钟）

```bash
bash run_smoke_test.sh
```

### 完整训练

```bash
# 逐阶段
bash run_stage.sh --stage 1
bash run_stage.sh --stage 2
# ...

# 一键全流程
bash run_all.sh

# Smoke 全流程
bash run_all.sh --smoke
```

### 评测

```bash
bash eval.sh --mode smoke       # 组件级验证
bash eval.sh --mode full        # 完整评测
```

## 模型选型理由

| 组件 | 选择 | 理由 |
|------|------|------|
| LLM | Qwen3-8B | 中文能力强、Apache 2.0 协议、MS-Swift 原生支持 |
| 视觉 | SigLIP-SO400M | 400M 参数性能/效率平衡、无需 [CLS] token |
| 音频 | Whisper-large-v3 | 中英双语 ASR、1500 帧标准输出 |
| 语音合成 | CosyVoice2 | 流式推理、5s 零样本克隆、多语种 |
| 视觉投影 | MLP (2层+GELU) | 简洁高效，LLaVA 验证有效 |
| 音频投影 | Conv-gMLP (4x↓) | 时序降采样+门控，Qwen2.5-Omni 验证 |

详细对比见 [scheme/training_scheme.md](scheme/training_scheme.md)

## 评测基准

| 能力 | 评测集 | 指标 | 选用理由 |
|------|--------|------|----------|
| 图像理解 | MMBench | Accuracy | 覆盖 20+ 能力维度，综合性最强 |
| 专业知识 | MMMU | Accuracy | 大学水平多模态问答 |
| OCR | TextVQA | Accuracy | 评估文字识别+理解 |
| 中文 ASR | AISHELL-1 test | CER | 标准中文语音识别基准 |
| 英文 ASR | LibriSpeech test-clean | WER | 标准英文语音识别基准 |
| 文本 | MMLU / C-Eval | Accuracy | 英文/中文知识评测 |
| 数学 | GSM8K | Accuracy | 数学推理能力 |

## 参考资料

- **论文对比**: [docs/training_comparison.md](docs/training_comparison.md)
- **训练方案**: [scheme/training_scheme.md](scheme/training_scheme.md)
- **方案草稿**: [Doc/草稿/08_Omni模型训练方案.md](../Doc/草稿/08_Omni模型训练方案.md) (3168行)

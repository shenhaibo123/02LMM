# Omni 全模态训练方案 —— 基于 MS-Swift

> 基于 11 篇 Omni 论文方法总结，使用 MS-Swift 框架实现完整的多模态大模型训练流程。

---

## 架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                    Qwen2.5-Omni-7B 基座                      │
│                                                             │
│  SigLIP-so400m ──→ Visual Projector (MLP) ──┐              │
│                                              │              │
│  Whisper-large-v3 → Audio Projector ─────────┼→ LLM Backbone│
│                     (Conv-GMLP 4x↓)         │   (Qwen-7B)  │
│                                              │              │
│  Text Tokenizer ─→ Token Embedding ─────────┘      │       │
│                                                     ↓       │
│                                              ┌──── Thinker ─┤
│                                              │      ↓       │
│                                              │   Talker ────┤
│                                              │   (Speech)   │
│                                              └──────────────┘
└─────────────────────────────────────────────────────────────┘
```

## 五阶段训练流程

```
Stage 1          Stage 2          Stage 3            Stage 4          Stage 5
视觉-语言对齐 → 音频-语言对齐 → 全模态联合 SFT → 语音生成训练 → DPO/GRPO 对齐

冻结: 编码器+LLM  冻结: 编码器+LLM  冻结: 编码器       冻结: 视觉编码器   冻结: 编码器
训练: 视觉投影    训练: 音频投影    训练: 投影层+LLM   训练: 音频+LLM     训练: 全部可训练
Loss: LM          Loss: LM         Loss: LM           Loss: CTC+LM      Loss: DPO
LR: 1e-3          LR: 1e-3         LR: 2e-5           LR: 5e-4          LR: 5e-6
```

## 快速开始

### 1. 环境安装

```bash
cd Omni/swift_train
bash setup_env.sh
```

### 2. Smoke Test（单卡快速验证）

```bash
bash scripts/run_smoke_test.sh
```

### 3. 完整训练

```bash
# 逐阶段训练
bash scripts/run_stage1.sh   # 视觉-语言对齐
bash scripts/run_stage2.sh   # 音频-语言对齐
bash scripts/run_stage3.sh   # 全模态联合 SFT
bash scripts/run_stage4.sh   # 语音生成（自定义 trainer）
bash scripts/run_stage5.sh   # DPO 偏好对齐

# 或一键全流程
bash scripts/run_all_stages.sh
```

### 4. 评测

```bash
bash eval/run_eval.sh
```

---

## 目录结构

```
swift_train/
├── README.md                   # 本文档
├── PAPERS_SUMMARY.md           # 论文方法总结
├── requirements.txt            # 依赖列表
├── setup_env.sh                # 环境安装脚本
│
├── configs/                    # 训练配置
│   ├── stage1_vision_align.yaml
│   ├── stage2_audio_align.yaml
│   ├── stage3_multimodal_sft.yaml
│   ├── stage4_speech_gen.yaml
│   ├── stage5_dpo.yaml
│   └── deepspeed_zero2.json
│
├── data/                       # 数据处理
│   ├── prepare_datasets.py     # 数据下载与预处理
│   ├── convert_to_swift.py     # 转换为 MS-Swift 格式
│   └── DATA_SOURCES.md         # 数据源说明
│
├── scripts/                    # 训练脚本
│   ├── run_stage1.sh
│   ├── run_stage2.sh
│   ├── run_stage3.sh
│   ├── run_stage4.sh
│   ├── run_stage5.sh
│   ├── run_smoke_test.sh
│   └── run_all_stages.sh
│
└── eval/                       # 评测
    ├── eval_multimodal.py      # 多模态评测脚本
    ├── eval_audio.py           # 音频/ASR 评测
    └── run_eval.sh             # 评测启动脚本
```

---

## 数据需求

| 阶段 | 数据类型 | 规模 | 来源 |
|------|----------|------|------|
| Stage 1 | 图像-文本对 | 500K~5M | LLaVA-CC3M, ShareGPT4V |
| Stage 2 | 音频-文本对 | 200K~1M | AISHELL, LibriSpeech, WenetSpeech |
| Stage 3 | 多模态混合 SFT | 500K~2M | LLaVA-Instruct, AudioCaps, VideoChat |
| Stage 4 | 文本-语音对 | 100K~500K | LibriTTS, AISHELL-3 |
| Stage 5 | 偏好对数据 | 10K~50K | UltraFeedback, 自建偏好数据 |

## 硬件需求

| 配置 | GPU | 预估时间 | 适用场景 |
|------|-----|----------|----------|
| Smoke Test | 1×A100-80G | ~30 分钟 | 流程验证 |
| 最小训练 | 4×A100-80G | ~7 天 | 小规模实验 |
| 标准训练 | 8×A100-80G | ~14 天 | 完整训练 |
| 大规模 | 32×A100-80G | ~5 天 | 高效完整训练 |

## 评测指标

| 能力 | 基准 | 指标 |
|------|------|------|
| 图像理解 | MMBench, MMMU, TextVQA | Accuracy |
| 音频理解 (ASR) | AISHELL-1, LibriSpeech | WER, CER |
| 视频理解 | Video-MME, MVBench | Accuracy |
| 语音生成 | 自建测试集 | MOS, WER |
| 文本生成 | MMLU, C-Eval, GSM8K | Accuracy |
| 多模态综合 | MMBench-Audio, SEEDBench | Accuracy |

## 参考论文

详见 [PAPERS_SUMMARY.md](./PAPERS_SUMMARY.md)

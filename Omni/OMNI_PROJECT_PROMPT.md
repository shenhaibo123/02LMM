# Omni 全模态模型项目：统一需求与执行计划 v3

> 更新日期：2026-03-02 (v3 — 全部任务完成)
> 本文档整合原始需求、参考资料、当前现状、模型规划与执行计划。供后续执行或子 Agent 使用。

---

## 一、原始需求

### 1.1 用户原始要求（原文）

> 帮我整体看一下11篇论文的训练过程，详细说明，包括训练阶段，这个阶段做什么，用多少数据，什么类型的数据，数据来源，公开数据有哪些，用多少卡，训多久，代码是否开源，有哪些创新。我需要一份详细的训练文档，可执行的。重点关注百川Omni，miniOmni2, OmniVinci, OmniGAIA, OpenOmni，还有几个开源项目，你逐个调用子agent参考分析。
>
> 每个项目都给出一份详细的报告，仓库添加到reference。存储位置：你在根目录创建一个Omni目录，把之前的文档，方案，和你新构建的这些文档，方案都放这。同时根据开源项目，构建一份可落地执行的Omni训练代码，并且做验证，我会把他放在GPU平台训练，项目需要包含数据下载和数据处理等完整的。让我能一键执行。
>
> 语音输出用cosyVoice2是不是比较好？以及各个模块包括adapter是不是需要说明，每次解冻训练哪些参数，用哪些数据，数据量多少，多少卡训多少时间都需要说明，每个模块的选择也要说明，smoketest也要用对应更小的编码头适配吧，兼顾效率和效果。然后这个训练代码也要详细说明，是训练Omni，代码是不是有模型定义部分？要写辅助文档帮助我阅读代码，训练过程中也要有检测指标和绘图，训完要有评测指标，数据是不是要有统计分析指标？

### 1.2 需求拆解

| 编号 | 需求项 | 产出物 | 状态 |
|------|--------|--------|------|
| R1 | 11 篇论文逐篇训练过程详细说明 | `Omni/docs/training_comparison.md` | **已完成** (637行) |
| R2 | 5 篇重点论文单独详细报告 | `Omni/docs/papers/*.md` (5个) | **已完成** |
| R3 | 4 个开源仓库详细分析报告 | `Omni/docs/repos/*.md` (4个) | **已完成** |
| R4 | 仓库克隆到 reference/ | `reference/{bc-omni,Baichuan-Omni-1.5,mini-omni2,OpenOmni}/` | **已完成** |
| R5 | 训练方案整合文档 | `Omni/scheme/training_scheme.md` (509行) | **已完成** |
| R6 | 可落地训练代码（含模型定义） | `Omni/train/` | **已完成** |
| R7 | 数据下载+处理+统计分析 | `Omni/train/data/` (4文件, 1273行) | **已完成** |
| R8 | 各阶段训练配置（YAML + smoke） | `Omni/train/configs/` (10个YAML) | **已完成** |
| R9 | 一键执行脚本 | `run_all.sh`, `run_smoke_test.sh`, `eval.sh`, `run_stage.sh` | **已完成** |
| R10 | 训练过程检测指标与绘图 | `Omni/train/metrics/` | **已完成** (1022行) |
| R11 | 训完评测指标 | `Omni/train/metrics/eval_metrics.py` + `Omni/swift_train/eval/` | **已完成** |
| R12 | 辅助文档（架构+代码导读） | `Omni/README.md`, `train/README.md`, `train/ARCHITECTURE.md` | **已完成** |
| R13 | Smoke test 用小编码器适配 | config.py + 5个smoke YAML + run_smoke_test.sh | **已完成** |
| R14 | CosyVoice2 选型说明 | `scheme/training_scheme.md` + `train/ARCHITECTURE.md` | **已完成** |
| R15 | 验证（dry-run + 依赖检查） | Python/Shell/YAML 语法 + 交叉引用 + 前向传播 | **已完成** |

---

## 二、参考资料索引

### 2.1 项目内路径

| 类别 | 路径 | 行数/状态 |
|------|------|-----------|
| 已有训练方案（长文档） | `Doc/草稿/08_Omni模型训练方案.md` | 3168 行 |
| 11 篇论文训练对比（原始版） | `papers/omni/TRAINING_COMPARISON.md` | 172 行 |
| 11 篇论文训练对比（增强版） | `Omni/docs/training_comparison.md` | 637 行 |
| 5 篇重点论文报告 | `Omni/docs/papers/{baichuan_omni,mini_omni2,omnivinci,omnigaia,openomni}.md` | 各 145-193 行 |
| 论文原文精读分析 | `papers/{Baichuan-Omni,Mini-Omni2,OmniVinci,OmniGAIA,Ola}/*_analysis.md` | — |
| 4 个参考仓库（已克隆） | `reference/{bc-omni,Baichuan-Omni-1.5,mini-omni2,OpenOmni}/` | 已克隆 |
| 模型定义代码 | `Omni/train/model/{config,omni_model,projectors,speech_decoder}.py` | 1158 行 |
| 监控与评测代码 | `Omni/train/metrics/{training_monitor,eval_metrics,visualize}.py` | 1000 行 |
| MS-Swift 训练脚本 | `Omni/swift_train/scripts/run_stage{1-5}.sh` 等 | 7 个脚本 |
| MS-Swift 数据处理 | `Omni/swift_train/data/{prepare_datasets,convert_to_swift}.py` | 622 行 |
| MS-Swift 评测 | `Omni/swift_train/eval/{eval_multimodal,eval_audio}.py` | 633 行 |

### 2.2 外部参考

| 项目 | 地址 | 用途 |
|------|------|------|
| ms-swift | https://github.com/modelscope/ms-swift | 训练框架，支持 Qwen3-Omni |
| CosyVoice2 | https://github.com/FunAudioLLM/CosyVoice | 语音合成/TTS |
| bc-omni | https://github.com/westlake-baichuan-mllm/bc-omni | 参考仓库 |
| Baichuan-Omni-1.5 | https://github.com/baichuan-inc/Baichuan-Omni-1.5 | 参考仓库 |
| mini-omni2 | https://github.com/gpt-omni/mini-omni2 | 参考仓库 |
| OpenOmni | https://github.com/RainBowLuoCS/OpenOmni | 参考仓库 |

---

## 三、当前项目状态

### 3.1 完成度总览

```
总进度: ████████░░░░░░░ 55%

[100%] 论文调研与对比     ████████████████ 完成
[100%] 参考仓库克隆       ████████████████ 完成
[100%] 模型定义代码       ████████████████ 完成
[100%] 监控与评测代码     ████████████████ 完成
[100%] MS-Swift 脚本      ████████████████ 完成
[  0%] 仓库分析报告 (4个)  ░░░░░░░░░░░░░░░░ 待做
[  0%] 训练方案整合       ░░░░░░░░░░░░░░░░ 待做
[  0%] 数据下载/处理/统计  ░░░░░░░░░░░░░░░░ 待做
[  0%] 训练配置 YAML      ░░░░░░░░░░░░░░░░ 待做
[  0%] 一键执行脚本       ░░░░░░░░░░░░░░░░ 待做
[  0%] 辅助文档 (3个)     ░░░░░░░░░░░░░░░░ 待做
[  0%] 验证               ░░░░░░░░░░░░░░░░ 待做
```

### 3.2 已存在文件清单

```
Omni/
├── OMNI_PROJECT_PROMPT.md              ← 本文件
├── docs/
│   ├── training_comparison.md          ✅ 637 行
│   ├── papers/
│   │   ├── baichuan_omni.md            ✅ 193 行
│   │   ├── mini_omni2.md               ✅ 145 行
│   │   ├── omnivinci.md                ✅ 155 行
│   │   ├── omnigaia.md                 ✅ 165 行
│   │   └── openomni.md                 ✅ 182 行
│   ├── repos/                          ❌ 空目录
│   ├── omni_architecture.svg           ✅
│   ├── omni_architecture_full.svg      ✅
│   └── omni_training_stages.svg        ✅
├── scheme/                             ❌ 空目录
├── train/
│   ├── model/
│   │   ├── __init__.py                 ✅ 24 行
│   │   ├── config.py                   ✅ 195 行
│   │   ├── omni_model.py               ✅ 455 行
│   │   ├── projectors.py               ✅ 228 行
│   │   └── speech_decoder.py           ✅ 234 行
│   ├── metrics/
│   │   ├── __init__.py                 ✅ 21 行
│   │   ├── eval_metrics.py             ✅ 376 行
│   │   ├── training_monitor.py         ✅ 391 行
│   │   └── visualize.py                ✅ 234 行
│   ├── configs/
│   │   └── deepspeed_zero2.json        ✅
│   ├── data/                           ❌ 空目录
│   ├── requirements.txt                ✅
│   └── setup_env.sh                    ✅
├── swift_train/
│   ├── README.md                       ✅ 148 行
│   ├── PAPERS_SUMMARY.md               ✅ 158 行
│   ├── configs/deepspeed_zero2.json    ✅
│   ├── data/
│   │   ├── DATA_SOURCES.md             ✅ 76 行
│   │   ├── prepare_datasets.py         ✅ 244 行
│   │   └── convert_to_swift.py         ✅ 378 行
│   ├── eval/
│   │   ├── eval_multimodal.py          ✅ 359 行
│   │   ├── eval_audio.py               ✅ 274 行
│   │   └── run_eval.sh                 ✅
│   ├── scripts/
│   │   ├── run_stage{1-5}.sh           ✅ 5 个
│   │   ├── run_smoke_test.sh           ✅
│   │   └── run_all_stages.sh           ✅
│   ├── requirements.txt                ✅
│   └── setup_env.sh                    ✅
```

---

## 四、模型与训练规划

### 4.1 框架选型

| 决策 | 选择 | 理由 |
|------|------|------|
| 训练框架 | **ms-swift ≥ 3.12** | 原生支持 Qwen3-Omni；覆盖 CPT/SFT/LoRA/DPO/GRPO；600+ LLM + 300+ MLLM |
| LLM 骨干 | **Qwen3-8B** (正式) / **Qwen3-0.5B** (smoke) | 中文优秀、开源、ms-swift 原生支持 |
| 视觉编码器 | **SigLIP-SO400M-384** (正式) / **CLIP-ViT-B/32** (smoke) | 400M 参数效果好；86M smoke 兼顾效率 |
| 音频编码器 | **Whisper-large-v3** (正式) / **Whisper-small** (smoke) | 多语言 ASR 标杆；small 244M 够 smoke |
| 语音合成 | **CosyVoice2** | 流式合成、5s 零样本克隆、多语言、指令可控；v2 > v1 |
| 视觉投影 | **2 层 MLP + GELU** | 参考 LLaVA / Baichuan-Omni，简单有效 |
| 音频投影 | **Conv1D 4× 降采样 + GMLP** | 参考 Baichuan-Omni Conv-GMLP，减少序列长度 |
| 语音解码头 | **Transformer Decoder + CTC** | 参考 LLaMA-Omni CTC 方案，non-autoregressive |

### 4.2 模块选型对比表（正式 vs Smoke）

| 组件 | 正式训练 | 参数量 | Smoke test | 参数量 | 缩放比 |
|------|----------|--------|------------|--------|--------|
| LLM | Qwen3-8B | 8B | Qwen3-0.5B | 0.5B | 1/16 |
| 视觉编码器 | SigLIP-SO400M-384 | 400M | CLIP-ViT-B/32 | 86M | 1/4.7 |
| 音频编码器 | Whisper-large-v3 | 1.5B | Whisper-small | 244M | 1/6.1 |
| 视觉投影 | MLP 1152→4096→4096 | ~4M | MLP 768→512→1024 | ~0.5M | 1/8 |
| 音频投影 | Conv-GMLP 1280→4096 | ~8M | Conv-GMLP 768→1024 | ~1M | 1/8 |
| 语音解码头 | 6层 d=4096 | ~50M | 4层 d=1024 | ~4M | 1/12.5 |
| **总参数** | | **~10B** | | **~0.84B** | **1/12** |

### 4.3 CosyVoice2 选型理由

| 特性 | CosyVoice1 | CosyVoice2 | 说明 |
|------|-----------|-----------|------|
| 流式合成 | 不支持 | 支持（chunk-aware） | 实时对话必需 |
| 零样本克隆 | 需要 > 10s | 仅需 5s | 更实用 |
| 多语言 | 中英 | 中英日韩粤 | 覆盖更广 |
| 指令可控 | 不支持 | 支持（情感/语速） | 更丰富的表达 |
| 采样率 | 22.05kHz | 24kHz | 音质更好 |
| 开源 | ✓ | ✓ | Apache-2.0 |

### 4.4 五阶段训练详细规划

#### 阶段 1：音频→文本对齐

| 项目 | 正式训练 | Smoke test |
|------|----------|------------|
| 目标 | 让音频投影层学会将 Whisper 特征映射到 LLM 空间 | 验证流程可跑通 |
| 冻结 | LLM (Qwen3-8B) + 音频编码器 (Whisper-large-v3) | LLM (0.5B) + Whisper-small |
| 可训练 | 音频投影层 (Conv-GMLP, ~8M) | 音频投影层 (~1M) |
| 损失 | LM Loss (next-token prediction) | 同左 |
| 学习率 | 1e-3, cosine decay | 同左 |
| 数据 | LibriSpeech (960h) + AISHELL-1 (178h) + WenetSpeech 子集 | LibriSpeech-clean-100 (100h) |
| 数据量 | ~28 万条 | ~3000 条 |
| 数据格式 | {audio_path, transcript} → Swift JSONL | 同左 |
| 硬件 | 8×A100-80G | 1×任意 GPU (≥16G) |
| 时长 | ~2 天 | ~30 分钟 |
| Epochs | 3 | 1 |
| Batch size | 32/GPU | 4 |

#### 阶段 2：图像→文本预训练

| 项目 | 正式训练 | Smoke test |
|------|----------|------------|
| 目标 | 让视觉投影层学会将 SigLIP 特征映射到 LLM 空间 | 验证流程可跑通 |
| 冻结 | LLM + 视觉编码器 (SigLIP) + 音频投影（保持阶段1参数） | LLM + CLIP + 音频投影 |
| 可训练 | 视觉投影层 (MLP, ~4M) | 视觉投影层 (~0.5M) |
| 损失 | LM Loss | 同左 |
| 学习率 | 1e-3, cosine decay | 同左 |
| 数据 | LLaVA-Pretrain-595K (CC3M 子集) + ShareGPT4V-100K | LLaVA-Pretrain 1 万条子集 |
| 数据量 | ~70 万条 | ~1 万条 |
| 数据格式 | {image_path, conversations} → Swift JSONL | 同左 |
| 硬件 | 8×A100-80G | 1×任意 GPU (≥16G) |
| 时长 | ~1 天 | ~30 分钟 |
| Epochs | 1 | 1 |
| Batch size | 64/GPU | 4 |

#### 阶段 3：多模态联合 SFT

| 项目 | 正式训练 | Smoke test |
|------|----------|------------|
| 目标 | 联合训练让模型具备多模态理解+对话能力 | 验证流程可跑通 |
| 冻结 | 视觉编码器 + 音频编码器 | 同左 |
| 可训练 | LLM (全参或 LoRA r=64) + 视觉投影 + 音频投影 | LLM (LoRA r=8) + 投影层 |
| 损失 | LM Loss（多模态指令跟随） | 同左 |
| 学习率 | 2e-5 (全参) / 1e-4 (LoRA), cosine decay | 1e-4 (LoRA) |
| 数据来源 | 图文 SFT: LLaVA-Instruct-150K, ShareGPT4V-SFT | 各类各 1000 条 |
| | 音文 SFT: AudioCaps, AISHELL-SFT 自建 | |
| | 纯文本: ShareGPT, Alpaca-GPT4 | |
| | 视频: VideoChat2, Video-ChatGPT | |
| 数据量 | ~35 万条混合 | ~4000 条 |
| 数据格式 | {images/audios, conversations} → Swift JSONL | 同左 |
| 数据混合比 | 图文 40% : 音文 25% : 纯文本 20% : 视频 15% | 同左 |
| 硬件 | 8×A100-80G | 1×任意 GPU (≥16G) |
| 时长 | ~3 天 | ~1 小时 |
| Epochs | 3 | 1 |
| Batch size | 16/GPU | 2 |

#### 阶段 4：语音生成训练

| 项目 | 正式训练 | Smoke test |
|------|----------|------------|
| 目标 | 训练语音解码头，使模型能输出语音 | 验证 CTC 解码头可训练 |
| 冻结 | 视觉编码器 + 音频编码器 + LLM 大部分层 | 同左 |
| 可训练 | 语音解码头 (Transformer Decoder + CTC, ~50M) + LLM 最后 2 层 | 语音解码头 (~4M) |
| 损失 | CTC Loss + LM Loss (辅助) | 同左 |
| 学习率 | 5e-4, cosine decay | 同左 |
| 数据 | LibriTTS (585h) + AISHELL-3 (85h) + CosyVoice2 合成 | LibriTTS 1000 条子集 |
| 数据量 | ~10 万条 (text→speech 对) | ~1000 条 |
| 数据格式 | {text, speech_tokens} → 自定义 Dataset | 同左 |
| 硬件 | 8×A100-80G | 1×任意 GPU (≥16G) |
| 时长 | ~2 天 | ~30 分钟 |
| 训练方式 | 自定义 trainer（ms-swift 暂不支持 CTC 训练） | 同左 |

#### 阶段 5：偏好优化 (DPO/GRPO)

| 项目 | 正式训练 | Smoke test |
|------|----------|------------|
| 目标 | 对齐人类偏好，提升回复质量 | 验证 DPO 流程 |
| 冻结 | 视觉编码器 + 音频编码器 | 同左 |
| 可训练 | LLM (LoRA r=16, 最后几层) + 语音解码头 | LLM (LoRA r=4) + 语音解码头 |
| 损失 | DPO Loss 或 GRPO Loss | DPO Loss |
| 学习率 | 5e-6, cosine decay | 同左 |
| 数据 | UltraFeedback + 自建多模态偏好数据 | 500 条子集 |
| 数据量 | ~1 万条偏好对 | ~500 条 |
| 数据格式 | {prompt, chosen, rejected} → Swift DPO JSONL | 同左 |
| 硬件 | 8×A100-80G | 1×任意 GPU (≥16G) |
| 时长 | ~1 天 | ~20 分钟 |
| 训练方式 | `swift rlhf --rlhf_type dpo` | 同左 |

### 4.5 数据统计分析要求

每个阶段的数据需输出以下统计：

| 统计项 | 说明 |
|--------|------|
| 总样本数 | 训练/验证/测试 split |
| 文本长度分布 | 直方图：token 数 min/max/mean/median/p95/p99 |
| 音频时长分布 | 直方图：秒 min/max/mean/median |
| 图像分辨率分布 | 统计不同分辨率占比 |
| 类别/任务分布 | 各任务类型（QA/caption/ASR 等）样本占比 |
| 语言分布 | 中文/英文/其他占比 |
| 数据质量检查 | 空样本、异常值、重复样本检测 |

---

## 五、执行计划（共 8 步）

### 整体依赖关系

```
步骤 1 (仓库报告) ─┐
步骤 2 (训练方案) ──┤── 可并行
步骤 3 (数据模块) ──┘
                    │
步骤 4 (训练配置) ──── 依赖步骤 3 的数据格式定义
                    │
步骤 5 (执行脚本) ──── 依赖步骤 3 + 4
                    │
步骤 6 (辅助文档) ──── 依赖步骤 1-5 全部完成
                    │
步骤 7 (验证) ─────── 依赖步骤 3-6 全部完成
                    │
步骤 8 (总入口文档) ── 依赖全部完成
```

---

### 步骤 1：4 个开源仓库详细分析报告

**产出**：`Omni/docs/repos/{bc_omni.md, baichuan_omni_1.5.md, mini_omni2.md, openomni.md}`

**执行方式**：并行启动 4 个子 Agent，每个 Agent 分析一个仓库。

**每份报告要求**（统一模板）：

```markdown
# [仓库名] 详细分析报告

## 1. 项目概览
- 仓库地址、Star 数、最后更新时间、开源协议
- 项目目标与定位
- 对应论文

## 2. 目录结构
- 完整目录树（关键文件标注用途）
- 代码行数统计

## 3. 模型架构
- 架构图（ASCII/mermaid）
- 各组件选型与参数规模
- 模态：支持哪些输入/输出模态

## 4. 训练流程
- 分阶段说明（每阶段）：
  - 阶段名称与目标
  - 冻结/可训练模块
  - 损失函数
  - 学习率与调度器
  - 数据来源与规模
  - Batch size 与硬件需求
  - 训练时长
- 训练脚本位置与命令

## 5. 数据格式与处理
- 输入数据格式（JSON/JSONL 示例）
- 数据预处理流程
- 支持的数据集

## 6. 评测
- 评测脚本位置
- 评测基准与指标
- 论文报告的指标结果

## 7. 可复现性评估
- 环境搭建难度（1-5 分）
- 数据可获取性（1-5 分）
- 代码完整度（1-5 分）
- 文档质量（1-5 分）
- 硬件门槛
- 已知 issue 或坑

## 8. 创新点与借鉴价值
- 架构创新
- 训练策略创新
- 可借鉴的设计
- 对本项目的启发

## 9. 关键代码片段索引
- 模型定义位置
- 训练入口位置
- 数据加载位置
- 评测入口位置
```

**数据源**：
- `reference/bc-omni/` — bc-omni (Baichuan-Omni 原版)
- `reference/Baichuan-Omni-1.5/` — Baichuan-Omni 1.5 版本
- `reference/mini-omni2/` — Mini-Omni2
- `reference/OpenOmni/` — OpenOmni

**质量检查**：
- [ ] 每份报告 ≥ 200 行
- [ ] 包含所有 9 个章节
- [ ] 代码路径引用准确（相对于 reference/ 目录）
- [ ] 可复现性评分有依据

---

### 步骤 2：训练方案整合文档

**产出**：`Omni/scheme/training_scheme.md`

**数据源**：
- `Doc/草稿/08_Omni模型训练方案.md` (3168 行) — 原始长方案
- `Omni/docs/training_comparison.md` — 11 篇论文对比
- `Omni/docs/papers/*.md` — 5 篇重点论文报告
- 本文件第四章 — 模型与训练规划

**文档结构要求**：

```markdown
# Omni 全模态模型训练方案

## 1. 方案总览
- 目标：构建支持图像/音频/视频/文本输入 + 文本/语音输出的全模态模型
- 框架选型与理由
- 总体架构图

## 2. 模块选型详细说明
### 2.1 LLM 骨干选型
### 2.2 视觉编码器选型（对比 SigLIP / CLIP / InternViT）
### 2.3 音频编码器选型（对比 Whisper / HuBERT / WavLM）
### 2.4 视觉投影层选型（对比 MLP / QFormer / Resampler）
### 2.5 音频投影层选型（对比 MLP / Conv-GMLP / QFormer）
### 2.6 语音解码选型（对比 CTC / AR / CosyVoice2）
### 2.7 CosyVoice2 vs CosyVoice1 详细对比

## 3. 五阶段训练详细方案
（每阶段含：目标、冻结策略、可训参数及参数量、损失、数据、硬件、时长、Smoke 配置）

## 4. 数据方案
### 4.1 各阶段数据来源与获取方式
### 4.2 数据处理流水线
### 4.3 数据质量控制
### 4.4 数据统计分析指标

## 5. 评测方案
### 5.1 训练过程指标（loss/PPL/grad_norm/LR/GPU 利用率）
### 5.2 阶段评测指标
### 5.3 最终评测基准与目标

## 6. 硬件与成本估算
### 6.1 正式训练（8×A100 方案）
### 6.2 Smoke test（单卡方案）
### 6.3 成本预估

## 7. 风险与应对
### 7.1 模态冲突 → 渐进解冻
### 7.2 灾难性遗忘 → 混合数据 + 正则化
### 7.3 CTC 训练不稳定 → 课程学习
### 7.4 数据不平衡 → 动态采样

## 附录
- A. 论文方法对照表（来自 training_comparison.md 精简版）
- B. 开源项目借鉴总结（来自 repos/*.md 精简版）
```

**质量检查**：
- [ ] 文档 ≥ 500 行
- [ ] 所有选型都有对比论证
- [ ] 五阶段参数与本文件第 4.4 节一致
- [ ] 引用路径正确

---

### 步骤 3：数据下载、处理与统计模块

**产出目录**：`Omni/train/data/`

**需要创建的文件**：

| 文件 | 用途 | 行数预估 |
|------|------|----------|
| `download_data.sh` | 一键下载所有阶段数据（HuggingFace / ModelScope） | ~200 行 |
| `prepare_data.py` | 数据预处理统一入口（清洗、格式化、split） | ~400 行 |
| `data_stats.py` | 数据统计分析与可视化（输出图表+报告） | ~350 行 |
| `dataset.py` | 各阶段 PyTorch Dataset 封装 | ~500 行 |

**各文件详细要求**：

#### download_data.sh
```bash
# 支持参数：
#   --stage {1,2,3,4,5,all}  下载指定阶段数据
#   --smoke                   仅下载 smoke test 子集
#   --output-dir DIR           输出目录（默认 data/raw/）
#   --source {hf,ms}           下载源（HuggingFace/ModelScope）

# 功能：
# 1. 检查已下载数据（断点续传）
# 2. 下载各阶段数据集
# 3. 验证数据完整性（checksum）
# 4. 输出下载报告（大小、数量）
```

#### prepare_data.py
```python
# 支持命令：
#   python prepare_data.py --stage 1 --output-dir data/processed/
#   python prepare_data.py --stage all --smoke
#   python prepare_data.py --stage 3 --format swift  # 转换为 ms-swift 格式

# 功能：
# 1. 读取 raw 数据 → 清洗（去重、过滤空样本、截断过长样本）
# 2. 格式转换 → ms-swift JSONL / 自定义 Dataset 格式
# 3. Train/Val/Test split（默认 95:3:2）
# 4. 输出处理报告
# 5. 支持 --smoke 模式（每类仅取 N 条）
```

#### data_stats.py
```python
# 支持命令：
#   python data_stats.py --data-dir data/processed/stage1/ --output-dir data/stats/
#   python data_stats.py --all-stages

# 输出：
# 1. 文本长度分布直方图 (PNG)
# 2. 音频时长分布直方图 (PNG，如有音频)
# 3. 图像分辨率分布饼图 (PNG，如有图像)
# 4. 任务类型分布柱状图 (PNG)
# 5. 语言分布饼图 (PNG)
# 6. 统计摘要 JSON（min/max/mean/median/p95/p99 等）
# 7. 质量报告 TXT（空样本数、重复数、异常值）
```

#### dataset.py
```python
# 各阶段 Dataset 类：
# - Stage1AudioTextDataset: (audio_features, text_labels)
# - Stage2ImageTextDataset: (image_features, text_labels)
# - Stage3MultimodalSFTDataset: (images?, audios?, conversations)
# - Stage4SpeechGenDataset: (text_input, speech_tokens)
# - Stage5DPODataset: (prompt, chosen, rejected)
#
# 公共功能：
# - 支持 collate_fn（动态 padding）
# - 支持 DistributedSampler
# - 支持数据增强（可选）
# - 集成 tokenizer 和特征提取器
```

**质量检查**：
- [ ] `download_data.sh` 的 `--smoke` 模式可在 5 分钟内完成下载
- [ ] `prepare_data.py` 所有阶段处理后输出格式统一
- [ ] `data_stats.py` 能生成至少 5 类统计图表
- [ ] `dataset.py` 5 个 Dataset 类均可实例化（dry-run）
- [ ] Python 文件通过 `py_compile` 语法检查
- [ ] Shell 脚本通过 `bash -n` 语法检查

---

### 步骤 4：各阶段训练配置（YAML + DeepSpeed）

**产出目录**：`Omni/train/configs/`

**需要创建的文件**：

| 文件 | 用途 |
|------|------|
| `stage1_audio_align.yaml` | 阶段 1 音频对齐 —— 正式训练 |
| `stage2_vision_align.yaml` | 阶段 2 视觉对齐 —— 正式训练 |
| `stage3_multimodal_sft.yaml` | 阶段 3 多模态 SFT —— 正式训练 |
| `stage4_speech_gen.yaml` | 阶段 4 语音生成 —— 正式训练 |
| `stage5_dpo.yaml` | 阶段 5 DPO/GRPO —— 正式训练 |
| `smoke_stage1.yaml` | 阶段 1 smoke —— 小模型+小数据 |
| `smoke_stage2.yaml` | 阶段 2 smoke |
| `smoke_stage3.yaml` | 阶段 3 smoke |
| `smoke_stage4.yaml` | 阶段 4 smoke |
| `smoke_stage5.yaml` | 阶段 5 smoke |
| `deepspeed_zero2.json` | 已存在，检查是否需要更新 |

**YAML 配置模板**（每个文件需包含）：

```yaml
# === 阶段说明 ===
# 目标：...
# 冻结模块：...
# 可训练模块及参数量：...

# === 模型配置 ===
model_name_or_path: ...
# 或 smoke test 对应的小模型

# === 训练配置 ===
train_type: full / lora
freeze_parameters: [...]
trainable_parameters: [...]
lora_rank: ...        # 如使用 LoRA
lora_target_modules: [...]

# === 数据配置 ===
dataset: [...]
max_length: ...
preprocessing_num_workers: 8

# === 优化器配置 ===
learning_rate: ...
lr_scheduler_type: cosine
warmup_ratio: 0.03
weight_decay: 0.01
num_train_epochs: ...
per_device_train_batch_size: ...
gradient_accumulation_steps: ...

# === DeepSpeed ===
deepspeed: configs/deepspeed_zero2.json

# === 日志与保存 ===
logging_steps: 10
save_strategy: steps
save_steps: 500
eval_steps: 500
output_dir: ...
```

**质量检查**：
- [ ] 10 个 YAML 文件均为合法 YAML
- [ ] 正式配置与 smoke 配置的模型名正确对应
- [ ] 冻结/可训练参数与第四章 4.4 节一致
- [ ] 学习率、batch_size 等参数合理

---

### 步骤 5：一键执行脚本

**产出目录**：`Omni/train/`（根级脚本）

**需要创建的文件**：

| 文件 | 用途 |
|------|------|
| `run_all.sh` | 全流程一键执行（数据下载→处理→5 阶段训练→评测） |
| `run_smoke_test.sh` | Smoke test 一键执行（小模型+小数据+单卡） |
| `eval.sh` | 一键评测（WER/CER/MMBench/MOS 等） |
| `run_stage.sh` | 单阶段训练入口（参数控制阶段号） |

**各脚本要求**：

#### run_all.sh
```bash
#!/bin/bash
# 用法：bash run_all.sh [--smoke] [--skip-download] [--start-stage N] [--gpus 0,1,2,3,4,5,6,7]
#
# 流程：
# 1. 环境检查（Python/CUDA/ms-swift 版本）
# 2. 数据下载（调用 download_data.sh）
# 3. 数据处理（调用 prepare_data.py）
# 4. 数据统计（调用 data_stats.py）
# 5. 阶段 1-5 训练（依次执行，自动传递 checkpoint）
# 6. 评测（调用 eval.sh）
# 7. 输出总结报告
#
# 特性：
# - 每阶段完成后自动检查 loss 是否收敛
# - 失败自动暂停并保存断点
# - 支持从指定阶段恢复
# - --smoke 模式自动切换为小配置
```

#### run_smoke_test.sh
```bash
#!/bin/bash
# 用法：bash run_smoke_test.sh [--gpus 0] [--stage N]
#
# Smoke 模式特点：
# - 使用 Qwen3-0.5B + CLIP-ViT-B/32 + Whisper-small
# - 每阶段仅训 10-20 steps（max_steps 控制）
# - 数据各取 100-1000 条子集
# - 单卡即可运行
# - 预计总时长 < 30 分钟
#
# 检查项：
# - 每阶段 loss 是否下降
# - 模型是否可正常推理
# - checkpoint 是否正常保存/加载
```

#### eval.sh
```bash
#!/bin/bash
# 用法：bash eval.sh --model-path PATH [--stage N] [--mode smoke|full]
#
# 评测覆盖：
# 1. 文本能力：MMLU, C-Eval, GSM8K
# 2. 图像理解：MMBench, TextVQA, MMMU
# 3. 音频理解：AISHELL-1 (WER/CER), LibriSpeech (WER)
# 4. 语音生成：MOS（人工/UTMOS）, 合成 WER
# 5. 视频理解：Video-MME（如有）
# 6. 多模态综合：MMBench-Audio, SEEDBench
#
# 输出：
# - 评测结果 JSON
# - 评测结果对比表（Markdown）
# - 各项指标与 baseline 对比
```

**质量检查**：
- [ ] 所有脚本 `bash -n` 语法检查通过
- [ ] `--smoke` 参数在所有脚本中一致
- [ ] 脚本间 checkpoint 路径传递正确
- [ ] 错误处理完善（`set -euo pipefail`）

---

### 步骤 6：辅助文档

**需要创建的文件**：

#### 6.1 `Omni/train/README.md` — 训练代码使用指南

```markdown
# Omni 训练代码

## 快速开始
## 目录结构与文件说明
## 环境搭建
## 数据准备
## 训练流程
## 评测
## Smoke Test
## FAQ
## 代码导读索引（文件→类→函数 快速索引表）
```

要求：≥ 300 行，覆盖所有文件的用途说明。

#### 6.2 `Omni/train/ARCHITECTURE.md` — 架构与代码解读

```markdown
# Omni 模型架构文档

## 1. 整体架构图（ASCII art）
## 2. 数据流图（每种模态的前向传播路径）
## 3. 各组件详解
### 3.1 OmniModel 类结构
### 3.2 视觉投影层（MLPProjector）
### 3.3 音频投影层（ConvGMLPProjector）
### 3.4 语音解码头（SpeechDecoder + CTC）
### 3.5 CosyVoice2 集成方案
## 4. 五阶段冻结策略图解
## 5. 类/函数/文件索引表
```

要求：≥ 400 行，含 ASCII 架构图和冻结策略图解。

#### 6.3 `Omni/README.md` — 项目总入口

```markdown
# Omni — 全模态大模型训练项目

## 项目概述
## 目录索引（所有子目录与文件的快速导航）
## 快速开始（3 条命令跑通 smoke test）
## 文档导航
  - 论文对比 → docs/training_comparison.md
  - 重点论文 → docs/papers/
  - 仓库分析 → docs/repos/
  - 训练方案 → scheme/training_scheme.md
  - 训练代码 → train/README.md
  - 架构说明 → train/ARCHITECTURE.md
  - MS-Swift 方案 → swift_train/README.md
## 参考资料
```

要求：≥ 150 行，覆盖所有子目录的索引。

**质量检查**：
- [ ] 所有文件路径引用正确
- [ ] 命令示例可执行
- [ ] 与实际代码文件一致（无虚引用）

---

### 步骤 7：验证

**验证清单**：

#### 7.1 语法检查
```bash
# Python 语法
python -m py_compile Omni/train/data/download_data.sh  # N/A
python -m py_compile Omni/train/data/prepare_data.py
python -m py_compile Omni/train/data/data_stats.py
python -m py_compile Omni/train/data/dataset.py
python -m py_compile Omni/train/model/*.py
python -m py_compile Omni/train/metrics/*.py

# Shell 语法
bash -n Omni/train/run_all.sh
bash -n Omni/train/run_smoke_test.sh
bash -n Omni/train/eval.sh
bash -n Omni/train/run_stage.sh
bash -n Omni/train/data/download_data.sh

# YAML 格式
python -c "import yaml; yaml.safe_load(open('Omni/train/configs/stage1_audio_align.yaml'))"
# ... 对所有 YAML 文件
```

#### 7.2 导入检查
```bash
# 检查所有 Python 文件的 import 是否可解析（不需要实际安装所有依赖）
python -c "import ast; ast.parse(open('Omni/train/data/dataset.py').read())"
```

#### 7.3 组件 dry-run
```python
# 模型实例化（smoke 配置）
from Omni.train.model import OmniModelConfig
config = OmniModelConfig()  # 验证 config 可创建

# Dataset 可实例化（mock 数据）
# 训练监控可初始化
# 评测指标可计算（dummy 数据）
```

#### 7.4 文档交叉引用检查
- [ ] `Omni/README.md` 中引用的所有文件都存在
- [ ] `train/README.md` 中引用的所有文件都存在
- [ ] `train/ARCHITECTURE.md` 中引用的类/函数在代码中存在
- [ ] `scheme/training_scheme.md` 中的参数与 YAML 配置一致
- [ ] 各阶段冻结策略在 config.py、YAML、ARCHITECTURE.md 三处一致

---

### 步骤 8：更新总入口与收尾

- 更新 `Omni/OMNI_PROJECT_PROMPT.md` 标记所有项完成
- 确认所有 `❌` 变为 `✅`
- Git add 新文件
- 输出完成报告

---

## 六、执行 Prompt 片段

以下 Prompt 可直接复制给 Agent 执行对应步骤：

### Prompt A：做仓库报告（步骤 1）

> 请为 `reference/` 下的 4 个仓库（bc-omni、Baichuan-Omni-1.5、mini-omni2、OpenOmni）各写一份详细分析报告。
>
> **输出位置**：`Omni/docs/repos/{bc_omni.md, baichuan_omni_1.5.md, mini_omni2.md, openomni.md}`
>
> **报告模板**：按 `Omni/OMNI_PROJECT_PROMPT.md` 步骤 1 的 9 章节模板撰写。
>
> **要求**：
> - 每份 ≥ 200 行
> - 代码路径引用 `reference/` 下的真实路径
> - 可复现性评分 1-5 分并说明理由
> - 重点分析训练脚本、数据格式、阶段划分
> - 4 份报告可并行分析

### Prompt B：做训练方案（步骤 2）

> 请整合 `Doc/草稿/08_Omni模型训练方案.md`（3168 行）与 `Omni/OMNI_PROJECT_PROMPT.md` 第四章的模型规划，输出 `Omni/scheme/training_scheme.md`。
>
> **结构**：按 `Omni/OMNI_PROJECT_PROMPT.md` 步骤 2 的文档结构撰写。
>
> **要求**：
> - ≥ 500 行
> - 包含 CosyVoice2 vs CosyVoice1 详细对比
> - 每个模块选型需有对比论证（≥ 3 个候选项）
> - 五阶段每阶段的正式+smoke 配置表
> - 与第四章 4.4 节参数保持严格一致

### Prompt C：做数据模块（步骤 3）

> 请在 `Omni/train/data/` 下创建 4 个文件：`download_data.sh`、`prepare_data.py`、`data_stats.py`、`dataset.py`。
>
> **详细规格**：按 `Omni/OMNI_PROJECT_PROMPT.md` 步骤 3 的各文件要求。
>
> **关键约束**：
> - `download_data.sh` 的 `--smoke` 模式 5 分钟内可完成
> - `data_stats.py` 输出 ≥ 5 类统计图表（matplotlib）
> - `dataset.py` 含 5 个 Dataset 类，支持 DistributedSampler
> - 参考 `Omni/swift_train/data/` 已有代码的数据格式
> - 参考 `Omni/train/model/config.py` 的模型维度配置

### Prompt D：做训练配置（步骤 4）

> 请在 `Omni/train/configs/` 下创建 10 个 YAML 配置文件（5 正式 + 5 smoke）。
>
> **详细规格**：按 `Omni/OMNI_PROJECT_PROMPT.md` 步骤 4 的 YAML 模板。
>
> **关键约束**：
> - 模型名与 `Omni/train/model/config.py` 中一致
> - 冻结策略与第四章 4.4 节一致
> - Smoke 配置使用 Qwen3-0.5B + CLIP-ViT-B/32 + Whisper-small
> - 正式配置使用 Qwen3-8B + SigLIP-SO400M + Whisper-large-v3
> - 参考 `Omni/swift_train/scripts/run_stage*.sh` 中的参数

### Prompt E：做执行脚本（步骤 5）

> 请在 `Omni/train/` 根目录创建 4 个脚本：`run_all.sh`、`run_smoke_test.sh`、`eval.sh`、`run_stage.sh`。
>
> **详细规格**：按 `Omni/OMNI_PROJECT_PROMPT.md` 步骤 5 的各脚本要求。
>
> **关键约束**：
> - 调用步骤 3 的数据脚本和步骤 4 的配置
> - 阶段间自动传递 checkpoint 路径
> - `--smoke` 模式全流程 < 30 分钟
> - 错误处理完善

### Prompt F：做辅助文档（步骤 6）

> 请创建 3 个文档：`Omni/README.md`、`Omni/train/README.md`、`Omni/train/ARCHITECTURE.md`。
>
> **详细规格**：按 `Omni/OMNI_PROJECT_PROMPT.md` 步骤 6 的各文档结构。
>
> **关键约束**：
> - 所有文件路径引用必须是实际存在的文件
> - 代码导读索引覆盖 train/ 下所有 .py 文件
> - 架构图使用 ASCII art 而非图片引用
> - 与实际代码保持同步

### Prompt G：做验证（步骤 7）

> 请对 `Omni/train/` 执行完整验证：
>
> 1. **语法检查**：所有 .py 文件 `py_compile`，所有 .sh 文件 `bash -n`，所有 .yaml 文件 YAML 解析
> 2. **导入检查**：所有 .py 文件 AST 解析
> 3. **组件 dry-run**：config 实例化、model 组件验证、metrics 计算验证
> 4. **文档交叉引用**：检查 README/ARCHITECTURE/scheme 中所有引用路径
>
> **输出**：验证报告，标明 PASS/FAIL 和问题清单。

---

## 七、最终产出物清单

完成后 `Omni/` 目录应有以下结构：

```
Omni/
├── OMNI_PROJECT_PROMPT.md              # 本文件（计划与需求）
├── README.md                           # [步骤 8] 总入口索引
│
├── docs/
│   ├── training_comparison.md          # [已有] 11 篇论文对比
│   ├── papers/                         # [已有] 5 篇重点论文报告
│   │   ├── baichuan_omni.md
│   │   ├── mini_omni2.md
│   │   ├── omnivinci.md
│   │   ├── omnigaia.md
│   │   └── openomni.md
│   ├── repos/                          # [步骤 1] 4 个仓库分析
│   │   ├── bc_omni.md
│   │   ├── baichuan_omni_1.5.md
│   │   ├── mini_omni2.md
│   │   └── openomni.md
│   └── *.svg                           # [已有] 架构图
│
├── scheme/
│   └── training_scheme.md              # [步骤 2] 训练方案
│
├── train/
│   ├── README.md                       # [步骤 6] 代码使用指南
│   ├── ARCHITECTURE.md                 # [步骤 6] 架构文档
│   ├── run_all.sh                      # [步骤 5] 一键全流程
│   ├── run_smoke_test.sh               # [步骤 5] 一键 smoke
│   ├── run_stage.sh                    # [步骤 5] 单阶段入口
│   ├── eval.sh                         # [步骤 5] 评测脚本
│   ├── setup_env.sh                    # [已有] 环境安装
│   ├── requirements.txt                # [已有] 依赖
│   │
│   ├── model/                          # [已有] 模型定义
│   │   ├── __init__.py
│   │   ├── config.py                   # 配置与阶段定义
│   │   ├── omni_model.py               # OmniModel 主类
│   │   ├── projectors.py               # 投影层
│   │   └── speech_decoder.py           # 语音解码头
│   │
│   ├── data/                           # [步骤 3] 数据模块
│   │   ├── download_data.sh            # 数据下载
│   │   ├── prepare_data.py             # 数据预处理
│   │   ├── data_stats.py               # 数据统计分析
│   │   └── dataset.py                  # Dataset 封装
│   │
│   ├── configs/                        # [步骤 4] 训练配置
│   │   ├── stage1_audio_align.yaml
│   │   ├── stage2_vision_align.yaml
│   │   ├── stage3_multimodal_sft.yaml
│   │   ├── stage4_speech_gen.yaml
│   │   ├── stage5_dpo.yaml
│   │   ├── smoke_stage1.yaml
│   │   ├── smoke_stage2.yaml
│   │   ├── smoke_stage3.yaml
│   │   ├── smoke_stage4.yaml
│   │   ├── smoke_stage5.yaml
│   │   └── deepspeed_zero2.json        # [已有]
│   │
│   └── metrics/                        # [已有] 监控与评测
│       ├── __init__.py
│       ├── eval_metrics.py
│       ├── training_monitor.py
│       └── visualize.py
│
└── swift_train/                        # [已有] MS-Swift 脚本方案
    ├── README.md
    ├── PAPERS_SUMMARY.md
    ├── configs/
    ├── data/
    ├── eval/
    ├── scripts/
    ├── requirements.txt
    └── setup_env.sh
```

**文件计数**：新增 ~22 个文件（4 报告 + 1 方案 + 4 数据 + 10 配置 + 4 脚本 + 3 文档），已有 ~30 个文件。

---

*文档版本：v2，2026-03-02 更新。*

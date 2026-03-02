# Omni 全模态模型训练方案

> 版本：v1.0 | 日期：2026-03-02
>
> 本文档基于 11 篇精读论文（Baichuan-Omni、OpenOmni、Qwen2.5-Omni、LLaMA-Omni/Omni2、Mini-Omni2、VITA、OmniVinci、InternOmni、AnyGPT、Ola）的训练策略，结合项目已有模型代码（`Omni/train/model/`）和 MS-Swift 方案（`Omni/swift_train/`），整合为一份完整、可落地的训练方案。

---

## 1. 方案目标

### 1.1 最终目标

构建一个全模态（文本 + 图像 + 音频 + 视频）理解与语音生成模型，支持：

- **输入**：文本、图像、音频、视频（含音轨）的任意组合
- **输出**：文本回复 + 流式语音合成
- **交互**：低延迟语音对话，支持打断

### 1.2 基座模型

| 用途 | 模型 | 说明 |
|------|------|------|
| 正式训练 | Qwen3-8B | 8B 参数，中文能力优秀，ms-swift 原生支持 |
| Smoke Test | Qwen3-0.5B | 0.5B 参数，单卡可跑，验证流程 |

### 1.3 目标能力清单

| 能力 | 说明 | 对应训练阶段 |
|------|------|-------------|
| 语音识别（ASR） | 中英文语音转文本 | Stage 1 |
| 图像理解 | 图片描述、VQA、OCR | Stage 2 |
| 多模态对话 | 图文/音文/视频混合输入的多轮对话 | Stage 3 |
| 语音合成（TTS） | 文本/多模态输入生成自然语音 | Stage 4 |
| 偏好对齐 | 减少幻觉、提升回复质量 | Stage 5 |
| 视频理解 | 短/长视频内容理解与推理 | Stage 3 |

---

## 2. 模型选型与对比

### 2.1 LLM 骨干选型

| 候选 | 参数量 | 优势 | 劣势 | 结论 |
|------|--------|------|------|------|
| **Qwen3-8B** | 8B | 中文能力业界领先；开源协议宽松（Apache-2.0）；ms-swift 原生支持 SFT/LoRA/DPO/GRPO；Qwen2.5-Omni 已验证全模态可行性 | 英文能力略逊于 LLaMA3 | **选用** |
| LLaMA3-8B-Instruct | 8B | 英文能力强；社区生态丰富；LLaMA-Omni/Omni2 已验证语音对话可行性 | 中文能力不如 Qwen；开源协议限制商业使用 | 备选 |
| Baichuan2-7B | 7B | 中文能力好；Baichuan-Omni 已验证全模态架构 | 社区活跃度下降；ms-swift 支持不如 Qwen 完善 | 不选 |

**选择 Qwen3-8B 的理由**：

1. **中文能力**：Qwen3 系列在 C-Eval、CMMLU 等中文评测上持续领先，对中文为主的项目至关重要。
2. **开源协议**：Apache-2.0，无商业限制。
3. **工具生态**：ms-swift 对 Qwen 系列有原生最优支持，涵盖 CPT/SFT/LoRA/DPO/GRPO 全流程。
4. **全模态验证**：Qwen2.5-Omni 已证明 Qwen 系列 LLM 可作为全模态 Thinker 骨干，训练策略可直接参考。

### 2.2 视觉编码器选型

| 候选 | 分辨率 | 输出维度 | 参数量 | 优势 | 劣势 | 结论 |
|------|--------|----------|--------|------|------|------|
| **SigLIP-SO400M-384** | 384x384 | 1152 | 400M | Baichuan-Omni / Ola / Qwen2.5-Omni 均采用 SigLIP 系列；Sigmoid 损失优于 Softmax；无需负样本 | 仅支持 384 分辨率（需动态裁剪处理高分辨率） | **选用** |
| CLIP-ViT-L/14-336 | 336x336 | 1024 | 304M | OpenAI 原版，社区支持广；Mini-Omni2 等采用 | 性能不如 SigLIP；对比学习依赖大 batch 负样本 | 备选（smoke） |
| EVA-CLIP-ViT-E/14 | 224x224 | 1024 | 4.4B | 参数量大，理论上限高 | 参数量过大，推理慢；较少 Omni 论文采用 | 不选 |
| InternViT-6B | 448x448 | 3200 | 6B | InternOmni 采用；高分辨率；输出维度大 | 6B 参数太重，显存压力大；与 Qwen 生态不匹配 | 不选 |

**选择 SigLIP-SO400M 的理由**：

1. **论文验证**：Baichuan-Omni（SigLIP-400M）、Ola（OryxViT 基于 SigLIP-400M 初始化）、Qwen2.5-Omni（SigLIP2-So400m）三篇核心论文均采用 SigLIP 系列，可直接复用其训练策略。
2. **效率**：400M 参数在效果与效率间平衡良好，适合 8B LLM 骨干。
3. **Smoke Test 适配**：降级为 CLIP-ViT-B/32（86M），维度从 1152 降至 768，可在单卡上快速验证。

### 2.3 音频编码器选型

| 候选 | 参数量 | 采样率 | 优势 | 劣势 | 结论 |
|------|--------|--------|------|------|------|
| **Whisper-large-v3** | 1.5B | 16kHz | 多语言 ASR 标杆（99 种语言）；Baichuan-Omni / Qwen2.5-Omni / Ola 均采用；输出维度 1280 | 参数量大；推理慢 | **选用** |
| HuBERT-Large | 300M | 16kHz | 自监督预训练；语音表征质量高 | 不擅长多语言；需额外 ASR 头；Omni 论文中较少采用 | 不选 |
| WavLM-Large | 300M | 16kHz | 噪声环境鲁棒性好 | 同 HuBERT，Omni 生态中较少使用 | 不选 |
| Qwen-Audio | 未公开 | 16kHz | Qwen 生态原生 | 非独立开源；无法单独替换 | 不选 |

**选择 Whisper-large-v3 的理由**：

1. **多语言能力**：覆盖 99 种语言，天然支持中英文 ASR。
2. **论文共识**：Baichuan-Omni、Qwen2.5-Omni、OpenOmni、Ola 等主流方案均以 Whisper 系列为音频编码器。
3. **Smoke Test 适配**：降级为 Whisper-small（244M），维度从 1280 降至 768，适合单卡验证。

### 2.4 语音合成选型

| 候选 | 特点 | 优势 | 劣势 | 结论 |
|------|------|------|------|------|
| **CosyVoice2** | Flow-matching + 声码器 | 流式合成（chunk-aware）；5 秒零样本克隆；中英日韩粤多语种；指令可控（情感/语速）；24kHz 高音质；Apache-2.0 开源 | 需额外部署声码器 | **选用** |
| VITS2 | 端到端 TTS | 单阶段合成简单；推理快 | 零样本克隆能力弱；不支持流式 | 不选 |
| SoundStorm | 并行解码 | 生成速度快 | 需 SoundStream tokenizer；开源实现不完整 | 不选 |
| MegaTTS2 | 扩散模型 TTS | 自然度高 | 延迟较大；不支持流式；开源支持不足 | 不选 |

**重点说明选 CosyVoice2 的理由**：

CosyVoice2 是本方案语音合成的核心选择，原因如下：

| 特性 | CosyVoice1 | **CosyVoice2** | 对本方案的价值 |
|------|-----------|-----------|--------------|
| 流式合成 | 不支持 | **支持（chunk-aware causal）** | 实时语音对话的硬性需求 |
| 零样本克隆 | 需 > 10s 参考音频 | **仅需 5s** | 更实用的声音克隆 |
| 多语言 | 中英 | **中英日韩粤** | 覆盖东亚主要语种 |
| 指令可控 | 不支持 | **支持（情感/语速/风格）** | 丰富的表达能力 |
| 采样率 | 22.05kHz | **24kHz** | 更高音质 |
| 开源 | Apache-2.0 | **Apache-2.0** | 无商业限制 |

此外，LLaMA-Omni2 已验证 CosyVoice2 可与 Qwen 系列 LLM 配合进行端到端语音对话训练，证明了技术路线的可行性。

### 2.5 投影层设计选型

投影层负责将编码器输出映射到 LLM 嵌入空间，不同阶段采用不同设计。

| 方案 | 结构 | 压缩率 | 优势 | 劣势 | 本方案采用阶段 |
|------|------|--------|------|------|---------------|
| **2 层 MLP + GELU** | Linear→GELU→Linear | 1x（仅对齐） | 最简单、训练稳定；LLaVA / Baichuan-Omni / Ola 验证有效 | 无压缩，序列长 | 视觉投影（Stage 1-2） |
| Q-Former | 可学习 query + 交叉注意力 | 可变（如 32x） | BLIP-2 原创；高压缩 | 参数多（~80M）；训练复杂；收敛慢 | 不采用 |
| Perceiver Resampler | 可学习 latent + 交叉注意力 | 可变 | 灵活控制输出 token 数 | 实现复杂 | 不采用（3D-Resampler 留作扩展） |
| **Conv1D + gMLP** | Conv1D 4x 降采样 + 门控 MLP | 4x | Baichuan-Omni 验证：8x 压缩仅降 0.3% ASR WER；显著减少音频序列长度 | 比纯 MLP 稍复杂 | 音频投影（Stage 1-2） |

**选择策略**：

- **视觉投影**：2 层 MLP + GELU（1152→4096→4096），参考 LLaVA 和 Baichuan-Omni，简单有效。
- **音频投影**：Conv1D 4x 降采样 + gMLP（1280→4096），参考 Baichuan-Omni Conv-GMLP 方案，在保留语音细节的同时将序列长度降为 1/4。

---

## 3. 五阶段训练计划

本方案采用**渐进式模态对齐**策略，参考 Baichuan-Omni（6 子阶段）、OpenOmni（5 阶段）、Ola（3 阶段渐进）的训练思路，结合项目已有代码（`Omni/train/model/config.py` 中的 `STAGE_CONFIGS`），设计 5 个训练阶段。

### Stage 1: 音频-语言对齐

| 项目 | 正式训练 | Smoke Test |
|------|----------|------------|
| **目标** | 将 Whisper 音频特征映射到 LLM 嵌入空间，使模型获得基本 ASR 能力 | 验证流程可跑通 |
| **冻结模块** | LLM (Qwen3-8B) + 音频编码器 (Whisper-large-v3) + 视觉编码器 + 语音解码头 | LLM (Qwen3-0.5B) + Whisper-small + CLIP-ViT-B/32 |
| **可训练模块** | 音频投影层 (Conv-gMLP, ~8M params) | 音频投影层 (~1M params) |
| **损失函数** | Language Modeling Loss (next-token prediction) | 同左 |
| **学习率** | 1e-3, cosine decay | 同左 |
| **Warmup** | 500 steps | 100 steps |
| **权重衰减** | 0.01 | 同左 |
| **梯度裁剪** | 1.0 | 同左 |
| **数据来源** | LibriSpeech (960h) + AISHELL-1 (178h) + WenetSpeech 子集 (2000h) | LibriSpeech-clean-100 (100h) |
| **数据量** | ~28 万条音频-转录对 | ~3000 条 |
| **数据格式** | `{audio_path, transcript}` → Swift JSONL | 同左 |
| **Batch size** | 32/GPU x 4 acc = 128 有效 batch | 4/GPU |
| **Epochs** | 1 | 1 |
| **硬件** | 8 x A100-80G | 1 x GPU (>=16G) |
| **预计时长** | ~2 天 | ~30 分钟 |
| **验收指标** | 投影 loss < 0.5; LibriSpeech dev-clean WER 可接受 | loss 下降趋势正常 |

**超参来源**：投影层学习率 1e-3 参考 Baichuan-Omni Stage I、OpenOmni Stage 1 和 LLaVA Pretrain 阶段的设置；batch size 参考 InternOmni Stage 1（64 GPU）按比例缩放。

### Stage 2: 图像-语言预训练

| 项目 | 正式训练 | Smoke Test |
|------|----------|------------|
| **目标** | 将 SigLIP 视觉特征映射到 LLM 嵌入空间，使模型获得基本图像理解能力 | 验证流程可跑通 |
| **冻结模块** | LLM + 视觉编码器 (SigLIP) + 音频编码器 + 音频投影（保持 Stage 1 参数）+ 语音解码头 | 同左（对应 smoke 模型） |
| **可训练模块** | 视觉投影层 (MLP, ~4M params) | 视觉投影层 (~0.5M params) |
| **损失函数** | Language Modeling Loss | 同左 |
| **学习率** | 1e-3, cosine decay | 同左 |
| **Warmup** | 500 steps | 100 steps |
| **数据来源** | LLaVA-Pretrain-595K (CC3M 子集) + ShareGPT4V-100K | LLaVA-Pretrain 1 万条子集 |
| **数据量** | ~70 万条图文对 | ~1 万条 |
| **数据格式** | `{image_path, conversations}` → Swift JSONL | 同左 |
| **Batch size** | 32/GPU x 4 acc = 128 有效 batch | 4/GPU |
| **Epochs** | 1 | 1 |
| **硬件** | 8 x A100-80G | 1 x GPU (>=16G) |
| **预计时长** | ~1 天 | ~30 分钟 |
| **验收指标** | 投影 loss < 0.5; 简单图像描述能力 | loss 下降趋势正常 |

**超参来源**：学习率 1e-3 参考 LLaVA Pretrain 阶段和 Baichuan-Omni Stage I 图像分支。数据选用 LLaVA-Pretrain-595K（HuggingFace: `liuhaotian/LLaVA-Pretrain`）为业界标准图文对齐数据集。

### Stage 3: 多模态联合 SFT

| 项目 | 正式训练 | Smoke Test |
|------|----------|------------|
| **目标** | 联合训练让模型具备多模态理解 + 指令跟随 + 对话能力 | 验证流程可跑通 |
| **冻结模块** | 视觉编码器 (SigLIP) + 音频编码器 (Whisper) | 同左 |
| **可训练模块** | LLM (全参或 LoRA r=64) + 视觉投影 + 音频投影 | LLM (LoRA r=8) + 投影层 |
| **损失函数** | Language Modeling Loss（多模态指令跟随） | 同左 |
| **学习率** | 2e-5 (全参) 或 1e-4 (LoRA), cosine decay | 1e-4 (LoRA) |
| **Warmup** | 1000 steps | 50 steps |
| **数据来源** | 见下方数据混合表 | 各类各 1000 条 |
| **数据量** | ~35 万条混合 | ~4000 条 |
| **数据混合比** | 图文 40% : 音文 25% : 纯文本 20% : 视频 15% | 同左 |
| **Batch size** | 8/GPU x 8 acc = 64 有效 batch | 2/GPU |
| **Epochs** | 3 | 1 |
| **硬件** | 8 x A100-80G | 1 x GPU (>=16G) |
| **预计时长** | ~3 天 | ~1 小时 |
| **验收指标** | MMBench >= 60; LibriSpeech WER 不退化; 多模态对话质量 | loss 下降；基本对话能力 |

**Stage 3 数据混合详情**：

| 类别 | 比例 | 数据集 | 规模 | HuggingFace ID / 来源 |
|------|------|--------|------|----------------------|
| 图文 SFT | 40% | LLaVA-Instruct-150K | 150K | `liuhaotian/LLaVA-Instruct-150k` |
| | | ShareGPT4V-SFT | 100K | `Lin-Chen/ShareGPT4V` |
| 音文 SFT | 25% | AudioCaps | 50K | `audiocaps` |
| | | AISHELL-SFT（自建） | 40K | AISHELL-1 转 SFT 格式 |
| 纯文本 | 20% | ShareGPT | 50K | `anon8231489123/ShareGPT_Vicuna_unfiltered` |
| | | Alpaca-GPT4 | 20K | `vicgalle/alpaca-gpt4` |
| 视频 | 15% | VideoChat2-IT | 50K | `OpenGVLab/VideoChat2-IT` |
| | | Video-ChatGPT | 30K | `MBZUAI/VideoInstruct-100K` |

**超参来源**：全参学习率 2e-5 参考 Baichuan-Omni SFT 阶段、LLaVA Fine-tuning 阶段和 OpenOmni Stage 3 的设置。数据混合比参考 Baichuan-Omni 600K SFT 数据的模态分布。

### Stage 4: 语音生成训练

| 项目 | 正式训练 | Smoke Test |
|------|----------|------------|
| **目标** | 训练语音解码头，使模型能输出语音 token，接入 CosyVoice2 生成语音 | 验证 CTC 解码头可训练 |
| **冻结模块** | 视觉编码器 + 音频编码器 + 视觉投影 | 同左 |
| **可训练模块** | 语音解码头 (Transformer Decoder + CTC, ~50M) + 音频投影 + LLM 最后 2 层 | 语音解码头 (~4M) |
| **损失函数** | CTC Loss + LM Loss (辅助, 权重 0.3) | 同左 |
| **学习率** | 5e-4, cosine decay | 同左 |
| **Warmup** | 500 steps | 50 steps |
| **数据来源** | LibriTTS (585h) + AISHELL-3 (85h) + CosyVoice2 合成数据 | LibriTTS 1000 条子集 |
| **数据量** | ~10 万条 text-speech 对 | ~1000 条 |
| **数据格式** | `{text, speech_tokens}` → 自定义 Dataset | 同左 |
| **Batch size** | 16/GPU x 4 acc = 64 有效 batch | 4/GPU |
| **Epochs** | 2 | 1 |
| **硬件** | 8 x A100-80G | 1 x GPU (>=16G) |
| **预计时长** | ~2 天 | ~30 分钟 |
| **训练方式** | 自定义 trainer（ms-swift 暂不支持 CTC 训练） | 同左 |
| **验收指标** | CTC loss 收敛; 合成语音可听; WER 可接受 | loss 下降趋势 |

**超参来源**：CTC 训练学习率 5e-4 参考 LLaMA-Omni 的 CTC 语音解码器训练设置和 OpenOmni Stage 4 的 CTC 训练配置。LibriTTS（HuggingFace: `openslr/libri-light`）和 AISHELL-3 为标准 TTS 训练数据集。

### Stage 5: 偏好优化 (DPO/GRPO)

| 项目 | 正式训练 | Smoke Test |
|------|----------|------------|
| **目标** | 对齐人类偏好，提升回复质量，减少幻觉 | 验证 DPO 流程 |
| **冻结模块** | 视觉编码器 + 音频编码器 | 同左 |
| **可训练模块** | LLM (LoRA r=16) + 视觉投影 + 音频投影 + 语音解码头 | LLM (LoRA r=4) + 语音解码头 |
| **损失函数** | DPO Loss (beta=0.1) 或 GRPO Loss | DPO Loss |
| **学习率** | 5e-6, cosine decay | 同左 |
| **Warmup** | 200 steps | 20 steps |
| **数据来源** | UltraFeedback + 自建多模态偏好数据 | 500 条子集 |
| **数据量** | ~1 万条偏好对 (chosen + rejected) | ~500 条 |
| **数据格式** | `{prompt, chosen, rejected}` → Swift DPO JSONL | 同左 |
| **Batch size** | 4/GPU x 16 acc = 64 有效 batch | 2/GPU |
| **Epochs** | 1 | 1 |
| **硬件** | 8 x A100-80G | 1 x GPU (>=16G) |
| **预计时长** | ~1 天 | ~20 分钟 |
| **训练方式** | `swift rlhf --rlhf_type dpo` | 同左 |
| **验收指标** | 偏好胜率 > 60%; 幻觉率下降; 回复质量提升 | DPO loss 下降 |

**超参来源**：DPO 学习率 5e-6 参考 OpenOmni Stage 5 的 CTC-DPO 设置和 Qwen2.5-Omni Talker DPO 阶段的配置。beta=0.1 为 DPO 标准参数。

---

## 4. 数据规划

### 4.1 各阶段数据源

| 阶段 | 数据集 | 规模 | 格式 | HuggingFace ID / 来源 | 许可证 |
|------|--------|------|------|-----------------------|--------|
| **Stage 1** | LibriSpeech | 960h, ~28万条 | FLAC 16kHz + 转录文本 | `openslr/librispeech_asr` | CC BY 4.0 |
| | AISHELL-1 | 178h, 400 说话人 | WAV 16kHz + 转录文本 | `openslr.org/33` | Apache-2.0 |
| | WenetSpeech 子集 | 2000h | WAV 16kHz + 转录文本 | `Wenetspeech/wenetspeech` | 开源 |
| **Stage 2** | LLaVA-Pretrain-595K | 595K 图文对 | JSON + 图像 | `liuhaotian/LLaVA-Pretrain` | CC BY 4.0 |
| | ShareGPT4V | 100K 图文对话 | JSON + 图像 | `Lin-Chen/ShareGPT4V` | Apache-2.0 |
| **Stage 3** | LLaVA-Instruct-150K | 150K 图文指令 | JSON 多轮对话 | `liuhaotian/LLaVA-Instruct-150k` | CC BY 4.0 |
| | AudioCaps | 50K 音频描述 | WAV + 文本 | `audiocaps` | MIT |
| | VideoChat2-IT | 50K 视频指令 | JSON + 视频帧 | `OpenGVLab/VideoChat2-IT` | Apache-2.0 |
| | ShareGPT | 50K 纯文本对话 | JSON | `anon8231489123/ShareGPT_Vicuna_unfiltered` | CC BY 4.0 |
| | Alpaca-GPT4 | 20K 纯文本指令 | JSON | `vicgalle/alpaca-gpt4` | CC BY 4.0 |
| **Stage 4** | LibriTTS | 585h, 多说话人 | WAV 24kHz + 文本 | `openslr.org/60` | CC BY 4.0 |
| | AISHELL-3 | 85h, 218 说话人 | WAV 44.1kHz + 文本 | `openslr.org/93` | Apache-2.0 |
| **Stage 5** | UltraFeedback | 64K 偏好对 | JSON (chosen/rejected) | `openbmb/UltraFeedback` | MIT |

### 4.2 数据质量控制

**清洗标准**：

| 检查项 | 规则 | 工具 |
|--------|------|------|
| 空样本过滤 | 文本字段为空或长度 < 5 token 的样本丢弃 | 自定义脚本 |
| 文本长度截断 | 超过 max_seq_length (4096) 的文本截断 | tokenizer |
| 音频时长过滤 | 音频 < 0.5s 或 > 30s 的样本丢弃 | librosa |
| 图像分辨率过滤 | 图像短边 < 224 的样本丢弃 | PIL |
| 语言检测 | 非目标语言（中/英）的样本标记并低权重 | fastText lid |
| 毒性过滤 | 有害内容检测并丢弃 | 毒性分类器 |

**去重策略**：

- **文本去重**：MinHash LSH (Jaccard 阈值 0.8) 全局去重
- **图像去重**：感知哈希 (pHash) 去重
- **音频去重**：基于 Whisper 转录文本的文本去重

**质量过滤规则**：

- 图文对：CLIP 相似度 >= 0.25
- 音文对：ASR 置信度 >= 0.8（若有）
- SFT 数据：回复长度 >= 20 token，非模板化回复

### 4.3 Smoke Test 数据

| 阶段 | 数据量 | 来源 | 说明 |
|------|--------|------|------|
| Stage 1 | 3000 条 | LibriSpeech-clean-100 子集 | 取前 3000 条最短音频 |
| Stage 2 | 10000 条 | LLaVA-Pretrain 子集 | 随机采样 10K |
| Stage 3 | 4000 条 | 各数据集各 1000 条 | 图文/音文/纯文本/视频各 1K |
| Stage 4 | 1000 条 | LibriTTS 子集 | 取前 1000 条 |
| Stage 5 | 500 条 | UltraFeedback 子集 | 随机采样 500 |

---

## 5. 评测方案

### 5.1 评测集选用与理由

#### 图像理解

| 评测集 | 选用理由 | 指标 | 基线（7B 模型） |
|--------|----------|------|-----------------|
| **MMBench** | 覆盖 20+ 能力维度的综合图像理解评测，Baichuan-Omni / Qwen2.5-Omni / OmniVinci 等论文均报告该基准，便于横向对比；含中文版 MMBench-CN | Accuracy (%) | Qwen2.5-VL-7B: ~70 |
| **MMMU** | 专家级多学科多模态理解，需要领域知识 + 视觉理解；人类表现约 60-70%，是衡量模型深层理解能力的关键基准 | Accuracy (%) | Qwen2.5-VL-7B: ~45 |
| **TextVQA** | 专注 OCR 能力测试，需要从图像中读取文字并回答问题；可验证 Stage 2 视觉投影对文字识别的对齐效果 | Accuracy (%) | Baichuan-Omni: ~70 |

#### 音频理解

| 评测集 | 选用理由 | 指标 | 基线 |
|--------|----------|------|------|
| **AISHELL-1 test** | 中文 ASR 标准基准；178h 训练集匹配，可直接评估 Stage 1 音频对齐效果 | CER (%) | Whisper-large-v3: ~3.0 |
| **LibriSpeech test-clean** | 英文 ASR 标准基准；业界通用报告指标，所有 Omni 论文均报告 | WER (%) | Whisper-large-v3: ~2.0 |
| **MMAU** | 10K 音频、27 种技能的多任务音频理解；覆盖语音/环境音/音乐三大类，超越简单 ASR，测试模型对音频内容的深层理解 | Accuracy (%) | Qwen2-Audio: 52.5 |

#### 视频理解

| 评测集 | 选用理由 | 指标 | 基线 |
|--------|----------|------|------|
| **Video-MME** | CVPR 2025 视频理解基准；覆盖 11 秒到 1 小时视频，测试时序推理和长程依赖；Qwen2.5-Omni、OmniVinci 等均在此基准报告，是最权威的视频理解评测之一 | Accuracy (%) | Qwen2.5-Omni: 71.9 |

#### 语音生成

| 评测集 | 选用理由 | 指标 | 基线 |
|--------|----------|------|------|
| **自建测试集** | 覆盖中英文短句/长句/情感表达；通过 UTMOS（自动评分）和人工 MOS 评估；参考 Qwen2.5-Omni 的 seed-tts-eval 设计 | MOS (1-5); WER (%) | CosyVoice2: MOS ~4.0 |

#### 文本能力（防退化）

| 评测集 | 选用理由 | 指标 | 基线 |
|--------|----------|------|------|
| **MMLU** | 英文多任务理解标杆；5-shot 评测；用于验证多模态训练后文本能力不退化 | Accuracy (%) | Qwen3-8B: ~70 |
| **C-Eval** | 中文多学科评测；与 MMLU 互补，覆盖中文知识；Baichuan-Omni 特别强调中文能力 | Accuracy (%) | Qwen3-8B: ~75 |
| **GSM8K** | 数学推理基准；验证推理能力不因多模态训练退化 | Accuracy (%) | Qwen3-8B: ~80 |

#### 全模态综合

| 评测集 | 选用理由 | 指标 | 基线 |
|--------|----------|------|------|
| **OmniBench** | 视/听/文三模态理解基准；选择题形式，测试全模态感知能力 | Accuracy (%) | Qwen2.5-Omni: 58.4 |
| **DailyOmni** | 1197 条日常生活音视频 QA；测试音视频联合理解，OmniVinci 在此基准超越 Qwen2.5-Omni 19 分 | Accuracy (%) | OmniVinci: 72+ |

### 5.2 评测时机

| 里程碑 | 阶段 | 必测项 | 通过标准 |
|--------|------|--------|----------|
| M1 | Stage 1 结束 | LibriSpeech test-clean WER; AISHELL-1 CER | WER 不劣于 Whisper 基线太多（容许 +5%） |
| M2 | Stage 2 结束 | MMBench 子集抽检; TextVQA 子集 | 基本图像描述能力 |
| M3 | Stage 3 结束 | MMBench; MMLU; LibriSpeech WER; C-Eval | 全模态理解无崩溃；MMLU 下降 < 2% |
| M4 | Stage 4 结束 | 语音合成 MOS; 合成 WER | MOS >= 3.5; 可听 |
| M5 | Stage 5 结束 | **全清单评测** | 见上表基线 |

**全流程结束后的完整评测**：运行 5.1 节所有评测集，输出统一对比表格，与 Baichuan-Omni、Qwen2.5-Omni 等论文报告值对比。

---

## 6. 硬件与资源规划

| 配置 | GPU | 显存需求/卡 | 预计总时长 | 适用场景 |
|------|-----|------------|-----------|----------|
| **Smoke Test** | 1 x A100-80G 或 1 x RTX 4090-24G | 16-24 GB | ~3 小时（全 5 阶段） | 验证流程正确性，使用 Qwen3-0.5B + CLIP-ViT-B/32 + Whisper-small |
| **最小训练** | 4 x A100-80G | 60-70 GB/卡 | ~5 天（全 5 阶段） | 小规模验证效果，可使用 LoRA 降低显存 |
| **标准训练** | 8 x A100-80G | 65-75 GB/卡 | ~9 天（全 5 阶段） | 正式训练，全参 SFT 或 LoRA r=64 |

**各阶段资源估算（标准 8xA100 配置）**：

| 阶段 | 可训练参数量 | 显存/卡 | 预计时长 | GPU 小时 |
|------|-------------|---------|----------|----------|
| Stage 1 | ~8M | ~60 GB | ~2 天 | ~384 |
| Stage 2 | ~4M | ~60 GB | ~1 天 | ~192 |
| Stage 3 | ~8B (全参) 或 ~50M (LoRA) | ~75 GB | ~3 天 | ~576 |
| Stage 4 | ~50M + LLM 最后 2 层 | ~70 GB | ~2 天 | ~384 |
| Stage 5 | ~16M (LoRA) | ~65 GB | ~1 天 | ~192 |
| **总计** | | | **~9 天** | **~1728** |

**DeepSpeed 配置**：使用 ZeRO Stage 2（已有配置 `Omni/train/configs/deepspeed_zero2.json`），Stage 3 全参训练时可切换 ZeRO Stage 3 进一步降低显存。

**混合精度**：全程使用 BF16 混合精度训练。

---

## 7. 风险与应对

### 7.1 训练不稳定（梯度爆炸 / Loss Spike）

| 风险 | 表现 | 应对措施 |
|------|------|----------|
| 梯度爆炸 | loss 突然飙升至 NaN/Inf | 梯度裁剪 max_grad_norm=1.0（已在 `StageConfig` 中配置）；启用 BF16 混合精度；遇到 spike 时回滚到上一个 checkpoint 并降低学习率至 0.5x |
| Loss 震荡 | loss 不收敛或剧烈波动 | 增大 batch size（增加梯度累积步数）；延长 warmup 步数；检查数据是否有噪声样本 |
| CTC 训练不稳定 | Stage 4 CTC loss 不收敛 | 采用课程学习：先用短音频（<5s）训练，逐步增加长度；CTC blank token 初始化偏置设为正值 |

### 7.2 模态冲突（Catastrophic Forgetting）

| 风险 | 表现 | 应对措施 |
|------|------|----------|
| 文本能力退化 | MMLU/C-Eval 下降 > 3% | Stage 3 混合 20% 纯文本数据（已在数据混合中设定）；每 500 步检查 MMLU，若下降触发报警 |
| 单模态退化 | 加入新模态后原模态能力下降 | 采用渐进式模态引入（参考 Ola）：先图文→再视频→再音频→全模态；冻结已对齐的投影层 |
| 模态间干扰 | 不同模态的梯度方向冲突 | 参考 Baichuan-Omni 渐进多分支对齐：各模态分支独立对齐后再联合；可选视-听对比损失（OmniVinci OmniAlignNet）辅助对齐 |

### 7.3 资源不足时的降级方案

| 资源等级 | 调整措施 | 预期影响 |
|---------|----------|----------|
| **4 x A100 → 2 x A100** | Stage 3 改用 LoRA r=32；增大梯度累积步数 | 训练时间约 2x；效果略降但可接受 |
| **无 A100 → 4 x RTX 4090** | 全程 LoRA；Batch size 减半；ZeRO Stage 3 | 训练时间约 3x；显存勉强够用 |
| **单卡** | 使用 Smoke Test 配置（Qwen3-0.5B）；仅跑 Stage 1-3 验证；Stage 4-5 暂缓 | 仅验证流程，不追求效果 |

### 7.4 数据相关风险

| 风险 | 应对 |
|------|------|
| 数据集下载中断 | `download_data.sh` 支持断点续传；先下载 Smoke Test 子集验证流程 |
| 数据质量不均 | `data_stats.py` 输出统计报告；抽检低质量样本并过滤 |
| 数据许可问题 | 优先使用 CC BY / Apache-2.0 / MIT 许可的数据集；避免使用限制商业用途的数据 |

---

## 8. 与现有方案的关系

### 8.1 与 swift_train/ 方案的异同

`Omni/swift_train/` 是基于 MS-Swift 框架的训练脚本方案，与本文档的训练方案在以下方面保持一致，也有差异：

| 维度 | swift_train/ 方案 | 本训练方案（train/） | 关系 |
|------|-------------------|---------------------|------|
| 训练框架 | MS-Swift（`swift sft`/`swift rlhf`） | 自定义 Trainer + 可选 MS-Swift | 互补：swift_train 用于 SFT/DPO 标准流程；train/ 用于自定义 CTC 等 |
| 模型定义 | 依赖 MS-Swift 内置模型 | 自定义 `OmniModel`（`train/model/`） | train/ 更灵活，可自定义投影层和语音解码头 |
| 五阶段参数 | `scripts/run_stage{1-5}.sh` | `STAGE_CONFIGS` in `config.py` | 参数保持一致（lr/bs/freeze 策略对齐） |
| 数据处理 | `data/prepare_datasets.py` + `convert_to_swift.py` | `data/prepare_data.py` + `dataset.py` | swift_train 输出 Swift JSONL；train/ 输出自定义 Dataset |
| 评测 | `eval/eval_multimodal.py` + `eval_audio.py` | `metrics/eval_metrics.py` | 评测指标和基准对齐 |

**使用建议**：Stage 1-3 和 Stage 5 可使用 swift_train/ 方案（MS-Swift 原生支持）；Stage 4 语音生成因需 CTC loss 须使用 train/ 方案。

### 8.2 与 Doc/草稿/08_Omni 方案的继承关系

`Doc/草稿/08_Omni模型训练方案.md`（3168 行）是项目初期的详细训练方案草稿，本文档是其精炼和整合版本：

| 继承内容 | 说明 |
|---------|------|
| 数据集选型 | 沿用其数据集调研结论（LibriSpeech、WenetSpeech、LLaVA-Pretrain 等） |
| 评测集分析 | 沿用其对 MMBench、Video-MME、LibriSpeech 等评测集的详细分析 |
| 六阶段 → 五阶段 | 原方案为六阶段（含独立的编码器微调和长上下文阶段），本方案精简为五阶段，将编码器微调并入 Stage 2，长上下文作为 Stage 3 的扩展 |
| 突破方向 | 原方案的稀疏注意力、3D-Resampler、全双工等突破方向作为后续扩展保留 |
| 论文反思 | 第 5-6 章的 11 篇论文逐篇对比反思和完善建议已融入本方案的各选型理由和训练策略中 |

### 8.3 为什么最终选择这套方案

1. **论文共识**：五阶段渐进式训练（对齐→预训练→SFT→语音生成→偏好对齐）是 Baichuan-Omni、OpenOmni、Ola 等主流论文的共同选择，经过充分验证。
2. **可落地性**：所有数据集均可从 HuggingFace 获取；超参数参考论文实际设置；硬件需求合理（8xA100 标准配置）。
3. **灵活性**：通过 `OmniModelConfig.resolve(smoke=True/False)` 支持正式训练和 Smoke Test 无缝切换；五阶段独立，可从任意阶段开始/恢复。
4. **效率**：参考 OmniVinci（0.2T token 超越 1.2T 基线）的高效数据策略，优先使用高质量数据集；参考 LLaMA-Omni（4 GPU < 3 天）的轻量级训练范式。
5. **生态兼容**：与 MS-Swift 框架兼容（Stage 1-3, 5）；自定义训练器覆盖 CTC 等特殊需求（Stage 4）。

---

## 附录

### A. 论文方法对照表

| 论文 | 训练阶段 | 总数据规模 | GPU 配置 | 语音生成 | 关键创新 |
|------|----------|-----------|----------|---------|---------|
| Baichuan-Omni | 2 大阶段 (6 子阶段 + SFT) | SFT: 600K | 未公开 | 无 | Conv-GMLP 音频投影；渐进多分支对齐 |
| OpenOmni | 5 阶段 | ~1.6M + 8Kh 语音 | 8xA100 | CTC + CosyVoice | 隐式跨模态对齐；CTC-DPO 情感偏好 |
| Qwen2.5-Omni | Thinker 3+1 / Talker 3 | ~1.2T token | 未公开 | Talker 双轨 AR + DiT | Thinker-Talker；TMRoPE；滑窗 DiT |
| LLaMA-Omni | 1 阶段 | 200K | **4 GPU** / **<3 天** | 非自回归 CTC | 极低资源；CTC 语音解码 |
| LLaMA-Omni2 | 1 阶段 | 200K | 未公开 | CosyVoice2 AR | 多规模统一；CosyVoice2 流式 |
| Mini-Omni2 | 3 阶段 | limited | 未公开 | SNAC 7 层并行 | 0.5B 全模态；命令式打断 |
| OmniVinci | 联合训练 | 24M / 0.2T token | 未公开 | 外接 TTS | OmniAlignNet 对比对齐；0.2T 超越 1.2T |
| InternOmni | 2 阶段 | S1: 26M / S2: 1.9M | S1: 64GPU / S2: 32GPU | 无 | 仅训 MLP_audio；~45h 完成 |
| Ola | 3 阶段渐进 | relatively small + 324K | 未公开 | CosyVoice 句级流式 | 视频作跨模态桥梁；渐进式模态对齐 |

### B. 项目文件索引

| 文件路径 | 用途 |
|---------|------|
| `Omni/train/model/config.py` | 模型配置与五阶段 StageConfig 定义 |
| `Omni/train/model/omni_model.py` | OmniModel 主模型类 |
| `Omni/train/model/projectors.py` | 视觉/音频投影层实现 |
| `Omni/train/model/speech_decoder.py` | 语音解码头实现 |
| `Omni/train/metrics/training_monitor.py` | 训练过程监控 |
| `Omni/train/metrics/eval_metrics.py` | 评测指标计算 |
| `Omni/train/metrics/visualize.py` | 训练曲线可视化 |
| `Omni/swift_train/scripts/run_stage{1-5}.sh` | MS-Swift 各阶段训练脚本 |
| `Omni/swift_train/data/prepare_datasets.py` | 数据下载与准备 |
| `Omni/swift_train/eval/eval_multimodal.py` | 多模态评测 |
| `Omni/docs/training_comparison.md` | 11 篇论文训练对比 |
| `Omni/swift_train/PAPERS_SUMMARY.md` | 论文方法总结 |
| `Doc/草稿/08_Omni模型训练方案.md` | 原始详细训练方案草稿 (3168 行) |

---

*本文档整合自项目内 5 份核心文档和 11 篇论文的训练策略分析，所有选型均有论文依据和对比论证。如需进一步细化某个阶段的训练配置，请参考对应的 YAML 配置文件和训练脚本。*

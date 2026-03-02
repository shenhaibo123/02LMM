# Omni 模型架构设计说明

## 1. 架构总览

```
┌──────────────────────────────────────────────────────────────┐
│                      OmniModel                                │
│                                                               │
│  ┌─────────────┐    ┌──────────────┐                         │
│  │ SigLIP-SO400M│    │ Whisper-v3   │                         │
│  │ 视觉编码器   │    │ 音频编码器   │                         │
│  │ (384×384)    │    │ (30s, 1500帧)│                         │
│  └──────┬──────┘    └──────┬───────┘                         │
│         │                  │                                  │
│  ┌──────▼──────┐    ┌──────▼───────┐                         │
│  │ MLPProjector│    │Conv-gMLP Proj│                         │
│  │ (2层 MLP)   │    │ (4× 降采样)  │                         │
│  │ 1152→4096   │    │ 1280→4096    │                         │
│  └──────┬──────┘    └──────┬───────┘                         │
│         │                  │          ┌──────────────┐       │
│         └──────────┬───────┘          │ Text Embed   │       │
│                    │                  │ (Tokenizer)  │       │
│                    │                  └──────┬───────┘       │
│                    │                         │               │
│              ┌─────▼─────────────────────────▼─────┐         │
│              │         [视觉] + [音频] + [文本]      │         │
│              │              LLM Backbone             │         │
│              │           (Qwen3-8B, 4096d)          │         │
│              └───────────┬──────────────┬───────────┘         │
│                          │              │                     │
│                   ┌──────▼──────┐ ┌─────▼──────────┐         │
│                   │  LM Head    │ │ SpeechDecoder   │         │
│                   │ (文本输出)  │ │ (Transformer+CTC│         │
│                   │             │ │  4097 tokens)   │         │
│                   └─────────────┘ └────────────────┘         │
└──────────────────────────────────────────────────────────────┘
```

## 2. 各组件设计

### 2.1 视觉编码器: SigLIP-SO400M

**选型对比:**

| 候选 | 参数量 | 分辨率 | 输出维度 | 优势 | 劣势 |
|------|--------|--------|----------|------|------|
| **SigLIP-SO400M** | 400M | 384 | 1152 | Sigmoid 对比学习，无需 [CLS]，性能/效率平衡 | 非最新 |
| CLIP-ViT-L/14 | 300M | 336 | 1024 | 广泛使用，生态好 | Softmax 对比损失，batch 依赖 |
| EVA-CLIP | 1B | 448 | 1408 | 性能最强 | 参数量过大 |
| InternViT-6B | 6B | 448 | 3200 | 极高分辨率 | 资源消耗过高 |

**选择 SigLIP 的理由:**
- Sigmoid 对比损失，不依赖 batch 内负样本，训练更稳定
- 400M 参数在性能和效率间取得平衡
- LLaVA-1.6、Qwen2.5-Omni 等均验证了 SigLIP 的有效性
- 输出为 patch 序列（无 [CLS]），天然适合 LLM 嵌入拼接

### 2.2 音频编码器: Whisper-large-v3

**选型对比:**

| 候选 | 参数量 | 语言 | 输出帧数 | 优势 | 劣势 |
|------|--------|------|----------|------|------|
| **Whisper-large-v3** | 1.55B | 100+ | 1500 | 中英双语 ASR 最优，标准化 30s 窗口 | 参数量大 |
| HuBERT-large | 300M | 英文 | 可变 | 语音表征学习 | 缺乏中文 |
| WavLM-large | 300M | 英文 | 可变 | 鲁棒性强 | 缺乏中文 |
| Qwen-Audio | 300M | 多语 | 可变 | 阿里生态 | 独立使用不方便 |

**选择 Whisper 的理由:**
- 中英双语 ASR 性能最优（WER < 5% on LibriSpeech）
- 标准化 30s 窗口 → 1500 帧 Mel 特征，便于批处理
- OpenOmni、Baichuan-Omni 等均使用 Whisper，方法经过验证
- Encoder-only 使用（丢弃 Decoder），参数量可控

### 2.3 视觉投影层: MLPProjector

```
Input (B, S, 1152)
  → Linear(1152, 4096) → GELU → Dropout
  → Linear(4096, 4096) → LayerNorm
Output (B, S, 4096)
```

**设计理由:**
- 2 层 MLP + GELU 是 LLaVA 验证的经典设计
- 相比 Q-Former (需要额外 learnable queries)，MLP 更简洁
- LayerNorm 稳定特征分布，防止与 LLM 嵌入空间的 scale 不匹配
- 初始化: truncated normal (std=0.02)，训练初期更稳定

### 2.4 音频投影层: ConvGMLPProjector

```
Input (B, 1500, 1280)
  → Conv1d(1280, 4096, kernel=4, stride=4)  # 4× 时序降采样
  → LayerNorm
  → gMLP Block × 2                           # 空间门控
  → Linear(4096, 4096) → LayerNorm
Output (B, 375, 4096)
```

**gMLP Block 内部:**
```
Input → LayerNorm → Linear → GELU → SpatialGating → Linear → Dropout → Residual
                                         │
                              Split half features → Spatial Linear (seq×seq) → Multiply
```

**设计理由:**
- Whisper 1500 帧直接输入 LLM 序列过长（占用过多 context window）
- Conv1d stride=4 降采样至 375 帧，压缩 4 倍序列长度
- gMLP 的 Spatial Gating 机制捕捉帧间时序依赖
- 参考 Qwen2.5-Omni 的 Conv+MLP 设计，但增加了 gMLP 增强表达力

### 2.5 Speech Decoder: Transformer + CTC

```
Input hidden_states (B, S, 4096)
  → Linear(4096, 1024) → LayerNorm              # 降维
  → Positional Embedding (learnable)
  → TransformerDecoder × 6 layers                # Pre-LN, 8 heads
  → Linear(1024, 4097)                           # 4096 speech tokens + 1 blank
  → CTC Loss (blank=4096)
```

**设计理由:**
- CTC 不需要自回归解码，推理速度快
- 4096 speech tokens 对应 CosyVoice2 词表
- blank token (id=4096) 用于 CTC 对齐
- Transformer Decoder 6 层足够建模 token 序列依赖
- Pre-LN 比 Post-LN 训练更稳定

### 2.6 CosyVoice2 语音合成

**选型对比:**

| 候选 | 特点 | 零样本克隆 | 流式 | 多语种 |
|------|------|-----------|------|--------|
| **CosyVoice2** | 流式+指令控制 | 5s | 是 | 中英日韩 |
| CosyVoice v1 | 第一代 | 10s | 否 | 中英 |
| VITS2 | End-to-end | 否 | 否 | 单语 |
| SoundStorm | 非自回归 | 3s | 否 | 英文 |

**选择 CosyVoice2 的理由:**
- **流式推理**: 首包延迟 < 150ms，适合实时对话
- **零样本克隆**: 仅需 5s 参考音频
- **指令可控**: 支持情感、语速、风格控制
- **多语种**: 中英日韩，匹配 Qwen3 的多语言能力
- OpenOmni 已验证 CosyVoice 在 Omni 模型中的有效性

## 3. 数据流

### 训练时

```
原始数据 (HuggingFace/本地)
  → prepare_data.py (下载+格式检测)
  → convert_to_swift_format() (→ MS-Swift JSONL)
  → OmniDataset (tokenize + 构造 input_ids/labels)
  → DataLoader (batch + shuffle)
  → OmniModel.forward() (多模态拼接 → LLM → loss)
  → TrainingMonitor.log_step() (JSONL 日志)
```

### 推理时

```
用户输入 (文本 / 图像 / 音频)
  → encode_vision() / encode_audio() / tokenize()
  → OmniModel.forward(inputs_embeds=...)
  → LLM 输出 logits → 文本生成
  → generate_speech(hidden_states) → 语音 token
  → CosyVoice2 decode → 波形
```

## 4. 训练策略

### 渐进式模态对齐

采用"先冻结后解冻"的渐进式策略，与 Baichuan-Omni、OpenOmni 的实践一致：

1. **Stage 1-2 (对齐)**: 仅训练投影层，让模态特征对齐到 LLM 空间
   - 为什么不直接联合训练？新投影层梯度噪声大，会破坏预训练 LLM
   - 参考: LLaVA Stage 1 仅训练 projector

2. **Stage 3 (SFT)**: 使用 LoRA (r=64) 微调 LLM
   - 为什么不 full finetune？8B 模型全量微调显存需求过大
   - LoRA r=64 已覆盖足够的参数子空间
   - 参考: Baichuan-Omni 的 SFT 阶段

3. **Stage 4 (语音)**: 训练 Speech Decoder + LLM 末 2 层
   - 为什么只解冻末 2 层？语音生成需要 LLM 输出层适配，但不应过多改变表征
   - CTC Loss 不需要自回归，训练效率高
   - 参考: OpenOmni 的 Text2Speech 阶段

4. **Stage 5 (DPO)**: 小学习率 (5e-6) 偏好对齐
   - 为什么用 DPO 而非 RLHF？DPO 不需要 reward model，实现简单
   - LoRA r=16 (比 Stage 3 小) 防止过拟合少量偏好数据
   - 参考: OpenOmni Stage 3-2 的 Emotional DPO

### DeepSpeed ZeRO-2

选择 ZeRO-2 而非 ZeRO-3 的理由：
- ZeRO-2 分片优化器状态 + 梯度，通信开销适中
- ZeRO-3 分片模型参数，但对模态编码器（冻结组件）分片收益低
- 8×A100-80G 下 ZeRO-2 显存足够

## 5. 设计决策记录

| 决策 | 选择 | 替代方案 | 理由 |
|------|------|----------|------|
| 训练框架 | MS-Swift | HuggingFace Trainer, LLaVA Trainer | 原生支持 Qwen3 全模态 |
| 投影层 | MLP (视觉), Conv-gMLP (音频) | Q-Former, Perceiver | 简洁，论文验证有效 |
| LLM 微调 | LoRA | Full finetune, QLoRA | 显存友好，效果接近全量 |
| 语音生成 | CTC | AR, NAR | 推理快，不需自回归 |
| 偏好对齐 | DPO | PPO, GRPO | 实现简单，无需 reward model |
| 分布式 | DeepSpeed ZeRO-2 | FSDP, ZeRO-3 | 通信开销适中，冻结组件兼容好 |
| Smoke 策略 | 全组件缩小 | 仅缩小 LLM | 端到端验证更可靠 |

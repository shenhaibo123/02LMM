# Omni 多模态论文方法总结

> 本文档总结 11 篇代表性 Omni 多模态论文的核心方法，作为训练方案的理论基础。

---

## 一、总体技术路线分类

### 1.1 渐进式模态对齐（推荐路线）

| 论文 | 阶段数 | 核心思路 |
|------|--------|----------|
| **Baichuan-Omni** | 2 阶段 (6 子阶段 + SFT) | 先图像→再视频→再音频→联合对齐 → SFT |
| **OpenOmni** | 5 阶段 | Speech2Text → Image2Text → Finetune → Text2Speech → DPO |
| **Ola** | 3 阶段 | 图文 → 加入语音 → 加入视频，视频作为跨模态桥梁 |
| **Mini-Omni2** | 3 阶段 | 编码器适配 → 多模态理解 → 语音输出 |

### 1.2 联合训练

| 论文 | 核心思路 |
|------|----------|
| **OmniVinci** | 对比对齐 Loss + LM Loss 联合训练，0.2T token 达到 SOTA |
| **Qwen2.5-Omni** | Thinker(3 阶段) + Talker(3 阶段) 双轨训练 |

### 1.3 端到端单阶段

| 论文 | 核心思路 |
|------|----------|
| **LLaMA-Omni** | 200K 数据，4 GPU，<3 天，非自回归 CTC 解码 |
| **AnyGPT** | 统一离散 token 化，标准 LM 训练 |

---

## 二、各论文核心方法详解

### 2.1 Baichuan-Omni (7B)
- **架构**: SigLIP-so400m + Whisper-large-v3 + 7B LLM
- **关键创新**: Conv-GMLP 音频投影层（8x 压缩，ASR 仅降 0.3%）
- **训练策略**: 6 子阶段渐进对齐 + 600K 混合 SFT
  - I-1: 图像-文本对齐（冻结编码器和 LLM）
  - I-2: 图像-文本微调（解冻 LLM）
  - I-3: 视频分支引入
  - I-4: 音频-文本对齐
  - I-5: 音频-文本微调
  - I-6: 全模态联合对齐（Omni-Alignment）
  - II: 600K 多任务 SFT
- **数据**: PIN-14M, MINT-1T, LAION-5B + 600K SFT 指令

### 2.2 OpenOmni (7B) — 唯一完整开源训练代码
- **架构**: 7B LLM + 视觉/音频编码器 + 双模式语音生成(CTC+AR)
- **关键创新**: 情感 DPO（9K 情感偏好对）+ 语言桥接策略
- **训练策略**: 5 阶段渐进
  1. Speech2Text 预训练
  2. Image2Text 预训练
  3. 多模态联合微调
  4. Text2Speech 预训练
  5. 情感 DPO
- **数据**: WeNetSpeech + LibriSpeech (~1.6M 样本 + 8K 小时语音)
- **资源**: 8×A100-80G，最具可复现性

### 2.3 Qwen2.5-Omni (7B) — 最大规模
- **架构**: Thinker(理解) + Talker(生成) 双轨
- **关键创新**: TMRoPE 时间对齐 + 滑窗 DiT 语音生成
- **训练策略**: Thinker 3 阶段 + Talker 3 阶段
  - Thinker: 800B 图像/视频 token + 300B 音频 token (~1.2T 总量)
  - Talker: 语音生成 + DPO 对齐
- **数据**: ~1.2T tokens（最大规模）

### 2.4 OmniVinci (9B) — 数据效率冠军
- **架构**: OmniAlignNet + TEG + CRTE
- **关键创新**:
  - 对比对齐网络 (OmniAlignNet)：跨模态特征统一
  - 时序编码 (TEG+CRTE)：处理时序信息
  - 仅 0.2T token 训练，达到 1.2T 的性能（6x 数据效率）
- **训练**: 联合训练，对比 Loss + LM Loss
- **数据**: 24M 对话数据（含隐式+显式多模态标注）

### 2.5 LLaMA-Omni (7B) — 资源效率冠军
- **架构**: Whisper Encoder + Adapter + LLM + CTC Decoder
- **关键创新**: 非自回归 CTC 语音解码器，延迟极低
- **训练**: 单阶段端到端，4 GPU <3 天
- **数据**: 200K InstructS2S 数据

### 2.6 Mini-Omni2 (0.5B) — 最轻量
- **架构**: Qwen2-0.5B + CLIP ViT-B/32 + Whisper-Small
- **关键创新**: SNAC 7 层延迟并行生成 + 命令式中断
- **训练**: 3 阶段对齐
- **特点**: 0.5B 参数即可实现 GPT-4o 式全模态

### 2.7 Ola — 渐进模态对齐
- **关键创新**: 视频作为视觉和音频的跨模态桥梁
- **训练**: 3 阶段渐进（图文 → +语音 → +视频）
- **双音频编码器**: Whisper + BEATs

### 2.8 VITA (MoE 8×7B)
- **架构**: Mixtral 8×7B MoE (47B 总量 / 12B 活跃)
- **训练**: 2 阶段（对齐 + 指令微调）
- **数据**: ~5M SFT 指令

### 2.9 InternOmni — 最小扩展
- **关键创新**: 仅添加 MLP_audio 即完成音频扩展
- **训练**: S1: 26M 样本/64 GPU/30h + S2: 1.9M/32 GPU/15h
- **特点**: ~45h 总训练时间

### 2.10 AnyGPT — 统一离散化
- **关键创新**: SpeechTokenizer + SEED + Encodec 统一离散 token
- **数据**: 108K AnyInstruct 合成数据
- **训练**: 标准 LM 训练，无架构修改

### 2.11 OmniGAIA — Agent 基准
- **关键创新**: OmniDPO 轨迹错误定位 + 后见树搜索
- **基准**: 360 个 Agent 任务
- **发现**: 工具使用(59.4%错误) > 推理(64.4%) > 感知(30%)

---

## 三、关键设计决策对比

### 3.1 音频投影层设计

| 方案 | 论文 | 压缩率 | 特点 |
|------|------|--------|------|
| Conv-GMLP | Baichuan-Omni | 8x | ASR 仅降 0.3%，本方案采用 |
| MLP | InternOmni | 1x | 最简单，扩展性差 |
| Q-Former | BLIP-2 系 | 可变 | 参数多，训练复杂 |

### 3.2 语音生成方案

| 方案 | 论文 | 延迟 | 质量 |
|------|------|------|------|
| CTC 非自回归 | LLaMA-Omni | 极低 | 中等 |
| CTC + AR 双模式 | OpenOmni | 可选 | 高 |
| Thinker-Talker | Qwen2.5-Omni | 低 | 高 |
| SNAC 延迟并行 | Mini-Omni2 | 低 | 中等 |

### 3.3 数据效率排名

1. **LLaMA-Omni**: 200K 样本，4 GPU，<3 天
2. **OmniVinci**: 0.2T tokens（同等性能下最少）
3. **InternOmni**: 28M 样本，~45h
4. **OpenOmni**: ~1.6M 样本
5. **Qwen2.5-Omni**: ~1.2T tokens（最大规模）

---

## 四、本方案的设计选择

基于上述论文分析，本训练方案采用以下技术路线：

| 决策 | 选择 | 参考论文 |
|------|------|----------|
| **基座模型** | Qwen2.5-Omni-7B / Qwen3-Omni | 已具备全模态能力 |
| **训练框架** | MS-Swift | 原生支持 Qwen-Omni SFT/DPO/GRPO |
| **训练策略** | 5 阶段渐进 | Baichuan-Omni + OpenOmni |
| **音频投影** | Conv-GMLP 4x 降采样 | Baichuan-Omni |
| **语音生成** | CTC Decoder | LLaMA-Omni |
| **偏好对齐** | DPO + GRPO | OpenOmni + Qwen |
| **数据策略** | 公开数据集为主 + 合成增强 | OpenOmni 可复现路线 |

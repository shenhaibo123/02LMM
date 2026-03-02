# Omni 全模态模型训练对比文档（增强版）

> 覆盖 11 篇代表性论文，逐篇详述训练阶段、冻结/训练模块、损失函数、数据来源与规模、公开数据集、GPU 配置、开源情况与关键创新。

---

## 目录

1. [Baichuan-Omni](#1-baichuan-omni)
2. [OpenOmni](#2-openomni)
3. [Qwen2.5-Omni](#3-qwen25-omni)
4. [LLaMA-Omni](#4-llama-omni)
5. [LLaMA-Omni2](#5-llama-omni2)
6. [Mini-Omni2](#6-mini-omni2)
7. [VITA](#7-vita)
8. [OmniVinci](#8-omnivinci)
9. [InternOmni](#9-internomni)
10. [AnyGPT](#10-anygpt)
11. [Ola](#11-ola)
12. [总对比表格](#总对比表格)

---

## 1. Baichuan-Omni

| 项目 | 内容 |
|------|------|
| **论文** | Baichuan-Omni Technical Report (arXiv:2410.08565) |
| **机构** | 百川智能 |
| **参数量** | 7B（LLM 骨干）；视觉编码器 SigLIP-400M；音频编码器 Whisper-large-v3（1.5B） |

### 训练阶段

#### 阶段一：多模态对齐预训练

分为图像、视频、音频三条分支，最后做 Omni-Alignment 混合对齐。

**图像-语言分支（3 子阶段）**

| 子阶段 | 冻结模块 | 训练模块 | 损失函数 | 数据类型/来源/数据量 |
|--------|---------|---------|---------|---------------------|
| Stage I | LLM + 视觉编码器 | 视觉投影层（Mean Pool + 2-layer MLP） | Next-token prediction (NTP) | 图文描述对：PIN-14M, MINT-1T, LAION-5B, OBELIC 等；lr=1e-3 |
| Stage II | LLM | 投影层 + 视觉编码器 (SigLIP) | NTP | VQA/OCR/图表数据：Cauldron, Monkey, ArxivQA, TGDoc, MM-Self-Instruct, MMTab + 130K 合成 OCR/图表 QA；lr=1e-5 |
| Stage III | 无（全解冻） | LLM + 投影层 + 视觉编码器 | NTP | 交错图文数据 + 纯文本（维持 LLM 语言能力）；lr=1e-5 |

**视频-语言分支**

| 冻结模块 | 训练模块 | 损失函数 | 数据类型/来源/数据量 |
|---------|---------|---------|---------------------|
| 视觉编码器 + LLM | 视频投影层（2×2 Conv） | NTP | ShareGPT4Video, WebVid, NExTVideo, ActivityNet-QA；先用图文数据再混入视频对；1fps 最多 48 帧；lr=4e-6 |

**音频-语言分支**

| 冻结模块 | 训练模块 | 损失函数 | 数据类型/来源/数据量 |
|---------|---------|---------|---------------------|
| LLM | 音频编码器 (Whisper-large-v3) + Conv-GMLP 投影器 | NTP | 开源 + 自建 ASR 数据（长音频-文本序列，最长 4K token） |

**Omni-Alignment**

| 冻结模块 | 训练模块 | 损失函数 | 数据类型/来源/数据量 |
|---------|---------|---------|---------------------|
| 无（全参数联合训练） | 所有模块 | NTP | 混合高质量图文/视频/音频文本对 |

#### 阶段二：多模态监督微调（SFT）

| 冻结模块 | 训练模块 | 损失函数 | 数据类型/来源/数据量 |
|---------|---------|---------|---------------------|
| 无（全参数） | 全模型 | NTP | **约 600K 条，覆盖 200+ 任务**：纯文本多任务（vFLAN 过滤+中译）、图文理解、视频理解（VideoInstruct100K 语义去重+中译）、音频理解（TTS 合成+人工录音）、跨模态交互（图像-音频、视频-音频）；packing + flash-attention2 cuseg_len 隔离 |

### 公开数据集

- PIN-14M：图文预训练
- MINT-1T：图文预训练
- LAION-5B：图文预训练（[HuggingFace](https://huggingface.co/datasets/laion/laion5B-index)）
- OBELIC：交错图文（[HuggingFace](https://huggingface.co/datasets/HuggingFaceM4/OBELICS)）
- Cauldron：VQA 合集（[HuggingFace](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron)）
- ShareGPT4Video：视频描述（[HuggingFace](https://huggingface.co/datasets/ShareGPT4Video/ShareGPT4Video)）
- WebVid：视频-文本对
- ActivityNet-QA：视频问答
- vFLAN：文本指令数据（经过滤+中译）

### GPU 配置与训练时长

技术报告**未公开** GPU 数量与总训练时长。

### 代码开源

部分开源。模型权重公开，训练代码未完整开放。
- GitHub: https://github.com/westlake-baichuan-mllm/bc-omni

### 关键创新

1. **Conv-GMLP 音频投影器**：用卷积门控 MLP 替代传统 pooling 做音频下采样，在高压缩率下仍保留音频细节（下采样率 2→8，ASR WER 仅下降 0.3%）。
2. **渐进式多分支对齐**：图像、视频、音频分别对齐后再做 Omni-Alignment，避免模态干扰。
3. 首个开源 7B 全模态 MLLM，中文能力大幅超越 VITA（CMMLU +25.6 pp）。

---

## 2. OpenOmni

| 项目 | 内容 |
|------|------|
| **论文** | OpenOmni (arXiv:2501.04561, NeurIPS 2025) |
| **机构** | 未明确（学术合作） |
| **参数量** | 7B（LLM 骨干） |

### 训练阶段

#### 阶段一：语音-文本对齐

| 冻结模块 | 训练模块 | 损失函数 | 数据类型/来源/数据量 |
|---------|---------|---------|---------------------|
| 视觉相关模块 | 语音编码器 + 适配器 + LLM | Language Modeling (LM) loss | 双语语音约 **8,000 小时**：WeNetSpeech, LibriSpeech, AIShell-4 + O2S 短回复 |

#### 阶段二：图像-文本对齐（预训练 + 指令微调）

| 子步 | 冻结模块 | 训练模块 | 损失函数 | 数据类型/来源/数据量 |
|------|---------|---------|---------|---------------------|
| 预训练 | LLM | 视觉投影层 | NTP | LLaVA-Pretrain-595K |
| 指令微调 | 部分冻结 | 投影层 + LLM | NTP | MMEvol 指令数据 |

#### 阶段三：文本引导语音生成

| 子步 | 冻结模块 | 训练模块 | 损失函数 | 数据类型/来源/数据量 |
|------|---------|---------|---------|---------------------|
| 语音解码器训练 | LLM + 视觉/音频编码器 | 语音解码器 | CTC loss | **O2S-300K**（300K 条指令，CosyVoice 合成） |
| 情感 DPO | 解码器骨干 | 偏好层 | CTC-DPO | **EO2S-9K**（9 类情感偏好对，基于 Plutchik 情感，CosyVoice 正负样本） |

### 公开数据集

- WeNetSpeech：中文语音（[HuggingFace](https://huggingface.co/datasets/Wenetspeech/wenetspeech)）
- LibriSpeech：英文语音（[HuggingFace](https://huggingface.co/datasets/openslr/librispeech_asr)）
- AIShell-4：中文多通道语音
- LLaVA-Pretrain-595K：图文预训练（[HuggingFace](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain)）
- MMEvol：多模态指令
- O2S-300K：自建合成语音指令数据
- EO2S-9K：自建情感偏好数据

### GPU 配置与训练时长

**8 × NVIDIA A100-80G**；总训练天数未给出。

### 代码开源

开源。
- GitHub: https://github.com/RainBowLuoCS/OpenOmni

### 关键创新

1. **隐式跨模态对齐**：通过先语音-文本对齐、再图像-文本对齐，实现零样本从视觉到语音的跨模态泛化。
2. **CTC-DPO 情感偏好学习**：首次将 DPO 应用于语音情感控制，CTC 损失结合偏好优化。
3. 以约 **1.6M 样本**（VITA 的 1/3）在 OmniBench 上超越 5M 样本的 VITA。

---

## 3. Qwen2.5-Omni

| 项目 | 内容 |
|------|------|
| **论文** | Qwen2.5-Omni Technical Report (arXiv:2503.20215) |
| **机构** | 阿里巴巴 Qwen Team |
| **参数量** | 7B（Thinker LLM）；视觉编码器 675M（Qwen2.5-VL ViT）；音频编码器 Whisper-large-v3 |

### 训练阶段

#### Thinker 预训练（3 阶段）

| 阶段 | 冻结模块 | 训练模块 | 损失函数 | 数据类型/来源/数据量 |
|------|---------|---------|---------|---------------------|
| S1：编码器对齐 | LLM | 视觉/音频编码器 + 适配器 | NTP | 多模态对齐数据（规模未公开） |
| S2：全参数多模态预训练 | 无（全解冻） | 全部参数 | NTP | **800B** 图像/视频 token + **300B** 音频 token + **100B** 音视频 token（合计约 1.2T） |
| S3：长序列训练 | 无 | 全部参数 | NTP | 32K 长序列多模态数据 |

#### Thinker 后训练

| 冻结模块 | 训练模块 | 损失函数 | 数据类型/来源/数据量 |
|---------|---------|---------|---------------------|
| 编码器（冻结或低 lr） | LLM 主体 | NTP（ChatML 格式） | 多模态指令微调数据（规模未公开） |

#### Talker 训练（3 阶段）

| 阶段 | 冻结模块 | 训练模块 | 损失函数 | 数据类型/来源/数据量 |
|------|---------|---------|---------|---------------------|
| ICL 续写 | Thinker | Talker 双轨解码器 | NTP（语音 token） | 上下文续写语音数据 |
| DPO 稳定性 | Thinker | Talker | DPO loss（基于 WER + 停顿错误率） | 语音偏好对 |
| 多说话人微调 | Thinker | Talker | NTP | 多说话人语音数据 |

### 公开数据集

技术报告**未逐项列出**公开数据集。已知初始化来源：
- LLM：Qwen2.5
- 视觉编码器：Qwen2.5-VL
- 音频编码器：Whisper-large-v3（[HuggingFace](https://huggingface.co/openai/whisper-large-v3)）

### GPU 配置与训练时长

技术报告**未公开** GPU 数量与总训练时长。

### 代码开源

完全开源（模型权重 + 推理代码）。
- GitHub: https://github.com/QwenLM/Qwen2.5-Omni
- HuggingFace: https://huggingface.co/Qwen/Qwen2.5-Omni-7B

### 关键创新

1. **Thinker-Talker 架构**：解耦文本生成（Thinker）与语音生成（Talker），端到端训练避免干扰。Talker 接收 Thinker 的高层表示+文本 token 嵌入，双轨输入实现自然语音。
2. **TMRoPE（时间对齐多模态 RoPE）**：将位置编码分解为时间/高度/宽度三维，音视频每 2 秒交错排列，实现精确的跨模态时间对齐。
3. **滑动窗口 DiT**：限制感受野为 4 块（2 回看 + 1 前看），实现流式语音合成并降低首包延迟。

---

## 4. LLaMA-Omni

| 项目 | 内容 |
|------|------|
| **论文** | LLaMA-Omni (ICLR 2025, arXiv:2409.06666) |
| **机构** | 中国科学院 / UCAS |
| **参数量** | 8B（基于 Llama-3.1-8B-Instruct） |

### 训练阶段

#### 单阶段：端到端联合训练

| 冻结模块 | 训练模块 | 损失函数 | 数据类型/来源/数据量 |
|---------|---------|---------|---------------------|
| 语音编码器部分冻结 | 语音适配器 + LLM + 流式语音解码器（非自回归） | NTP（文本）+ CTC loss（语音） | **200K** 条语音指令-语音回复对（InstructS2S-200K） |

### 公开数据集

- InstructS2S-200K：自建语音指令数据（[HuggingFace](https://huggingface.co/datasets/ICTNLP/LLaMA-Omni-Data)）
- 底座：Llama-3.1-8B-Instruct（[HuggingFace](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)）

### GPU 配置与训练时长

**4 GPU，< 3 天**。

### 代码开源

完全开源。
- GitHub: https://github.com/ictnlp/LLaMA-Omni

### 关键创新

1. **极低资源训练**：仅 200K 数据、4 GPU、不到 3 天即完成语音对话模型训练，为最轻量的全模态训练方案之一。
2. **非自回归流式语音解码器**：基于 CTC 的非自回归解码实现低延迟流式语音输出。
3. 语音编码器 + 适配器 + LLM + 流式解码器联合端到端训练。

---

## 5. LLaMA-Omni2

| 项目 | 内容 |
|------|------|
| **论文** | LLaMA-Omni2 (ACL 2025, arXiv:2505.02625) |
| **机构** | 中国科学院 / UCAS |
| **参数量** | 0.5B / 1.5B / 3B / 7B / 14B / 32B（基于 Qwen2.5-Instruct 系列） |

### 训练阶段

#### 单阶段：端到端联合训练

| 冻结模块 | 训练模块 | 损失函数 | 数据类型/来源/数据量 |
|---------|---------|---------|---------------------|
| 语音编码器部分冻结 | 语音适配器 + LLM + CosyVoice 2 流式解码器 | NTP + 流式 AR 语音损失 | **200K** 多轮语音对话样本 |

### 公开数据集

- 多轮语音对话数据（与 LLaMA-Omni 同量级）
- 底座：Qwen2.5-Instruct 系列（[HuggingFace](https://huggingface.co/Qwen)）
- CosyVoice 2 解码器

### GPU 配置与训练时长

README **未公开** GPU 数量与训练时长。

### 代码开源

完全开源。
- GitHub: https://github.com/ictnlp/LLaMA-Omni2

### 关键创新

1. **多规模统一方案**：从 0.5B 到 32B 六种规模，验证了语音对话方案在不同参数量下的可扩展性。
2. **CosyVoice 2 流式解码**：引入 flow-matching 与声码器实现中英双语高质量流式语音。
3. 仅 200K 数据即可达到与数百万小时语音数据可比的效果。

---

## 6. Mini-Omni2

| 项目 | 内容 |
|------|------|
| **论文** | Mini-Omni2: Towards Open-source GPT-4o with Vision, Speech and Duplex Capabilities (arXiv:2410.11190) |
| **机构** | 独立研究（Zhifei Xie, Changqiao Wu） |
| **参数量** | 0.5B（LLM 骨干 Qwen2-0.5B）；视觉编码器 CLIP ViT-B/32；音频编码器 Whisper-v3-Small |

### 训练阶段

#### 阶段一：编码器适配（模态-语言空间对齐）

| 冻结模块 | 训练模块 | 损失函数 | 数据类型/来源/数据量 |
|---------|---------|---------|---------------------|
| LLM + 编码器（冻结） | 视觉适配器（LlamaMLP 单层）+ 音频适配器 | 对比/回归损失 | 图文对、语音-文本对（论文强调 limited dataset） |

#### 阶段二：多模态理解与文本生成

| 冻结模块 | 训练模块 | 损失函数 | 数据类型/来源/数据量 |
|---------|---------|---------|---------------------|
| 编码器 | 适配器 + LLM | NTP | 多模态指令→文本数据 |

#### 阶段三：多模态/文本 → 语音输出

| 冻结模块 | 训练模块 | 损失函数 | 数据类型/来源/数据量 |
|---------|---------|---------|---------------------|
| 编码器（冻结或低 lr） | LLM + 7 个 sub-LM-head（SNAC 7 层语音 token） | NTP（文本 + 语音 token） | 多模态输入 + 语音输出对；词汇表扩展至 181,120 |

### 公开数据集

论文**未详细列出**公开数据集。涉及：
- 预训练视觉编码器 CLIP ViT-B/32（[HuggingFace](https://huggingface.co/openai/clip-vit-base-patch32)）
- 预训练音频编码器 Whisper-v3-Small
- SNAC 语音编解码器

### GPU 配置与训练时长

论文**未公开** GPU 数量与总训练时长。

### 代码开源

完全开源。
- GitHub: https://github.com/gpt-omni/mini-omni2

### 关键创新

1. **极轻量全模态**：仅 0.5B 参数即实现视觉+语音+文本+双工交互，功能形态最接近 GPT-4o 的开源复现之一。
2. **命令式打断机制**：用户可通过特定命令实时打断模型语音输出，实现灵活的双工交互。
3. **并行延迟生成（Delayed Parallel Generation）**：主 LM-head 输出文本 token，同时 7 个 sub-LM-head 输出 SNAC 7 层语音 token，文本与语音同步生成。

---

## 7. VITA

| 项目 | 内容 |
|------|------|
| **论文** | VITA (OpenReview/arXiv, 8×7B MoE) |
| **机构** | 多机构合作 |
| **参数量** | 约 47B 总参 / 12B 激活参（Mixtral 8×7B MoE） |

### 训练阶段

#### 阶段一：多模态对齐

| 冻结模块 | 训练模块 | 损失函数 | 数据类型/来源/数据量 |
|---------|---------|---------|---------------------|
| LLM（冻结或低 lr） | 视觉/音频适配器 + 编码器 | NTP / 对齐损失 | 视觉-文本对齐 + 音频-文本对齐数据（规模未单独给出） |

#### 阶段二：指令微调

| 冻结模块 | 训练模块 | 损失函数 | 数据类型/来源/数据量 |
|---------|---------|---------|---------------------|
| 无（全参数） | 全模型 | NTP | 约 **5M** 条多模态指令数据，覆盖双语（中英） |

### 公开数据集

- Mixtral 8×7B 底座
- 双语指令与多模态对齐数据（未逐项列出）

### GPU 配置与训练时长

论文**未明确**；Long-VITA 提到单节点 8 GPU 用于预填充与上下文扩展。

### 代码开源

开源。
- GitHub: https://github.com/VITA-MLLM/VITA

### 关键创新

1. **大规模 MoE 全模态**：基于 Mixtral 8×7B，以 12B 激活参数实现全模态理解与双语交互。
2. **双语指令扩展**：中英双语多模态指令微调，覆盖广泛任务。
3. Long-VITA 进一步扩展长视频理解能力。

---

## 8. OmniVinci

| 项目 | 内容 |
|------|------|
| **论文** | OmniVinci: Enhancing Architecture and Data for Omni-Modal Understanding LLM (arXiv:2510.15870) |
| **机构** | NVIDIA |
| **参数量** | 9B（OmniVinci-9B） |

### 训练阶段

OmniVinci 采用**联合训练**范式，未拆分为多个独立阶段，而是在统一框架内同时优化对齐与生成。

#### 联合训练

| 冻结模块 | 训练模块 | 损失函数 | 数据类型/来源/数据量 |
|---------|---------|---------|---------------------|
| 无（全参数） | OmniAlignNet + TEG/CRTE + LLM | **对齐损失**（CLIP 式对称对比 L_o-align）+ **LLM 语言建模损失**（NTP） | **24M** 条单模态与全模态对话；**0.2T** 训练 token |

OmniAlignNet 对比损失细节：
- 视觉嵌入 E_v 与音频嵌入 E_a 通过可学习 query Q_v/Q_a 投影 → 三层自注意力 → L2 归一化 → 共享潜空间
- L_v→a = -1/N Σ_i log(exp(s_ii)/Σ_j exp(s_ij))，L_a→v 对称
- L_o-align = (L_v→a + L_a→v) / 2

### 公开数据集

- 自建 24M 对话数据管线（含隐式+显式全模态标注）
- 具体公开数据集未逐项列出

### GPU 配置与训练时长

论文**未公开** GPU 数量与总训练时长。

### 代码开源

完全开源。
- GitHub: https://github.com/NVlabs/OmniVinci
- HuggingFace: https://huggingface.co/nvidia/omnivinci

### 关键创新

1. **OmniAlignNet**：在共享全模态潜空间内通过对比学习加强视觉与音频嵌入对齐。
2. **TEG + CRTE 时间建模**：TEG（Temporal Embedding Grouping）按时间块分组编码相对时间顺序；CRTE（Constrained Rotary Time Embedding）注入与 RoPE 兼容的绝对时间编码。
3. **极高数据效率**：仅 0.2T token（Qwen2.5-Omni 1.2T 的 1/6）即在 DailyOmni 上超越 Qwen2.5-Omni +19.05。

---

## 9. InternOmni

| 项目 | 内容 |
|------|------|
| **论文** | InternOmni（InternVL 博客/扩展） |
| **机构** | 上海 AI Lab / 商汤等 |
| **参数量** | 基于 InternVL 架构（含 ViT + Whisper + LLM） |

### 训练阶段

#### 阶段一：音频-文本对齐

| 冻结模块 | 训练模块 | 损失函数 | 数据类型/来源/数据量 |
|---------|---------|---------|---------------------|
| ViT + MLP（视觉侧冻结） | MLP_audio（音频投影层） | NTP（audio+text → text） | **约 26M 条**：GigaSpeech, CommonVoice, Libriheavy, WENETSPEECH |

#### 阶段二：音频-图像指令微调

| 冻结模块 | 训练模块 | 损失函数 | 数据类型/来源/数据量 |
|---------|---------|---------|---------------------|
| ViT + Whisper | MLP_audio | NTP（audio+image → text） | **约 1.9M 条**：TextVQA, GQA, OKVQA, ALLAVA 等 |

### 公开数据集

- GigaSpeech：英文语音（[HuggingFace](https://huggingface.co/datasets/speechcolab/gigaspeech)）
- CommonVoice：多语言语音（[HuggingFace](https://huggingface.co/datasets/mozilla-foundation/common_voice_16_1)）
- Libriheavy：英文语音
- WENETSPEECH：中文语音（[HuggingFace](https://huggingface.co/datasets/Wenetspeech/wenetspeech)）
- TextVQA：图文 VQA（[HuggingFace](https://huggingface.co/datasets/facebook/textvqa)）
- GQA：图文问答
- OKVQA：知识型图文问答
- ALLAVA：图文指令数据

### GPU 配置与训练时长

- **Stage 1**：64 GPU，约 30 小时，4K steps
- **Stage 2**：32 GPU，约 15 小时，3K steps

### 代码开源

基于 InternVL 开源生态。
- GitHub: https://github.com/OpenGVLab/InternVL

### 关键创新

1. **极简扩展策略**：仅训练 MLP_audio 一个投影层即可为现有 VLM 增加音频理解能力，冻结视觉与 LLM 参数。
2. **两阶段高效训练**：总计约 45 小时即完成 28M+ 条数据的训练。
3. 为 InternVL 生态扩展全模态提供了轻量化路径。

---

## 10. AnyGPT

| 项目 | 内容 |
|------|------|
| **论文** | AnyGPT (ACL 2024, arXiv:2402.12226) |
| **机构** | 复旦大学 |
| **参数量** | 基于 LLaMA-2-7B |

### 训练阶段

AnyGPT 采用**统一离散化 + next-token 预测**范式，所有模态离散化为统一 token 序列，无额外多阶段训练。

#### 数据构建（2 步）

| 步骤 | 内容 |
|------|------|
| 步骤 1 | 生成多模态主题与文本对话（GPT-4 驱动） |
| 步骤 2 | 文本 → 多模态转换（SpeechTokenizer 语音、SEED 图像、Encodec-32k 音乐） |

#### 模型训练

| 冻结模块 | 训练模块 | 损失函数 | 数据类型/来源/数据量 |
|---------|---------|---------|---------------------|
| 无 | LLM（标准架构不修改） | NTP（统一离散序列） | **108K** 条多轮多模态对话（AnyInstruct），交织语音、文本、图像、音乐 |

### 公开数据集

- AnyInstruct-108K：自建多模态指令数据（[HuggingFace](https://huggingface.co/datasets/fnlp/AnyInstruct)）
- SpeechTokenizer：语音离散化（[HuggingFace](https://huggingface.co/fnlp/SpeechTokenizer)）
- SEED：图像离散化
- Encodec-32k：音乐离散化

### GPU 配置与训练时长

论文**未公开** GPU 数量与总训练时长。

### 代码开源

完全开源。
- GitHub: https://github.com/OpenMOSS/AnyGPT

### 关键创新

1. **全模态离散化统一**：将语音、图像、音乐全部离散化为 token，与文本共享同一个 next-token 预测框架，不修改 LLM 架构。
2. **生成式数据构建 AnyInstruct**：先用 GPT-4 生成文本对话，再自动转为多模态内容，解决多模态对话数据稀缺问题。
3. 证明了「统一离散化 + 标准 LLM 训练」范式在全模态场景下的可行性。

---

## 11. Ola

| 项目 | 内容 |
|------|------|
| **论文** | Ola: Pushing the Frontiers of Omni-Modal Language Model (arXiv:2502.04328) |
| **机构** | 多机构合作（Zuyan Liu 等） |
| **参数量** | 7B（LLM 骨干 Qwen2.5-7B）；视觉编码器 OryxViT（SigLIP-400M 初始化）；音频编码器 Whisper-v3-Large + BEATs 双编码器 |

### 训练阶段

Ola 采用**渐进式模态对齐**训练，以视频为桥梁连接图像与音频。

#### 阶段一：文本-图像训练

| 子步 | 冻结模块 | 训练模块 | 损失函数 | 数据类型/来源/数据量 |
|------|---------|---------|---------|---------------------|
| MLP 对齐 | LLM + 视觉编码器 | MLP 投影器 | NTP | 图文对齐数据 |
| 预训练 | 部分冻结 | 投影器 + 编码器 | NTP | 图文预训练数据 |
| SFT | 无 | 全参数 | NTP | 图文指令数据 |

#### 阶段二：图像+视频连续训练

| 冻结模块 | 训练模块 | 损失函数 | 数据类型/来源/数据量 |
|---------|---------|---------|---------------------|
| 无（或低 lr 冻结） | 全参数 | NTP | 图文 + 视频-文本数据，利用视频建立视觉时间建模 |

#### 阶段三：全模态训练

| 冻结模块 | 训练模块 | 损失函数 | 数据类型/来源/数据量 |
|---------|---------|---------|---------------------|
| 无 | 全参数 | NTP | 全模态数据：图文 + 视频 + 音频/语音 + **324K 跨模态视频-音频数据** |

### 公开数据集

论文**未详细列出**数据集名称，数据规模标注为「relatively small」。涉及：
- 图文预训练与 SFT 数据
- 视频-文本数据
- 音频/语音数据
- 324K 跨模态视频-音频数据

### GPU 配置与训练时长

论文**未公开** GPU 数量与总训练时长。

### 代码开源

完全开源（权重 + 代码 + 数据）。
- GitHub: https://github.com/Ola-Omni/Ola

### 关键创新

1. **视频作为跨模态桥梁**：视频天然包含视觉帧与伴随音频，以视频为中介对齐图像与音频比直接对齐更自然。
2. **渐进式模态对齐**：从差异最大的模态开始（图像-文本），逐步加入视频、音频，避免「大杂烩」训练导致模态冲突。渐进策略在 MMBench/VideoMME/LibriSpeech 上均优于直接混合与均衡采样。
3. **双音频编码器**：Whisper-v3-Large（语音语义）+ BEATs（音频事件/音乐），通道拼接提供更全面的音频理解。
4. **Local-Global Attention Pooling**：学习重要性权重实现 2x 视觉压缩，信息损失小于简单 pooling。

---

## 总对比表格

| 模型 | 机构 | 参数量 | 训练阶段数 | 总数据规模 | GPU 配置 | 训练时长 | 代码开源 | 关键创新 |
|------|------|--------|-----------|-----------|---------|---------|---------|---------|
| **Baichuan-Omni** | 百川智能 | 7B | 2 大阶段（对齐含 6 子阶段 + SFT） | 对齐：多来源大规模；SFT：600K | 未公开 | 未公开 | 部分 ([GitHub](https://github.com/westlake-baichuan-mllm/bc-omni)) | Conv-GMLP 音频投影；渐进多分支对齐 |
| **OpenOmni** | 学术合作 | 7B | 3 阶段 | ~1.6M 样本 + 8K h 语音 | 8×A100-80G | 未公开 | 是 ([GitHub](https://github.com/RainBowLuoCS/OpenOmni)) | 隐式跨模态对齐；CTC-DPO 情感偏好 |
| **Qwen2.5-Omni** | 阿里 Qwen | 7B | Thinker 3+1 / Talker 3 | ~1.2T token 预训练 | 未公开 | 未公开 | 是 ([GitHub](https://github.com/QwenLM/Qwen2.5-Omni)) | Thinker-Talker 架构；TMRoPE；滑窗 DiT |
| **LLaMA-Omni** | 中科院 | 8B | 1 阶段 | 200K | **4 GPU** | **< 3 天** | 是 ([GitHub](https://github.com/ictnlp/LLaMA-Omni)) | 极低资源；非自回归 CTC 语音解码 |
| **LLaMA-Omni2** | 中科院 | 0.5B–32B | 1 阶段 | 200K | 未公开 | 未公开 | 是 ([GitHub](https://github.com/ictnlp/LLaMA-Omni2)) | 多规模统一；CosyVoice 2 流式解码 |
| **Mini-Omni2** | 独立研究 | 0.5B | 3 阶段 | limited dataset | 未公开 | 未公开 | 是 ([GitHub](https://github.com/gpt-omni/mini-omni2)) | 0.5B 全模态；命令式打断双工 |
| **VITA** | 多机构 | 47B/12B 激活 (MoE) | 2 阶段 | ~5M 指令微调 | 未明确 | 未明确 | 是 ([GitHub](https://github.com/VITA-MLLM/VITA)) | 大规模 MoE 全模态；双语 |
| **OmniVinci** | NVIDIA | 9B | 联合训练 | 24M 对话 / 0.2T token | 未公开 | 未公开 | 是 ([GitHub](https://github.com/NVlabs/OmniVinci)) | OmniAlignNet 对比对齐；TEG+CRTE 时间建模；0.2T 超越 1.2T |
| **InternOmni** | 上海 AI Lab | InternVL 架构 | 2 阶段 | S1: 26M / S2: 1.9M | **S1: 64 GPU** / **S2: 32 GPU** | **S1: ~30h** / **S2: ~15h** | 是 ([GitHub](https://github.com/OpenGVLab/InternVL)) | 仅训 MLP_audio 极简扩展；~45h 完成 |
| **AnyGPT** | 复旦大学 | 7B | 无显式多阶段 | 108K (AnyInstruct) | 未公开 | 未公开 | 是 ([GitHub](https://github.com/OpenMOSS/AnyGPT)) | 全模态离散化统一 NTP；生成式数据构建 |
| **Ola** | 多机构 | 7B | 3 阶段（渐进式） | relatively small + 324K 跨模态 | 未公开 | 未公开 | 是 ([GitHub](https://github.com/Ola-Omni/Ola)) | 视频为桥梁；渐进式模态对齐；双音频编码器 |

### 补充维度对比

| 模型 | 输入模态 | 输出模态 | 语音生成方式 | 损失函数特色 |
|------|---------|---------|-------------|-------------|
| Baichuan-Omni | 文本+图像+视频+音频 | 文本 | 无端到端语音 | NTP |
| OpenOmni | 文本+图像+音频 | 文本+语音 | CosyVoice CTC | CTC + CTC-DPO |
| Qwen2.5-Omni | 文本+图像+视频+音频 | 文本+语音（流式） | Talker 双轨 AR + DiT | NTP + DPO（语音） |
| LLaMA-Omni | 语音 | 文本+语音 | 非自回归 CTC | NTP + CTC |
| LLaMA-Omni2 | 语音 | 文本+语音 | CosyVoice 2 AR | NTP + AR |
| Mini-Omni2 | 文本+图像+音频 | 文本+语音 | SNAC 7 层并行 | NTP（多头） |
| VITA | 文本+图像+视频+音频 | 文本 | 无端到端语音 | NTP |
| OmniVinci | 文本+图像+视频+音频 | 文本 | 外接 TTS | 对比对齐 + NTP |
| InternOmni | 图像+音频+文本 | 文本 | 无 | NTP |
| AnyGPT | 文本+语音+图像+音乐 | 文本+语音+图像+音乐 | 离散 token NTP | NTP（统一离散） |
| Ola | 文本+图像+视频+音频 | 文本+语音（流式） | CosyVoice 句级流式 | NTP |

---

## 效率与数据规模速览

| 效率维度 | 最高效方案 | 说明 |
|---------|----------|------|
| **训练资源最少** | LLaMA-Omni：4 GPU / < 3 天 / 200K 数据 | 语音对话最轻量方案 |
| **参数最小** | Mini-Omni2：0.5B | 全模态+双工 |
| **数据效率最高** | OmniVinci：0.2T token 超越 1.2T 基线 | 架构创新带来数据效率 |
| **对齐效率最高** | InternOmni：仅训 MLP_audio / ~45 小时 | 为现有 VLM 扩展音频的最快路径 |
| **数据规模最大** | Qwen2.5-Omni：~1.2T token | 工业级全量预训练 |
| **指令微调规模最大** | VITA：~5M 条 | 大规模多模态指令 |

---

*本文档基于 11 篇论文原文、arXiv 技术报告、GitHub README 及公开分析报告整理。部分信息（如 GPU 配置、具体数据量）因原文未公开而标注为「未公开」。如需进一步扩展为超参/脚本级训练复现指南，可在对应论文目录下单独撰写。*

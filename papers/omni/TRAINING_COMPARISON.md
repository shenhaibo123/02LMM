# Omni 全模态模型：训练数据、算力与阶段对比

本文档汇总 **开源项目** 的训练对比，以及 **11 篇代表性论文** 的逐篇训练过程说明（数据量、数据来源、卡数、时长、阶段与每阶段任务）。便于复现与选型时对照。

---

## 一、开源项目：训练数据量、卡数、时长、阶段对比

| 项目 | 总数据规模 | 主要数据来源 | 卡数/时长 | 阶段划分 | 每阶段任务简述 |
|------|------------|--------------|-----------|----------|----------------|
| **OpenOmni** | 对齐：8k h 语音 + LLaVA-Pretrain-595K + MMEvol；语音生成：O2S-300K（300K 条指令）、EO2S-9K（DPO）；相对 VITA 约 1.6M 样本（VITA 5M） | WeNetSpeech, LibriSpeech, AIShell-4, O2S 合成；LLaVA-Pretrain-595K；MMEvol；CosyVoice 合成 8k h 双语语音；Plutchik 情感 DPO | **8×A100-80G**（论文实验）；未给出总训练天数 | **3 阶段**：① 语音-文本对齐 ② 图像-文本对齐（预训练 + 指令微调）③ 文本引导语音生成（解码器训练 + 情感 DPO） | ① 语音→文本 LM 目标，建立语音理解 ② 图像→文本预训练 + MMEvol 指令微调，实现隐式全模态对齐 ③ CTC 训练语音解码器 + CTC-DPO 情感偏好 |
| **Baichuan-Omni** | 对齐：图像（PIN-14M/MINT-1T/LAION-5B/OBELIC 等 Stage I；Cauldron/Monkey/ArxivQA 等 Stage II–III）、视频（ShareGPT4Video/WebVid/NExTVideo/ActivityNet-QA 等）、音频（开源+自建 ASR）；SFT：**约 600K** 条，覆盖 200+ 任务 | 图像：PIN-14M, MINT-1T, LAION-5B, OBELIC, Cauldron, Monkey, ArxivQA, TGDoc, MM-Self-Instruct, MMTab 等；视频：ShareGPT4Video, WebVid, NExTVideo, ActivityNet-QA；音频：开源+自建 ASR；SFT：vFLAN（过滤+中译）、VideoInstruct100K、TTS 合成音频等 | **未在技术报告中给出** GPU 数量与总训练时长 | **2 大阶段**：① 多模态对齐预训练（图像 3 子阶段 + 视频 + 音频 + 全模态 Omni-Alignment）② 多模态监督微调（600K） | ① 图像：Stage I 仅训投影层 caption；Stage II 训投影+视觉编码器 VQA/OCR/图表；Stage III 解冻 LLM；视频：同视觉编码器+视频投影，先图文再混入视频对；音频：Whisper+Conv-GMLP 投影，仅训编码器+投影；最后 Omni-Alignment 混合高质量图文/视频/音频 ② 纯文本+图文+视频+音频+跨模态交互 SFT，packing 优化 |
| **Qwen2.5-Omni** | 预训练：**800B** 图像/视频 + **300B** 音频 + **100B** 音视频 token；后训练与 Talker 未在公开报告中给出具体样本数 | 技术报告未逐项列出；已知使用 Qwen2.5、Qwen2.5-VL、Whisper-large-v3 初始化；32K 长序列训练 | **未在公开报告中给出** 卡数与总时长 | **Thinker**：三阶段预训练（锁 LLM 训编码器+适配器 → 全参数 800B+300B+100B → 32K 长序列）+ ChatML 多模态指令微调；**Talker**：ICL 续写 → DPO → 多说话人微调 | 预训练：编码器与适配器对齐 → 全参数多模态 token 预训练 → 长序列；Thinker 后训练：多模态指令微调；Talker：上下文续写 → DPO 稳定性 → 多说话人 |
| **LLaMA-Omni2** | **200K** 多轮语音对话样本 | InstructS2S-200K 类数据；CosyVoice 2 流式解码器；Qwen2.5-0.5B/1.5B/3B/7B/14B/32B-Instruct | **未在 README 中给出**；LLaMA-Omni 一代：**4 GPU、<3 天** | 基于 Qwen2.5-Instruct 的语音编码器+流式 AR 语音解码；README 未拆分为多阶段训练描述 | 语音编码+适配器+LLM+CosyVoice2 解码器联合推理与微调；侧重推理与 Demo |
| **Mini-Omni2** | 有限数据（论文强调「limited dataset」）；未给出具体条数或 token 数 | 预训练视觉/听觉编码器；三阶段对齐用图文对、语音-文本对、多模态指令与语音输出数据 | **未在论文/检索中给出** 卡数与时长 | **3 阶段**：① 编码器适配（预训练视觉/听觉编码器与语言空间对齐）② 模态对齐（多模态→文本）③ 多模态微调（多模态输入输出，含语音生成） | ① 视觉/听觉编码器输出对齐到 LLM 表示空间 ② 多模态指令→文本 ③ 多模态→语音输出，支持实时与打断 |
| **VITA** | **约 5M** 样本（OpenOmni 论文中对比提及）；指令微调阶段 5M | Mixtral 8×7B 底座；双语指令扩展；多模态对齐+指令微调 | **未在论文中明确**；Long-VITA 提到单节点 8 GPU 的预填充与上下文扩展 | **2 阶段**：① 多模态对齐 ② 指令微调 | ① 视觉/音频与语言模型对齐 ② 多模态指令跟随 |
| **OmniVinci** | **24M** 条单模态与全模态对话；**0.2T** 训练 token（约为 Qwen2.5-Omni 1.2T 的 1/6） | 构建与合成管线生成 24M 对话；隐式+显式全模态标注 | **未在论文中给出** 卡数与总时长 | 架构上：模态编码 → OmniAlignNet 对齐 → TEG/CRTE 时间建模 → LLM；训练上未拆为多阶段描述 | 联合优化对齐损失（OmniAlignNet 对比）+ LLM 语言建模；数据效率高 |
| **InternOmni** | Stage1：**约 26M** 条（GigaSpeech, CommonVoice, Libriheavy, WENETSPEECH）；Stage2：**约 1.9M** 条（TextVQA, GQA, OKVQA, ALLAVA 等） | 音频：GigaSpeech, CommonVoice, Libriheavy, WENETSPEECH；图文：TextVQA, GQA, OKVQA, ALLAVA 等 | **Stage1：64 GPU，约 30 小时，4k steps**；**Stage2：32 GPU，约 15 小时，3k steps** | **2 阶段**：① 音频对齐（audio+text→text）② 音频-图像指令微调（audio+image→text） | ① 仅训 MLP_audio，冻结 ViT/MLP ② 仅训 MLP_audio，冻结 ViT/Whisper |
| **AnyGPT** | **108K** 条多轮多模态对话（AnyInstruct） | 生成式构造：先生成文本对话再转为多模态（语音/图像/音乐）；SpeechTokenizer, SEED, Encodec-32k | **未在论文/检索中给出** 卡数与时长 | **2 阶段数据构建**：① 生成多模态主题与文本对话 ② 文本→多模态转换；模型端为统一离散序列、next-token 预测，无额外训练阶段描述 | 所有模态离散化后统一 next-token 预测，不改 LLM 架构 |
| **Ola** | 渐进式扩展，对齐数据规模相对小（论文强调「relatively small」）；具体条数未给出 | 图像-文本 → 加入语音 → 加入视频；与现有视觉-语言模型扩展 | **未在论文/检索中给出** 卡数与时长 | **渐进 3 步**：① 图像+文本 ② +语音 ③ +视频；每步扩展模态，保持对齐数据量可控 | ① 图像-文本对齐 ② 加入语音数据 ③ 加入视频，形成全模态 |
| **LLaMA-Omni（一代）** | **200K** 语音指令与语音回复（InstructS2S-200K） | 语音指令-语音回复对，匹配语音交互场景 | **4 GPU，<3 天** | 端到端语音-语言模型训练；README 未再细分阶段 | 语音编码器+适配器+LLM+流式语音解码器联合训练 |

### 说明与数据来源

- **数据量**：部分项目给出的是「样本条数」，部分为「token 数」或「时长（小时）」；表中尽量保留原文表述。
- **卡数/时长**：多数论文未公开总 GPU 数与总训练天数，仅 InternOmni、LLaMA-Omni 有明确数字；其余标注为「未给出」。
- **阶段**：按各论文/README/技术报告描述的「训练流程」归纳，阶段命名与顺序以原文为准。

---

## 二、11 篇论文逐篇训练过程说明

### 1. Baichuan-Omni（arXiv:2410.08565）

| 项目 | 内容 |
|------|------|
| **训练过程** | 两阶段：先多模态对齐预训练（图像/视频/音频分别对齐再 Omni-Alignment），再 600K 多模态 SFT。 |
| **数据量** | 对齐：图像（PIN-14M、MINT-1T、LAION-5B、OBELIC 等 Stage I；Cauldron、Monkey、ArxivQA、TGDoc、MM-Self-Instruct、MMTab 等 Stage II–III），含 130k 合成 OCR/图表 QA；视频（ShareGPT4Video、WebVid、NExTVideo、ActivityNet-QA 等）；音频（开源+自建 ASR、多版本转录与过滤）。SFT：**约 600K**，覆盖 **200+ 任务**。 |
| **数据来源** | 见上；SFT 含 vFLAN（loss 过滤+中译）、VideoInstruct100K（语义去重+中译）、TTS 合成音频+人工录音、纯文本多任务、图文/视频/音频理解与跨模态交互。 |
| **卡数/时长** | 技术报告未给出。 |
| **阶段与每阶段任务** | **阶段一（多模态对齐）**：图像分支三子阶段（I：仅投影层 caption；II：投影+视觉编码器 VQA/OCR/图表/交错；III：解冻 LLM）；视频分支（同视觉编码器+视频投影，先图文再混视频对）；音频分支（Whisper-large-v3+Conv-GMLP 投影，仅编码器+投影）；最后 **Omni-Alignment** 混合高质量图文/视频/音频。**阶段二（SFT）**：600K 多任务 SFT（纯文本、图文、视频、音频、图像-音频交互），packing+cuseq_len 隔离。 |

---

### 2. OpenOmni（arXiv:2501.04561, NIPS 2025）

| 项目 | 内容 |
|------|------|
| **训练过程** | 两阶段框架：全模态对齐（语音-文本 + 图像-文本）→ 语音生成（解码器 + 情感 DPO）。实际训练拆为 3 步：语音-文本对齐 → 图像-文本预训练+指令微调 → 文本引导语音解码器+CTC-DPO。 |
| **数据量** | 语音-文本：WeNetSpeech、LibriSpeech、AIShell-4 + O2S 短回复，合计 **约 8,000 小时** 双语语音。图像-文本：LLaVA-Pretrain-**595K**；指令微调 MMEvol。语音生成：**O2S-300K**（300K 条指令，来自 MMEvol+UltraChat，答案用 CosyVoice 合成）；情感 DPO：**EO2S-9K**（9 类情感偏好对）。相对 VITA：约 **1.6M** 训练样本（VITA 5M）。 |
| **数据来源** | 语音：WeNetSpeech, LibriSpeech, AIShell-4；O2S-300K 为 MMEvol/UltraChat 答案 TTS；EO2S-9K 基于 Plutchik 情感、Qwen2-72B 标注+GPT-4o-mini 扩充，CosyVoice 正负样本。图像：LLaVA-Pretrain-595K，MMEvol。 |
| **卡数/时长** | **8×NVIDIA A100-80G**；总训练天数未给出。 |
| **阶段与每阶段任务** | **阶段 1**：语音-文本对齐，LM 目标，使 LLM 具备语音理解。**阶段 2**：图像-文本预训练（固定 LLM）+ 图像-文本指令微调（MMEvol），实现隐式全模态对齐（零样本从视觉到语音）。**阶段 3**：文本引导语音解码器（O2S-300K，CTC 损失）+ 情感 DPO（EO2S-9K，CTC-DPO）。 |

---

### 3. Qwen2.5-Omni（arXiv:2503.20215）

| 项目 | 内容 |
|------|------|
| **训练过程** | Thinker-Talker 端到端；Thinker 侧：三阶段预训练 + 多模态指令微调；Talker 侧：ICL 续写 → DPO → 多说话人微调。 |
| **数据量** | 预训练：**800B** 图像/视频 token + **300B** 音频 token + **100B** 音视频 token；32K 长序列训练。指令微调与 Talker 各阶段未在公开报告中给出具体样本数。 |
| **数据来源** | 技术报告未逐项列出；初始化：LLM 来自 Qwen2.5，视觉来自 Qwen2.5-VL，音频来自 Whisper-large-v3。 |
| **卡数/时长** | 未在公开报告中给出。 |
| **阶段与每阶段任务** | **Thinker 预训练**：(1) 锁 LLM，训编码器+适配器；(2) 全参数解冻，800B+300B+100B token；(3) 32K 长序列。**Thinker 后训练**：ChatML 格式多模态指令微调。**Talker**：(1) ICL 上下文续写；(2) DPO 稳定性；(3) 多说话人微调。 |

---

### 4. LLaMA-Omni（ICLR 2025, arXiv:2409.06666）

| 项目 | 内容 |
|------|------|
| **训练过程** | 端到端语音-语言模型：语音编码器+适配器+LLM+流式语音解码器联合训练，支持语音入、语音/文本出。 |
| **数据量** | **200K** 条语音指令与语音回复（InstructS2S-200K），针对语音交互场景构建。 |
| **数据来源** | InstructS2S-200K（语音指令-语音回复对）。底座：Llama-3.1-8B-Instruct。 |
| **卡数/时长** | **4 GPU，<3 天**。 |
| **阶段与每阶段任务** | 未再细分多阶段；单一训练阶段完成语音理解与流式语音生成联合优化。 |

---

### 5. LLaMA-Omni2（ACL 2025, arXiv:2505.02625）

| 项目 | 内容 |
|------|------|
| **训练过程** | 基于 Qwen2.5 系列（0.5B–32B-Instruct）的实时语音对话模型，引入流式 AR 语音解码（CosyVoice 2），支持中英双语。 |
| **数据量** | **200K** 多轮语音对话样本（与 LLaMA-Omni 同量级）；论文强调仅用 200K 即可达到与数百万小时语音数据可比的效果。 |
| **数据来源** | 多轮语音对话数据；CosyVoice 2 的 flow-matching 与声码器；Qwen2.5-0.5B/1.5B/3B/7B/14B/32B-Instruct。 |
| **卡数/时长** | README 未给出 GPU 数与训练时长。 |
| **阶段与每阶段任务** | 仓库以推理与 Gradio Demo 为主，未在 README 中拆分为多阶段训练；训练流程可参考 OpenOmni/LLaMA-Omni 的语音-文本对齐与语音解码器训练思路。 |

---

### 6. Mini-Omni2（arXiv:2410.11190）

| 项目 | 内容 |
|------|------|
| **训练过程** | 三阶段模态对齐：先对齐编码器与语言空间，再多模态→文本，最后多模态→语音输出；配合预训练视觉/听觉编码器与命令式打断机制。 |
| **数据量** | 论文强调「limited dataset」；未给出具体样本数或 token 数。 |
| **数据来源** | 预训练视觉/听觉编码器；三阶段分别使用图文对、语音-文本对、多模态指令与语音输出数据。 |
| **卡数/时长** | 未在论文/检索中给出。 |
| **阶段与每阶段任务** | **阶段 1**：编码器适配——视觉/听觉编码器输出对齐到 LLM 表示空间（投影+对比或回归）。**阶段 2**：多模态理解与文本生成（多模态指令→文本）。**阶段 3**：多模态/文本→语音输出，端到端语音生成，支持实时与打断。 |

---

### 7. VITA（OpenReview/arXiv, 8×7B MoE）

| 项目 | 内容 |
|------|------|
| **训练过程** | 两阶段：多模态对齐 → 指令微调；基于 Mixtral 8×7B，双语（含中文）扩展。 |
| **数据量** | 指令微调约 **5M** 样本（OpenOmni 论文对比提及）；对齐阶段规模未单独给出。 |
| **数据来源** | Mixtral 8×7B；双语指令与多模态对齐数据。 |
| **卡数/时长** | 论文未明确 GPU 数与总时长；Long-VITA 提到单节点 8 GPU。 |
| **阶段与每阶段任务** | **阶段 1**：多模态对齐——视觉/音频与语言模型对齐。**阶段 2**：指令微调——约 5M 条多模态指令跟随。 |

---

### 8. OmniVinci（arXiv:2510.15870, NVIDIA）

| 项目 | 内容 |
|------|------|
| **训练过程** | 架构上：模态编码 → OmniAlignNet 对比对齐 → TEG/CRTE 时间建模 → LLM；训练上联合优化对齐损失与 LLM 语言建模损失。 |
| **数据量** | **24M** 条单模态与全模态对话；**0.2T** 训练 token（约为 Qwen2.5-Omni 1.2T 的 1/6）。 |
| **数据来源** | 构建与合成管线生成 24M 对话；含隐式与显式全模态标注。 |
| **卡数/时长** | 论文未给出 GPU 数与总时长。 |
| **阶段与每阶段任务** | 未拆为多阶段描述；统一训练中完成 OmniAlignNet 对比学习、TEG/CRTE 时间编码与 LLM 自回归生成，数据效率高（0.2T 即超越 1.2T 基线）。 |

---

### 9. InternOmni（InternVL 博客/扩展）

| 项目 | 内容 |
|------|------|
| **训练过程** | 两阶段：先音频-文本对齐，再音频-图像指令微调；扩展 InternVL 的音频能力。 |
| **数据量** | Stage1：**约 26M** 条（GigaSpeech, CommonVoice, Libriheavy, WENETSPEECH）；Stage2：**约 1.9M** 条（TextVQA, GQA, OKVQA, ALLAVA 等）。 |
| **数据来源** | Stage1：GigaSpeech, CommonVoice, Libriheavy, WENETSPEECH；Stage2：TextVQA, GQA, OKVQA, ALLAVA 等开源图文指令数据。 |
| **卡数/时长** | **Stage1：64 GPU，约 30 小时，4k steps**；**Stage2：32 GPU，约 15 小时，3k steps**。 |
| **阶段与每阶段任务** | **阶段 1**：音频-文本对齐（audio+text→text），仅训 MLP_audio，冻结 ViT 与 MLP。**阶段 2**：音频-图像指令微调（audio+image→text），仅训 MLP_audio，冻结 ViT 与 Whisper。 |

---

### 10. AnyGPT（ACL 2024, arXiv:2402.12226）

| 项目 | 内容 |
|------|------|
| **训练过程** | 所有模态离散化为统一序列，标准 next-token 预测；数据侧两阶段构建 AnyInstruct，模型侧无额外多阶段训练。 |
| **数据量** | **108K** 条多轮多模态对话（AnyInstruct），交织语音、文本、图像、音乐等。 |
| **数据来源** | 生成式构造：先生成多模态主题与文本对话，再经 Text-to-Multimodality 转为真实多模态内容；SpeechTokenizer（语音）、SEED（图像）、Encodec-32k（音乐）。 |
| **卡数/时长** | 论文/检索未给出 GPU 数与时长。 |
| **阶段与每阶段任务** | **数据构建**：① 生成文本对话与多模态元素 ② 文本→多模态转换。**模型训练**：统一离散序列 + next-token 预测，不修改标准 LLM 架构与训练范式。 |

---

### 11. Ola（arXiv:2502.04328）

| 项目 | 内容 |
|------|------|
| **训练过程** | 渐进式模态对齐：从图像-文本开始，依次加入语音、视频，每步扩展一种模态，保持对齐数据量相对较小。 |
| **数据量** | 论文强调「relatively small」对齐数据；具体条数/ token 未给出。 |
| **数据来源** | 先图像-文本；再加入语音；最后加入视频；可与现有视觉-语言模型扩展结合。 |
| **卡数/时长** | 未在论文/检索中给出。 |
| **阶段与每阶段任务** | **步骤 1**：图像+文本对齐（差异最大的模态）。**步骤 2**：加入语音数据（连接语言与音频）。**步骤 3**：加入视频（连接所有模态）；每步保持跨模态对齐数据规模可控，便于从现有 VLM 扩展为全模态。 |

---

## 三、简要对比小结

- **数据规模**：从 200K（LLaMA-Omni/LLaMA-Omni2）到数千万条（InternOmni 26M、OmniVinci 24M）、再到数百亿 token（Qwen2.5-Omni 800B+300B+100B）；SFT 从 108K（AnyGPT）到 600K（Baichuan-Omni）、5M（VITA）。
- **算力**：仅 InternOmni、LLaMA-Omni 给出明确卡数与时长；多数工作未公开总 GPU 与总训练时间。
- **阶段数**：2 阶段（Baichuan-Omni、VITA、InternOmni）、3 阶段（OpenOmni、Mini-Omni2、Ola 渐进）较常见；Qwen2.5-Omni Thinker/Talker 各自多子阶段。
- **效率亮点**：OpenOmni 以约 1.6M 样本、7B 规模在 OmniBench 上超越 5M 样本、7×8B 的 VITA；OmniVinci 以 0.2T token 超越 1.2T 基线；LLaMA-Omni 以 4 GPU、<3 天、200K 数据完成语音对话训练。

如需把某一条目扩展为「数据格式/脚本/超参」级别，可在对应论文目录下单独写训练复现笔记（与 [paper-read](../../.cursor/skills/paper-read/SKILL.md) 精读报告并列）。

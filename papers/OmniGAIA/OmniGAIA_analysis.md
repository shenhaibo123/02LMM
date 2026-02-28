# OmniGAIA 论文精读分析报告

**论文标题**：OmniGAIA: Towards Native Omni-Modal AI Agents  
**作者**：Xiaoxi Li, Wenxiang Jiao, Jiarui Jin, Shijian Wang, Guanting Dong, Jiajie Jin, Hao Wang, Yinuo Wang, Ji-Rong Wen, Yuan Lu, Zhicheng Dou  
**机构**：中国人民大学、小红书、东南大学、浙江大学、清华大学  
**发布日期**：2026-02-26  
**来源链接**：
- arXiv: https://arxiv.org/abs/2602.22897
- PDF: https://arxiv.org/pdf/2602.22897
- GitHub (代码 & Demo): https://github.com/RUC-NLPIR/OmniGAIA
- HuggingFace (数据集 & 模型): https://huggingface.co/collections/RUC-NLPIR/omnigaia
- Leaderboard: https://huggingface.co/spaces/RUC-NLPIR/OmniGAIA-LeaderBoard

---

## 逐句翻译（中英对照）

### Abstract（摘要）

- **EN**: Human intelligence naturally intertwines omni-modal perception—spanning vision, audio, and language—with complex reasoning and tool usage to interact with the world.
- **中**: 人类智能自然地将全模态感知（涵盖视觉、音频与语言）与复杂推理和工具使用交织在一起，以此与世界交互。

- **EN**: However, current multi-modal LLMs are primarily confined to bi-modal interactions (e.g., vision-language), lacking the unified cognitive capabilities required for general AI assistants.
- **中**: 然而，当前的多模态大语言模型主要局限于双模态交互（如视觉-语言），缺乏通用 AI 助手所需的统一认知能力。

- **EN**: To bridge this gap, we introduce OmniGAIA, a comprehensive benchmark designed to evaluate omni-modal agents on tasks necessitating deep reasoning and multi-turn tool execution across video, audio, and image modalities.
- **中**: 为弥合这一差距，我们引入 OmniGAIA——一个综合性基准，旨在评估全模态智能体在需要跨视频、音频和图像模态进行深度推理与多轮工具执行的任务上的表现。

- **EN**: Constructed via a novel omni-modal event graph approach, OmniGAIA synthesizes complex, multi-hop queries derived from real-world data that require cross-modal reasoning and external tool integration.
- **中**: OmniGAIA 通过一种新颖的全模态事件图方法构建，从真实数据中合成需要跨模态推理和外部工具集成的复杂多跳查询。

- **EN**: Furthermore, we propose OmniAtlas, a native omni-modal foundation agent under tool-integrated reasoning paradigm with active omni-modal perception.
- **中**: 此外，我们提出 OmniAtlas——一种在工具集成推理范式下、具有主动全模态感知能力的原生全模态基础智能体。

- **EN**: Trained on trajectories synthesized via a hindsight-guided tree exploration strategy and OmniDPO for fine-grained error correction, OmniAtlas effectively enhances the tool-use capabilities of existing open-source models.
- **中**: OmniAtlas 通过后见之明引导的树搜索策略合成训练轨迹，并利用 OmniDPO 进行细粒度错误纠正训练，有效增强了现有开源模型的工具使用能力。

- **EN**: This work marks a step towards next-generation native omni-modal AI assistants for real-world scenarios.
- **中**: 本工作标志着迈向面向真实场景的下一代原生全模态 AI 助手的一步。

### 1 Introduction（引言）

- **EN**: Human intelligence seamlessly intertwines language, vision, and audio with long-horizon reasoning and tool use to understand the world and take actions.
- **中**: 人类智能将语言、视觉和音频与长程推理和工具使用无缝交织，以理解世界并采取行动。

- **EN**: Building general-purpose AI assistants therefore requires models that can jointly perceive across modalities, reason over long contexts, and interact with external tools for verification and knowledge acquisition.
- **中**: 因此，构建通用 AI 助手需要模型能够跨模态联合感知、在长上下文中推理，以及与外部工具交互以进行验证和知识获取。

- **EN**: Yet, despite rapid progress, multimodal LLM research is still dominated by bi-modal settings (e.g., vision–language or audio–language), which limits their ability to handle truly interwoven real-world modalities.
- **中**: 然而，尽管进展迅速，多模态 LLM 研究仍以双模态设置（如视觉-语言或音频-语言）为主导，这限制了它们处理真正交织在一起的现实世界模态的能力。

- **EN**: Emerging omni-modal foundation models (e.g., Qwen3-Omni) have begun to unify richer modalities, but most efforts primarily emphasize perception, leaving tool-integrated, agentic reasoning underexplored.
- **中**: 新兴的全模态基础模型（如 Qwen3-Omni）已开始统一更丰富的模态，但大多数工作主要强调感知，工具集成的智能体推理仍未充分探索。

- **EN**: Evaluation also lags behind: existing benchmarks are largely bi-modal and perception-centric (e.g., OmniBench, WorldSense, UNO-Bench), and thus do not adequately measure multi-hop omni-modal reasoning and multi-turn external tool use with verifiable open-form answers.
- **中**: 评估也落后了：现有基准主要是双模态和以感知为中心的（如 OmniBench、WorldSense、UNO-Bench），因此无法充分衡量多跳全模态推理和带有可验证开放式答案的多轮外部工具使用。

- **EN**: To bridge this gap, we introduce OmniGAIA, a challenging benchmark for native omni-modal agents.
- **中**: 为弥合这一差距，我们引入 OmniGAIA——一个面向原生全模态智能体的具有挑战性的基准。

- **EN**: OmniGAIA comprises 360 tasks across 9 real-world domains, covering both video-with-audio and image+audio settings, and explicitly requires multi-turn tool use (e.g., web search/browsing and code) to produce verifiable open-form answers.
- **中**: OmniGAIA 包含 360 个任务，覆盖 9 个真实领域，涵盖「视频+音频」和「图像+音频」两种设置，并明确要求多轮工具使用（如网页搜索/浏览和代码执行）以产生可验证的开放式答案。

- **EN**: To structure time-aligned multimodal cues and tool-related evidence for multi-hop reasoning, OmniGAIA is constructed via an omni-modal event-graph-driven pipeline: (1) we collect data and mine fine-grained signals from raw media; (2) we build an initial event graph that connects cross-modal entities/events and relations; (3) we expand the graph with next-hop evidence via cross-modal retrieval and external tools; and (4) we fuzzify key nodes/edges to generate multi-hop QA, followed by LLM screening and human verification for solvability and uniqueness.
- **中**: 为了构建时间对齐的多模态线索和用于多跳推理的工具相关证据，OmniGAIA 通过全模态事件图驱动的流水线构建：(1) 收集数据并从原始媒体中挖掘细粒度信号；(2) 构建连接跨模态实体/事件及其关系的初始事件图；(3) 通过跨模态检索和外部工具扩展图的下一跳证据；(4) 模糊化关键节点/边以生成多跳问答，随后进行 LLM 筛选和人工验证以确保可解性和唯一性。

- **EN**: Beyond benchmarking, we propose OmniAtlas, a native omni-modal foundation agent following the Tool-Integrated Reasoning (TIR) paradigm that naturally interleaves reasoning and tool calls.
- **中**: 除了基准测试之外，我们提出 OmniAtlas——一种遵循工具集成推理（TIR）范式、自然交错推理和工具调用的原生全模态基础智能体。

- **EN**: OmniAtlas further supports active omni-modal perception to selectively "look" or "listen" to the segments/regions in long media without blanket downsampling.
- **中**: OmniAtlas 还支持主动全模态感知，能够选择性地"观看"或"聆听"长媒体中的特定片段/区域，而非一刀切的下采样。

- **EN**: For training, we synthesize high-quality tool-integrated trajectories via hindsight-guided tree exploration, perform trajectory-level supervised learning, and further propose OmniDPO for fine-grained error correction.
- **中**: 在训练方面，我们通过后见之明引导的树搜索合成高质量的工具集成轨迹，进行轨迹级监督学习，并进一步提出 OmniDPO 用于细粒度错误纠正。

- **EN**: Experiments show that OmniGAIA is highly challenging: the strongest proprietary model (Gemini-3-Pro) reaches 62.5 Pass@1, while an open-source baseline (Qwen3-Omni) achieves 13.3.
- **中**: 实验表明 OmniGAIA 极具挑战性：最强的商业模型（Gemini-3-Pro）达到 62.5 Pass@1，而开源基线（Qwen3-Omni）仅达到 13.3。

- **EN**: Our OmniAtlas recipe substantially improves open models (e.g., Qwen3-Omni: 13.3→20.8).
- **中**: 我们的 OmniAtlas 训练方案大幅提升了开源模型的性能（例如 Qwen3-Omni: 13.3→20.8）。

- **EN**: Further analyses of fine-grained error types, tool-use behaviors, and perception strategies expose key limitations of current methods and point to promising directions for future omni-modal agents.
- **中**: 对细粒度错误类型、工具使用行为和感知策略的进一步分析揭露了当前方法的关键局限性，并指出了未来全模态智能体的有前景的研究方向。

### 2 Related Work（相关工作）

- **EN**: Building on advances in pure-text, vision-language, and audio-language foundation models, recent omni-modal models seek to unify text, vision, and audio within a single LLM backbone.
- **中**: 在纯文本、视觉-语言和音频-语言基础模型的进展基础上，近期的全模态模型寻求在单一 LLM 骨干中统一文本、视觉和音频。

- **EN**: A common approach adopts a unified tokenization-and-projection interface that maps heterogeneous visual and acoustic inputs into a shared token space.
- **中**: 一种常见方法采用统一的标记化和投影接口，将异构的视觉和声学输入映射到共享的 token 空间中。

- **EN**: Concurrent work further strengthens omni-modal reasoning behaviors.
- **中**: 同期工作进一步加强了全模态推理行为。

- **EN**: For evaluation, existing benchmarks largely emphasize short audios/videos and perception-centric tasks, leaving long-horizon reasoning and tool-integrated agency underexplored.
- **中**: 在评估方面，现有基准主要强调短音频/视频和以感知为中心的任务，长程推理和工具集成智能体能力仍未充分探索。

- **EN**: LLM-driven autonomous agents tackle real-world tasks by reasoning and acting through external tools that interface with their environment.
- **中**: LLM 驱动的自主智能体通过推理和调用与环境交互的外部工具来完成真实世界任务。

- **EN**: Existing approaches broadly fall into workflow-based paradigms and native agentic reasoning methods, and have shown strong performance on text-only tasks.
- **中**: 现有方法大致分为基于工作流的范式和原生智能体推理方法，并在纯文本任务上表现出色。

- **EN**: Moving beyond text, recent studies investigate vision-language agents for multimodal web search, long-form video understanding, and GUI navigation.
- **中**: 超越文本领域，近期研究探索了用于多模态网页搜索、长视频理解和 GUI 导航的视觉-语言智能体。

- **EN**: However, omni-modal foundation agents that natively fuse audio, vision, and language while performing long-horizon agentic reasoning remain underexplored.
- **中**: 然而，原生融合音频、视觉和语言同时执行长程智能体推理的全模态基础智能体仍未充分探索。

### 3 OmniGAIA: Benchmarking（基准构建）

- **EN**: OmniGAIA is a benchmark of challenging omni-modal agentic tasks designed to stress-test unified perception over vision, audio, and language, together with long-horizon reasoning and multi-turn tool use in realistic scenarios.
- **中**: OmniGAIA 是一个具有挑战性的全模态智能体任务基准，旨在压力测试视觉、音频和语言的统一感知能力，以及在现实场景中的长程推理和多轮工具使用。

- **EN**: To reflect the complexity of real-world omni-modal interactions, we construct OmniGAIA from two complementary settings: (i) video with audio, and (ii) image + audio pairs.
- **中**: 为反映现实世界全模态交互的复杂性，我们从两种互补设置构建 OmniGAIA：(i) 带音频的视频，(ii) 图像+音频配对。

- **EN**: For the video setting, we aggregate high-quality videos from multiple sources to ensure diversity in both content and duration.
- **中**: 对于视频设置，我们从多个来源聚合高质量视频以确保内容和时长的多样性。

- **EN**: We include FineVideo (43K videos spanning broad domains; average length 4 minutes).
- **中**: 我们纳入了 FineVideo（43K 个视频，涵盖广泛领域；平均时长 4 分钟）。

- **EN**: To evaluate long-context reasoning, we further incorporate LongVideoBench (∼1K videos) and LongVideo-Reason (∼1K videos), both containing videos around 10 minutes.
- **中**: 为评估长上下文推理，我们进一步引入 LongVideoBench（约 1K 个视频）和 LongVideo-Reason（约 1K 个视频），两者都包含约 10 分钟的视频。

- **EN**: For the image + audio setting, we use audio tracks from FineVideo to provide diverse acoustic environments, and draw images from COCO 2017, which contains 122K complex everyday-scene images with object detection and segmentation annotations.
- **中**: 对于图像+音频设置，我们使用 FineVideo 的音频轨道提供多样的声学环境，并从 COCO 2017 中提取图像，该数据集包含 122K 复杂日常场景图像及其目标检测和分割标注。

- **EN**: We employ a strong omni-modal model (Gemini-3-Flash) to extract fine-grained, time-aware signals from each modality for task construction.
- **中**: 我们使用一个强大的全模态模型（Gemini-3-Flash）从每种模态中提取细粒度、时间感知的信号用于任务构建。

- **EN**: For videos, we split each video into clips of at most 60 seconds to capture subtle temporal details, and generate both clip-level and full-video descriptions covering scenes, events, and non-speech ambient sounds.
- **中**: 对于视频，我们将每个视频分割为最多 60 秒的片段以捕获细微的时间细节，并生成覆盖场景、事件和非语音环境声的片段级和全视频描述。

- **EN**: For audio, we run timestamped automatic speech recognition (ASR), speaker diarization, and audio event detection; we also tag non-speech acoustic environments and produce global audio summaries.
- **中**: 对于音频，我们运行带时间戳的自动语音识别（ASR）、说话人分离和音频事件检测；还标注非语音声学环境并生成全局音频摘要。

- **EN**: For images, we apply optical character recognition (OCR), recognize objects and faces, and generate a holistic caption to summarize visual content.
- **中**: 对于图像，我们应用光学字符识别（OCR），识别物体和面孔，并生成整体描述以总结视觉内容。

- **EN**: To reliably synthesize complex multi-hop tasks, we build an omni-modal event graph that structures the discovered information into an explicit graph for each sample.
- **中**: 为可靠地合成复杂的多跳任务，我们构建了一个全模态事件图，将发现的信息结构化为每个样本的显式图。

- **EN**: This graph serves as the backbone of our event-graph-driven construction pipeline, enabling systematic evidence expansion and controllable information fuzzification for QA generation.
- **中**: 该图作为事件图驱动构建流水线的骨架，支持系统性的证据扩展和用于问答生成的可控信息模糊化。

- **EN**: Using the extracted information, we leverage a strong reasoning agent DeepSeek-V3.2 to automatically build an event graph that represents entities/events and their cross-modal relations.
- **中**: 利用提取的信息，我们借助强推理智能体 DeepSeek-V3.2 自动构建表示实体/事件及其跨模态关系的事件图。

- **EN**: Importantly, real-world logic is rarely a simple linear chain; it often exhibits branching (one-to-many), cascading (sequential), and mixed topologies.
- **中**: 重要的是，现实世界的逻辑很少是简单的线性链；它经常表现出分支（一对多）、级联（顺序）和混合拓扑结构。

- **EN**: The graph representation captures such structures and supports reliable synthesis of logically consistent, challenging tasks.
- **中**: 图表示能够捕获这些结构，并支持可靠地合成逻辑一致、具有挑战性的任务。

- **EN**: Given an initial event graph, we introduce Agentic Event Graph Expansion to proactively discover missing evidence and create tasks that truly require cross-modal association and external tool use.
- **中**: 给定初始事件图，我们引入智能体事件图扩展来主动发现缺失证据，创建真正需要跨模态关联和外部工具使用的任务。

- **EN**: To convert expanded graphs into truly challenging tasks, we propose QA generation via event fuzzification.
- **中**: 为将扩展后的图转化为真正具有挑战性的任务，我们提出通过事件模糊化进行问答生成。

- **EN**: Directly querying a graph node often reduces to trivial fact lookup. Instead, we select specific nodes/edges along long reasoning paths and apply fuzzy entities to mask or abstract key information.
- **中**: 直接查询图节点通常会退化为简单的事实查找。相反，我们沿长推理路径选择特定节点/边，并应用模糊实体来掩蔽或抽象关键信息。

- **EN**: This forces models to traverse the full logical path and integrate multi-source, multi-modal evidence to derive a unique answer.
- **中**: 这迫使模型遍历完整的逻辑路径并整合多来源、多模态的证据以得出唯一答案。

- **EN**: As shown in Figure 3, OmniGAIA comprises 360 omni-modal agentic tasks across 9 real-world domains, intentionally designed to stress long-horizon perception and tool-integrated reasoning.
- **中**: 如图 3 所示，OmniGAIA 包含 360 个全模态智能体任务，覆盖 9 个真实领域，刻意设计以压力测试长程感知和工具集成推理。

### 4 OmniAtlas: Omni-Modal Foundation Agent（全模态基础智能体）

- **EN**: In this section, we introduce OmniAtlas, a native omni-modal foundation agent that unifies vision, audio, and language perception with long-horizon reasoning and autonomous tool use.
- **中**: 在本节中，我们介绍 OmniAtlas——一种统一视觉、音频和语言感知与长程推理和自主工具使用的原生全模态基础智能体。

- **EN**: To enable OmniAtlas to acquire external knowledge and handle complex tasks, we integrate tools like web search, page browser, and code executor.
- **中**: 为使 OmniAtlas 能够获取外部知识并处理复杂任务，我们集成了网页搜索、网页浏览器和代码执行器等工具。

- **EN**: The agent adopts a tool-integrated reasoning paradigm, autonomously switching between internal reasoning and tool usage as needed.
- **中**: 该智能体采用工具集成推理范式，根据需要在内部推理和工具使用之间自主切换。

- **EN**: Formally, we define an agent trajectory as τ = [(s_t, a_t, o_t)] from t=0 to T, where s_t denotes the reasoning thought at step t, a_t the action (either a tool call or a final response), and o_t the observation returned by the tool.
- **中**: 形式化地，我们将智能体轨迹定义为 τ = [(s_t, a_t, o_t)]（t 从 0 到 T），其中 s_t 表示第 t 步的推理思考，a_t 为动作（工具调用或最终响应），o_t 为工具返回的观察结果。

- **EN**: For samples requiring active perception, the model can generate specialized tool calls to segment or re-examine specific portions of the media, rather than relying on a blanket downsampled input.
- **中**: 对于需要主动感知的样本，模型可以生成专门的工具调用来分割或重新检查媒体的特定部分，而不是依赖一刀切的下采样输入。

- **EN**: We synthesize training trajectories through a structured multi-step process.
- **中**: 我们通过结构化的多步过程合成训练轨迹。

- **EN**: We then propose OmniDPO for fine-grained trajectory-level preference learning that identifies and corrects the first error in a failed trajectory.
- **中**: 随后我们提出 OmniDPO，用于细粒度轨迹级偏好学习，能识别并纠正失败轨迹中的第一个错误。

### 5 Experiments（实验）

- **EN**: We evaluate both existing omni-modal models and our OmniAtlas agent on OmniGAIA.
- **中**: 我们在 OmniGAIA 上评估了现有全模态模型和我们的 OmniAtlas 智能体。

- **EN**: OmniGAIA is highly challenging for current models. Gemini-3-Pro achieves the highest score (62.5 Pass@1), while open-source models lag significantly.
- **中**: OmniGAIA 对当前模型极具挑战性。Gemini-3-Pro 取得最高分（62.5 Pass@1），而开源模型明显落后。

### 6 Conclusion（结论）

- **EN**: We introduce OmniGAIA, a benchmark for native omni-modal agents that requires multi-hop reasoning and multi-turn tool use over video-with-audio and image+audio inputs.
- **中**: 我们引入 OmniGAIA，一个面向原生全模态智能体的基准，要求在「视频+音频」和「图像+音频」输入上进行多跳推理和多轮工具使用。

- **EN**: Experiments show OmniGAIA remains challenging for current models, and that effective tool-use and long-horizon reasoning—rather than parameter scaling alone—are decisive bottlenecks.
- **中**: 实验表明 OmniGAIA 对当前模型仍极具挑战性，有效的工具使用和长程推理——而非仅靠参数规模增长——是决定性的瓶颈。

- **EN**: Our OmniAtlas recipe improves Qwen3-Omni from 13.3 to 20.8 Pass@1 while reducing tool-use and reasoning failures.
- **中**: 我们的 OmniAtlas 方案将 Qwen3-Omni 从 13.3 提升至 20.8 Pass@1，同时降低了工具使用和推理失败率。

---

## 第一章：方法核心

### 1. 方法动机

**驱动力**：人类的认知能力本质上是全模态的——我们同时通过视觉、听觉和语言来理解世界，并将这些感知与推理和工具使用（如搜索信息、计算数据）结合起来完成复杂任务。然而，当前的多模态 LLM 研究仍以**双模态交互**（如视觉-语言或音频-语言）为主导，这意味着模型只能处理两种模态的组合，无法处理现实世界中视觉、音频和语言紧密交织的场景。例如，一个旅行视频中，画面显示某座桥，旁白提到它让人想起电影《蓝调兄弟》——要回答"这座桥叫什么？拍摄《蓝调兄弟》时它已经矗立了多少年？"这类问题，模型需要同时理解视频画面（视觉感知）、语音内容（音频感知）、进行网络搜索（工具使用）、最后做算术运算（代码执行）。

**现有方法的具体局限性**：
1. **双模态局限**：主流多模态 LLM 如 GPT-4V、LLaVA 等主要处理图文或音文对，缺乏在同一推理链中融合视觉+音频+语言的能力。
2. **感知中心化**：即使有少数全模态模型（如 Qwen3-Omni），它们主要强调感知能力（"看到了什么""听到了什么"），而非将感知与长程推理和外部工具使用有机结合。
3. **评估缺失**：现有基准（OmniBench、WorldSense、UNO-Bench）多为选择题、短媒体、单轮任务，无法评估需要多跳推理、多轮工具调用、开放式答案的真实场景能力。OmniBench 上 Qwen3-Omni 可以达到 58.4（选择题），但在 OmniGAIA 上仅有 13.3（开放式），说明感知能力强不代表智能体能力强。
4. **训练方法薄弱**：缺乏针对全模态智能体的系统训练方案，开源模型在工具使用上几乎是"零基础"——Qwen3-Omni 平均每个任务仅调用 0.2 次工具。

**研究假设**：(1) 通过全模态事件图可以结构化地构建复杂的跨模态多跳任务，并确保任务逻辑一致且可解；(2) 后见之明引导的树搜索 + OmniDPO 细粒度纠错可以有效训练全模态智能体的工具使用和推理能力。

### 2. 方法设计

本文包含两个核心贡献：**OmniGAIA 基准**（任务构建方法）和 **OmniAtlas 智能体**（训练与推理方法）。

#### 2.1 OmniGAIA 基准构建 Pipeline

**总体流程**：原始媒体 → 细粒度信号挖掘 → 初始事件图构建 → 智能体驱动的事件图扩展 → 事件模糊化生成 QA → 质量检验（LLM + 人工）

**Step 1: 数据收集**  
从两种设置收集数据：(a) 带音频的视频（来自 FineVideo 43K 视频、LongVideoBench ~1K 视频、LongVideo-Reason ~1K 视频）；(b) 图像+音频配对（COCO 2017 图像 122K 张 + FineVideo 音频轨道）。视频涵盖广泛领域，时长从 20 秒到 2352 秒不等。**为何要双设置**：现实场景既有「一段视频自带音轨」的连续多模态，也有「一张图配一段音频」的离散组合；两种设置一起覆盖，才能全面考察模型在不同模态组合与时长下的表现。

**Step 2: 有价值信息发现（Discovering Valuable Information）**  
本步目的是把「媒体里有什么」变成机器可用的结构化信号，供后续建事件图使用。使用 Gemini-3-Flash 从每种模态提取细粒度、时间感知的信号：
- **视频**：将视频分割为 ≤60 秒的片段，生成片段级和全视频描述（覆盖场景、事件、非语音环境声）。分割的目的是确保捕获细微的时间细节，避免信息在长视频中被稀释。
- **音频**：运行带时间戳的 ASR（自动语音识别）、说话人分离和音频事件检测；标注非语音声学环境（如街道、室内、体育场、自然）并生成全局音频摘要。时间戳信息对于后续将音频事件与视觉事件对齐至关重要。
- **图像**：应用 OCR、物体识别、人脸识别，生成整体描述。这些信号为构建事件图提供了丰富的节点候选。

**Step 3: 全模态事件图构建（Omni-modal Event Graph Construction）**  
本步把前面挖到的「视频里有什么、音频里说了什么、图像里有什么」等信息，整理成一张**事件图**，方便后面自动出题。可以通俗理解为：事件图就像一张「知识地图」，上面有各种「点」（实体或事件）和「线」（它们之间的关系），且这些点可以来自不同模态。具体做法是：

1. **输入**：把 Step 2 得到的多模态信号（视频片段描述、ASR 文本、说话人/音频事件、图像 OCR/物体/描述等）一并交给 DeepSeek-V3.2。
2. **建图**：模型自动识别「实体」（如某座桥、某部电影名、某个人）和「事件」（如「旁白提到桥」「画面出现某地」），并在它们之间建立**跨模态关系边**（例如「画面中的桥」—「旁白提到的电影」—「该电影拍摄年份」）。
3. **为何用图而不是线性链**：现实中的逻辑往往不是 A→B→C 一条线，而是有分支（一个原因导致多个结果）、级联（多步顺序）或混合。图结构能自然表达这些拓扑，后续生成的多跳问题才能逻辑一致、有据可查。
4. **效果**：得到每个样本对应的「初始事件图」，为下一步的扩展和出题提供骨架。

**Step 4: 智能体驱动的事件图扩展（Agentic Event Graph Expansion）**  
初始事件图往往只包含「媒体里直接能看到的/听到的」信息，要出真正需要「查资料、算数」的多跳题，必须把图往外「长」——补上外部知识、时间线、数值等。做法是让一个**探索智能体**（仍用 DeepSeek-V3.2）像人做题一样：边推理、边查资料、边把新证据写进图里。具体步骤是：

1. **设定目标**：在已有事件图的基础上，智能体要判断「还缺什么信息才能构造一道有挑战的多跳题」。例如图中已有「视频里提到某座桥」「旁白说想起某部电影」，但还没有「桥的名字」「电影拍摄年份」，就需要去查。
2. **工具调用**：智能体可以自主选择调用以下工具之一或组合：  
   - **跨模态源链接**（`search_related_{video/audio/image}_info`）：从已有媒体库中再找相关视频/音频/图像片段，补足当前样本内跨模态关联；  
   - **网络知识**（`web_search` + `page_browser`）：搜网页、读页面，获取可验证的外部事实（如建造年份、上映时间）；  
   - **外部视觉**（`web_image_search` + `visual_question_answering`）：搜图并对其做视觉问答，把结果作为新节点加入图；  
   - **计算**（`code_executor`）：做算术、统计，把数值结果也作为图中的节点（如「桥的年龄 = 某年 − 建造年份」）。
3. **更新图结构**：每次工具返回后，智能体把新得到的事实整理成新的节点和边，**更新事件图**，使图的「信息边界」扩大。
4. **质量把关**：扩展完后会做图验证（事实是否一致、来源是否可靠）以及 LLM 自检（推理是否合理），必要时用工具再核实，保证扩展后的图能支撑高质量、可验证的多跳问答。

**Step 5: 事件模糊化生成 QA（Event Fuzzification）**  
如果题目直接问图里某个节点的具体值（如「这座桥叫什么？」「建于哪年？」），模型只需在图上「查表」就能答对，无法考察多跳推理和跨模态整合。**事件模糊化**的作用就是：把题目问法改得「绕一点」，让模型必须**先理清整条推理链、再结合多模态和工具证据**才能答对。具体做法是：

1. **选路径**：在扩展后的事件图中，挑出一条或几条**长推理路径**（例如：视频里的桥 → 旁白提到的电影 → 电影上映年份 → 桥的建造年份 → 计算「电影上映时桥已存在多少年」）。
2. **模糊关键节点/边**：对路径上的部分节点或边做「脱敏」——例如把具体桥名换成「视频中提到的那座让说话者想起某部电影的桥」，把具体电影名换成「那部电影」，这样模型无法直接查图得到答案，必须先从视频/音频里识别出桥和电影，再通过搜索/计算得到年份并算出最终数值。
3. **生成问句**：根据模糊后的路径，用自然语言生成一道多跳问答题（如「视频中提到的那座让说话者想起某部电影的桥，在拍摄那部电影时已矗立了多少年？」），确保答案唯一、可验证。
4. **效果**：题目既保留真实世界的复杂逻辑，又强制模型完成「感知 → 关联 → 检索 → 计算」的完整链路，而不是单点查表。

**Step 6: 质量检验**  
三层质量保障：(1) **LLM 筛选**：由 DeepSeek-V3.2 和 Gemini-3-Pro 组成评审委员会，自动评估每个 QA 对的自然性、全模态感知与工具使用的不可或缺性、答案正确性与唯一性；(2) **难度扩展**：对初步合格样本，可选地通过链接额外数据源或引入更复杂的计算步骤来增加难度；(3) **人工审核**：三名研究生级计算机科学审核员验证每个 QA 对。

#### 2.2 OmniAtlas 智能体设计

**工具集成推理（Tool-Integrated Reasoning, TIR）**  
TIR 可以通俗理解为：**边想边做、边做边想**——推理和调用工具写在同一条「时间线」里，模型先想一步、再决定是继续想还是调用工具，等工具返回后再根据结果继续想或再调工具。这样设计是因为真实任务往往需要「先搜再算」「先看某段再回答」，若把「所有推理」和「所有工具调用」拆成两阶段，模型无法根据工具结果动态改变思路。具体地：

1. **轨迹形式**：一条轨迹 τ 由多步组成，每步包含三项——s_t（本步的推理/思考）、a_t（本步动作：要么调用某个工具，要么直接给出最终答案）、o_t（若调用了工具，则为工具返回的观察结果）。即 τ = [(s_t, a_t, o_t)]_{t=0}^T。
2. **生成方式**：模型在每一步根据当前输入 x、历史思考与动作 (s_{<t}, a_{<t}) 以及历史工具返回 o_{<t}，生成本步的 (s_t, a_t)。因此工具的结果会立刻影响下一步的推理内容，形成「推理 ↔ 工具」的闭环。
3. **与固定工作流的区别**：传统做法可能是「先规划 N 步再执行」，TIR 不预设步骤数或顺序，完全由模型根据中间结果决定下一步，更贴近人类解题时的灵活决策。

**主动全模态感知（Active Omni-Modal Perception）**  
长视频、长音频若一次性全部塞进模型，要么超出上下文长度，要么被均匀下采样导致关键片段信息被稀释。**主动感知**的意思是：模型可以**按需**只「看」或「听」某一段时间或某一区域，而不是被动接受整段下采样结果。具体机制是：

1. **触发条件**：当任务或当前推理需要更细粒度的视觉/听觉信息时（例如「请确认 2:30–3:00 这段里是否出现了某座桥」），模型会生成一次**工具调用**，请求系统返回指定时间段的视频帧或音频片段。
2. **与「一刀切下采样」的对比**：若整段媒体被统一压缩成固定数量的 token，重要细节可能被平均掉；主动感知让模型先基于摘要或全局信息做初步判断，再在「怀疑」或「需要核实」的地方主动索要高分辨率片段，既省上下文又保关键信息。
3. **效果**：在长媒体任务上，模型可以像人一样「先扫一眼整体，再聚焦关键片段」，而不是被迫一次性处理全部原始数据。

**轨迹合成与监督学习（Trajectory Synthesis & SFT）**  
要训练模型「会推理、会调工具」，需要大量「正确示范」轨迹（即从问题到答案的完整步骤序列）。这些轨迹通过**后见之明引导的树搜索**自动合成，而不是完全靠人工写。原理可以概括为「用答案反推哪条路是对的，再把这些对的路径当训练数据」。具体步骤是：

1. **多候选探索**：用 DeepSeek-V3.2 作为探索代理，给定题目和媒体后，在**每一步**不只生成一个动作，而是生成多个候选（例如多种不同的推理+工具调用组合），形成一棵「探索树」。
2. **后见之明打分**：因为我们已经有人工标注的正确答案和参考解法，可以用「从当前步出发，能否最终走到正确答案」来给每个候选打分，从而知道哪条分支更有希望。
3. **采样的轨迹**：从树中选出那些**能走到正确答案**的完整路径，作为**正例轨迹**用于监督学习（SFT）；同时保留**失败轨迹**（走到错误或中途卡住），用于后面的 OmniDPO。
4. **步级质量把关**：除了「能否最终答对」，还用 Gemini-3 对轨迹中每一步的合理性做监督，避免虽然最终答对但中间步骤含糊或错误的轨迹进入训练集。

这样得到的正例轨迹既「工具使用丰富」又「推理链完整」，模型通过 SFT 学会模仿这些轨迹。

**OmniDPO（细粒度偏好学习）**  
SFT 只能学习「好的轨迹长什么样」，无法显式学习「错在哪里、该怎么改」。OmniDPO 用**偏好学习**在失败轨迹上做文章，但和传统 DPO 不同：不是整条轨迹打「好/坏」标签，而是**精确定位失败轨迹中的第一个错误**，并只对这一小段做「错 vs 纠错」的对比。通俗说就是：告诉模型「从这里开始你错了，应该这样改」，而不是笼统地说「这条轨迹不好」。具体做法是：

1. **输入**：一条**失败轨迹**（模型最终答错或中途出错），以及正确答案/参考解法。
2. **定位第一个错误**：用强模型（如 Gemini-3）分析这条轨迹，找出**第一个**出错的位置——可能是感知错了（看错/听错）、推理错了（逻辑推错）、或工具用错了（搜错关键词、算错）。只关心「第一个」，因为后面的错误往往是由它引发的。
3. **构造偏好对**：在该错误步之前，模型的行为是正确的；从这一步开始，原始轨迹是「错误延续」，再让 Gemini-3 生成一个「纠正后的后续」（正确感知/正确推理/正确工具调用）。于是得到两段前缀：**正例** = 正确前缀 + 纠错后的后续，**负例** = 正确前缀 + 原始错误后续。形成 (正, 负) 偏好对。
4. **训练目标**：用 DPO 等偏好学习目标，让模型对「正例前缀」的偏好高于「负例前缀」，从而学到「在这种情境下应该这样纠正」，而不是笼统地排斥整条轨迹。这样纠错信号更集中、学习更高效，实验中也验证了在 SFT 基础上能进一步提升 Pass@1 并降低各类错误率。

### 3. 与其他方法对比

| 对比维度 | OmniGAIA / OmniAtlas | 现有全模态基准 | 现有智能体方法 |
|----------|----------------------|----------------|----------------|
| 模态覆盖 | 视频+图像+音频 | 多数仅双模态（图+音或图+文） | 多数仅文本或图文 |
| 答案形式 | 开放式（需计算/搜索验证） | 多为选择题（MC） | 多为文本生成 |
| 工具使用 | 多轮工具集成（搜索/浏览/代码） | 不要求工具使用 | 部分支持但非全模态 |
| 推理深度 | 多跳（中位 6.5 步，最多 >10 步） | 单轮或浅层推理 | 变化较大 |
| 媒体时长 | 20-2352s（视频），20-657s（音频） | 多数 <60s | N/A |
| 训练方法 | 树搜索轨迹合成 + OmniDPO 细粒度纠错 | N/A | SFT / RL（多为粗粒度） |
| 主动感知 | 支持选择性查看/聆听特定片段 | N/A | 多为全量输入 |

### 4. 实验表现与优势

**基准难度**：OmniGAIA 对当前模型极具挑战性。具体数值如下：

| 模型 | 整体 Avg. | Easy | Medium | Hard |
|------|-----------|------|--------|------|
| Gemini-3-Pro | **62.5** | 78.7 | 57.5 | 47.4 |
| Gemini-3-Flash | 51.7 | 67.2 | 46.9 | 37.2 |
| Qwen3-Omni-30B | 13.3 | 19.7 | 10.6 | 9.0 |
| GPT-4.1 | 32.8 | 42.6 | 27.5 | 26.9 |
| OmniAtlas-30B (Ours) | **20.8** | 29.5 | 17.5 | 14.1 |
| OmniAtlas-7B (Ours) | **13.3** | 19.7 | 10.6 | 9.0→13.3 |

**关键发现**：
1. **商业 vs 开源差距巨大**：Gemini-3-Pro（62.5）远超所有开源模型（Qwen3-Omni-30B 仅 13.3），说明开源全模态模型在工具集成推理方面严重不足。
2. **OmniAtlas 训练有效**：OmniAtlas-SFT 将 Qwen3-Omni-30B 从 13.3 提升到 18.9，OmniDPO 进一步提升到 20.8（总提升 56%）。无效工具使用率从 81.1% 降至 59.4%。
3. **错误类型分析**：主要错误来源为无效工具使用（59.4%，OmniAtlas 后）和推理错误（64.4%）。视觉感知错误（30.3%）和音频感知错误（31.9%）相对较低。
4. **原生感知 vs 工具感知**：对强模型（Gemini-3-Flash），原生全模态感知（51.7）优于工具替代（50.0/43.3/46.4）且工具调用更少。对弱模型，工具感知在简单/中等任务上有帮助，但在困难任务上反而降低性能。
5. **工具调用行为**：Qwen3-Omni-30B 平均仅 0.2 次工具调用（几乎不使用工具），而 OmniAtlas-30B 大幅增加了工具使用频率。

### 5. 学习与应用

- **开源情况**：代码、数据集、模型和 Leaderboard 均已开源。
  - GitHub: https://github.com/RUC-NLPIR/OmniGAIA
  - HuggingFace: https://huggingface.co/collections/RUC-NLPIR/omnigaia
  - Leaderboard: https://huggingface.co/spaces/RUC-NLPIR/OmniGAIA-LeaderBoard
- **复现要点**：
  1. 基准构建需要 Gemini-3-Flash（信号提取）和 DeepSeek-V3.2（事件图构建与扩展）。
  2. OmniAtlas 训练需要：先用 DeepSeek-V3.2 和 Gemini-3 做树搜索轨迹合成，然后对开源全模态模型（如 Qwen3-Omni 或 Qwen2.5-Omni）进行 SFT + OmniDPO 微调。
  3. 推理时需要工具接口（网页搜索、浏览器、代码执行器）。
- **迁移建议**：OmniAtlas 训练方案可迁移到其他全模态/多模态模型上增强工具使用能力；OmniGAIA 基准可用于评估任何全模态智能体。事件图构建方法可扩展到更多领域和模态组合。

### 6. 总结

- **一句话**：全模态事件图基准 + 树搜索轨迹训练的原生全模态智能体。
- **速记 Pipeline**：多模态媒体 → Gemini-3 提取细粒度信号 → DeepSeek-V3.2 构建+扩展事件图 → 事件模糊化生成多跳 QA → OmniAtlas 工具集成推理（搜索/浏览/代码）→ 树搜索合成训练轨迹 → SFT + OmniDPO 纠错。

---

## 第二章：图与表

### Figure 1：OmniGAIA 任务示例

- **类型**：示例/效果图。
- **图中元素**：展示两个任务实例——(左) 图像+音频任务：一张伦敦反削减抗议的照片 + 一段描述 Rolling Jubilee 债务减免运动的音频，要求计算两事件之间的月数；(右) 视频+音频任务：一段参观 Joliet Iron Works 的视频，旁白提到一座可移动桥让他想起《蓝调兄弟》，要求找出桥名并计算拍摄时桥的年龄。每个示例都附有标注解法（多步骤，涉及视觉/音频分析 + 网络搜索 + 计算）。
- **与正文对应**：对应 Section 1 Introduction，直观说明 OmniGAIA 任务需要跨模态感知、多跳推理和外部工具使用。
- **解读**：这两个例子直观展示了 OmniGAIA 的核心挑战——模型不仅需要"看到"和"听到"，还需要通过搜索获取外部事实并进行计算，才能得出最终答案。

### Table 1：OmniGAIA 与现有基准对比

- **列名含义**：Benchmark（基准名）、Video/Image/Audio（是否支持该模态）、Multi-hop Reasoning（是否需要多跳推理）、External Tools（是否需要外部工具）、Multi-Domain（是否跨领域）、Video/Audio Duration（时长范围）、Answer Type（MC=选择题, Open=开放式）、Qwen3-Omni Accuracy（Qwen3-Omni 在该基准上的准确率）。
- **关键数据**：OmniGAIA 是唯一同时支持视频+图像+音频、多跳推理、外部工具、多领域、长媒体（视频 20-2352s、音频 20-657s）、开放式答案的基准。Qwen3-Omni 在 OmniGAIA 上仅 13.3，远低于在 OmniBench（58.4）或 Daily-Omni（75.8）上的表现。
- **论证作用**：证明现有基准的局限性（大多缺乏多跳推理和工具使用），以及 OmniGAIA 的独特价值和挑战性。

### Figure 2：OmniGAIA 构建流水线（架构图）

- **类型**：流程/架构图。
- **整体结构**：从左到右，分为 4 个主要阶段：(1) 数据收集 → (2) 有价值信息发现 → (3) 智能体全模态事件图构建 → (4) QA 生成与质量审核。
- **每个模块**：
  - **数据收集**：HuggingFace 公开数据源（FineVideo、LongVideoBench、LongVideo-Reason、COCO 2017），涵盖 100+ 领域和宽泛的视频/音频时长。
  - **信息发现**：对视频（事件与环境分析）、图像（理解+OCR+物体+人脸）、音频（ASR+说话人+事件检测）分别提取信号。
  - **事件图构建**：将发现的信息结构化为初始事件图 → 通过 DeepSeek-V3.2 推理+工具扩展 → 扩展事件图 → 图验证（包括 LLM 自反省+人工审核）。外部工具包括网络搜索/浏览器、代码执行器、视觉问答、跨模态检索。
  - **QA 生成**：事件模糊化（选择节点/边进行掩蔽或抽象）→ LLM + 人工验证（检查正确性、任务难度、答案唯一性）。
- **关键符号**：实线箭头表示数据流方向，虚线框表示外部工具接口，"Update"标注表示扩展过程中的迭代更新。
- **与 Method 对应**：对应 Section 3（Sections 3.1-3.6）完整的基准构建方法。
- **亮点**：事件图驱动的构建方法是本文最核心的贡献之一。它不仅能结构化多模态信息，还能通过智能体扩展发现新的推理路径，最终通过模糊化生成高难度的多跳问题。
- **改动**：相比现有基准的构建方法（通常是直接标注或 LLM 生成），OmniGAIA 引入了事件图作为中间表示、智能体驱动的图扩展、和事件模糊化三大新机制。
- **达成效果**：生成了 360 个高质量、逻辑一致、可验证的全模态多跳推理任务，最强商业模型仅 62.5%，开源模型仅 13.3%。

### Figure 3：OmniGAIA 统计信息

- **类型**：统计图（多子图）。
- **图中元素**：
  - (a) 左上：领域分布柱状图（9 个领域：Geography & Travel 19.2%, History & Society 18.6%, Technology 13.6%, Sports 10.3% 等）。
  - (a) 中上：问题词云。
  - (a) 右上：音频时长分布（p50=197s, p90=489s）和视频时长分布（p50=242s, p90=550s）。
  - (b) 左下：最需要的能力（Visual Perception 99.7%, Audio Perception 99.7%, Web search 98.6%, Multi-hop Planning 98.1%, Object Identification 91.1%, Code/Computation 74.4%, Contrastive Analysis 55.6%, Temporal Localization 31.7%, Geo-spatial Reasoning 21.7%）。
  - (b) 中下：难度分布（Easy 33.9%, Medium 44.4%, Hard 21.7%）和每任务步数（p50=6.5, p90=9.0）。
  - (b) 右下：工具分布（Search 98.3%, Code 68.3%, Browse 23.1%）和每任务工具数与图片数。
- **与正文对应**：Section 3.7 Statistics。
- **解读**：几乎所有任务都需要视觉+音频感知和网络搜索（≥98%），约 74% 需要代码计算，说明这是一个真正的"智能体"基准而非纯感知基准。中位步数 6.5 表示任务复杂度较高。

### Figure 4：OmniAtlas 训练策略（架构图）

- **类型**：架构/流程图。
- **整体结构**：分为左右两部分：(左) 轨迹合成与监督学习，(右) OmniDPO 细粒度错误纠正。
- **每个模块**：
  - **左侧——轨迹合成**：
    - 输入：Source Media + Annotations + Question
    - DeepSeek-V3.2 作为探索代理进行工具集成推理
    - 每步可分为 Exploration Step（探索性步骤，产生多候选）和 Vanilla Step（常规步骤）
    - Gemini-3 提供步级监督
    - 成功轨迹用于监督学习（SFT），失败轨迹丢弃或保留用于 OmniDPO
  - **右侧——OmniDPO**：
    - OmniAtlas 在训练后生成轨迹
    - Gemini-3 识别并纠正失败轨迹中的第一个错误
    - 生成正样本（纠正后）和负样本（原始错误）
    - 偏好学习优化
- **关键符号**：绿色实线箭头表示数据流，红色/黄色标注表示错误识别，蓝色标注表示纠正后的正确路径。
- **与 Method 对应**：Section 4（OmniAtlas 训练方法）。
- **亮点**：OmniDPO 的细粒度纠错——不是对整条轨迹打好/坏标签，而是精确定位第一个错误点并生成对比样本，使偏好学习更高效。
- **改动**：相比传统的 DPO/PPO，OmniDPO 在轨迹中精确定位错误并生成局部纠正，而非全局替换。
- **达成效果**：OmniDPO 在 SFT 基础上进一步提升 Pass@1（18.9→20.8），同时四类错误率全面下降。

### Table 2：主要实验结果

- **列名**：Method（方法）、模型名称、Difficulty Levels（Easy/Med./Hard）、Avg.（平均分）、Avg. Tool Calls（平均工具调用次数）。
- **关键数据**：
  - Gemini-3-Pro: 62.5 Avg., 10.1 Tool Calls
  - Gemini-3-Flash: 51.7 Avg., 4.4 Tool Calls
  - Qwen3-Omni-30B: 13.3 Avg., 0.2 Tool Calls（几乎不使用工具）
  - OmniAtlas-30B: 20.8 Avg., 3.6 Tool Calls
- **论证作用**：(1) 证明 OmniGAIA 的挑战性；(2) 证明 OmniAtlas 训练方案显著提升开源模型性能；(3) 揭示开源模型的工具使用严重不足。

### Table 3：原生感知 vs 工具感知消融实验

- **内容**：在 Gemini-3 和 Qwen-3 模型家族内，对比原生全模态感知、用音频感知模型作为工具、用视觉感知模型作为工具、双模态均用工具等配置下的性能。
- **关键发现**：
  - Gemini-3-Flash 原生感知最优（51.7）且工具调用最少（4.4）
  - Qwen-3-Omni 在 Easy/Med. 上可通过工具感知获得小幅提升，但 Hard 上下降
  - 工具感知总是增加交互成本

### Table 4：OmniAtlas 训练效果消融

- **列名**：Method、Visual Percept.（视觉感知错误率↓）、Audio Percept.（音频感知错误率↓）、Ineffect. Tool-Use（无效工具使用率↓）、Reason. Error（推理错误率↓）、Perform.（Pass@1↑）。
- **关键数据（Qwen-3-Omni-30B 组）**：
  - 基线：31.7 / 33.9 / 81.1 / 79.7 / 13.3
  - +OmniAtlas-SFT：32.2 / 35.8 / 65.3 / 68.1 / 18.9
  - +OmniDPO：30.3 / 31.9 / 59.4 / 64.4 / 20.8
- **论证作用**：SFT 贡献主要性能提升（无效工具使用 81.1%→65.3%），OmniDPO 进一步全面改善四类错误。

### Figure 5：错误类型分析（热力图）

- **类型**：热力图 / 统计图。
- **内容**：展示不同模型在四种错误类型上的细粒度分布。
- **与正文对应**：Section 5.3 Fine-Grained Error Analysis。

### Figure 6：工具使用行为分析

- **类型**：分布图。
- **内容**：展示不同模型的工具调用次数分布。
- **关键发现**：Qwen3-Omni 呈现极端的"不调用"模式，Gemini-3-Pro 呈现广泛的调用分布，OmniAtlas 从"不调用"转向更活跃的工具使用。

---

## 第三章：详细总结

- **基本信息**：OmniGAIA: Towards Native Omni-Modal AI Agents；Xiaoxi Li 等（中国人民大学、小红书）；arXiv:2602.22897；2026-02-26 提交。

- **技术背景与挑战**：人类认知需要全模态感知+推理+工具使用的统一，但当前多模态 LLM 以双模态交互为主，缺乏面向全模态智能体的评估基准和训练方法，现有基准主要是短媒体、选择题、感知导向。

- **论文亮点与贡献**：(1) OmniGAIA 首个需要多跳推理+多轮工具使用+开放式答案的全模态智能体基准（360 任务，9 领域，最强模型仅 62.5%）；(2) 事件图驱动的构建流水线，确保任务复杂且逻辑一致；(3) OmniAtlas 原生全模态智能体+主动感知+后见之明树搜索训练+OmniDPO 细粒度纠错，将 Qwen3-Omni 从 13.3 提升至 20.8。

- **方法详解**（步骤化，每步讲清是什么、为什么、怎么做、效果）：
  1. **数据收集**：从 FineVideo（43K 视频）、LongVideoBench / LongVideo-Reason（各约 1K 长视频）以及 COCO 2017（122K 图像）收集原始媒体；同时用 FineVideo 的音频轨道与 COCO 图像配对，构成「图像+音频」设置。双设置（视频+音频 / 图像+音频）保证基准在模态组合和时长上的多样性，能覆盖从几十秒到几十分钟的媒体。
  2. **信号挖掘**：用 Gemini-3-Flash 从每种模态中抽取细粒度、带时间信息的信号，为后续建图提供「原材料」。视频按 ≤60 秒切片段并生成片段级与全片描述（场景、事件、环境声）；音频做带时间戳的 ASR、说话人分离、音频事件检测及声学环境标注；图像做 OCR、物体与人脸识别并生成整体描述。这样做的目的是把「看到/听到什么」显式化，便于后续自动建图和出题。
  3. **事件图构建**：把上一步得到的多模态信号交给 DeepSeek-V3.2，自动建一张**事件图**——节点是实体或事件，边是跨模态关系（如「画面中的桥」与「旁白提到的电影」相连）。图能表达分支、级联等复杂逻辑，比线性链更贴近真实任务，为多跳问答提供逻辑一致的结构化骨架。
  4. **图扩展**：在初始事件图上，用 DeepSeek-V3.2 作为探索智能体，按需调用跨模态检索、网页搜索/浏览、图像搜索与 VQA、代码执行等工具，把外部知识和数值结果补充进图。智能体自主决定何时调用何种工具，每次调用后更新图结构并做事实与合理性校验，使图能支撑「必须查资料、算数才能答」的多跳题。
  5. **事件模糊化**：在扩展后的图中选取长推理路径，对路径上部分节点/边做模糊化（如用「视频里提到的那座桥」替代具体桥名），再据此生成自然语言问题。这样题目无法通过单点查图作答，必须完整走完「感知→关联→检索→计算」的链路，从而得到高难、可验证的多跳 QA。
  6. **质量检验**：先由 LLM（DeepSeek-V3.2 + Gemini-3-Pro）自动筛选题目的自然性、对全模态与工具使用的必要性、答案正确性与唯一性；通过者可再经难度扩展（加数据源或计算步骤）；最后由三名审核员做人工校验，保证 360 个任务均可解且答案唯一。
  7. **OmniAtlas 推理**：智能体采用**工具集成推理（TIR）**——在同一生成序列中交替输出「思考」与「工具调用/最终答案」，根据工具返回动态决定下一步，而非固定工作流。对长媒体支持**主动全模态感知**：按需请求特定时间段或区域的视频/音频片段，避免整段下采样导致的信息稀释。
  8. **轨迹合成**：用**后见之明引导的树搜索**生成训练轨迹：DeepSeek-V3.2 在每步生成多候选推理+动作，结合标注答案判断哪些分支能通向正确解，采样的成功路径作为正例轨迹；Gemini-3 对每一步做质量监督，保证轨迹既工具使用充分又推理合理。
  9. **SFT 训练**：用上述成功轨迹对开源全模态模型（如 Qwen3-Omni）做监督微调，使模型学会「何时推理、何时调工具、如何根据工具结果继续推理」，显著提升工具调用频率与任务通过率（如 13.3→18.9 Pass@1）。
  10. **OmniDPO**：在失败轨迹上定位**第一个错误**（感知/推理/工具使用），由强模型生成纠错后的后续，形成「正确前缀+纠错后续」为正例、「正确前缀+原始错误后续」为负例的偏好对；用偏好学习让模型学会在错误发生处如何纠正，比整条轨迹打好坏标签更精细，在 SFT 基础上进一步提升性能（18.9→20.8）并降低四类错误率。

- **实验设置**：评估 Gemini-3-Pro/Flash、GPT-4.1、Claude-4-Sonnet、Qwen3-Omni-30B/7B、Qwen2.5-Omni-7B 等模型；消融实验包括原生/工具感知对比、SFT/OmniDPO 逐步贡献、工具使用行为分析、错误类型分析。

- **实验结果分析**：
  - 商业模型（Gemini-3-Pro 62.5）远超开源（Qwen3-Omni 13.3），差距近 5 倍。
  - OmniAtlas 训练方案有效：13.3→20.8（+56%），无效工具使用率从 81.1%→59.4%。
  - SFT 贡献主要提升，OmniDPO 进一步全面改善。
  - 原生全模态感知对强模型更优，工具感知仅作为弱模型的补充手段。
  - 工具调用次数与性能非线性：>10 次调用不保证成功，说明存在"乱调"行为。

- **结论**：
  1. OmniGAIA 是首个真正评估全模态智能体能力的基准，揭示了当前模型在工具使用和长程推理上的严重不足。
  2. 事件图驱动的构建方法可扩展、可控，能系统性地生成高质量复杂任务。
  3. OmniAtlas 的训练方案（树搜索+SFT+OmniDPO）显著提升开源模型的工具使用能力。
  4. 有效的工具使用和长程推理——而非仅参数规模——是全模态智能体的决定性瓶颈。
  5. 一句话总结：全模态感知+推理+工具使用的统一评估与训练。

---

## 自检表

| # | 检查项 | 结果 |
|---|--------|------|
| C1 | 技术报告来源 | ✅ 列出 arXiv、PDF、GitHub、HuggingFace、Leaderboard 链接 |
| C2 | 逐句翻译覆盖度 | ✅ 覆盖 Abstract、Introduction、Related Work、Method（Sections 3-4）、Experiments、Conclusion 主要句子 |
| C3 | 逐句翻译质量 | ✅ 忠实原文、术语准确、中文通顺 |
| C4 | 方法设计详细度 | ✅ 每个模块（事件图构建6步、OmniAtlas 4个核心组件）均有 ≥3 句详细解释 |
| C5 | 公式解释完整度 | ✅ 轨迹定义公式 τ = [(s_t, a_t, o_t)] 已逐符号解释 |
| C6 | 图表完整性 | ✅ Figure 1-6 和 Table 1-4 均有对应小节 |
| C7 | 架构图规范 | ✅ Figure 2（构建流水线）和 Figure 4（训练策略）满足架构图 8 要素 |
| C8 | 数值证据 | ✅ 引用了 Pass@1、工具调用次数、错误率等具体数值 |
| C9 | 解释深度 | ✅ 事件图、模糊化、OmniDPO 等核心概念均有详细的背景+原理解释 |
| C10 | 报告自洽性 | ✅ 仅读报告可完整理解论文核心方法和实验结论 |

**自检通过，全部 10 项达标。**

---

## 本次检查与改写说明

本次对「方法原理」相关部分做了针对性改写，以满足逐步拆解、讲透、通俗易懂与详解充分四项标准。**具体改动**：（1）**第一章 §2.1 基准构建 Pipeline**：对 Step 3 全模态事件图构建、Step 4 智能体事件图扩展、Step 5 事件模糊化，补充了 1–4 的步骤化拆解，并加入「事件图」的通俗类比（知识地图、点与边）及「为什么用图」「模糊化如何迫使模型走完整链路」等原理说明；（2）**第一章 §2.2 OmniAtlas 智能体设计**：对 TIR、主动全模态感知、后见之明树搜索轨迹合成、OmniDPO 四个模块，分别重写为多步骤表述，并增加「是什么、为什么这样设计、具体怎么做、效果如何」以及通俗表述（如「边想边做」「用答案反推哪条路对」「从这里开始你错了、应该这样改」）；（3）**第三章 方法详解**：将原先每条仅一句的 1–10 条扩展为每条约 3–5 句的实质性解释，保留步骤顺序并补全原理与效果说明。整体未改动报告原有章节与图表结构，仅增强方法原理部分的拆解深度与可读性。

---

*本报告基于 arXiv:2602.22897 PDF 全文撰写，严格按 paper-read 技能三章结构+自检流程产出。*

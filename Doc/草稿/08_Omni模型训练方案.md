# 08 Omni 模型训练方案（详细版）

> 基于 11 篇已精读论文（MiniCPM-V-4.5、Qwen3-Omni、Ola、Baichuan-Omni、LongCat-Flash-Omni、MiniCPM-SALA、OmniVinci、OmniGAIA 等）的训练方案、数据策略与评测基准整理，以 **Qwen3-LLM + SigLIP + Whisper-medium + CosyVoice2** 为目标架构，给出完整的 Omni 模型训练方案。
> 
> **版本说明**：本版本为详细实施版，包含每个数据集的完整技术细节、评测集的构造与提升策略、可落地的训练配置与资源估算，以及突破方向的详细实施步骤。

---

## 目录

1. [数据收集与构造](#1-数据收集与构造)
   - 1.1 [开源数据集详细介绍](#11-开源数据集详细介绍)
   - 1.2 [数据组织与生产方案](#12-数据组织与生产方案)
2. [评测构建](#2-评测构建)
   - 2.1 [评测集详细分析](#21-评测集详细分析)
3. [训练方案](#3-训练方案)
   - 3.1 [可落地的六阶段训练方案](#31-可落地的六阶段训练方案)
4. [突破方向详细实施方案](#4-突破方向详细实施方案)
5. [逐篇论文对比反思与方案完善](#5-逐篇论文对比反思与方案完善)
6. [综合反思与终极方案](#6-综合反思与终极方案)
7. [完善版方案（可落地实施）](#7-完善版方案可落地实施)
   - 7.1 [完善版训练方案](#71-完善版训练方案)
   - 7.2 [完善版数据方案](#72-完善版数据方案)
   - 7.3 [完善版评测方案](#73-完善版评测方案)
   - 7.4 [技术可行性与优势总结](#74-技术可行性与优势总结)

---

## 1. 数据收集与构造

### 1.1 开源数据集详细介绍

#### 1.1.1 纯文本数据集

##### SlimPajama
- **数据量**：627B token（去重后约 500B+）
- **数据来源**：RedPajama、C4、Wikipedia、GitHub、BookCorpus 等多源混合
- **收集方法**：
  - 从 Common Crawl 快照中提取原始 HTML
  - 使用 jusText 进行正文提取
  - 使用 MinHash LSH 进行全局去重（14 组哈希函数，Jaccard 阈值 0.8）
  - 质量过滤：使用困惑度评分、语言检测（fastText）、毒性过滤
- **数据格式**：JSON Lines，每条包含 `text`、`meta`（来源、日期、URL）
- **组织方式**：按来源分片（C4、Wikipedia、GitHub 等），每片约 10GB
- **许可证**：Apache 2.0（不同子集可能有差异）
- **适用阶段**：S2 联合预训练（维持语言能力）、SFT（约 10% 混合防退化）

##### FineWeb
- **数据量**：15T+ tokens（2024 版本），持续增长
- **数据来源**：Common Crawl 2013–2024 快照
- **收集方法**：
  - 使用 datatrove 管道进行大规模清洗
  - 语言识别：fastText lid.176.bin，保留英语及多语言高质量内容
  - 去重：MinHash + SimHash，全局去重率约 70%
  - 质量过滤：使用自定义质量评分模型（基于教育内容训练）
  - 毒性/PII 过滤：使用多分类器识别有害内容
- **数据格式**：Parquet 格式，含 `text`、`id`、`metadata`（timestamp, URL, language_score）
- **组织方式**：按 CC 快照月份分片，支持流式读取
- **许可证**：ODC-BY 1.0
- **适用阶段**：S2 预训练（高质量通用语料）

##### CulturaX
- **数据量**：多语言，覆盖 167 种语言，总量约 6T tokens
- **数据来源**：Common Crawl、mC4 等
- **收集方法**：
  - 语言特定清洗管道
  - 每个语言独立的质量评估与过滤
  - 保留文化多样性内容（传统文本、地方新闻等）
- **数据格式**：JSON Lines，含语言标签、质量分数
- **许可证**：ODC-BY
- **适用阶段**：多语言 Omni 模型训练

---

#### 1.1.2 图文配对 / 交错数据集

##### OmniCorpus（ICLR 2025 Spotlight）
- **数据量**：
  - 总计：8.6B 图像 + 1,696B（1.7T）文本 token
  - 是 MMC4/OBELICS 的 15 倍规模
  - 三部分组成：
    - **OmniCorpus-CC**：Common Crawl 提取，约 7B 图像
    - **OmniCorpus-CW**：中文网页，约 1B 图像
    - **OmniCorpus-YT**：YouTube 视频帧提取，约 0.6B 图像
- **数据来源**：
  - 英文/非英文网站（CC）
  - 中文网站（CW）
  - YouTube 视频帧（YT）
- **收集方法**：
  - **数据引擎**：高效过滤与提取大规模高质量文档
  - **图文交错**：保持自然文档格式，图像与文本按原始阅读顺序交错
  - **可降级性**：可降级为纯文本语料或图文对（灵活支持不同训练需求）
  - **质量控制**：基于 CLIP 相似度、图像分辨率、OCR 可读性等过滤
- **数据格式**：
  - 图像：原始分辨率保存，WebP/PNG 格式
  - 文本：与图像交错的 Markdown-like 格式
  - 元数据：URL、抓取时间、语言、CLIP 相似度分数
- **组织方式**：
  - HuggingFace Dataset 格式
  - 分片：按域名/来源分片，每片含数千到数万文档
- **下载方式**：
  - GitHub: https://github.com/OpenGVLab/OmniCorpus
  - HuggingFace: https://huggingface.co/collections/OpenGVLab/omnicorpus
- **许可证**：CC0 / 各子集依原始来源
- **适用阶段**：
  - S1 对齐：图文 caption 数据
  - S2 预训练：图文交错数据维持多模态能力
  - 文档统一学习：动态损坏范式可应用于此

##### LAION-5B
- **数据量**：5.85B 图文对
- **数据来源**：Common Crawl 抓取的图像及周围文本
- **收集方法**：
  - 从 CC 提取图像 URL 及 ALT 文本
  - 使用 CLIP 过滤：图像-文本相似度阈值
  - 去重：图像感知哈希去重
  - 安全过滤：NSFW 内容过滤
- **数据格式**：
  - 图像：原始 URL（需下载），提供 downsampled 版本
  - 文本：原始 ALT 文本或周围文本
  - 元数据：CLIP 相似度、图像尺寸、URL
- **组织方式**：Parquet 文件，按相似度分桶
- **下载**：https://laion.ai/blog/laion-5b/
- **许可证**：Creative Commons 混合
- **适用阶段**：S1 大规模图文对齐预训练

##### DataComp-1B
- **数据量**：1.28B 图文对（12.8 亿）
- **特点**：竞争性数据集，提供标准化评估框架
- **收集方法**：
  - 从 Common Crawl 提取
  - 多维度质量评估（CLIP 分数、分辨率、文本长度等）
- **适用阶段**：S1 对齐，可与 LAION-5B 互补使用

##### ShareGPT4V / ALLaVA-Instruct
- **数据量**：
  - ShareGPT4V：约 100K 高质量图文对话
  - ALLaVA：约 1M 指令数据
- **来源**：GPT-4V 生成的指令遵循数据
- **收集方法**：
  - 使用 GPT-4V 对图文对生成详细描述
  - 构造多轮对话、指令遵循、推理任务
- **数据格式**：
  - 图像文件
  - JSON：含 `conversations` 列表（role/content）
- **适用阶段**：SFT（高质量指令数据）

##### DocVQA / ChartQA / InfoVQA
- **数据量**：
  - DocVQA：约 12K 文档，50K QA 对
  - ChartQA：约 20K 图表，9K QA 对
  - InfoVQA：约 30K 信息图，15K QA 对
- **数据来源**：真实文档、图表、信息图
- **收集方法**：
  - 人工标注 QA 对
  - 涵盖 OCR、推理、数值计算等
- **数据格式**：
  - 图像：文档/图表扫描件
  - JSON：bounding boxes、QA 对、答案类型
- **适用阶段**：
  - Phase 2 编码器微调（OCR 密集数据）
  - SFT（文档理解任务）

##### OCRBench 训练集
- **数据量**：约 10M 文本定位与识别样本
- **来源**：合成 + 真实文档
- **适用阶段**：Phase 2 OCR 能力增强

---

#### 1.1.3 视频–文本数据集

##### OmniCorpus-YT
- **数据量**：约 0.6B 视频帧 + 字幕文本
- **来源**：YouTube 视频
- **收集方法**：
  - 从 YouTube 提取视频帧（每秒 1-2 帧）
  - 提取字幕/ASR 文本
  - 视频-文本对齐：基于时间戳
- **数据格式**：
  - 视频帧：JPEG/PNG，时间戳标注
  - 字幕：SRT/VTT 格式，含时间戳
- **适用阶段**：S2 视频-文本对齐预训练

##### VALID（Video-Audio-Large-Interleaved-Dataset）
- **数据量**：约 720,000 个 Creative Commons 许可 YouTube 视频
- **来源**：YouTube CC 许可视频
- **收集方法**：
  - 爬取 YouTube CC 视频
  - 提取多语言转录（Whisper 或其他 ASR）
  - 音视频-文本三模态对齐
- **数据格式**：
  - 视频片段：≤10 秒/段，可更长
  - 多语言转录文本
  - 视频帧 caption
- **模态覆盖**：视频帧 + 音频 + 多语言文本
- **组织方式**：
  - 每记录可含多个视频/音频/图像，交错组织
  - HuggingFace: ontocord/MixtureVitae-VALID
- **许可证**：CC 许可（依原始视频）
- **适用阶段**：
  - S2 三模态联合预训练
  - 跨模态对齐训练
- **状态**：预览版，持续上传中

##### FineVideo
- **数据量**：
  - 43,751 视频
  - 总计 3,400+ 小时（3.4k hours）
  - 122 内容类别
  - 600GB 数据
- **来源**：YouTube Commons
- **收集方法**：
  - **多阶段管道**：
    1. 原始数据集过滤（YouTube Commons）
    2. 视频动态性过滤（去除静态/低动态内容）
    3. 内容分类（122 类别）
    4. 内容选择
    5. 丰富标注生成
    6. 细粒度视频-元数据对齐
  - 标注内容：
    - 场景分割
    - 角色描述与列表
    - 活动描述
    - 情绪分析
    - 上下文相关性评分
    - 叙事进展细节
    - 音频-视觉相关性评分
    - 动态性评分
- **数据格式**：
  - 视频：MP4 格式，保留原始 FPS
  - 元数据：JSON，含帧率、角色列表、场景边界、QA 对
- **组织方式**：
  - HuggingFace Dataset: HuggingFaceFV/finevideo
  - 支持流式读取或全量下载
  - 可按类别过滤
- **许可证**：YouTube Commons 许可
- **适用阶段**：
  - OmniGAIA 等基准的数据源
  - S3 长上下文训练
  - 高质量视频理解训练
- **GitHub**：https://github.com/huggingface/fineVideo

##### LongVideoBench / LongVideo-Reason
- **数据量**：各约 1,000 视频
- **视频时长**：平均约 10 分钟，最长可达 1 小时
- **来源**：电影、纪录片、教育视频等
- **收集方法**：
  - 精选长视频内容
  - 人工标注长程 QA 对（需要理解整个视频才能回答）
- **数据格式**：
  - 视频分段
  - QA 对含时间戳引用
- **适用阶段**：S3 长上下文扩展、长视频理解评测

##### ShareGPT-Video / VideoChat2-IT / LLaVA-Video
- **数据量**：
  - ShareGPT-Video：约 10K 视频指令对
  - VideoChat2-IT：约 100K 指令
  - LLaVA-Video：约 200K 指令
- **来源**：GPT-4V 或其他模型生成的视频指令数据
- **收集方法**：
  - 视频采样（帧提取）
  - 使用强模型生成详细描述与 QA
  - 人工审核质量
- **数据格式**：
  - 视频帧序列
  - JSON 指令（多轮对话、描述、QA）
- **适用阶段**：SFT（视频指令遵循）

---

#### 1.1.4 音频–文本数据集

##### LibriSpeech
- **数据量**：约 1,000 小时（原文为 960h，实际约 1000h）
- **语言**：英语
- **来源**：LibriVox 公共领域有声书（Project Gutenberg 文本）
- **收集方法**：
  - **两阶段对齐**：
    1. 文本预处理：转为大写、去除标点、展开缩写
    2. 音频分割：使用 SRILM 工具包将长录音与文本对齐，切分为短 utterance
  - 志愿者朗读，多种口音与朗读风格
- **音频格式**：
  - 采样率：16 kHz
  - 格式：FLAC（无损压缩）
  - 声道：单声道
- **数据集划分**：
  - train-clean-100：100 小时，清洁语音
  - train-clean-360：360 小时，清洁语音
  - train-other-500：500 小时，更具挑战性（口音、背景噪声）
  - dev/test：clean 和 other 各一套
- **文本格式**：
  - 转录文本文件
  - 分段边界对齐
- **组织方式**：
  - 按说话人分目录
  - 每个 utterance 独立文件
- **下载**：https://www.openslr.org/12
- **许可证**：CC BY 4.0
- **适用阶段**：
  - S1 音频-文本对齐
  - ASR 基准评测

##### WenetSpeech 系列
- **数据量**：
  - 原始 WenetSpeech：10,000+ 小时中文
  - WenetSpeech-Chuan：10,000 小时四川话
  - WenetSpeech-Yue：21,800 小时粤语
  - WenetSpeech-Wu：8,000 小时吴语
- **语言**：中文普通话 + 多种方言
- **来源**：YouTube 短视频、直播、播客
- **收集方法**：
  - **Chuan-Pipeline 六阶段框架**：
    1. 音频收集（从视频提取）
    2. 说话人属性标注（年龄、性别）
    3. 语音质量标注（DNSMOS、SNR）
    4. 自动语音识别
    5. 文本后处理
    6. 识别器输出投票（多模型融合）
  - 丰富标注：
    - 说话人人口统计（年龄、性别）
    - 情感表达分类
    - 语音质量指标（DNSMOS、SNR）
    - ASR 置信度分数
- **音频格式**：
  - 采样率：16 kHz
  - 格式：WAV/FLAC
  - 分段：短片段（几秒到几十秒）
- **文本格式**：
  - 转录文本（人工校对子集）
  - 置信度分数
- **组织方式**：
  - HuggingFace: ASLP-lab/wenetspeech-chuan
  - 按说话人/来源分片
- **许可证**：开源，具体见各子集说明
- **适用阶段**：
  - S1 中文 ASR 对齐
  - 多方言训练
  - 语音质量研究

##### AISHELL 系列
- **数据量**：
  - AISHELL-1：178 小时，400 说话人
  - AISHELL-2：1,000+ 小时，多场景
  - AISHELL-4：120 小时，8 通道环形麦克风阵列会议场景
- **语言**：中文普通话
- **来源**：
  - AISHELL-1/2：朗读语音
  - AISHELL-4：真实会议录音
- **收集方法**：
  - AISHELL-4：
    - 8 通道环形麦克风阵列
    - 真实会议场景（多人对话）
    - 包含说话人分割标注
- **音频格式**：16 kHz，多通道（AISHELL-4）
- **标注**：
  - TextGrid 格式转录
  - 说话人标注（AISHELL-4）
- **适用阶段**：
  - AISHELL-1/2：S1 ASR 对齐
  - AISHELL-4：会议场景、说话人分割训练

##### GigaSpeech / Common Voice / Fleurs
- **GigaSpeech**：
  - 10,000 小时英语 ASR
  - 来源：播客、有声书、YouTube
- **Common Voice**：
  - Mozilla 众包数据集
  - 多语言，社区贡献
  - 验证：多人验证机制
- **Fleurs**：
  - 102 语言 ASR
  - 基于 FLORES 翻译基准的语音版本
- **适用阶段**：多语言 ASR 训练

##### MusicCaps / AudioSet
- **MusicCaps**：
  - 5.5K 音乐片段，人工标注丰富描述
  - 来源：YouTube 音乐
  - 适用：音乐理解、音乐-文本对齐
- **AudioSet**：
  - Google 提供的 632 音频事件类别
  - 2M+ 10 秒片段
  - 多标签分类
  - 适用：音频事件检测、环境音理解

##### MMAU（Massive Multi-Task Audio Understanding）
- **数据量**：10,000 精心筛选的音频片段
- **任务覆盖**：27 种技能
  - 12 信息提取任务
  - 15 推理任务
- **领域**：语音、环境音、音乐
- **难度**：需要专家级领域知识
- **性能基准**：
  - Gemini 2.0 Flash：59.93%
  - Qwen2-Audio：52.50%
- **适用阶段**：音频理解能力评测、SFT 数据构造参考

##### Emilia
- **数据量**：大规模 TTS 训练集（小时级）
- **内容**：多说话人、多语言、情感标注
- **适用阶段**：语音合成（TTS）SFT

---

#### 1.1.5 跨模态（音视频联合）数据集

##### E-MM1（Encord Multimodal Dataset）
- **数据量**：
  - 107M 组（1.07 亿组）
  - 约 1B（10 亿）数据对
  - 5 种模态：音频、图像、视频、文本、3D 点云
- **来源**：
  - 5.3M 音频样本
  - 3.5M 图像
  - 828K 3D 对象
  - 5.4M 视频片段
- **收集方法**：
  - **三层次结构**：
    1. **预训练池**：>100M 自动生成样本，来自公共数据源
    2. **后训练子集**：1M 人工评分对，确保质量
    3. **评估集**：3.5K 基于共识的数据点，用于零样本跨模态能力评估
  - 每组包含：caption + 来自其他 4 种模态的匹配项（quintuples）
  - 人工标注：1M 人工标注确保质量
- **数据格式**：
  - 图像：标准格式
  - 音频：WAV/MP3
  - 视频：MP4
  - 3D 点云：标准 3D 格式
  - 文本：caption 描述
- **组织方式**：
  - 按 quintuples 组织
  - 支持跨模态检索任务
- **网站**：https://e-mm1.github.io/
- **许可证**：开源（具体见网站）
- **适用阶段**：
  - S2 跨模态对齐
  - 跨模态检索预训练
  - 物理 AI 应用

##### VALID（已在上文视频部分介绍）
- **重申**：音视频文本三模态交错数据集
- **适用阶段**：三模态联合预训练

##### FineVideo（已在上文视频部分介绍）
- **重申**：含音频轨道的视频数据集
- **适用阶段**：音视频联合理解训练

---

### 1.2 数据组织与生产方案

#### 1.2.1 数据组织总览

```
数据根目录/
├── phase1_adapter_alignment/          # Phase 1: Adapter 对齐
│   ├── image_text_pairs/              # 图文 caption 对
│   │   ├── images/                    # 图像文件
│   │   ├── captions.json              # {image_id: caption}
│   │   └── metadata.json              # 来源、质量分数
│   ├── audio_text_pairs/              # 音文 ASR 对
│   │   ├── audio/                     # 音频片段
│   │   │   ├── librispeech/
│   │   │   ├── wenetspeech/
│   │   │   └── aishell/
│   │   ├── transcripts.json           # {audio_id: text}
│   │   └── manifests.csv              # 路径、时长、采样率
│   └── metadata/                      # 数据统计信息
│
├── phase2_encoder_finetune/           # Phase 2: 编码器微调
│   ├── ocr_dense/                     # OCR 密集数据
│   │   ├── docvqa/
│   │   ├── chartqa/
│   │   ├── infovqa/
│   │   └── ocr_synthetic/             # 合成 OCR 数据
│   ├── audio_events/                  # 音频事件数据
│   │   ├── audioset/
│   │   └── musiccaps/
│   └── metadata/
│
├── phase3_multimodal_pretrain/         # Phase 3: 多模态联合预训练
│   ├── text_only/                     # 纯文本（防退化）
│   │   ├── slimpajama/
│   │   └── fineweb/
│   ├── image_text_interleaved/        # 图文交错
│   │   ├── omnicorpus/
│   │   │   ├── cc/
│   │   │   ├── cw/
│   │   │   └── yt/
│   │   └── laion5b_subset/
│   ├── video_text/                    # 视频-文本
│   │   ├── omnicorpus_yt/
│   │   ├── valid/
│   │   └── finevideo/
│   ├── audio_text/                    # 音频-文本
│   │   └── asr_datasets/              # LibriSpeech, WenetSpeech等
│   └── cross_modal/                   # 跨模态（音视频联合）
│       ├── emm1/
│       └── valid/
│
├── phase4_long_context/               # Phase 4: 长上下文
│   ├── long_video/                    # 长视频
│   │   ├── longvideobench/
│   │   └── finevideo_long_segments/
│   ├── long_audio/                    # 长音频
│   │   └── wenetspeech_long/
│   └── long_conversations/            # 长对话
│
├── phase5_sft/                        # Phase 5: SFT
│   ├── sft_stage1_general/            # 通用 SFT
│   │   ├── sharegpt4v/
│   │   ├── llava_instruct/
│   │   ├── videochat2_it/
│   │   └── text_only_10pct/           # 10% 纯文本
│   ├── sft_stage2_long_cot/           # Long-CoT SFT
│   │   ├── reasoning_qa/
│   │   ├── math_vqa/
│   │   └── video_reasoning/
│   └── metadata/
│
└── phase6_rl/                         # Phase 6: RLHF/DPO
    ├── preference_pairs/              # 偏好对
    ├── tool_use_trajectories/         # 工具使用轨迹（OmniAtlas风格）
    └── tts_finetune/                   # TTS 微调数据
        └── emilia_multilingual/
```

#### 1.2.2 各阶段数据组织与生产详细方案

##### Phase 1: Adapter 对齐（可落地方案）

**目标**：训练视觉/音频投影器，将多模态特征对齐到 LLM 嵌入空间

**数据组织**：

1. **图文 Caption 数据（约 100M 样本）**
   - **来源组合**：
     - LAION-5B 高 CLIP 相似度子集（50M）
     - OmniCorpus-CC 图文对（30M）
     - DataComp-1B 高质量子集（20M）
   - **筛选标准**：
     - 图像：分辨率 ≥ 224×224，CLIP 相似度 ≥ 0.25
     - 文本：长度 10–200 tokens，语言检测通过
     - 去重：图像感知哈希去重，文本 n-gram 去重
   - **格式**：
     ```json
     {
       "image_path": "path/to/image.jpg",
       "caption": "A detailed description...",
       "clip_score": 0.32,
       "source": "laion5b"
     }
     ```

2. **音文 ASR 数据（约 50K 小时）**
   - **来源组合**：
     - LibriSpeech（全部 1,000h，clean+other）
     - WenetSpeech 普通话子集（8,000h）
     - AISHELL-1/2（约 1,200h）
   - **音频处理**：
     - 统一重采样到 16 kHz
     - 分段：最大 30 秒/段
     - 提取 80-dim mel 频谱特征（用于 Whisper）
   - **格式**：
     ```json
     {
       "audio_path": "path/to/audio.wav",
       "transcript": "The spoken text content",
       "duration": 12.5,
       "source": "librispeech",
       "split": "train"
     }
     ```

**生产流程**：

```python
# 数据处理 pipeline 示例
class Phase1DataPipeline:
    def prepare_image_text_data(self):
        # 1. 下载 LAION-5B metadata
        # 2. 筛选高 CLIP 分数样本
        # 3. 下载图像（可用 img2dataset）
        # 4. 生成 caption 配对文件
        pass
    
    def prepare_audio_text_data(self):
        # 1. 下载 LibriSpeech, WenetSpeech
        # 2. 统一转换为 16kHz WAV
        # 3. 分段（vad 或固定窗口）
        # 4. 生成 transcript 配对
        pass
```

**数据加载配置**：
```yaml
# phase1_dataloader.yaml
image_text:
  batch_size: 256
  num_workers: 8
  image_size: 384
  max_seq_length: 512
  
audio_text:
  batch_size: 128
  num_workers: 8
  audio_length: 30  # seconds
  sample_rate: 16000
  whisper_model: "whisper-medium"
```

---

##### Phase 2: 编码器微调（可落地方案）

**目标**：增强编码器感知能力（OCR、细节、音频事件）

**数据组织**：

1. **OCR 密集数据（约 500K 样本）**
   - **来源**：
     - DocVQA（12K 文档，50K QA）
     - ChartQA（9K QA）
     - InfoVQA（15K QA）
     - OCRBench 训练集（10M 定位+识别）
   - **合成数据（关键创新）**：
     - 使用 MiniCPM-V 4.5 动态损坏范式
     - 生成流程：
       1. 获取 PDF/文档图像
       2. 随机选择文本区域
       3. 应用三等级损坏：
          - 低：高斯噪声 σ=10
          - 中：模糊核 5×5 + 噪声 σ=25
          - 高：完全遮蔽（mask）
       4. 生成训练目标：预测原始文本
   - **格式**：
     ```json
     {
       "original_image": "path/to/doc.png",
       "corrupted_image": "path/to/corrupted.png",
       "damage_level": "medium",
       "target_text": "Original text content",
       "bbox": [x1, y1, x2, y2]
     }
     ```

2. **音频事件/音乐数据（约 5K 小时）**
   - **来源**：
     - AudioSet（2M+ 片段，632 类）
     - MusicCaps（5.5K 音乐，丰富描述）
     - MMAU（10K 音频，27 种技能）
   - **处理**：
     - 提取事件标签
     - 生成自然语言描述

**生产脚本示例**：
```python
# OCR 动态损坏数据生成
def generate_ocr_training_data(doc_image_path):
    # 1. OCR 检测文本区域（使用 PaddleOCR/EasyOCR）
    text_regions = detect_text_regions(doc_image_path)
    
    for region in text_regions:
        level = random.choice(['low', 'medium', 'high'])
        corrupted = apply_damage(doc_image_path, region, level)
        
        yield {
            'corrupted_image': corrupted,
            'target_text': region['text'],
            'level': level,
            'bbox': region['bbox']
        }
```

---

##### Phase 3: 多模态联合预训练（可落地方案）

**目标**：全模态联合理解，各模态均衡

**数据组织（约 1–2T tokens）**：

**数据混合比例（关键！）**：
```yaml
phase3_data_mix:
  text_only: 0.40          # 40% 纯文本（防退化）
  image_text_interleaved: 0.25   # 25% 图文交错
  video_text: 0.15         # 15% 视频-文本
  audio_text: 0.10         # 10% 音频-文本
  cross_modal: 0.10        # 10% 跨模态（音视频联合）
```

**各子集细节**：

1. **纯文本（40% = ~800B tokens）**
   - SlimPajama: 300B
   - FineWeb: 400B
   - CulturaX 多语言: 100B

2. **图文交错（25% = ~500B tokens）**
   - OmniCorpus-CC: 300B
   - OmniCorpus-CW: 100B
   - OmniCorpus-YT: 100B
   - 处理：保持交错格式，图像 token 数控制（如每图 64–256 tokens）

3. **视频-文本（15% = ~300B tokens）**
   - VALID: 100B
   - FineVideo: 100B
   - OmniCorpus-YT 视频: 100B
   - 处理：
     - 采样：1 FPS 或动态采样
     - 最大帧数：48–64 帧
     - 视频 token 压缩：使用 3D-Resampler 或平均池化

4. **音频-文本（10% = ~200B tokens）**
   - WenetSpeech: 100B
   - LibriSpeech + GigaSpeech: 60B
   - MusicCaps + AudioSet 描述: 40B

5. **跨模态（10% = ~200B tokens）**
   - E-MM1 音视频文本组: 100B
   - VALID 三模态: 60B
   - 合成跨模态数据（见下文）: 40B

**合成跨模态数据生产（参考 Baichuan-Omni）**：
```python
def generate_cross_modal_data(qa_text, num_timbres=44):
    """
    将 QA 文本按 1:3 拆分，前 1/4 转为音频输入
    """
    words = qa_text.split()
    split_point = len(words) // 4
    
    audio_input_text = ' '.join(words[:split_point])
    target_text = ' '.join(words[split_point:])
    
    # 使用 TTS 生成音频（多音色）
    timbre_id = random.randint(0, num_timbres - 1)
    audio = tts_synthesize(audio_input_text, timbre_id)
    
    return {
        'audio_input': audio,
        'target_text': target_text,
        'timbre_id': timbre_id
    }
```

**渐进式模态加入（参考 Ola）**：
- **Week 1–2**：仅图文数据（Phase 1 数据 + 图文交错）
- **Week 3–4**：加入视频-文本（静态图→动态视频迁移）
- **Week 5–6**：加入音频-文本
- **Week 7+**：加入跨模态音视频联合数据

---

##### Phase 4: 长上下文扩展（可落地方案）

**目标**：支持 32K+ 上下文长度

**数据组织**：

1. **长视频数据（约 50K 样本）**
   - 来源：LongVideoBench、FineVideo 长片段（>10 分钟）
   - 处理：
     - 最大 256 帧采样
     - 帧 token 压缩（3D-Resampler 到 256 tokens）
     - 总序列长度控制在 32K 以内

2. **长音频数据（约 20K 样本）**
   - 来源：WenetSpeech 长会议录音（>5 分钟）
   - 处理：
     - Whisper 编码后最大 8K audio tokens
     - 配合视频时采用时间交错（每 2 秒一块）

3. **长对话数据（约 30K 样本）**
   - 构造多轮图文音视频对话
   - 平均 20+ 轮次

---

##### Phase 5: SFT（可落地方案）

**目标**：指令对齐、多任务能力

**数据组织（500K–1M 条）**：

**两阶段 SFT**：

**Stage 1: 通用 SFT（300K 条）**
- 图文指令（100K）：ShareGPT4V、LLaVA-Instruct
- 视频指令（80K）：VideoChat2-IT、ShareGPT-Video
- 音频指令（50K）：语音 QA、ASR 校正
- 纯文本（70K）： Alpaca、Dolly（防退化）

**Stage 2: Long-CoT SFT（200K 条）**
- 数学推理（50K）：MathVista、复杂图表推理
- 长视频推理（80K）：长视频 QA、时序推理
- 跨模态推理（70K）：音视频联合推理

**数据格式（统一对话格式）**：
```json
{
  "id": "unique_id",
  "modality": "image+audio+text",
  "images": ["path/to/img1.jpg"],
  "audio": "path/to/audio.wav",
  "conversations": [
    {"from": "human", "value": "<image><audio>\nWhat is happening in this scene?"},
    {"from": "gpt", "value": "Based on the video and audio..."}
  ]
}
```

---

##### Phase 6: RLHF/DPO（可落地方案）

**目标**：质量提升、减幻觉、语音自然度

**数据组织**：

1. **Thinker 侧偏好数据**
   - 构造方式：
     - 同一 prompt 采样 4 个响应
     - GPT-4 评分选择最佳/最差
     - 或使用规则奖励（准确率 + 格式 + 重复惩罚）
   - 数据量：约 50K 偏好对

2. **Talker 侧语音偏好数据**
   - 多语言 TTS 质量对
   - 说话人相似度对
   - 数据量：约 20K 对

3. **OmniAtlas 风格工具使用轨迹（可选）**
   - 后见之明树搜索生成成功轨迹
   - 轨迹级监督学习
   - OmniDPO 纠错训练

---

## 2. 评测构建

### 2.1 评测集详细分析

#### 2.1.1 图像理解评测集

##### MMBench / MMBench-1.1 / MMBench-CN
- **官方链接**：https://mmbench.opencompass.org.cn/
- **题目数量**：约 3,000 QA 对（各版本略有不同）
- **题型**：
  - 全部为多选题（4 选 1）
  - 20+ 能力维度：感知（粗粒度/细粒度识别）、认知（逻辑/推理）、粗粒度/细粒度理解
- **难度分布**：
  - 简单：直接识别（约 30%）
  - 中等：需要推理（约 50%）
  - 困难：多跳推理、知识结合（约 20%）
- **考察能力**：
  - 细粒度视觉识别（物体部件、属性）
  - 视觉推理（空间关系、逻辑推理）
  - 跨域知识应用
- **数据构造方法**：
  - 从 COCO、OpenImages 等选取图像
  - 人工标注 + 模板生成结合
  - 选项设计：干扰项与正确答案视觉相似
- **如何提升表现**：
  - 训练数据：增加细粒度定位数据（如 RefCOCO）
  - 数据增强：多尺度图像输入
  - 后处理：选项对比（对比每个选项与图像的匹配度）

##### MMMU / MMMU-Pro
- **官方链接**：https://mmmu-benchmark.github.io/
- **题目数量**：
  - MMMU：约 11.5K（多学科多模态理解）
  - MMMU-Pro：更严格筛选的 subset
- **题型**：
  - 多学科选择题（艺术、商业、健康、科学、人文、社科）
  - 需要专业知识 + 视觉理解
- **难度**：
  - 专家级难度，大学生水平题目
  - 人类表现约 60-70%
- **考察能力**：
  - 领域专业知识
  - 图表/示意图理解
  - 数学公式识别与推理
- **数据构造**：
  - 从大学考试、教科书收集
  - 人工筛选确保需要视觉信息才能回答
- **提升策略**：
  - 训练数据增加教科书、学术论文图文对
  - 使用课程学习（curriculum learning）逐步增加难度
  - 多模态 CoT（思维链）推理训练

##### MathVista / MATH-Vision
- **题目数量**：
  - MathVista：约 6K 数学图形 QA
  - MATH-Vision：约 3K
- **题型**：
  - 数学图表、几何图形、函数图像的理解与推理
  - 多选 + 开放数字答案
- **考察能力**：
  - 数学图表解析
  - 几何推理
  - 数值计算
- **提升策略**：
  - 增加几何、图表、函数图像的预训练数据
  - 训练时使用程序辅助（PAL）或工具使用（计算器）

##### HallusionBench
- **题目数量**：约 500 对（图像 + 问题）
- **题型**：
  - 二分类：幻觉 vs 非幻觉
  - 设计陷阱：问题引导错误答案
- **考察能力**：
  - 幻觉检测与避免
  - 细粒度事实验证
- **数据构造**：
  - 人工设计引导性问题
  - 使用易混淆的图像对
- **提升策略**：
  - RLAIF-V（RLAIF for Vision）减幻觉训练
  - 增加否定陈述训练数据
  - 事实验证模块微调

---

#### 2.1.2 视频理解评测集

##### Video-MME / VideoMME（CVPR 2025）
- **官方链接**：https://video-mme.github.io/
- **题目数量**：
  - 900 视频
  - 2,700 人工标注 QA 对
- **视频时长**：
  - 短：< 2 分钟
  - 中：4–15 分钟
  - 长：30–60 分钟
  - 范围：11 秒到 1 小时
- **题型**：
  - 多选题（4 选 1）
  - 6 大视觉领域，30 子领域
- **考察能力**：
  - 时序推理（动作顺序、因果关系）
  - 长程依赖（长视频信息整合）
  - 音视频联合理解（可选字幕/音频）
- **数据构造**：
  - 专家人工标注（反复观看）
  - 确保需要视频内容才能回答（文本-only 无法回答）
- **性能基准**：
  - Gemini 1.5 Pro：75%
  - GPT-4o：71.9%
- **提升策略**：
  - 训练数据：增加长视频（>10 分钟）样本
  - 时间采样策略：动态采样关键帧
  - 3D-Resampler 等 token 压缩技术
  - 音视频联合训练（字幕+音频轨道）

##### MVBench / LongVideoBench / LVBench
- **MVBench**：多视频理解任务合集
- **LongVideoBench**：
  - 约 1K 长视频（平均 10 分钟）
  - 测试长时序推理
- **LVBench**：长视频问答
- **提升策略**：
  - 32K+ 长上下文训练
  - 时间位置编码（TM-RoPE）

---

#### 2.1.3 音频理解评测集

##### LibriSpeech
- **标准 ASR 基准**
- **划分**：
  - test-clean：清洁测试集
  - test-other：挑战性测试集
- **评估指标**：WER（Word Error Rate）
- **提升策略**：
  - 增加训练数据多样性（口音、噪声）
  - 数据增强：速度扰动、SpecAugment
  - 更大的音频编码器（Whisper-large vs medium）

##### WenetSpeech
- **中文 ASR 基准**
- **子集**：
  - test_net：网络视频
  - test_meeting：会议场景
- **评估指标**：CER（Character Error Rate）
- **提升策略**：
  - 方言数据混合训练
  - 长音频上下文建模

##### MMAU / MMAU-Pro
- **题目数量**：
  - MMAU：10K 音频，27 种技能
  - MMAU-Pro：5.3K 实例，49 种技能
- **题型**：
  - 多选 + 开放
  - 语音、环境音、音乐三大类
  - 信息提取 + 推理两类任务
- **难度**：
  - 专家级，需要领域知识
  - Gemini 2.0 Flash：59.93%
  - Qwen2-Audio：52.50%
- **考察能力**：
  - 音频内容理解
  - 音乐理论推理
  - 环境音场景推理
- **提升策略**：
  - 增加 AudioSet、MusicCaps 训练数据
  - 音频-文本联合预训练
  - 领域知识注入（音乐理论、声学知识）

##### VoiceBench
- **语音交互评测**
- **测试维度**：
  - AlpacaEval、CommonEval、WildVoice、SD-QA、MMSU 等
- **提升策略**：
  - 语音对话数据 SFT
  - 多轮对话能力训练

---

#### 2.1.4 全模态联合评测集

##### DailyOmni
- **题目数量**：1,197 多选 QA 对
- **视频**：684 段日常生活视频（30 秒–60 秒片段）
- **题型**：6 类任务
  1. 音视频事件对齐
  2. 事件序列
  3. 推理
  4. 推断
  5. 对比分析
  6. 上下文理解
- **数据构造**：
  - 来源：AudioSet、Video-MME、FineVideo
  - 自动生成 + 人工审核（Gemini 2.0 Flash、Qwen2.5-VL-7B、Whisper-Large-V2）
- **考察能力**：
  - 音视频联合理解
  - 时序对齐推理
- **提升策略**：
  - 训练时音视频时间交错（每 2 秒一块）
  - TM-RoPE 时间对齐位置编码
  - 显式跨模态对齐模块（OmniAlignNet）

##### WorldSense
- **题目数量**：3,172 QA 对
- **视频**：1,662 音视频同步视频
- **覆盖**：8 大领域，67 子类别
- **难度**：
  - 当前最佳模型：65.1%（仍较低）
- **考察能力**：
  - 空间推理
  - 常识推理
  - 物理世界理解
- **提升策略**：
  - 增加物理、地理、常识知识预训练
  - 多跳推理训练

##### OmniGAIA（2026）
- **题目数量**：360 QA 对
- **领域**：9 个真实领域（地理、历史、科技、体育、艺术、电影、科学、金融、食品）
- **模态覆盖**：
  - 99.7% 需要视觉感知
  - 99.7% 需要音频感知
  - 98.6% 需要网页搜索
  - 74.4% 需要代码/计算
- **难度分布**：
  - Easy：33.9%
  - Medium：44.4%
  - Hard：21.7%
- **媒体时长**：
  - 视频平均：242.2 秒
  - 音频平均：197.0 秒
- **题型**：开放式答案（非多选），需工具使用
- **数据构造（事件图方法）**：
  1. 数据收集：FineVideo、LongVideoBench、COCO 2017
  2. 信息发现：Gemini-3-Flash 提取事件、音频分析、图像理解
  3. 事件图构建：DeepSeek-V3.2 迭代扩展
  4. QA 生成：事件模糊化 + 人工验证
- **性能基准**：
  - Gemini-3-Pro：62.5 Pass@1
  - Qwen3-Omni-30B：13.3 Pass@1
  - OmniAtlas（训练后）：20.8 Pass@1
- **考察能力**：
  - 多跳推理
  - 工具使用（搜索、代码）
  - 全模态感知
  - 开放式答案生成
- **提升策略**：
  - **OmniAtlas 训练方案**：
    1. SFT：工具集成轨迹训练（13.3→18.9）
    2. OmniDPO：细粒度纠错（18.9→20.8）
  - 增加工具使用训练数据
  - 后见之明树搜索生成成功轨迹

##### OmniBench
- **多模态推理基准**
- **覆盖**：视/听/文三模态
- **特点**：选择题，感知为主
- **注意**：OmniBench 高分 ≠ 智能体能力强（如 Qwen3-Omni OmniBench 58.4，但 OmniGAIA 仅 13.3）

---

## 3. 训练方案

### 3.1 可落地的六阶段训练方案

#### 3.1.1 硬件与资源估算总览

| 阶段 | 目标参数量 | GPU 需求 | 显存/卡 | 预计时间 | 总 GPU 小时 |
|------|-----------|---------|---------|----------|-------------|
| Phase 1 | 7B–30B | 8× A100 80GB | 约 60GB | 2–3 天 | ~500 |
| Phase 2 | 7B–30B | 8× A100 80GB | 约 70GB | 3–5 天 | ~800 |
| Phase 3 | 7B | 32× A100 80GB | 约 70GB | 2–3 周 | ~16,000 |
| Phase 3 | 30B | 64× A100 80GB | 约 75GB | 3–4 周 | ~43,000 |
| Phase 4 | 7B–30B | 16× A100 80GB | 约 80GB | 5–7 天 | ~2,000 |
| Phase 5 | 7B–30B | 8× A100 80GB | 约 70GB | 3–5 天 | ~800 |
| Phase 6 | 7B–30B | 8× A100 80GB | 约 65GB | 2–3 天 | ~500 |
| **总计 7B** | | | | **约 4–5 周** | **~20,600** |
| **总计 30B** | | | | **约 6–8 周** | **~47,600** |

---

#### 3.1.2 Phase 1: Adapter 对齐（详细配置）

**目标**：训练视觉/音频投影器

**冻结组件**：
- LLM（Qwen3）
- 视觉编码器（SigLIP）
- 音频编码器（Whisper）

**可训练组件**：
- 视觉投影器（2 层 MLP 或 3D-Resampler）
- 音频投影器（MLP 或 Conv-GMLP）

**数据配置**：
```yaml
# phase1_config.yaml
data:
  image_caption:
    source: ["laion5b_high_quality", "omnicorpus_cc", "datacomp"]
    samples: 100_000_000
    batch_size: 256
    max_seq_len: 512
    
  audio_transcript:
    source: ["librispeech", "wenetspeech", "aishell"]
    hours: 50_000
    batch_size: 128
    max_audio_len: 30  # seconds

model:
  llm: "Qwen3-7B/30B"
  visual_encoder: "siglip-so400m-patch14-384"
  audio_encoder: "whisper-medium"
  
  visual_projector:
    type: "mlp"  # or "3d_resampler"
    hidden_dim: 2048
    output_dim: 4096  # LLM hidden size
    
  audio_projector:
    type: "conv_gmlp"  # or "mlp"
    downsampling_rate: 4
```

**训练配置**：
```python
training_config = {
    "optimizer": "AdamW",
    "learning_rate": 1e-3,  # 投影层可用较高 LR
    "weight_decay": 0.01,
    "warmup_steps": 2000,
    "max_steps": 50_000,
    "gradient_accumulation_steps": 4,
    "mixed_precision": "bf16",
    "gradient_clipping": 1.0,
}
```

**代码框架**（HuggingFace Trainer）：
```python
from transformers import Trainer, TrainingArguments

# 加载冻结的编码器
llm = AutoModelForCausalLM.from_pretrained("Qwen3-7B")
vision_encoder = AutoModel.from_pretrained("siglip-so400m")
audio_encoder = WhisperModel.from_pretrained("whisper-medium")

# 冻结
for param in llm.parameters():
    param.requires_grad = False
for param in vision_encoder.parameters():
    param.requires_grad = False

# 初始化投影器
visual_projector = MLPProjector(vision_dim→llm_dim)
audio_projector = ConvGMLP(audio_dim→llm_dim)

# 训练
trainer = Trainer(
    model=MultiModalModel(llm, vision_encoder, audio_encoder, 
                          visual_projector, audio_projector),
    args=training_args,
    train_dataset=phase1_dataset,
)
trainer.train()
```

**验证指标**：
- 投影层 loss 收敛（< 0.5）
- COCO caption 验证集 CIDEr 分数
- LibriSpeech dev-clean WER

---

#### 3.1.3 Phase 2: 编码器微调（详细配置）

**冻结**：LLM
**可训练**：投影器 + 编码器（视觉/音频）

**数据配置**：
```yaml
phase2_data:
  ocr_dense:
    docvqa: 50_000
    chartqa: 9_000
    infovqa: 15_000
    ocr_synthetic: 1_000_000  # 动态损坏生成
    
  audio_events:
    audioset: 500_000
    musiccaps: 5_500
```

**学习率**：
- 编码器：1e-5（较低，保护预训练权重）
- 投影器：1e-4

**动态损坏数据生成代码**：
```python
class DynamicCorruptionDataset(Dataset):
    """MiniCPM-V 4.5 风格动态损坏"""
    
    def __getitem__(self, idx):
        doc = self.load_document(idx)
        
        # 随机选择损坏等级
        level = random.choice(['low', 'medium', 'high'])
        
        if level == 'low':
            # 高斯噪声 σ=10
            corrupted = add_gaussian_noise(doc.image, sigma=10)
        elif level == 'medium':
            # 模糊 + 噪声
            corrupted = gaussian_blur(doc.image, kernel=5)
            corrupted = add_gaussian_noise(corrupted, sigma=25)
        else:  # high
            # 完全遮蔽文本区域
            mask = detect_text_regions(doc.image)
            corrupted = apply_mask(doc.image, mask)
        
        return {
            'image': corrupted,
            'text_target': doc.text,  # 预测原文
            'corruption_level': level,
        }
```

---

#### 3.1.4 Phase 3: 多模态联合预训练（详细配置）

**可训练**：全部参数（或 LoRA 高效微调）

**数据混合配置（关键！）**：
```yaml
phase3_data_mix:
  total_tokens: 2_000_000_000_000  # 2T
  
  components:
    text_only:
      ratio: 0.40
      sources:
        slimpajama: 300B
        fineweb: 400B
        culturax: 100B
    
    image_text_interleaved:
      ratio: 0.25
      sources:
        omnicorpus_cc: 300B
        omnicorpus_cw: 100B
        omnicorpus_yt: 100B
      
    video_text:
      ratio: 0.15
      sources:
        valid: 100B
        finevideo: 100B
        omnicorpus_yt_video: 100B
      sampling: 1_fps
      max_frames: 64
      
    audio_text:
      ratio: 0.10
      sources:
        wenetspeech: 100B
        librispeech: 60B
        musiccaps_audioset: 40B
        
    cross_modal:
      ratio: 0.10
      sources:
        emm1: 100B
        valid_multimodal: 60B
        synthetic_cross_modal: 40B  # TTS 生成
```

**渐进式模态加入（Ola 策略）**：
```python
class ProgressiveModalityScheduler:
    """渐进式模态加入调度器"""
    
    def get_modality_mix(self, step):
        if step < 10_000:
            # 第 1-2 周：仅图文
            return {'image_text': 0.6, 'text_only': 0.4}
        elif step < 20_000:
            # 第 3-4 周：加入视频
            return {
                'image_text': 0.4,
                'video_text': 0.2,
                'text_only': 0.4
            }
        elif step < 30_000:
            # 第 5-6 周：加入音频
            return {
                'image_text': 0.25,
                'video_text': 0.15,
                'audio_text': 0.10,
                'text_only': 0.40,
                'cross_modal': 0.10
            }
        else:
            # 之后：全模态均衡
            return {
                'text_only': 0.40,
                'image_text': 0.25,
                'video_text': 0.15,
                'audio_text': 0.10,
                'cross_modal': 0.10
            }
```

**训练配置**：
```python
phase3_config = {
    "optimizer": "AdamW",
    "learning_rate": 1e-5,  # 全参数训练用低 LR
    "lr_scheduler": "cosine_with_restarts",
    "warmup_ratio": 0.01,
    "weight_decay": 0.1,
    "gradient_accumulation_steps": 8,
    "max_grad_norm": 1.0,
    "bf16": True,
    "gradient_checkpointing": True,  # 节省显存
    "max_steps": 100_000,  # 约 2T tokens
    "save_steps": 1000,
    "eval_steps": 500,
}
```

**跨模态数据合成（Baichuan-Omni 风格）**：
```python
def synthesize_cross_modal_data(qa_dataset, num_timbres=44):
    """
    将 QA 数据转换为跨模态训练数据
    前 1/4 转语音作为音频输入，后 3/4 为预测目标
    """
    synthesized = []
    
    for qa in qa_dataset:
        text = qa['question'] + ' ' + qa['answer']
        words = text.split()
        split = len(words) // 4
        
        audio_text = ' '.join(words[:split])
        target_text = ' '.join(words[split:])
        
        # 使用多种 TTS 音色
        for timbre_id in range(num_timbres):
            audio = tts_synthesize(
                audio_text, 
                timbre_id=timbre_id,
                model="cosyvoice2"  # 或其他 TTS
            )
            
            synthesized.append({
                'audio_input': audio,
                'text_target': target_text,
                'timbre_id': timbre_id,
                'source_qa': qa['id']
            })
    
    return synthesized
```

**关键检查点**：
- 每 500 步验证：MMLU（文本不退化）、MMBench（视觉）、LibriSpeech（音频）
- 安全线：MMLU 下降 < 2% 视为可接受
- 模态失衡检测：各模态 loss 应均衡下降

---

#### 3.1.5 Phase 4: 长上下文扩展（详细配置）

**目标**：32K+ 上下文

**数据组织**：
```yaml
phase4_data:
  long_video:
    source: ["longvideobench", "finevideo_long"]
    min_duration: 600  # 10 minutes
    max_frames: 256
    token_compression: "3d_resampler"  # 到 256 tokens
    
  long_audio:
    source: ["wenetspeech_long"]
    min_duration: 300  # 5 minutes
    max_audio_tokens: 8192
    
  long_conversation:
    num_samples: 30_000
    avg_turns: 20
    max_total_length: 32768
```

**位置编码扩展**：
```python
# 从 4K/8K 扩展到 32K
# 方法：NTK-aware 插值 或 直接微调

class ContextExtension:
    def __init__(self, model, original_max_pos=8192, target_max_pos=32768):
        self.model = model
        self.original = original_max_pos
        self.target = target_max_pos
        
    def apply_ntk_scaling(self, base=10000):
        """NTK-aware 位置编码扩展"""
        scaling_factor = self.target / self.original
        # 修改 RoPE 的 base
        self.model.config.rope_scaling = {
            "type": "dynamic",
            "factor": scaling_factor
        }
```

**训练配置**：
- 学习率：5e-6（更低，稳定长上下文）
- 序列并行（Sequence Parallelism）：必须启用
- 上下文并行（Context Parallel）：DeepSpeed Ulysses 或 Ring Attention

---

#### 3.1.6 Phase 5: SFT（详细配置）

**两阶段 SFT**：

**Stage 1: 通用 SFT（300K 条）**
```yaml
stage1_data:
  image_instruction:
    sharegpt4v: 50_000
    llava_instruct: 50_000
    
  video_instruction:
    videochat2_it: 40_000
    sharegpt_video: 40_000
    
  audio_instruction:
    voice_qa: 30_000
    asr_correction: 20_000
    
  text_only:  # 10% 防退化
    alpaca: 35_000
    dolly: 35_000
```

**Stage 2: Long-CoT SFT（200K 条）**
```yaml
stage2_data:
  math_reasoning:
    mathvista_complex: 25_000
    chartqa_advanced: 25_000
    
  video_reasoning:
    longvideobench_qa: 40_000
    temporal_reasoning: 40_000
    
  cross_modal_reasoning:
    dailyomni_style: 35_000
    worldsense_style: 35_000
```

**统一对话格式**：
```json
{
  "id": "sft_001",
  "modality": "image+audio+text",
  "images": ["path/to/frame_001.jpg", "path/to/frame_002.jpg"],
  "audio_segments": [
    {"path": "audio_001.wav", "start": 0, "end": 10},
    {"path": "audio_002.wav", "start": 10, "end": 20}
  ],
  "conversations": [
    {
      "from": "human",
      "value": "<image><image><audio>\nDescribe what is happening in this video segment."
    },
    {
      "from": "gpt",
      "value": "In the first frame, we see... The audio indicates..."
    },
    {
      "from": "human", 
      "value": "What happens next?"
    },
    {
      "from": "gpt",
      "value": "Based on the sequence..."
    }
  ]
}
```

**训练配置**：
```python
sft_config = {
    "learning_rate": 2e-5,
    "batch_size": 128,
    "gradient_accumulation": 4,
    "max_seq_length": 8192,  # Stage 1
    # Stage 2: max_seq_length: 32768
    "warmup_ratio": 0.03,
    "num_epochs": 3,
}
```

---

#### 3.1.7 Phase 6: RLHF/DPO（详细配置）

**Thinker 侧（多模态理解）**：

**GRPO 混合 RL（MiniCPM-V 4.5 风格）**：
```python
class MixedReasoningGRPO:
    """
    短推理与长推理交替的 GRPO
    """
    def compute_rewards(self, response, mode):
        # 准确率奖励
        r_acc = accuracy_reward(response, ground_truth)
        
        # 格式奖励（CoT 格式正确性）
        r_format = format_reward(response, mode)
        
        # 重复惩罚
        r_rep = repetition_penalty(response)
        
        # 最终答案偏好（仅评估最终答案正确性）
        r_rm = preference_reward(response) if mode == 'long' else 0
        
        # 混合奖励
        return r_acc + r_format + r_rep + 0.5 * r_rm
    
    def rollout(self, prompt):
        # 随机选择模式
        mode = random.choice(['short', 'long'])
        
        if mode == 'short':
            # 快速回答模式
            response = self.model.generate(
                prompt, 
                max_new_tokens=256,
                reasoning_style='direct'
            )
        else:
            # 长 CoT 模式
            response = self.model.generate(
                prompt,
                max_new_tokens=2048,
                reasoning_style='step_by_step'
            )
        
        return response, mode
```

**RLAIF-V（减幻觉）**：
```python
class RLAIFV:
    """
    原子声明级验证的 RLAIF
    """
    def verify_response(self, response, image):
        # 1. 将响应分解为原子声明
        claims = self.extract_claims(response)
        
        # 2. 对每个声明进行图像验证
        verified_claims = []
        for claim in claims:
            is_supported = self.vlm_verify(claim, image)
            verified_claims.append((claim, is_supported))
        
        # 3. 构造偏好对
        if all(supported for _, supported in verified_claims):
            return 'positive'
        else:
            # 构造负例：替换不支持的声明
            negative_response = self.replace_unsupported_claims(response, verified_claims)
            return 'negative', negative_response
```

**Talker 侧（语音生成）四阶段**：

```yaml
# Talker 训练配置
talker_training:
  stage1_mapping:
    desc: "多模态语义到语音映射"
    data: "tts_pairs_50k"
    lr: 1e-4
    
  stage2_cpt:
    desc: "CPT + 长上下文"
    data: "long_speech_20k"
    lr: 5e-5
    
  stage3_multilingual_dpo:
    desc: "多语言 DPO"
    data: "multilingual_preferences_30k"
    lr: 1e-6
    
  stage4_speaker:
    desc: "说话人微调"
    data: "speaker_specific_5k"
    lr: 5e-7
```

**OmniAtlas 风格工具使用训练（可选）**：
```python
class OmniAtlasTraining:
    """
    后见之明树搜索 + 轨迹级 SFT + OmniDPO
    """
    def hindsight_tree_search(self, task):
        """
        从失败任务中学习成功轨迹
        """
        # 1. 执行树搜索探索多种策略
        trajectories = self.tree_search_explore(task, num_paths=16)
        
        # 2. 识别成功路径
        successful = [t for t in trajectories if t.success]
        
        # 3. 若成功，返回成功轨迹用于 SFT
        if successful:
            return self.select_best(successful)
        
        # 4. 若全部失败，分析最接近成功的路径
        best_failed = self.select_most_promising(trajectories)
        
        # 5. 使用强模型（GPT-4）完成剩余步骤
        completed = self.strong_model_complete(best_failed)
        
        return completed
    
    def omnidpo_training(self, failed_trajectory):
        """
        细粒度错误纠正 DPO
        """
        # 1. 定位第一个错误位置
        error_step = self.identify_first_error(failed_trajectory)
        
        # 2. 强模型生成纠正后的后续
        corrected_continuation = self.generate_correction(
            failed_trajectory[:error_step]
        )
        
        # 3. 构造偏好对
        positive = failed_trajectory[:error_step] + corrected_continuation
        negative = failed_trajectory
        
        # 4. DPO 训练
        self.dpo_update(positive, negative)
```

---

## 4. 突破方向详细实施方案

### 4.1 稀疏注意力支持更长上下文

#### 目标
- 支持 1M token 上下文
- 256K 长度下 3.5× 推理加速
- 训练成本降低 75%

#### 技术方案（MiniCPM-SALA / InfLLM-V2）

**架构设计**：
```python
class HybridSparseLinearAttention(nn.Module):
    """
    25% InfLLM-V2 稀疏 + 75% Lightning 线性注意力
    """
    def __init__(self, num_layers=32):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # 1:3 比例分配
        for i in range(num_layers):
            if i % 4 == 0:  # 每 4 层中的第 1 层
                self.layers.append(InfLLMV2Attention())
            else:
                self.layers.append(LightningAttention())
```

**InfLLM-V2 关键特性**：
- **零参数**：复用稠密注意力参数
- **动态切换**：短序列用稠密，长序列自动切换稀疏
- **块选择优化**：硬件感知的块选择，减少 HBM I/O

**Lightning Attention 关键特性**：
- **线性复杂度**：O(N) 而非 O(N²)
- **核技巧**：将注意力分解为 intra-block（常规）+ inter-block（线性核）

**实施步骤**：

**Step 1: 准备基础模型**
```bash
# 下载预训练的稠密注意力模型
# 如 Qwen3-7B/30B
huggingface-cli download Qwen/Qwen3-7B
```

**Step 2: 实现稀疏注意力层**
```python
# 安装 InfLLM-V2
pip install infllm-v2

# 或手动实现核心逻辑
class InfLLMV2Attention(nn.Module):
    def forward(self, hidden_states, attention_mask):
        # 1. 选择最相关的 KV 块
        selected_blocks = self.select_relevant_blocks(
            hidden_states, 
            num_blocks=16  # 只选 16 个最相关块
        )
        
        # 2. 只在选定块上做稠密注意力
        attn_output = self.dense_attention_on_blocks(
            hidden_states, 
            selected_blocks
        )
        
        return attn_output
```

**Step 3: 混合架构转换**
```python
def convert_to_hybrid(model, sparse_layer_indices):
    """
    将稠密模型转换为混合稀疏-线性模型
    """
    for i, layer in enumerate(model.model.layers):
        if i in sparse_layer_indices:
            # 替换为 InfLLM-V2
            new_attn = InfLLMV2Attention.from_dense(layer.self_attn)
            layer.self_attn = new_attn
        else:
            # 替换为 Lightning Attention
            new_attn = LightningAttention.from_dense(layer.self_attn)
            layer.self_attn = new_attn
    
    return model
```

**Step 4: HALO 蒸馏训练**
```python
class HALODistillation:
    """
    从稠密注意力蒸馏到混合注意力
    """
    def training_step(self, batch):
        # 1. 教师模型（冻结的稠密注意力）输出
        with torch.no_grad():
            teacher_logits = self.teacher(**batch).logits
        
        # 2. 学生模型（混合注意力）输出
        student_logits = self.student(**batch).logits
        
        # 3. KL 散度蒸馏损失
        distill_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction='batchmean'
        ) * (T * T)
        
        # 4. 标准语言建模损失
        lm_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            batch['labels'].view(-1)
        )
        
        return 0.5 * distill_loss + 0.5 * lm_loss
```

**Step 5: 长上下文微调**
```yaml
# long_context_finetune.yaml
data:
  max_seq_length: 32768  # 32K
  source: "longvideobench_text"
  
training:
  lr: 5e-6
  steps: 10_000
  batch_size: 4  # 小 batch，大序列
  gradient_accumulation: 8
  
optimization:
  sequence_parallel: True
  context_parallel: "ulysses"  # DeepSpeed Ulysses
```

**Step 6: HyPE 混合位置编码**
```python
class HybridPositionalEncoding(nn.Module):
    """
    协调短上下文和长上下文性能
    """
    def forward(self, seq_len):
        if seq_len <= 8192:
            # 短序列：标准 RoPE
            return self.standard_rope(seq_len)
        else:
            # 长序列：NTK-aware 插值 + 特殊处理
            return self.hype_long_rope(seq_len)
```

**资源估算**：
- 稠密预训练模型：已存在（如 Qwen3-7B）
- HALO 蒸馏：约 5B tokens，8× A100 3 天
- 长上下文微调：约 10K steps，8× A100 2 天
- **总计额外成本**：约 75% 低于从头训练混合模型

**验证指标**：
- Passkey 检索（1M 上下文中的关键信息检索）
- LongVideoBench 准确率
- 256K 长度下的推理延迟（目标：< 3.5× 加速）

---

### 4.2 更大视觉分辨率通过稀疏降低 token

#### 目标
- 视频 96× token 压缩（6 帧 448×448 → 64 tokens）
- 支持高分辨率输入（896×896+）而不爆显存

#### 技术方案（3D-Resampler / Local-Global Attention Pooling）

**方案 A: 3D-Resampler（MiniCPM-V 4.5）**

**架构**：
```python
class Unified3DResampler(nn.Module):
    """
    统一图像和视频 3D 重采样
    """
    def __init__(self, num_queries=64, dim=1024):
        super().__init__()
        self.queries = nn.Embedding(num_queries, dim)
        self.cross_attn = CrossAttention(dim)
        
    def forward_image(self, image_features):
        """
        图像：2D 重采样
        输入: [H/p, W/p, D]  (如 32×32=1024 tokens)
        输出: [64, D]  (64 tokens，16× 压缩)
        """
        # 可学习查询
        queries = self.queries.weight.unsqueeze(0).expand(
            image_features.size(0), -1, -1
        )
        
        # 交叉注意力压缩
        output = self.cross_attn(queries, image_features)
        return output
    
    def forward_video(self, video_features):
        """
        视频：3D 重采样（时空联合）
        输入: [T, H/p, W/p, D]  (如 6×32×32=6144 tokens)
        输出: [64, D]  (64 tokens，96× 压缩)
        """
        # 沿时间维度分包
        # 每包多帧，时空联合压缩
        
        # 1. 时间位置编码
        temporal_pos = self.temporal_embedding(torch.arange(T))
        video_features = video_features + temporal_pos
        
        # 2. 3D 交叉注意力
        queries = self.queries.weight.unsqueeze(0).expand(
            video_features.size(0), -1, -1
        )
        
        # 展平时空做交叉注意力
        video_flat = video_features.flatten(1, 3)  # [B, T*H*W, D]
        output = self.cross_attn(queries, video_flat)
        
        return output
```

**实施步骤**：

**Step 1: 图像 2D Resampler 训练**
```yaml
# stage1_2d_resampler.yaml
model:
  visual_encoder: "siglip-so400m"
  resampler_queries: 64
  
training:
  data: "laion5b_caption"
  epochs: 3
  lr: 1e-3
  
freeze:
  - visual_encoder
  - llm
trainable:
  - resampler
```

**Step 2: 升级到 3D（轻量 SFT）**
```python
# 关键：图像到视频的自然迁移
class UpgradeTo3D:
    def upgrade(self, model_2d):
        """
        将 2D Resampler 升级为 3D
        只需添加时间维度处理
        """
        # 1. 复制 2D 权重
        model_3d = Unified3DResampler()
        model_3d.load_state_dict(model_2d.state_dict(), strict=False)
        
        # 2. 初始化时间编码
        nn.init.normal_(model_3d.temporal_embedding.weight, std=0.02)
        
        return model_3d
```

**Step 3: 视频数据微调**
```yaml
# stage3_video_sft.yaml
data:
  source: "finevideo_omnicorpus_yt"
  sampling: 1_fps
  max_frames: 64
  
training:
  lr: 1e-4
  epochs: 1
  # 轻量微调，保持图像能力
```

**方案 B: Local-Global Attention Pooling（Ola）**

**架构**：
```python
class LocalGlobalAttentionPooling(nn.Module):
    """
    2× 压缩且信息损失小于简单 pooling
    """
    def forward(self, features):
        """
        输入: [H*W, D]
        输出: [H*W/2, D]
        """
        # 1. 双线性插值下采样（全局特征）
        global_feat = F.interpolate(
            features.transpose(0, 1).unsqueeze(0),
            scale_factor=0.5,
            mode='bilinear'
        ).squeeze(0).transpose(0, 1)  # [H*W/4, D]
        
        # 2. 拼接原始特征和全局特征
        combined = torch.cat([features, global_feat.repeat_interleave(4, 0)], dim=-1)
        
        # 3. MLP + Softmax 预测重要性权重
        importance = self.mlp(combined).softmax(dim=0)
        
        # 4. 加权下采样
        weighted = features * importance
        output = weighted.view(H, W, D).permute(2, 0, 1).unsqueeze(0)
        output = F.adaptive_avg_pool2d(output, (H//2, W//2))
        output = output.squeeze(0).permute(1, 2, 0).flatten(0, 1)
        
        return output
```

**比较与选择**：

| 方案 | 压缩率 | 实现复杂度 | 适用场景 |
|------|--------|-----------|---------|
| 3D-Resampler | 96× 视频 / 16× 图像 | 中等 | 追求极致效率、长视频 |
| Local-Global | 2× | 低 | 平衡效率与质量 |
| MLP 投影 | 1×（仅对齐） | 最低 | 快速原型验证 |

**推荐**：
- 资源允许：使用 3D-Resampler
- 平衡选择：Local-Global Attention Pooling

---

### 4.3 更自然的声音

#### 目标
- 低首包延迟（234 ms）
- 流式语音生成
- 自然语气与情感
- 多语言/跨语言支持

#### 技术方案（Thinker-Talker + 多码本 Codec）

**架构**：
```
输入（多模态）
  ↓
Thinker（LLM）
  ├─→ 文本输出
  └─→ 高层表示（含语气/情感）→ Talker
                              ↓
                         多码本语音 Codec
                              ↓
                         流式声码器
                              ↓
                         音频波形输出
```

**实施步骤**：

**Step 1: Thinker 训练**
```python
# Thinker 就是标准的多模态 LLM
# 训练方式同 Phase 1-3
# 额外：高层表示输出头

class Thinker(nn.Module):
    def forward(self, multimodal_input):
        # 标准前向
        hidden_states = self.llm(multimodal_input)
        
        # 文本输出
        text_logits = self.lm_head(hidden_states)
        
        # 高层表示（给 Talker）
        high_level_repr = self.repr_head(hidden_states)
        
        return text_logits, high_level_repr
```

**Step 2: 多码本语音 Codec 训练**
```python
class MultiCodebookCodec(nn.Module):
    """
    多码本离散化语音表示
    参考：Qwen3-Omni 多码本、LongCat 4 码本
    """
    def __init__(self, num_codebooks=4, codebook_size=1024):
        super().__init__()
        self.codebooks = nn.ModuleList([
            VectorQuantization(codebook_size, dim=256)
            for _ in range(num_codebooks)
        ])
        
    def encode(self, audio_waveform):
        """
        音频 → 多码本离散 token
        """
        # 1. 编码为连续特征
        features = self.encoder(audio_waveform)
        
        # 2. 多层向量量化
        codes = []
        residual = features
        for vq in self.codebooks:
            quantized, indices = vq(residual)
            codes.append(indices)
            residual = residual - quantized
            
        return codes  # [B, T, num_codebooks]
    
    def decode(self, codes):
        """
        多码本 token → 音频
        """
        # 1. 从码本查表
        features = sum(
            self.codebooks[i].embed(codes[:, :, i])
            for i in range(len(self.codebooks))
        )
        
        # 2. 解码为波形
        audio = self.decoder(features)
        return audio
```

**Step 3: Talker 四阶段训练**

```python
class TalkerTrainingPipeline:
    def stage1_mapping(self):
        """多模态语义 → 语音映射"""
        config = {
            'data': 'tts_pairs_50k',
            'lr': 1e-4,
            'epochs': 5,
            'objective': 'cross_entropy',  # 预测 codec token
        }
        self.train(config)
    
    def stage2_cpt(self):
        """CPT + 长上下文"""
        config = {
            'data': 'long_speech_20k',  # 长段语音
            'lr': 5e-5,
            'epochs': 3,
            'max_seq_length': 8192,  # 长上下文一致性
        }
        self.train(config)
    
    def stage3_multilingual_dpo(self):
        """多语言 DPO"""
        # 构造偏好对
        # 正例：高质量多语言 TTS
        # 负例：机器翻译腔/口音不正确的语音
        config = {
            'data': 'multilingual_preferences_30k',
            'lr': 1e-6,
            'algorithm': 'dpo',
            'beta': 0.1,
        }
        self.train(config)
    
    def stage4_speaker_finetune(self):
        """说话人微调"""
        config = {
            'data': 'speaker_specific_5k',  # 特定说话人数据
            'lr': 5e-7,
            'epochs': 10,
        }
        self.train(config)
```

**Step 4: 流式解码实现**
```python
class StreamingSpeechDecoder:
    """
    滑动窗口流式解码
    参考：Qwen2.5-Omni（2 回看 + 1 前看）
    """
    def __init__(self, lookback=2, lookahead=1):
        self.lookback = lookback
        self.lookahead = lookahead
        self.buffer = []
        
    def decode_chunk(self, codec_chunk):
        """
        codec_chunk: 当前接收的 codec token
        """
        self.buffer.append(codec_chunk)
        
        if len(self.buffer) >= self.lookback + 1 + self.lookahead:
            # 有足够上下文，可以解码
            window = self.buffer[-(self.lookback + 1 + self.lookahead):]
            
            # DiT / ConvNet 解码
            audio_chunk = self.dit_decode(window)
            
            # 移除最旧（如果超过 lookback）
            if len(self.buffer) > self.lookback + 1 + self.lookahead:
                self.buffer.pop(0)
                
            return audio_chunk
        
        return None  # 等待更多上下文
```

**首包延迟优化**：
- 目标：234 ms
- 方法：
  1. 分块预填充（chunked prefill）
  2. Thinker-Talker 异步执行
  3. 小窗口流式解码（4 块窗口）

---

### 4.4 全双工对话

#### 目标
- 边听边说（实时交互）
- 支持打断（barge-in）
- 低延迟（< 500ms 响应）

#### 技术方案（分块预填充 + VAD + 流式解码）

**架构**：
```
用户语音输入
  ↓
VAD (Voice Activity Detection)
  ↓（检测到语音）
音频编码器（Whisper）分块处理
  ↓
视觉编码器（并行，流式处理视频帧）
  ↓
Thinker（流式 LLM 推理）
  ↓
Talker（流式语音生成）
  ↓
用户听到回复
（同时用户可打断，循环）
```

**实施步骤**：

**Step 1: VAD 实现**
```python
class PersonalizedVAD:
    """
    个性化 VAD，降低误打断
    参考：FireRedChat
    """
    def __init__(self):
        self.base_vad = SileroVAD()
        self.speaker_profile = None
        
    def set_speaker_profile(self, audio_samples):
        """学习特定说话人声纹"""
        self.speaker_profile = self.extract_speaker_embedding(audio_samples)
    
    def detect(self, audio_chunk):
        # 基础 VAD
        is_speech = self.base_vad(audio_chunk)
        
        if is_speech and self.speaker_profile:
            # 验证是主要说话人（降低背景其他人干扰）
            embedding = self.extract_speaker_embedding(audio_chunk)
            similarity = cosine_sim(embedding, self.speaker_profile)
            
            if similarity < 0.7:
                return False  # 不是主要说话人，忽略
        
        return is_speech
```

**Step 2: 音频边界预测**
```python
class AudioBoundaryPredictor(nn.Module):
    """
    预测音频输入的起止边界
    参考：Baichuan-Omni
    """
    def forward(self, audio_features):
        """
        二分类：语音开始、语音结束
        """
        start_logits = self.start_classifier(audio_features)
        end_logits = self.end_classifier(audio_features)
        
        return start_logits, end_logits
```

**Step 3: 分块预填充实现**
```python
class ChunkedPrefill:
    """
    分块预填充，边录边算
    """
    def __init__(self, model, chunk_size=2.0):  # 2 秒一块
        self.model = model
        self.chunk_size = chunk_size
        self.kv_cache = None
        
    def process_audio_chunk(self, audio_chunk, is_last=False):
        """
        处理一块音频，更新 KV cache
        """
        # 1. 编码这块音频
        audio_tokens = self.model.audio_encoder(audio_chunk)
        
        # 2. 预填充（不生成，只计算 KV）
        with torch.no_grad():
            outputs = self.model.llm(
                inputs_embeds=audio_tokens,
                past_key_values=self.kv_cache,
                use_cache=True,
            )
            self.kv_cache = outputs.past_key_values
        
        # 3. 如果是最后一块，开始生成
        if is_last:
            return self.generate_response()
        
        return None  # 继续等待
```

**Step 4: 语义轮次检测**
```python
class SemanticTurnDetector:
    """
    检测用户说话轮次是否结束（语义级，非仅 VAD）
    """
    def detect_turn_end(self, text_so_far, audio_features):
        """
        结合文本和音频特征判断轮次结束
        """
        # 文本线索：检测到完整句子（句号、问号等）
        text_complete = self.is_complete_sentence(text_so_far)
        
        # 音频线索：VAD 检测到足够长停顿
        pause_detected = self.detect_pause(audio_features, threshold=0.5)
        
        # 综合判断
        if text_complete and pause_detected:
            return True
        
        return False
```

**Step 5: 自然独白数据训练**
```python
class NaturalMonologueDataset(Dataset):
    """
    含停顿和等待间隔的自然独白数据
    参考：FLM-Audio
    """
    def __getitem__(self, idx):
        # 真实对话录音，保留自然停顿
        audio = load_with_pauses(self.paths[idx])
        
        # 标注：说话时段 vs 等待时段
        segments = self.annotate_speaking_and_waiting(audio)
        
        return {
            'audio': audio,
            'segments': segments,  # [(start, end, 'speak'/'wait'), ...]
        }
```

**Step 6: 双通道训练**
```python
def full_duplex_training_step(self, batch):
    """
    模拟全双工场景训练
    """
    # 输入：用户语音（含可能的打断点）
    user_audio = batch['user_audio']
    
    # 模型开始生成回复
    for t in range(max_response_length):
        # 1. 生成下一个 token
        next_token = self.model.generate_step(...)
        
        # 2. 模拟用户可能在任何时刻打断
        if self.should_simulate_barge_in(t, batch):
            # 添加打断音频到输入
            barge_in_audio = batch['barge_in_audio']
            self.model.process_interrupt(barge_in_audio)
            
        # 3. 继续生成或响应打断
        ...
```

**延迟优化检查清单**：
- [ ] VAD 延迟 < 100ms
- [ ] 音频编码分块 < 2 秒
- [ ] Thinker TTFT < 150ms
- [ ] Talker 首音频包 < 100ms
- [ ] 总端到端延迟 < 500ms

---

### 4.5 Omni 模型优势场景数据构造

#### 目标
- 构造 VLM/纯语音模型做不了的数据
- 覆盖视频会议、音乐分析、安防等多模态场景

#### 数据构造方案

**场景 1: 视频会议总结**
```python
def generate_meeting_summary_data():
    """
    构造视频会议理解数据
    需要：画面（共享屏幕/人脸）+ 音频（多人语音）
    """
    # 1. 收集真实会议录像（脱敏）
    meeting_video = load_meeting_recording()
    
    # 2. 人工或强模型标注：
    # - 议程提取
    # - 决策点识别
    # - 行动项分配
    summary = {
        'agenda': ['项目 A 进展', '预算讨论'],
        'decisions': ['批准预算 10 万'],
        'action_items': [
            {'who': '张三', 'task': '提交详细方案', 'deadline': '下周三'},
        ]
    }
    
    # 3. 构造多轮问答
    qa_pairs = [
        {
            'question': '会议决定预算多少？',
            'answer': '会议批准了 10 万预算。',
            'evidence': {
                'video_timestamp': 1200,  # 第 20 分钟
                'audio_transcript': '我提议预算 10 万...通过',
                'screen_content': 'budget_slide.png'
            }
        }
    ]
    
    return meeting_video, qa_pairs
```

**场景 2: 音乐视频分析**
```python
def generate_music_video_analysis_data():
    """
    音乐视频理解：画面 + 音乐
    """
    music_video = load_music_video()
    
    # 需要理解：
    qa_pairs = [
        {
            'question': '这段音乐的情绪如何？',
            'answer': '欢快、激昂，配合画面中的舞蹈场景',
            'requires': ['audio_emotion', 'visual_dance']
        },
        {
            'question': '吉他独奏出现在画面的什么时候？',
            'answer': '第 45 秒，此时画面聚焦到吉他手',
            'requires': ['audio_event', 'visual_focus']
        }
    ]
    
    return music_video, qa_pairs
```

**场景 3: 安防监控事件检测**
```python
def generate_security_monitoring_data():
    """
    安防监控：视频 + 环境音
    """
    surveillance_video = load_surveillance()
    
    # 异常事件检测
    events = [
        {
            'timestamp': 3600,
            'event': '玻璃破碎',
            'evidence': {
                'audio': 'glass_breaking_sound',
                'visual': 'broken_window',
                'combined_confidence': 0.95
            }
        }
    ]
    
    return surveillance_video, events
```

**场景 4: OmniGAIA 风格多跳+工具使用**
```python
class OmniGAIAStyleDataGenerator:
    """
    构造 OmniGAIA 风格的多跳全模态数据
    """
    def generate(self, video_with_audio):
        # 1. 提取细粒度信号（同 OmniGAIA 事件图）
        signals = self.extract_signals(video_with_audio)
        
        # 2. 构建事件图
        graph = self.build_event_graph(signals)
        
        # 3. 多跳扩展（使用工具）
        expanded = self.multi_hop_expand(graph)
        
        # 4. 模糊化生成 QA
        qa = self.fuzzify_and_generate_qa(expanded)
        
        # 5. 人工验证
        verified = self.human_verify(qa)
        
        return verified
    
    def extract_signals(self, video):
        """细粒度信号提取"""
        return {
            'visual_events': self.detect_visual_events(video),
            'audio_events': self.detect_audio_events(video),
            'asr_transcript': self.transcribe(video.audio),
            'speaker_ids': self.diarize(video.audio),
        }
    
    def multi_hop_expand(self, graph):
        """多跳扩展（需要工具使用）"""
        # 例如：
        # - 识别到"埃菲尔铁塔" → 搜索确认位置在巴黎
        # - 听到"2024 奥运会" → 搜索确认时间
        # - 结合得出：这是 2024 巴黎奥运会开幕式
        
        tools = ['web_search', 'calculator', 'calendar']
        
        for hop in range(max_hops):
            # 规划下一步证据获取
            next_action = self.plan_next_evidence(graph, tools)
            
            # 执行工具调用
            result = self.execute_tool(next_action)
            
            # 更新图
            graph = self.expand_graph(graph, result)
        
        return graph
```

**数据量估算**：
- 每种场景：10K–50K 样本
- 总计：100K+ 高质量全模态场景数据

---

## 附录：代码仓库与资源链接

### 核心实现参考

| 组件 | 仓库 | 关键文件 |
|------|------|---------|
| MiniCPM-V 4.5 | https://github.com/OpenBMB/MiniCPM-V | `modeling_minicpmv.py` |
| MiniCPM-o | https://github.com/OpenBMB/MiniCPM-o | `processing_minicpmo.py` |
| Qwen2.5-Omni | https://github.com/QwenLM/Qwen2.5-Omni | `modeling_qwen2_5_omni.py` |
| Qwen3-Omni | https://github.com/QwenLM/Qwen3-Omni | 官方实现 |
| InfLLM-V2 | https://github.com/OpenBMB/infllmv2_cuda_impl | 稀疏注意力 CUDA 实现 |
| OmniVinci | https://github.com/NVlabs/OmniVinci | OmniAlignNet 实现 |
| OmniGAIA | https://github.com/RUC-NLPIR/OmniGAIA | 事件图构建代码 |

### 数据集下载

| 数据集 | HuggingFace 链接 |
|--------|-----------------|
| OmniCorpus | OpenGVLab/OmniCorpus |
| OmniCorpus-YT | OpenGVLab/OmniCorpus-YT |
| VALID | ontocord/MixtureVitae-VALID |
| FineVideo | HuggingFaceFV/finevideo |
| E-MM1 | https://e-mm1.github.io/ |
| WenetSpeech-Chuan | ASLP-lab/wenetspeech-chuan |
| LibriSpeech | openslr/librispeech_asr |
| MMAU | Sakshi113/MMAU |

---

## 5. 逐篇论文对比反思与方案完善

本节对 11 篇 Omni 论文逐篇与当前方案（Qwen3-LLM + SigLIP + Whisper-medium + CosyVoice2，六阶段训练）做对比，从数据、评测、训练三维度反思，并给出可落地的改进建议。

---

### 5.1 MiniCPM-V 4.5

**论文核心方案概要**  
MiniCPM-V 4.5 是 8B 高效 MLLM，聚焦三方面：统一 3D-Resampler 实现图像/视频高度紧凑编码（视频 96×、图像约 16× 压缩）；文档知识与 OCR 统一为「从损坏文档图像预测原文」的单一目标，无需外部 PDF 解析；混合 RL 同时优化短推理与长推理模式，rollout 随机交替两种模式。在 OpenCompass/VideoMME 等上超越 GPT-4o-latest 与 Qwen2.5-VL 72B，且显存与推理时间显著更低。

**与我们方案的对比**

| 维度 | 论文做法 | 我们当前方案 | 差距/优劣 |
|------|----------|--------------|-----------|
| 数据 | 动态视觉损坏三等级（低/中/高）统一文档知识+OCR；无重解析器 | 文档阶段用 DocVQA/ChartQA + 动态损坏脚本 | 我们已有损坏思路，可细化三等级与比例、与文档阶段数据配比 |
| 评测 | OpenCompass、VideoMME、30B 以下 SOTA | 有 Video-MME、MMBench 等，未强调 OpenCompass 全量 | 建议加入 OpenCompass 全量或子集，统一对比口径 |
| 训练 | 混合 RL：短/长推理模式随机交替 rollout，联合优化 | S3 用 GRPO + RLAIF-V，未显式分短/长推理模式 | 我们可增加「短/长推理模式」可控设计与混合 RL 目标 |
| 架构 | 3D-Resampler 统一图/视频，96× 视频压缩 | 3D-Resampler + Local-Global，视频压缩率未写死 96× | 我们架构兼容，可对标其压缩率与帧数设定做消融 |

**可行性分析**  
动态损坏三等级、混合 RL 均为算法与数据配置层面，无需新硬件；OpenCompass 评测为开源流程，可落地。

**效果预估**  
采纳后：文档/OCR 与知识学习更统一，减少解析依赖；Video-MME 等视频指标更稳；短/长推理可切换，用户体验与复杂推理兼顾。

**对我们方案的改进建议**  
1. 在文档阶段明确「低/中/高」损坏比例（如 3:4:3）与采样策略，与 DocVQA/ChartQA/InfoVQA 混合。  
2. S3 增加「短推理 / 长推理」两种行为标签，GRPO rollout 时按比例随机选模式，损失中显式加入模式一致性项。  
3. 评测清单中加入 OpenCompass 全量或选定子集，与 MiniCPM-V 4.5 报告对齐。  
4. 3D-Resampler 超参（每包帧数、每包 token 数）对齐 96× 视频压缩做一组消融，记录显存与精度。

---

### 5.2 Qwen3-Omni

**论文核心方案概要**  
Qwen3-Omni 为 Thinker–Talker 双 MoE 全模态模型：Thinker 负责多模态理解与文本生成并输出高层语义给 Talker；Talker 以多码本自回归 + MTP 补全帧内码本 + 因果 ConvNet Code2Wav 实现流式语音，冷启动首包约 234 ms。采用 TM-RoPE、自研 AuT（12.5 Hz）、SigLIP2-So400m 视觉编码；预训练三阶段（编码器对齐→通用→长上下文），Thinker 后训练含 SFT、强-弱蒸馏、GSPO，Talker 四阶段（多模态到语音映射、CPT+长上下文、多语言 DPO、说话人微调）。目标全模态无退化、多码本低延迟语音。

**与我们方案的对比**

| 维度 | 论文做法 | 我们当前方案 | 差距/优劣 |
|------|----------|--------------|-----------|
| 数据 | 预训练三阶段数据分层；Talker 四阶段专用数据（映射、长上下文、多语言 DPO、说话人） | 六阶段数据（S0–S5），语音有 CosyVoice2 与 Thinker-Talker 设计 | 我们缺「Talker 四阶段」的显式数据划分与多语言 DPO 数据 |
| 评测 | 36 个音频/音视频基准，文本/视觉/音乐/语音生成全覆盖 | 有 MMAU、DailyOmni、LibriSpeech 等，未列 36 项全表 | 建议对齐其评测集列表，补音乐与多语言 TTS 评测 |
| 训练 | Thinker MoE + Talker MoE；TM-RoPE；Talker 四阶段训练 | Qwen3-LLM + CosyVoice2；TMRoPE；Talker 为 CosyVoice2+Thinker-Talker 思路 | 我们为单 LLM+单语音模型，非双 MoE；可借鉴其 Talker 阶段切分与 MTP/Code2Wav 设计 |
| 架构 | AuT 12.5 Hz；多码本+MTP+因果 Code2Wav | Whisper-medium；CosyVoice2 多码本与流式 | 我们采用现成 Whisper+CosyVoice2，首包延迟可对标 234 ms 做优化 |

**可行性分析**  
双 MoE 与自研 AuT 成本高；我们沿用 Whisper+CosyVoice2 可行。TM-RoPE、Talker 四阶段数据与训练顺序可直接借鉴；MTP/因果 Code2Wav 需看 CosyVoice2 是否支持或需轻量适配。

**效果预估**  
采纳阶段切分与评测对齐后：语音与多模态路线更清晰；与 Qwen3-Omni 可比性更强；全模态无退化可作长期目标。

**对我们方案的改进建议**  
1. 将「Talker」侧训练显式拆为四阶段：多模态→语音映射、长上下文 CPT、多语言 DPO、说话人微调，并配数据与检查点。  
2. 引入 TM-RoPE（或与现有 TMRoPE 对齐），在长音视频与流式输入上做对比实验。  
3. 评测补充 Qwen3-Omni 使用的 36 个音频/音视频基准中的开源子集，以及音乐理解、多语言 TTS 指标。  
4. 若 CosyVoice2 支持 MTP 或因果 Code2Wav，在文档中写明并做首包延迟测试；否则标注为与 234 ms 的差距与后续优化点。

---

### 5.3 Ola

**论文核心方案概要**  
Ola 强调全模态训练中的模态关系与顺序：以视频为跨模态桥梁，连接图像与音频；渐进式训练从「差异最大的模态」到「对齐更紧的模态」，先单/双模态打好基础再在视频桥上做联合对齐。架构与数据、训练策略系统改进，目标与专用单模态模型竞争，权重与数据开源。

**与我们方案的对比**

| 维度 | 论文做法 | 我们当前方案 | 差距/优劣 |
|------|----------|--------------|-----------|
| 数据 | 渐进式各阶段对应不同模态配比；视频作为图文与音文之间的桥梁数据 | S0–S5 按阶段分，但未显式强调「视频为桥」的配比与顺序 | 我们可显式增加「视频桥」阶段的数据占比与顺序 |
| 评测 | 与专用单模态模型对比，全模态综合 | 有各模态评测，未强调「与专用模型并排」的表格 | 建议增加与同规模 VL/ASR 专用模型的并排对比 |
| 训练 | 渐进：差异大→对齐紧；视频居中 | 六阶段为 S0 基础→S1 多模态→…→S5 专项，顺序偏能力而非模态关系 | 我们可在一阶段内或阶段间注入「先图/音单模态→视频联合→全模态」的课程 |

**可行性分析**  
不改变总体架构，仅调整数据配比与训练顺序，完全可落地。

**效果预估**  
更清晰的「视频桥」与渐进顺序有望提升跨模态一致性与音视频联合理解，并减少模态间相互压制。

**对我们方案的改进建议**  
1. 在 S1/S2 中显式划分：先以图文+音文为主，再提高「视频+文本」「视频+音频」占比，最后再全模态混合，并在文档中写明各阶段视频占比目标。  
2. 设计「视频桥」专用数据子集：带对齐字幕或描述的短视频，同时用于视觉与听觉对齐。  
3. 评测报告增加与同规模专用 VL、ASR 模型的并排对比（如 7B/8B VL、同规模 ASR），突出全模态不牺牲单模态。

---

### 5.4 Baichuan-Omni

**论文核心方案概要**  
Baichuan-Omni 为首个开源 7B 全模态 MLLM：两阶段训练——先多模态对齐（视觉+音频投影到 LLM 空间），再跨图像/视频/音频/文本的多任务微调；Conv-GMLP 等架构改进；目标全模态理解与实时交互，作为社区基线。

**与我们方案的对比**

| 维度 | 论文做法 | 我们当前方案 | 差距/优劣 |
|------|----------|--------------|-----------|
| 数据 | 两阶段：对齐阶段数据 + 多任务微调阶段数据，简洁清晰 | 六阶段 S0–S5，阶段更多、数据组织更细 | 我们更细粒度，可保留；同时可抽象为「对齐→多任务」两阶段视图便于对外表述 |
| 评测 | 全模态与多模态基准，7B 竞争基线 | 评测集较全，未单独强调 7B/8B 基线对比 | 若我们为 7B/8B 级，可明确标注「7B 全模态基线」并与之对比 |
| 训练 | 对齐→多任务微调；Conv-GMLP 投影 | 多阶段含 S0 预训练、S1–S2 SFT、S3 RL、S4 长上下文、S5 专项 | 我们阶段更丰富；投影层可评估 Conv-GMLP 类结构是否带来收益 |

**可行性分析**  
两阶段视图为归纳方式，不改变实现；Conv-GMLP 需在现有投影上做替换实验，成本可控。

**效果预估**  
明确「对齐→多任务」视图利于与 Baichuan-Omni 及社区对齐话术；Conv-GMLP 若有效可提升小模型下的多模态稳定性。

**对我们方案的改进建议**  
1. 在文档中增加「两阶段视图」：阶段 1 = 对齐（对应我们 S0–S1），阶段 2 = 多任务微调与强化（对应 S2–S5），便于与 Baichuan-Omni 等对比。  
2. 在投影层做 Conv-GMLP 与现有 MLP 的对比实验（收敛速度、显存、精度），若有利则纳入默认配置。  
3. 若模型为 7B/8B，在评测与摘要中明确写「7B 全模态基线」，并引用 Baichuan-Omni 等做对比。

---

### 5.5 LongCat-Flash-Omni

**论文核心方案概要**  
LongCat-Flash-Omni 为 560B 参数（激活约 27B）的开源全模态模型，基于 Shortcut-connected MoE + zero-computation experts 实现低延迟实时音视频交互。采用课程式渐进训练：从较简单到更复杂的模态序列建模任务过渡，在保持强单模态能力下获得全面多模态能力。训练基础设施上提出模态解耦并行，多模态训练时维持超 90% 纯文本训练吞吐。

**与我们方案的对比**

| 维度 | 论文做法 | 我们当前方案 | 差距/优劣 |
|------|----------|--------------|-----------|
| 数据 | 课程式渐进：先单/双模态短序列，再多模态长序列、复杂跨模态任务 | S0–S5 按阶段分，有渐进但未显式「简单→复杂模态序列」的课程 | 我们可显式设计「模态序列复杂度」课程（单模态→双模态→全模态混合、短→长） |
| 评测 | 全模态基准、单模态不退化、实时音视频交互指标 | 有全模态与单模态评测，未单独强调「吞吐/延迟」与纯文本对比 | 可增加多模态训练时吞吐占比、推理延迟与纯文本基线的对比 |
| 训练 | 模态解耦并行；560B MoE 仅 27B 激活 | 未涉及超大规模并行与模态解耦 | 我们若为 7B/8B 可忽略解耦并行；课程式渐进可直接借鉴 |

**可行性分析**  
课程式渐进为数据与阶段设计，无需 560B 资源；我们按「简单模态/短序列→复杂模态/长序列」调整阶段顺序即可落地。

**效果预估**  
采纳课程式渐进后训练更稳、单模态退化风险降低；若未来做大规模多模态训练，模态解耦并行思路可预留为扩展方向。

**对我们方案的改进建议**  
1. 在 S1–S2 显式定义「模态序列复杂度」：先单模态/短序列图文、音文，再双模态视频+文本、视频+音频，最后全模态长序列混合。  
2. 文档中增加「课程表」：各阶段最小/最大序列长度、模态组合类型、占比目标。  
3. 若有分布式多模态训练，调研并记录「模态解耦并行」在小规模下的收益（如按模态分数据并行），便于后续扩展。

---

### 5.6 MiniCPM-SALA

**论文核心方案概要**  
MiniCPM-SALA 为 9B 混合注意力架构：25% InfLLM-V2 稀疏注意力 + 75% Lightning 线性注意力，层选择算法确定稀疏层位置；HyPE 混合位置编码协调短/长上下文；Transformer-to-hybrid 持续训练框架，相比从头训练约省 75% 成本。单 A6000D 上 256K 时推理约 3.5× 全注意力加速，支持最长 1M token。

**与我们方案的对比**

| 维度 | 论文做法 | 我们当前方案 | 差距/优劣 |
|------|----------|--------------|-----------|
| 数据 | 持续训练：预训练 Transformer 转为混合模型，无需从头 | S4 长上下文阶段用长文本/长视频数据 | 我们已有 S4；可明确「先全注意力预训练再转混合」的路径以省成本 |
| 评测 | 256K/1M 长度、推理速度、与全注意力能力对比 | 长上下文有 LongVideoBench 等，未写 256K/1M 与速度 | 建议增加 256K 序列下的吞吐/延迟、与 Qwen3-8B 全注意力的对比 |
| 训练 | 1:3 稀疏-线性混合；HyPE；Transformer-to-hybrid | 突破方向含 HybridSparseLinearAttention、InfLLM-V2+Lightning | 我们方案已对齐；需落实层比例与 HyPE 是否替代/兼容 TMRoPE |

**可行性分析**  
InfLLM-V2 与 Lightning 有开源实现；HyPE 与 TMRoPE 在 Omni 中可并存（文本/位置用 HyPE 或 RoPE，多模态时间用 TMRoPE）。Transformer-to-hybrid 需在现有 Qwen3 上做转换实验，成本可控。

**效果预估**  
采纳后长上下文显存与速度显著改善，1M 级上下文可纳入评测与产品目标。

**对我们方案的改进建议**  
1. 突破方向「稀疏注意力」中明确：层比例 1:3（InfLLM-V2 : Lightning）、层选择算法、QK-Normalization，并写进配置与伪代码。  
2. 评估 HyPE 与 TMRoPE 的兼容方案：长文本用 HyPE、多模态时间用 TMRoPE，或在统一框架下做消融。  
3. 长上下文阶段采用「先全注意力预训练再转混合」的 Transformer-to-hybrid 路径，并估算训练成本节省比例。  
4. 评测增加 256K token 序列的推理速度（与同规模全注意力对比）及 1M 可运行性验证。

---

### 5.7 OmniVinci

**论文核心方案概要**  
OmniVinci 强调架构与数据：OmniAlignNet 在共享潜空间做视觉-音频对比学习；TEG 按时间分组编码相对时间；CRTE 注入绝对时间且与 RoPE 兼容。数据管线生成 24M 单模态与全模态对话，仅 0.2T token 即超过 Qwen2.5-Omni（1.2T）在 DailyOmni（+19.05）、MMAR（+1.7）、Video-MME（+3.9）。

**与我们方案的对比**

| 维度 | 论文做法 | 我们当前方案 | 差距/优劣 |
|------|----------|--------------|-----------|
| 数据 | 24M 对话、单模态+全模态合成管线、0.2T 高效利用 | 各阶段数据量更大、来源多，未强调「少 token 高回报」 | 我们可增加「高质量全模态对话」的筛选与合成管线，对标 0.2T 级高效配方 |
| 评测 | DailyOmni、MMAR、Video-MME 等，直接对比 Qwen2.5-Omni | 有 DailyOmni、Video-MME，未列 MMAR | 建议补 MMAR 评测；与 Qwen2.5-Omni 同设置对比以便引用 |
| 训练 | OmniAlignNet 对比损失；TEG+CRTE 时间编码 | 多模态对齐有投影与 TMRoPE，无显式视-听对比、无 TEG/CRTE | 我们可增加视-听对比损失与 TEG/CRTE 类时间编码作为可选模块 |

**可行性分析**  
OmniAlignNet 为额外对齐模块与损失，TEG/CRTE 为位置/时间编码扩展，均可在我们现有骨干上叠加；24M 级对话合成需数据管线与质量把控，可分批落地。

**效果预估**  
视-听对比与 TEG/CRTE 有望提升音视频联合理解与时间敏感任务；少 token 高回报配方可降低总训练成本。

**对我们方案的改进建议**  
1. 在视觉-音频对齐阶段增加「OmniAlignNet 式」对比损失：batch 内 (V_i, A_i) 正样本、跨样本负样本，与现有投影联合训练。  
2. 评估 TEG（按时间分组排序 token）与 CRTE（绝对时间编码与 RoPE 兼容）的接入方式，与现有 TMRoPE 做消融（保留其一或组合）。  
3. 构建「高质量全模态对话」子集：单模态+全模态混合、24M 量级目标，并记录对应 token 量与 DailyOmni/Video-MME 表现曲线。  
4. 评测清单加入 MMAR，并与 Qwen2.5-Omni 在相同数据/设置下对比，便于复现与引用。

---

### 5.8 OmniGAIA

**论文核心方案概要**  
OmniGAIA 为全模态智能体基准与 OmniAtlas 智能体：事件图驱动构建 360 任务、9 领域，需多轮工具调用与可验证开放式答案；OmniAtlas 采用 TIR（工具集成推理）、主动全模态感知（选择性看/听片段）；训练用后见之明引导树搜索合成轨迹 + 轨迹级 SFT + OmniDPO 细粒度纠错。Gemini-3-Pro 62.5 Pass@1，Qwen3-Omni 13.3，OmniAtlas 方案将 Qwen3-Omni 提至 20.8。

**与我们方案的对比**

| 维度 | 论文做法 | 我们当前方案 | 差距/优劣 |
|------|----------|--------------|-----------|
| 数据 | 事件图驱动多跳 QA、工具证据扩展、节点/边模糊化、LLM+人工筛 | S5 专项含智能体/工具，未采用事件图与多跳 QA 构造 | 我们可引入事件图思路构造「跨模态多跳+工具」数据，用于 S5 |
| 评测 | OmniGAIA 360 任务、Pass@1、工具使用与感知策略分析 | 有 OmniGAIA 等基准，未强调工具使用与错误类型分析 | 建议将 OmniGAIA 列为必测，并做工具调用率、错误类型拆解 |
| 训练 | 轨迹级 SFT + OmniDPO 细粒度纠错；主动感知（选段看/听） | S3 有 GRPO/RLAIF-V，无轨迹级 DPO、无主动感知训练 | 我们可增加轨迹级 DPO（OmniDPO 式）与主动感知数据/损失 |

**可行性分析**  
事件图构建需标注或半自动管线，可先做小规模试点；OmniDPO 为 DPO 在轨迹/细粒度错误上的扩展，实现成本适中；主动感知需在推理或训练中接入「选段」逻辑，为架构与数据联合改动。

**效果预估**  
采纳后智能体与工具使用能力可量化提升（如 OmniGAIA Pass@1），与 SOTA 商业模型差距可缩小。

**对我们方案的改进建议**  
1. S5 智能体/工具阶段：引入「事件图驱动」数据构造——从音视频抽取实体/事件与关系，扩展工具证据，模糊化节点/边生成多跳 QA，并做可解性与唯一性筛选。  
2. 在 S3 或 S5 增加轨迹级 DPO（OmniDPO）：对工具调用轨迹做正负样本对，细粒度纠错，与现有 GRPO 并存或分阶段使用。  
3. 评测固定包含 OmniGAIA，报告 Pass@1 及工具使用率、感知策略、错误类型统计，便于与 OmniAtlas 对比。  
4. 研究「主动全模态感知」：长媒体上按需选段看/听而非全局下采样，在数据构造或模型接口上预留扩展点。

---

### 5.9 Mini-Omni2

**论文核心方案概要**  
Mini-Omni2 以「最接近 GPT-4o 形态的开源复现」为目标：集成预训练视觉与听觉编码器，三阶段训练——先模态与语言空间对齐，再多模态理解与文本生成，最后文本/多模态→语音输出；支持实时端到端语音回复。交互上引入基于命令的打断机制（如「停」「打断一下」），使模型能识别并立即响应，实现更灵活的双工交互。

**与我们方案的对比**

| 维度 | 论文做法 | 我们当前方案 | 差距/优劣 |
|------|----------|--------------|-----------|
| 数据 | 三阶段对应不同数据：对齐→多模态文本→语音输出，有限数据即可 | 六阶段 S0–S5，数据更细但未强调「有限数据」下的最小可行集 | 我们可抽象出「最小三阶段数据清单」便于复现与对外表述 |
| 评测 | 视觉、音频单模态保持、端到端语音响应、双工交互 | 有各模态评测与全双工突破方向，未单独测「命令式打断」 | 可增加打断命令识别率、打断到停播延迟等指标 |
| 训练 | 三阶段严格顺序：对齐→多模态文本→语音 | 我们 S1–S2 多模态 SFT、Talker 侧可拆为多阶段 | 我们阶段更细，可保留；同时标注与 Mini-Omni2 三阶段的对应关系 |

**可行性分析**  
三阶段视图与命令式打断均为设计与数据标注层面；打断机制需在推理链路中接入命令检测与中断逻辑，工程可落地。

**效果预估**  
明确三阶段对应关系利于与社区对齐；命令式打断可显著提升双工体验与产品化程度。

**对我们方案的改进建议**  
1. 在文档中增加「三阶段视图」与 Mini-Omni2 的对应：阶段 1=对齐（对应 S0–S1），阶段 2=多模态文本（对应 S2），阶段 3=语音输出（对应 Talker 各阶段）。  
2. 全双工突破方向中增加「命令式打断」：定义打断命令集（中英文）、在 SFT 数据中加入打断/续说样本、推理时实时检测用户语音中的命令并触发停播/清空状态。  
3. 评测增加「打断识别率」「打断到停播延迟」等指标，并与无打断基线对比。

---

### 5.10 Ming-Flash-Omni

**论文核心方案概要**  
Ming-Flash-Omni 基于 Ling-Flash-2.0 的稀疏 MoE（100B 总参数、6.1B 激活）统一视觉、语音与语言。在上下文 ASR 上 12 个基准全部刷新；图像生成侧高保真文字渲染与编辑时场景/身份一致性；新增生成式分割（独立分割 + 生成中的空间控制与编辑一致性）。单架构内同时实现感知与生成 SOTA。

**与我们方案的对比**

| 维度 | 论文做法 | 我们当前方案 | 差距/优劣 |
|------|----------|--------------|-----------|
| 数据 | 上下文 ASR、方言 ASR、文生图、生成式分割等多类数据联合 | 我们有 ASR、语音、视觉数据，未强调「上下文 ASR」与「生成式分割」数据 | 若要做上下文 ASR 与生成式分割，需补充对应数据与任务 |
| 评测 | 12 个上下文 ASR、文生图、生成式分割基准 | 有 LibriSpeech/WenetSpeech、视觉评测，未列 12 个上下文 ASR 与生成式分割 | 若目标包含上述能力，需将评测集纳入 |
| 训练 | 稀疏 MoE 100B；统一多模态感知与生成 | 我们为稠密 LLM + 多模态编码器，无文生图与生成式分割 | 我们定位不同；可借鉴「上下文 ASR」数据构造与评测思路 |

**可行性分析**  
我们方案若不做文生图与生成式分割，可仅借鉴上下文 ASR 与方言感知的数据/评测思路；若做，需单独设计生成与分割模块与数据。

**效果预估**  
采纳上下文 ASR 与方言数据后，语音识别在专有名词与多口音场景下更稳；生成式分割为可选扩展方向。

**对我们方案的改进建议**  
1. 在语音阶段明确「上下文 ASR」：利用对话/文档上下文（如专有名词列表、领域词表）构造上下文相关 ASR 数据，并在评测中加入 1–2 个上下文 ASR 基准。  
2. 若资源允许，评估「方言/多口音」数据比例与方言感知 ASR 评测，作为可选增强。  
3. 文生图与生成式分割列为可选扩展方向，在文档中注明与 Ming-Flash-Omni 的差异及当前不做的理由（如聚焦理解与语音生成）。

---

### 5.11 Qwen2.5-Omni

**论文核心方案概要**  
Qwen2.5-Omni 为 Thinker–Talker 端到端多模态模型：音视频分块处理、TMRoPE 时间对齐、音视频按 2 秒块交错排列；Thinker 生成文本与高层表示，Talker 双轨自回归生成语音 token；滑动窗口 DiT 限制感受野以降低首包延迟。与同规模 Qwen2.5-VL 相当、超越 Qwen2-Audio，OmniBench 等 SOTA；语音指令跟随与 MMLU/GSM8K 文本能力相当；seed-tts-eval 上超越 CosyVoice 2。

**与我们方案的对比**

| 维度 | 论文做法 | 我们当前方案 | 差距/优劣 |
|------|----------|--------------|-----------|
| 数据 | 多模态预训练与 SFT、Talker 训练数据；2 秒块时间交错 | 我们已有 TMRoPE、分块与交错思路；Talker 用 CosyVoice2 | 我们方案与 Qwen2.5-Omni 同源思路，可对齐 2 秒块与 seed 评测 |
| 评测 | OmniBench、MMLU、GSM8K、seed-tts-eval、语音指令跟随 | 有 OmniBench、MMLU 等，未强调 seed-tts-eval 与语音指令跟随 | 建议将 seed-tts-eval 与「语音指令跟随 vs 文本」对比列入必测 |
| 训练 | Thinker-Talker；滑动窗口 DiT；TMRoPE | 我们采用 Thinker-Talker + CosyVoice2；TMRoPE；突破方向有流式与首包 | 我们可对比滑窗 DiT 与 CosyVoice2 解码方式，优化首包延迟 |

**可行性分析**  
我们与 Qwen2.5-Omni 架构思路一致，可直接对标其评测与 2 秒块设置；seed-tts-eval 为公开评测，可落地。

**效果预估**  
对齐 seed-tts-eval 与语音指令跟随评测后，可与 Qwen2.5-Omni 直接对比；滑窗解码与首包优化可进一步缩小与官方实现的延迟差距。

**对我们方案的改进建议**  
1. 评测清单固定包含 seed-tts-eval（test-zh / test-en / test-hard）及「语音指令跟随 vs 文本指令」能力对比（如 MMLU/GSM8K 语音输入 vs 文本输入）。  
2. 在音视频输入处明确「2 秒块时间交错」策略，与文档中 TMRoPE 与分块描述一致。  
3. 若 CosyVoice2 支持滑窗或因果解码，在突破方向中写明并做首包延迟测试；否则标注与 Qwen2.5-Omni 滑窗 DiT 的差异及后续优化方向。

---

## 6. 综合反思与终极方案

### 6.1 11 篇反思的共性发现

**数据维度**  
- **少 token 高回报**：OmniVinci 0.2T 即超 Qwen2.5-Omni 1.2T 说明高质量对话与对齐数据比粗放规模更重要；建议我们做「高质量全模态对话」筛选与合成管线，并设 0.2T 级高效配方实验。  
- **阶段与模态顺序**：Ola、LongCat、Mini-Omni2 等均强调「先单/双模态、再全模态」「先对齐、再多任务」的渐进顺序；我们已有 S0–S5，需显式写出「模态序列复杂度」课程与「视频为桥」的占比。  
- **工具与智能体**：OmniGAIA 事件图驱动多跳 QA 与工具轨迹、OmniDPO 细粒度纠错；我们 S5 可引入事件图构造与轨迹级 DPO。  
- **文档与 OCR**：MiniCPM-V 4.5 的动态损坏三等级统一文档知识+OCR，我们可直接采用并细化比例。

**评测维度**  
- **统一对比口径**：多篇与 Qwen2.5-Omni、OpenCompass、DailyOmni、Video-MME、OmniGAIA、seed-tts-eval 等对齐；我们应固定上述评测子集并统一报告格式。  
- **单模态不退化**：与同规模 VL/ASR 单模态模型并排对比、语音指令跟随 vs 文本指令对比，需在报告中显式呈现。  
- **长上下文与效率**：MiniCPM-SALA 的 256K/1M、推理速度对比；我们长上下文与突破方向需增加吞吐/延迟与 256K 级评测。

**训练维度**  
- **混合 RL 与短/长推理**：MiniCPM-V 4.5 的混合 RL、可控短/长推理；我们 S3 可增加双模式 rollout 与损失。  
- **视-听对比与时间编码**：OmniVinci 的 OmniAlignNet、TEG、CRTE；我们可在对齐阶段增加视-听对比，并评估 TEG/CRTE 与 TMRoPE 的组合。  
- **Transformer-to-hybrid**：MiniCPM-SALA 的持续训练省约 75% 成本；我们长上下文阶段可采用「先全注意力再转混合」的路径。  
- **命令式打断与双工**：Mini-Omni2 的命令式打断；我们全双工突破方向需落实命令集与打断逻辑。

### 6.2 当前方案的主要薄弱环节

1. **数据**：缺少「0.2T 级高效配方」的显式设计与高质量全模态对话筛选管线；事件图与多跳工具数据未纳入 S5；文档阶段动态损坏未细化三等级比例。  
2. **评测**：OpenCompass 全量/子集、MMAR、seed-tts-eval、语音指令跟随 vs 文本、OmniGAIA 工具与错误分析、256K 速度与 1M 可运行性等尚未全部列入或未统一格式。  
3. **训练**：S3 未显式区分短/长推理模式与混合 RL；Talker 四阶段与 OmniDPO 轨迹级 DPO 未在正文中写清；视-听对比与 TEG/CRTE 为可选未落实。  
4. **突破方向**：命令式打断未具体到命令集与接口；稀疏/混合注意力的层比例与 HyPE 与 TMRoPE 的兼容方案未完全确定。

### 6.3 完善版终极方案要点

**架构**  
- 保持 Qwen3-LLM + SigLIP + Whisper-medium + CosyVoice2；在视觉-音频对齐阶段增加 OmniAlignNet 式视-听对比损失（可选）。  
- 评估 TEG + CRTE 与 TMRoPE 的组合或分工（如 TMRoPE 主、CRTE 补绝对时间）。  
- 长上下文：明确 HybridSparseLinearAttention 1:3 比例、层选择、HyPE 与 TMRoPE 兼容方案；采用 Transformer-to-hybrid 持续训练路径。

**数据**  
- 增加「高质量全模态对话」子集与 24M 级合成管线目标；设 0.2T 级高效配方实验并记录 DailyOmni/Video-MME 曲线。  
- 文档阶段：动态损坏低/中/高 3:4:3（可调），与 DocVQA/ChartQA/InfoVQA 混合。  
- S1–S2：显式「模态序列复杂度」课程（单模态→双模态→全模态、短→长）与「视频为桥」占比。  
- S5：事件图驱动多跳工具 QA 构造、轨迹级数据；Talker 侧显式四阶段数据划分。

**训练**  
- S3：增加短/长推理双模式、混合 RL rollout 与损失；增加 OmniDPO 轨迹级 DPO（与 GRPO 分阶段或并存）。  
- Talker：显式四阶段（多模态→语音映射、CPT+长上下文、多语言 DPO、说话人微调）并配检查点。  
- 全双工：定义打断命令集、SFT 中加入打断/续说样本、推理链路接入命令检测与中断。

**评测**  
- 固定清单：OpenCompass 子集、DailyOmni、Video-MME、MMAR、OmniGAIA、OmniBench、seed-tts-eval、语音 vs 文本指令跟随（MMLU/GSM8K）、256K 推理速度与 1M 可运行性、打断识别与延迟。  
- 报告格式：与 Qwen2.5-Omni、OmniVinci、MiniCPM-V 4.5 等可对齐的表格与设置。

**突破方向优先级**  
1. 稀疏/混合注意力 + 长上下文（显存与速度收益大，可复用 MiniCPM-SALA 方案）。  
2. 高质量数据与 0.2T 高效配方（提升训练性价比）。  
3. 视-听对比与 TEG/CRTE（提升音视频联合理解）。  
4. 命令式打断与全双工（产品化与体验）。  
5. 事件图与 OmniDPO（智能体与工具使用）。

### 6.4 风险与缓解

- **数据与标注成本**：事件图、多跳 QA、打断标注需人力或半自动；先做小规模试点再扩量。  
- **架构改动**：视-听对比、TEG/CRTE、HyPE 与 TMRoPE 并存可能增加调参；先做消融再全量接入。  
- **评测一致性**：不同论文评测设置略有差异；我们固定自选子集并注明与各论文的差异。

### 6.5 资源估算更新

- 在现有六阶段资源估算基础上，增加「0.2T 高效配方」实验的 GPU 与数据预算（约 1/5–1/6 全量）。  
- Transformer-to-hybrid 长上下文阶段：按「约 25% 相对从头训练」的节省比例，更新训练时长与卡数估算。  
- 其余阶段（S0–S5）按原方案；若增加 OmniDPO 与事件图数据，S5 数据量与训练时间约上浮 10–20%，需在迭代中细化。

---

## 7. 完善版方案（可落地实施）

本章在原有方案（第 1–4 章）与反思（第 5–6 章）基础上，给出**可直接执行**的完善版训练方案、数据方案与评测方案，确保技术可行、能落地、有优势、表述清楚。

---

### 7.1 完善版训练方案

#### 7.1.1 阶段总览与两视图对应

| 两阶段视图（对外表述） | 六阶段实施（内部执行） | 核心目标 |
|------------------------|------------------------|----------|
| **阶段一：多模态对齐** | S0 基础 + Phase 1 Adapter 对齐 + Phase 2 编码器微调 | 视觉/音频投影与编码器对齐到 LLM 空间；文档 OCR+知识统一 |
| **阶段二：多任务与强化** | Phase 3 联合预训练 + Phase 4 长上下文 + Phase 5 SFT + Phase 6 RL/DPO | 全模态理解、长上下文、指令遵循、偏好与工具 |

**与论文对齐**：阶段一 ≈ Baichuan-Omni / Mini-Omni2 的「对齐阶段」；阶段二 ≈ 其「多任务微调」+ 我们的 SFT/RL 与 Talker 四阶段。

#### 7.1.2 各阶段可执行配置（含反思改进）

**Phase 1：Adapter 对齐**

- **必选**：图文 100M 级 + 音文 50K 小时；视觉投影 2 层 MLP 或 3D-Resampler，音频投影 MLP 或 Conv-GMLP。
- **新增（来自反思）**：
  - **视-听对比损失（可选）**：在 batch 内对同一样本的视觉摘要与音频摘要做 CLIP 式对比损失（OmniVinci 思路），与 caption/ASR 损失联合训练；权重 0.1–0.2，避免压制主任务。
  - **验收**：投影 loss < 0.5；COCO caption 验证 CIDEr；LibriSpeech dev WER 不劣于基线。

**Phase 2：编码器微调**

- **必选**：OCR 密集（DocVQA/ChartQA/InfoVQA + 动态损坏合成）；音频事件（AudioSet/MusicCaps/MMAU）。
- **新增（来自反思）**：
  - **文档动态损坏三等级比例**：低:中:高 = **3:4:3**（可调）；低=高斯噪声 σ=10，中=模糊+噪声，高=文本区域遮蔽；目标为「从损坏图预测原文」。
  - **验收**：DocVQA/ChartQA 验证集准确率；LibriSpeech test-clean WER。

**Phase 3：多模态联合预训练**

- **必选**：文本 40% + 图文 25% + 视频 15% + 音频 10% + 跨模态 10%；渐进式模态调度（先图文→再视频→再音频+跨模态）。
- **新增（来自反思）**：
  - **「视频为桥」显式占比**：在总 token 中，带音轨的短视频（≤2 分钟）或 2 秒块音视频交错数据占比 ≥ **8%**，与 VALID/E-MM1 等来源对齐。
  - **0.2T 高效配方实验（可选）**：单独跑一版约 **0.2T token** 的高质量子集（24M 级全模态对话 + 精选图文/音视频），记录 DailyOmni、Video-MME 曲线，用于验证「少 token 高回报」。
  - **验收**：验证集 perplexity；抽检 DailyOmni/Video-MME 若干题不下降。

**Phase 4：长上下文**

- **必选**：长视频（LongVideoBench/FineVideo 长段）、长音频（WenetSpeech 长段）、长对话。
- **新增（来自反思）**：
  - **Transformer-to-hybrid 路径**：先以**全注意力**在 32K–64K 长度上预训练 1–2 个 checkpoint，再转为 **HybridSparseLinearAttention**（1:3 InfLLM-V2 : Lightning），在相同或更长数据上持续训练，目标节省约 **25%** 总训练成本（相对从头混合架构）。
  - **HyPE 与 TMRoPE**：长文本用 HyPE（若采用 MiniCPM-SALA 方案），多模态时间维保留 TMRoPE；兼容方案为「LLM 层用 HyPE，多模态 token 时间用 TMRoPE」。
  - **验收**：256K 长度推理可通过；LongVideoBench 或等价长视频 QA 指标。

**Phase 5：SFT**

- **必选**：通用 SFT（ShareGPT4V/LLaVA-Instruct/VideoChat2）+ 约 10% 纯文本防退化；Long-CoT 与推理类数据。
- **新增（来自反思）**：
  - **短/长推理双模式数据**：构造「短答」与「长链式推理」两类样本，在数据中打标签（`reasoning_mode: short | long`），供 Phase 6 混合 RL 使用。
  - **命令式打断数据**：加入 **打断/续说** 样本：用户说「停」「打断一下」「换个话题」等，目标为模型停止生成或清空状态；命令集见 7.2.2。
  - **验收**：MMBench/MMMU 抽检；人工抽检打断回复正确性。

**Phase 6：RL/DPO 与 Talker**

- **必选**：GRPO + RLAIF-V 偏好学习；CosyVoice2 / Thinker-Talker 语音输出。
- **新增（来自反思）**：
  - **混合 RL**：rollout 时按比例（如 50%:50%）随机选择**短推理**或**长推理**模式，与偏好损失联合优化；保证单一模型既可快速回答又可链式推理。
  - **OmniDPO（轨迹级 DPO）**：在工具/智能体轨迹数据上，构造「正确轨迹 vs 错误轨迹」偏好对，做轨迹级 DPO；与 GRPO 可同阶段或紧接其后，权重 0.3–0.5。
  - **Talker 四阶段数据**：显式划分 (1) 多模态→语音映射数据；(2) CPT+长上下文语音数据；(3) 多语言 DPO 语音偏好数据；(4) 说话人微调数据；每阶段对应独立 dataloader 与 checkpoint。
  - **验收**：偏好胜率；OmniGAIA 或工具类基准 Pass@1；seed-tts-eval 与语音自然度。

#### 7.1.3 检查点与里程碑

| 里程碑 | 阶段 | 必测指标 | 通过标准 |
|--------|------|----------|----------|
| M1 | Phase 1 结束 | 投影 loss、COCO CIDEr、LibriSpeech WER | loss<0.5，WER 不劣于基线 |
| M2 | Phase 2 结束 | DocVQA/ChartQA、LibriSpeech | 明显优于 Phase 1 |
| M3 | Phase 3 结束 | 验证 perplexity、DailyOmni/Video-MME 抽检 | 全模态理解无崩溃 |
| M4 | Phase 4 结束 | 256K 可跑、LongVideoBench | 长上下文可用 |
| M5 | Phase 5 结束 | MMBench/MMMU、打断抽检 | 指令遵循与打断正确 |
| M6 | Phase 6 结束 | 偏好胜率、OmniGAIA、seed-tts-eval | 达预设基线 |

---

### 7.2 完善版数据方案

#### 7.2.1 数据总览与 0.2T 高效配方

- **全量路线**：按第 1 章与 1.2 节组织，Phase 3 约 2T token，其余阶段按 1.2 节比例。
- **0.2T 高效配方（可选，用于验证与快速迭代）**：
  - **目标**：用约 **0.2T token** 达到 DailyOmni、Video-MME 等与全量可比或可接受的性能（参考 OmniVinci）。
  - **构成**：
    - 高质量全模态对话：**24M 条**级别，单模态 + 全模态混合；来源为 GPT-4o/Gemini 蒸馏、ShareGPT4V/ALLiAVA 扩展、自建音视频 QA 转对话。
    - 图文：OmniCorpus/LAION 高 CLIP 子集，约 50B token。
    - 音视频：VALID + E-MM1 + 自建 2 秒块交错，约 30B token。
    - 纯文本：SlimPajama/FineWeb 高质量子集，约 80B token。
  - **执行**：单独建 dataloader 与 config（如 `phase3_02t_recipe.yaml`），记录各 checkpoint 在 DailyOmni/Video-MME 上的曲线，与全量对比。

#### 7.2.2 模态课程与「视频为桥」占比

- **模态序列复杂度课程**（Phase 3 内或 S1–S2 等效）：
  1. **第 1 段**（约 0–25% 步数）：单模态/双模态为主——图文 60% + 纯文本 40%。
  2. **第 2 段**（约 25–50%）：加入视频——图文 40% + 视频 20% + 纯文本 40%。
  3. **第 3 段**（约 50–75%）：加入音频——图文 25% + 视频 15% + 音频 10% + 纯文本 40% + 跨模态 10%。
  4. **第 4 段**（约 75–100%）：全模态均衡，与 3.1.4 节配置一致。
- **「视频为桥」**：带音轨的视频或 2 秒块音视频交错数据，在 Phase 3 总 token 中 ≥ **8%**；来源：VALID、E-MM1、OmniCorpus-YT、自建 2s-chunk。

#### 7.2.3 文档阶段与 OCR 数据

- **动态损坏比例**：低:中:高 = **3:4:3**；每 batch 按此比例采样。
- **数据源**：DocVQA、ChartQA、InfoVQA、OCRBench 训练集；合成时对同一文档可生成多等级损坏样本，目标均为「预测原文」。
- **与 Phase 2 的衔接**：Phase 2 的 OCR 密集数据中，动态损坏样本占比建议 ≥ **50%**，其余为原始 QA。

#### 7.2.4 S5 智能体/工具与事件图（可选）

- **事件图驱动多跳 QA**（OmniGAIA 思路）：
  1. 从音视频/图文数据中抽取**实体、事件、关系**，构建初始事件图。
  2. 通过跨模态检索或外部工具（搜索/代码）扩展**下一跳证据**。
  3. 对关键节点/边做**模糊化**生成多跳问答，再经 LLM 筛选与人工抽检（可解性、唯一性）。
- **规模**：先 **1K–5K** 条高质量多跳工具 QA 试点，再扩到 10K+；与 Phase 6 的 OmniDPO 轨迹数据共用来源。
- **轨迹级 DPO**：从工具执行轨迹中构造「正确轨迹 vs 错误轨迹」偏好对，格式与现有 DPO 一致，额外字段标注轨迹 ID 与工具调用序列。

#### 7.2.5 命令式打断与全双工

- **打断命令集（中英文）**：
  - 中文：`停`、`打断一下`、`别说了`、`换个话题`、`等一下`。
  - 英文：`stop`、`interrupt`、`wait`、`change topic`、`hold on`。
- **数据**：在 SFT 数据中增加「用户说打断命令 → 模型停止或输出 [INTERRUPTED]」样本，建议 **2K–5K** 条；推理时用小型分类器或关键词检测识别命令并触发停播/清状态。
- **与评测衔接**：见 7.3 节「打断识别率」「打断到停播延迟」。

#### 7.2.6 Talker 四阶段数据划分

| Talker 阶段 | 数据内容 | 建议规模 |
|-------------|----------|----------|
| 1 多模态→语音映射 | 多模态输入（图/音/视频+文本）→ 语音输出；TTS 或录音对齐 | 约 50K–100K 条 |
| 2 CPT+长上下文 | 长上下文多模态输入 → 语音回复；会议/长视频摘要朗读 | 约 20K–50K 条 |
| 3 多语言 DPO | 多语言语音偏好对（好/坏自然度、口音一致性） | 约 10K–20K 对 |
| 4 说话人微调 | 目标说话人语音克隆/适配数据 | 按产品需求 |

---

### 7.3 完善版评测方案

#### 7.3.1 固定评测清单（必测）

**图像理解**  
- MMBench / MMBench-CN、MMMU（或 MMMU-Pro 子集）、MathVista、HallusionBench。

**视频理解**  
- Video-MME、LongVideoBench（或等价长视频 QA）；可选 MVBench。

**音频理解**  
- LibriSpeech（test-clean / test-other）WER、WenetSpeech CER；MMAU 或 MMAU-Pro；可选 VoiceBench。

**全模态 / 音视频**  
- DailyOmni、OmniBench、WorldSense；**OmniGAIA**（Pass@1 + 工具使用率 + 错误类型统计）。

**语音生成与语音指令**  
- **seed-tts-eval**（test-zh / test-en / test-hard）WER 或自然度；**语音指令跟随**：MMLU、GSM8K 等用**语音输入** vs **文本输入**的准确率对比（目标：语音≈文本）。

**长上下文与效率**  
- 256K token 序列的**推理吞吐**（tokens/s）及**显存占用**；与同规模全注意力模型对比；可选 1M 可运行性验证。

**全双工**  
- **打断识别率**：命中命令即识别为打断的比例；**打断到停播延迟**：从检测到命令到播放停止的 ms 数。

**对齐论文的补充**  
- OpenCompass 子集（与 MiniCPM-V 4.5 等对齐）；MMAR（与 OmniVinci 对齐）。

#### 7.3.2 报告格式与单模态对比

- **表格格式**：每个基准一行，列包括：基准名称、我们模型得分、对比模型（Qwen2.5-Omni、Qwen2.5-VL、同规模 VL/ASR 等）、数据来源/版本、备注。
- **单模态不退化**：在同一表中增加「同规模纯 VL」与「同规模纯 ASR」的并排结果（如 Qwen2.5-VL-7B、Whisper-medium 单独评测），证明 Omni 未明显牺牲单模态。
- **语音 vs 文本**：MMLU/GSM8K 等报告「文本输入准确率」与「语音输入准确率」两列，目标为两者接近。

#### 7.3.3 评测节奏

- **Phase 1–2 结束**：LibriSpeech、COCO caption、DocVQA/ChartQA。
- **Phase 3 结束**：DailyOmni、Video-MME、OmniBench、验证集 perplexity。
- **Phase 4 结束**：LongVideoBench、256K 吞吐与显存。
- **Phase 5 结束**：MMBench、MMMU、打断抽检。
- **Phase 6 结束**：全清单 + OmniGAIA、seed-tts-eval、语音 vs 文本、偏好胜率。

---

### 7.4 技术可行性与优势总结

#### 7.4.1 为何能落地

- **架构**：Qwen3-LLM、SigLIP、Whisper-medium、CosyVoice2 均为开源或可获取；3D-Resampler、TMRoPE、HyPE、InfLLM-V2、Lightning Attention 有公开实现或论文复现路径。
- **数据**：所用数据集（OmniCorpus、VALID、LAION、LibriSpeech、DocVQA 等）均可在 HuggingFace 或官方渠道获取；0.2T 配方与 24M 对话为规模目标，可先小规模验证再扩量。
- **训练**：六阶段与两阶段视图对应清晰；每阶段有明确冻结/可训练组件、数据配置与验收指标；Transformer-to-hybrid 与 OmniDPO 均为在现有框架上增加模块或损失，无需新硬件。
- **评测**：清单中基准均为公开；报告格式统一后可直接对表填数，便于与 Qwen2.5-Omni、OmniVinci、MiniCPM-V 4.5 等对比。

#### 7.4.2 相对原方案与论文的差异化优势

| 维度 | 完善版方案优势 |
|------|----------------|
| **数据** | 0.2T 高效配方可验证「少 token 高回报」；文档三等级损坏比例明确；模态课程与视频为桥占比可复现；Talker 四阶段与事件图数据有清晰划分。 |
| **训练** | 两阶段视图便于对外对标 Baichuan/Mini-Omni2；混合 RL 短/长推理、OmniDPO、视-听对比、Transformer-to-hybrid 均为可选项，可按资源分步启用；里程碑与验收标准明确。 |
| **评测** | 固定清单覆盖主流论文对比口径；单模态不退化与语音 vs 文本显式呈现；长上下文 256K 与打断指标可直接用于产品化判断。 |
| **可复现** | 每阶段有 YAML/配置与数据目录结构；检查点与评测节奏绑定，便于团队与社区复现和迭代。 |

#### 7.4.3 实施顺序建议

1. **先跑通基线**：Phase 1–2 按 7.1.2 执行，验收 M1–M2；文档损坏 3:4:3 与视-听对比可选第一轮就加。
2. **Phase 3 双轨**：主轨全量 2T；副轨 0.2T 高效配方，记录 DailyOmni/Video-MME 曲线。
3. **Phase 4**：全注意力长上下文 → Transformer-to-hybrid 转换，验收 256K 与 LongVideoBench。
4. **Phase 5–6**：SFT 加入打断与短/长推理标签；Phase 6 启用混合 RL、OmniDPO、Talker 四阶段；评测按 7.3 全清单跑一轮。
5. **迭代**：根据 OmniGAIA、seed-tts-eval、语音 vs 文本结果，决定是否加强事件图数据、多语言 DPO 或上下文 ASR 数据。

---

*本文档为详细实施版本，所有配置参数、代码片段、资源估算均基于公开论文与开源实现。实际实施时请根据具体硬件环境调整。*

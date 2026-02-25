# MiniMind 与 02LMM 代码结构总览

> 初稿由 Vibe Writing 大模型生成，并结合 MiniMind 官方仓库与本项目实际结构整理。

这一篇从「上帝视角」看一眼整个 MiniMind/02LMM 仓库：先说整体架构，再按目录拆解每个核心文件的内容、作用和关系，方便你在阅读 Doc 其他文章时，随时能找到对应的代码入口。

## 1. 整体架构：从数据到模型，再到服务

站在 02LMM 的视角，可以把整个项目分成四层：

- 数据层：`dataset/`
- 模型层：`model/`
- 训练层：`trainer/`
- 服务与工具层：`scripts/`、`eval_llm.py`

外加两块支撑：

- 可视化与素材：`images/`
- 学习与笔记：`Doc/`

一个典型的实验闭环大致是：

1. 准备或下载数据（`dataset/` + `Doc/进阶实践/prepare`）
2. 选择模型结构（`model/model_minimind.py`，或者带 LoRA 的 `model/model_lora.py`）
3. 选定训练脚本（例如 `trainer/train_pretrain.py` / `trainer/train_full_sft.py` 等）
4. 跑完训练后，用 `scripts/serve_openai_api.py` / `scripts/chat_openai_api.py` 或 `scripts/web_demo.py` 做推理体验
5. 在 `Doc/` 中记录实验配置、现象和结论

后续在「进阶实践」和「深度探索」两大模块里，你会频繁在这些目录之间穿梭。

## 2. 根目录与全局文件

根目录下的关键文件：

- `README.md` / `README_en.md`：项目介绍、与上游 MiniMind 的关系、目录总览。
- `requirements.txt`：项目依赖列表，用于创建 Python/conda 环境。
- `LICENSE` / `CODE_OF_CONDUCT.md`：开源协议与行为规范。
- `eval_llm.py`：对训练好的模型做简单评测或交互测试的脚本入口（通常会加载某个 checkpoint，然后跑一组问题）。

这些文件更多是「项目门面」，真正的代码逻辑集中在下面几类目录。

## 3. 数据层：dataset/

目录位置：`dataset/`。

核心文件：

- `dataset/dataset.md`：
  - 人类可读的说明文档，约定了数据集放置位置（根目录下的 `dataset/`）以及不同阶段数据的大致用途。
- `dataset/__init__.py`：
  - 声明这是一个 Python 包，通常会从这里导出常用的数据集构造函数或类。
- `dataset/lm_dataset.py`：
  - 「语言建模数据集」的核心实现。
  - 典型职责包括：
    - 读取 jsonl / txt /其他格式的原始数据文件。
    - 通过 `model/tokenizer.json` / `tokenizer_config.json` 对文本进行分词。
    - 按最大序列长度切分为训练样本，构造 `input_ids` 和 `labels`。
    - 封装成 PyTorch `Dataset` / `DataLoader` 可直接被 `trainer/*` 使用。

在训练脚本中（例如 `trainer/train_pretrain.py`），你通常会看到类似：

- 导入 `from dataset.lm_dataset import ...`
- 指定数据路径、最大长度、batch size 等参数
- 将得到的数据集对象传入训练循环或 Trainer 封装中

## 4. 模型层：model/

目录位置：`model/`。

这是 MiniMind 的「心脏」，包含模型结构、LoRA 扩展以及分词器配置。

### 4.1 分词器与配置

- `model/tokenizer.json`：
  - 具体的 tokenizer 词表与 merge 规则，一般由 `trainer/train_tokenizer.py` 训练得到。
- `model/tokenizer_config.json`：
  - 与分词器相关的配置，比如特殊 token（bos/eos/pad）、大小写处理方式等。

这些文件在加载模型或数据时会被用到，例如：

- 推理脚本中使用 `AutoTokenizer.from_pretrained` 风格的接口加载本地 tokenizer。
- 数据集构造时用它来把原始文本转成 token 序列。

### 4.2 MiniMind 模型主体：model_minimind.py

文件位置：`model/model_minimind.py`。

核心内容可以分三层理解：

- 配置层：`MiniMindConfig`
  - 继承自 `transformers.PretrainedConfig`，定义了所有可调超参数：
    - `hidden_size`、`num_hidden_layers`、`num_attention_heads` 等基本 Transformer 结构参数。
    - `max_position_embeddings`、`rope_theta`、`rope_scaling` 等与 RoPE/YaRN 相关的配置。
    - `use_moe`、`n_routed_experts`、`num_experts_per_tok` 等 MoE（Mixture of Experts）相关参数。
  - 这一层可以理解为「模型的超参数描述文件」。

- 编码层：`MiniMindModel`
  - 负责构建真正的 Transformer 网络：
    - 词嵌入：`self.embed_tokens`
    - 多层 Block：`MiniMindBlock` 列表
    - 最终归一化层：`RMSNorm`
    - RoPE 位置编码预计算：`precompute_freqs_cis` + 注册到 buffer 的 `freqs_cos` / `freqs_sin`
  - 在前向中会：
    - 将 `input_ids` 送入 embedding
    - 按顺序通过每一层 `MiniMindBlock`
    - 返回最终的 `hidden_states` 与缓存的 `past_key_values`（用于增量推理）

- 语言建模头：`MiniMindForCausalLM`
  - 继承自 `transformers.PreTrainedModel` + `GenerationMixin`。
  - 在 `MiniMindModel` 之上加一层 `lm_head` 全连接，用于输出每个 token 的 logits。
  - forward 中：
    - 调用底层 `MiniMindModel` 得到 hidden states。
    - 通过 `lm_head` 得到 logits。
    - 如果提供了 `labels`，计算交叉熵 loss（带 shift，标准自回归训练方式）。
  - 支持 `generate` 等高层 API，方便在推理脚本里直接调用。

除此之外，文件中还包含：

- `RMSNorm`、`Attention`、`FeedForward`、`MOEFeedForward` 等模块：
  - 这些是构成 Transformer Block 的内部组件。
  - `Attention` 中可以看到是否使用 Flash Attention、如何应用 RoPE 等。
- `precompute_freqs_cis`、`apply_rotary_pos_emb` 等实用函数：
  - 负责生成并应用 RoPE 位置编码。

### 4.3 LoRA 扩展：model_lora.py

文件位置：`model/model_lora.py`。

用途：

- 在 MiniMind 基础上引入 LoRA（Low-Rank Adaptation）参数高效微调。
- 典型做法是：
  - 将注意力或 MLP 中的部分线性层替换为带低秩适配的结构。
  - 冻结原始权重，只训练新增的 LoRA 权重。
- 与 `trainer/train_lora.py` 搭配使用，适合在显存有限时做快速实验。

## 5. 训练层：trainer/

目录位置：`trainer/`。

这一层是「把模型跑起来」的地方，不同脚本对应不同训练阶段或算法。

### 5.1 公共工具：trainer_utils.py

- 封装训练通用逻辑：
  - 日志与可视化（如 wandb/swanlab）
  - checkpoint 保存与恢复
  - 学习率调度、梯度累积等
- 其他训练脚本会频繁导入这里的工具函数或类，以避免重复造轮子。

### 5.2 预训练与 SFT

- `trainer/train_pretrain.py`：
  - 从头预训练 MiniMind 语言模型。
  - 一般流程：
    - 加载 `dataset/lm_dataset.py` 构造的预训练语料数据集。
    - 初始化 `MiniMindForCausalLM` 模型。
    - 使用交叉熵 loss 在大规模无标注语料上训练。
  - 这是「让模型学会基本语言分布」的阶段。

- `trainer/train_full_sft.py`：
  - 对全参数模型做监督微调（SFT）。
  - 使用指令数据（指令 + 回复形式）让模型学会「听懂指令，给出有用回答」。

- `trainer/train_lora.py`：
  - 基于 LoRA 的参数高效微调，通常针对 SFT 场景。
  - 与 `model/model_lora.py` 协同工作，将 LoRA 插入到模型中。

### 5.3 对齐与强化学习：DPO / PPO / GRPO / SPO

- `trainer/train_dpo.py`：
  - 实现 DPO（Direct Preference Optimization）训练流程。
  - 使用成对偏好数据（优/劣回答），让模型朝「更好的回答」方向更新。

- `trainer/train_ppo.py`：
  - 基于 PPO 算法的强化学习训练脚本。
  - 使用奖励模型 + 反馈信号，对语言模型进行策略优化。

- `trainer/train_grpo.py` / `trainer/train_spo.py`：
  - GRPO / SPO 等 RLAIF 派生算法的实现。
  - 这些脚本主要组合：
    - 语言模型（policy）
    - 奖励模型或规则
    - 序列采样与回放缓冲

- `trainer/train_reason.py`：
  - 针对「推理增强模型」的训练脚本（例如模仿 DeepSeek-R1/Reason 风格）。
  - 可能会结合链式思维（CoT）或思考轨迹的特殊训练流程。

### 5.4 分词器与蒸馏

- `trainer/train_tokenizer.py`：
  - 针对语料库训练自定义 tokenizer。
  - 输出的结果就是 `model/tokenizer.json` 与 `tokenizer_config.json`。

- `trainer/train_distillation.py`：
  - 模型蒸馏训练脚本。
  - 将大模型的知识迁移到更小的 MiniMind 学生模型上。

## 6. 服务与工具层：scripts/ 与 eval_llm.py

目录位置：`scripts/`。

主要文件：

- `scripts/serve_openai_api.py`：
  - 提供兼容 OpenAI-API 协议的 HTTP 服务。
  - 可以接在 FastGPT、Open-WebUI 等上游 UI 工具上，作为后端模型服务。

- `scripts/chat_openai_api.py`：
  - 一个简单的 CLI Chat 客户端。
  - 通过 OpenAI-API 协议与本地/远程服务端交互，方便快速测试对话能力。

- `scripts/web_demo.py`：
  - 基于 Streamlit 的轻量 Web Demo。
  - 启动后可以通过浏览器访问，体验简易聊天界面。

- `scripts/convert_model.py`：
  - 模型格式转换脚本。
  - 典型用途是将训练好的权重导出为适配 `llama.cpp`、`vllm`、`ollama` 等推理引擎的格式。

根目录的 `eval_llm.py` 则偏向于：

- 载入某个 checkpoint，对一组 benchmark 任务或自定义问题做快速评估。
- 可以视作「介于训练脚本和推理服务之间」的轻量工具。

## 7. 学习与文档层：Doc/

目录位置：`Doc/`。

- `Doc/基础知识/`：
  - 01：大模型与 LLM 概览 —— 讲清楚大模型从哪来、能做什么、和传统 ML 的区别。
  - 02：Transformer 与注意力机制 —— 对照 `model/model_minimind.py` 中的实现理解结构。
  - 03～06：从 tokenizer、预训练、对齐，到推理部署的整体路线。

- `Doc/进阶实践/`：
  - `prepare/环境与数据准备_从零跑通MiniMind.md`：
    - 对应脚本：`Doc/进阶实践/prepare/setup_env_and_data.sh`
    - 负责一键创建 conda 环境并下载核心数据集。
  - `代码结构总览_MiniMind与02LMM.md`（本文）：
    - 从全局视角梳理 `dataset/`、`model/`、`trainer/`、`scripts/` 的职责与联系。
  - 未来可以在这里继续添加：
    - 预训练实战记录
    - SFT / LoRA / DPO / RLAIF 调参经验
    - 从训练到部署的完整闭环示例

- `Doc/深度探索/`：
  - 用于记录更开放、更具探索性质的实验，例如：
    - 改动模型结构（增加专家数、改变 RoPE 策略等）
    - 与 MiniMind-V 或其他多模态项目联动
    - 02LMM 未来的多模态尝试

## 8. 如何利用这篇文档继续深入？

建议把这篇当作「导航地图」：

- 当你在基础知识模块看到某个概念时：
  - 想看具体实现：按照这里的目录指引，跳到对应的代码文件。
- 当你在进阶实践中设计实验时：
  - 先在这里确认数据流转路径（dataset → trainer → model）。
  - 再去改对应脚本和配置。
- 当你在深度探索中想改模型结构或训练策略时：
  - 先在这里找到相关模块（例如 Attention、MoE、RoPE）。
  - 再结合上游 MiniMind 的文档与论文去做大胆尝试。

后续你可以在本文档的基础上，继续为每个小节补充「你自己的理解」和「具体实验链接」，让它逐渐变成一份属于 02LMM 的代码读书笔记。


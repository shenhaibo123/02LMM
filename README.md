<div align="center">
  <h1>02LMM: From Zero to Large Multimodal Model</h1>
  <p>从零训练超小语言模型 → 全流程实践 → 多模态大模型</p>
</div>

<div align="center">

中文 | [English](./README_en.md)

</div>

---

## 项目简介

**02LMM** 是一个面向学习者的大模型全流程实践项目。从 25.8M 参数的超小语言模型出发，覆盖：

- **预训练（Pretrain）** — 从零学习自回归语言建模
- **监督微调（SFT / LoRA）** — 让模型学会对话
- **强化学习对齐（DPO / PPO / GRPO / SPO）** — 偏好优化与安全对齐
- **模型蒸馏** — 小模型向大模型学习
- **训练观测** — 集成 Loss/PPL/梯度范数/表征多样性/输出分布等 30+ 指标
- **标准评测** — 基于 lm-evaluation-harness 的自动化评测（HellaSwag / ARC / MMLU / GSM8K 等）
- **推理部署** — OpenAI 兼容 API、Web Demo，兼容 llama.cpp / vLLM / ollama

> **设计原则**
> 1. 核心算法全部 PyTorch 原生实现，不依赖第三方训练框架的抽象接口
> 2. 模块间解耦 — 指标监控、数据处理、模型定义各自独立，互不污染
> 3. GPU + macOS 双平台 — 小规模实验在 Mac (MPS) 完成，大规模训练切换到 CUDA

### 参考与致谢

本项目结构与代码参考了以下优秀开源项目，在此致谢：

| 项目 | 描述 | 参考内容 |
|------|------|---------|
| [MiniMind](https://github.com/jingyaogong/minimind) | 超小语言模型全流程实践 | 模型结构、训练流程、数据管线 |
| [QiChat](https://github.com/Morton-Li/QiChat) | Decoder-only 对话模型训练框架 | 训练监控指标体系（ModelProbe / TrainingTracker） |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | 通用 LLM 评测框架 | 标准评测集成 |

如需在本地克隆上述参考项目（不修改，仅对照），可在项目根目录执行：

```bash
bash scripts/clone_reference_repos.sh
```

克隆结果位于 `reference/` 目录（已加入 .gitignore）。

---

## 快速开始

### macOS / Linux 环境安装（终端）

**本机需 Python 3.10+**（项目依赖如 matplotlib 等按 3.10 要求）。若当前 `python3 --version` 低于 3.10，请先升级本机 Python，再创建 venv；升级失败再考虑改项目兼容低版本。

```bash
# 0) 检查本机 Python 版本（可选）
python3 scripts/ensure_python.py   # 若 < 3.10 会打印升级说明

# 1) 建议使用虚拟环境（请用 3.10+ 的解释器创建）
python3 -m venv .venv
# 若本机默认是 3.9，可用：python3.10 -m venv .venv 或 pyenv local 3.10 后再创建
source .venv/bin/activate
python -m pip install -U pip

# 2) 安装项目依赖
python -m pip install -r requirements.txt

# 3) 评测（可选，推荐）
python -m pip install lm_eval

# 4) OpenAI 兼容 API 服务端（可选）
python -m pip install fastapi uvicorn
```

> 说明：`requirements.txt` 中固定了 `torch==2.6.0`。如果你在 **Linux + CUDA** 环境，需要根据你的 CUDA 版本安装对应的 PyTorch Wheel。\n+> 典型做法是先装 PyTorch（CUDA 版），再装其余依赖；或在安装 PyTorch 时指定官方 index。\n+
```bash
# Linux + CUDA（示例：CUDA 12.4）
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip install lm_eval
```

### 最小预训练（macOS / Linux / GPU 均可）

```bash
# macOS (Apple Silicon MPS)
python trainer/train_pretrain.py --device mps --batch_size 4 --max_seq_len 128 --epochs 1 --hidden_size 64 --num_hidden_layers 2

# Linux / GPU (CUDA)
python trainer/train_pretrain.py --device cuda:0 --batch_size 32 --max_seq_len 340 --epochs 1

# CPU（通用兜底）
python trainer/train_pretrain.py --device cpu --batch_size 4 --max_seq_len 128 --epochs 1 --hidden_size 64 --num_hidden_layers 2 --num_workers 0
```

训练完成后会在 `out/` 目录生成权重文件，在 `logs/` 目录生成指标 JSONL 与曲线图（本地输出，不依赖 TensorBoard）。

### 评测模型

```bash
# 基于本地权重
python scripts/eval_model_benchmark.py --backend hf --model-path ./model --tasks hellaswag arc_easy --limit 50 --device cpu

# 基于 OpenAI 兼容 API
python scripts/serve_openai_api.py  # 先启动服务
python scripts/eval_model_benchmark.py --backend api --tasks hellaswag gsm8k
```

### 启动对话服务

```bash
python scripts/serve_openai_api.py --device cuda:0  # 或 --device mps / --device cpu
```

---

## 项目结构

```
02LMM/
├── model/                      # 模型定义
│   ├── model_minimind.py       #   K Transformer (Dense + MoE)（文件名沿用历史实现）
│   └── model_lora.py           #   LoRA 适配器
├── metrics/                    # 训练监控指标（解耦模块）
│   ├── probes.py               #   模型探针：输出分布 + 表征多样性
│   ├── tracker.py              #   Loss / PPL / 梯度 / 资源 跟踪器
│   └── visualize.py            #   训练曲线绘图工具
├── trainer/                    # 训练脚本
│   ├── train_pretrain.py       #   预训练
│   ├── train_full_sft.py       #   全参数 SFT
│   ├── train_lora.py           #   LoRA 微调
│   ├── train_dpo.py            #   DPO 对齐
│   ├── train_ppo.py            #   PPO 强化学习
│   ├── train_grpo.py           #   GRPO
│   ├── train_spo.py            #   SPO
│   ├── train_reason.py         #   推理模型训练
│   ├── train_distillation.py   #   知识蒸馏
│   ├── train_tokenizer.py      #   Tokenizer 训练
│   └── trainer_utils.py        #   训练工具函数
├── dataset/                    # 数据集处理
│   └── lm_dataset.py           #   Pretrain / SFT / DPO / RLAIF 数据集
├── scripts/                    # 工具脚本
│   ├── eval_model_benchmark.py #   lm-evaluation-harness 评测
│   ├── serve_openai_api.py     #   OpenAI API 服务端
│   ├── web_demo.py             #   Streamlit Web Demo
│   ├── convert_model.py        #   模型格式转换
│   └── ...                     #   Tokenizer 对比/评估等工具
├── Doc/                        # 学习文档
├── eval_llm.py                 # 交互式推理入口
└── requirements.txt
```

---

## 模型配置

| 模型 | 参数量 | hidden_size | num_layers | 推理占用 |
|------|--------|-------------|------------|---------|
| K-Small | 26M | 512 | 8 | ~0.5 GB |
| K-MoE | 145M | 640 | 8 (MoE) | ~1.0 GB |
| K-Base | 104M | 768 | 16 | ~1.0 GB |

支持自定义更小的配置（如 hidden_size=64, num_layers=2）用于 Mac 上的快速实验。

---

## 训练监控指标

本项目集成了完整的训练观测体系（参考 [QiChat](https://github.com/Morton-Li/QiChat) 的 ModelProbe 设计），所有指标以解耦模块形式封装于 `metrics/` 目录，可被任意训练脚本调用：

### 输出分布指标
- `logits_entropy` — 输出熵（不确定性）
- `top1_acc` / `top5_acc` — Top-1 / Top-5 准确率
- `confidence_gap` — Top-1 与 Top-2 概率差
- `topk_mass` — Top-k 概率质量
- `p_true_mean` — 真实标签平均概率

### 表征多样性指标
- `cosine_sim_intra` / `cosine_sim_inter` — 样本内/跨样本余弦相似度
- `participation_ratio` — 有效维度比（表征塌缩检测）

### 训练过程指标
- Loss（当前值 / 滑动平均 / 最小值）
- Perplexity（当前值 / 滑动平均）
- 梯度裁剪率 / 梯度范数
- 学习率 / 训练速度 / 显存占用

---

## 评测与部署

### 评测说明

评测基于 **[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)**（`lm_eval`）。本仓库的 `scripts/eval_model_benchmark.py` 是对其的薄封装：统一入口、默认任务和参数，并支持本地 HF 模型与 OpenAI 兼容 API 两种后端。`--tasks` 传入的即 lm_eval 的任务名，完整列表见 [lm-evaluation-harness 文档](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md)。

**安装：** `pip install lm_eval`（可选，仅评测时需要。）

#### 支持的 `--tasks` 示例

| 类型 | 任务名（可作 `--tasks` 参数） | 说明 |
|------|-------------------------------|------|
| **英文 · 常识/推理** | `hellaswag` | 常识推理，4 选 1 |
| | `arc_easy`, `arc_challenge` | 科学常识（简单/困难） |
| | `piqa` | 物理直觉，2 选 1 |
| | `winogrande` | 常识共指消解 |
| **英文 · 知识/推理** | `mmlu` | 多学科知识（57 子任务） |
| | `gsm8k` | 小学数学推理 |
| | `truthfulqa_mc2` | 真实性多选 |
| **中文 · 客观题** | `ceval*` | C-Eval 中文综合 |
| | `cmmlu*` | C-MMLU 中文多学科 |
| | `aclue*` | A-CLUE 中文理解 |
| | `tmmlu*` | TMMLU+ 中文 |

未指定 `--tasks` 时，脚本使用默认英文任务集：`hellaswag arc_easy arc_challenge piqa winogrande mmlu gsm8k truthfulqa_mc2`。中文任务需显式传入（如 `--tasks ceval* cmmlu*`）。`*` 为 lm_eval 通配符，会展开为对应多子任务。

#### 常用命令

```bash
# 本地 HF 模型，默认英文任务，限制条数
python scripts/eval_model_benchmark.py --backend hf --model-path ./model --tasks hellaswag arc_easy mmlu --limit 100

# 冒烟测试（少量任务 + 少量样本）
python scripts/eval_model_benchmark.py --backend hf --model-path ./model --smoke --device cpu --limit 10

# 中文客观题（同上脚本，传入中文任务）
python scripts/eval_model_benchmark.py --backend hf --model-path ./model --tasks ceval* cmmlu* aclue* tmmlu* --device cuda:0

# 先起 OpenAI 兼容服务，再评测在线模型
python scripts/serve_openai_api.py --device cuda:0
python scripts/eval_model_benchmark.py --backend api --tasks hellaswag gsm8k
```

也可直接用 lm_eval 命令行（与脚本等价，仅参数写法不同）：

```bash
lm_eval --model hf --model_args pretrained=./model,device=cuda,dtype=auto --tasks hellaswag arc_easy mmlu --batch_size 8 --trust_remote_code
```

### 部署与长上下文

- **RoPE 长度外推（YaRN）**：`eval_llm.py --inference_rope_scaling`；Transformers 模型可在 `config.json` 里配置 `rope_scaling`。

#### 模型转换（PyTorch ↔ Transformers）

- `scripts/convert_model.py` 用于在 **PyTorch 原生权重** 与 **Transformers 格式** 之间互相转换，可选导出带 `rope_scaling` 的 `LlamaConfig`。
- 一般推荐将训练好的权重先转换为 Transformers 格式，再接入评测与下游推理/部署框架。

#### 基于 OpenAI 兼容 API 的服务

- `scripts/serve_openai_api.py` 提供兼容 OpenAI API 的最简聊天接口，方便接入 FastGPT、Open-WebUI、Dify 等三方 UI。

**Transformers 模型目录示例：**

```bash
<model-root>/
├── config.json
├── generation_config.json
├── pytorch_model.bin 或 model.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

**启动服务与测试：**

```bash
# 启动聊天服务端
python scripts/serve_openai_api.py

# 测试接口
python scripts/chat_openai_api.py
```

**API 示例（兼容 OpenAI Chat Completions）：**

```bash
curl http://<ip>:<port>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model-identifier",
    "messages": [
      { "role": "user", "content": "世界上最高的山是什么？" }
    ],
    "temperature": 0.7,
    "max_tokens": 512,
    "stream": true
  }'
```

#### 其他部署路径（简要指引）

- **vLLM**：高吞吐推理框架，可直接加载 Transformers 模型，例如：

  ```bash
  vllm serve ./<model-root> --model-impl transformers --served-model-name "minimind" --port 8998
  ```

- **llama.cpp**：将 Transformers 模型转换为 GGUF 格式并量化，适合本地命令行与多端推理：
  1. 参考官方文档安装 llama.cpp，确保与本仓库位于同级目录。
  2. 使用 `convert_hf_to_gguf.py` 将 Transformers 模型转换为 `.gguf`。
  3. 可选：使用 `llama-quantize` 做 4bit/8bit 量化。
  4. 使用 `llama-cli` 进行命令行推理。

- **Ollama**：在有 GGUF 模型的前提下，编写 `modelfile` 并通过 `ollama create` / `ollama run` 启动本地服务，也可直接使用社区发布的模型。

- **MNN**：面向端侧的推理引擎，可在其 `transformers/llm/export` 工具中将 Transformers 模型导出为 MNN/HQQ 量化格式，用于 Mac 或移动端测试。

> 以上三方框架的详细参数和高级用法，请以各自官方文档为准。

---

## 许可证

本项目采用 [Apache 2.0](LICENSE) 许可证。

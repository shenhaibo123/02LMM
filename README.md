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

### 评测（看模型能力）

- **本仓库自带评测脚本**：  
  - `scripts/eval_model_benchmark.py --backend hf --model-path <本地 Transformers 模型> --tasks hellaswag arc_easy mmlu --limit 100`  
  - 或先用 `scripts/serve_openai_api.py` 起一个 OpenAI 兼容服务，再用 `--backend api` 评测在线模型。
- **中文客观题榜单（C-Eval / C-MMLU / A-CLUE / TMMLU+）**：  
  - 使用 `lm-evaluation-harness` 直接评测：  
    ```bash
    lm_eval --model hf \
      --model_args pretrained=<模型路径>,device=cuda,dtype=auto \
      --tasks ceval* cmmlu* aclue* tmmlu* \
      --batch_size 8 --trust_remote_code
    ```
  - 本项目 K 系列与 [MiniMind2](https://github.com/jingyaogong/minimind) 结构一致，可直接参考其官方中文榜单成绩（单位：准确率，%）：  

    | 模型（等价本仓库）      | 参数量 | C-Eval | C-MMLU | A-CLUE | TMMLU+ |
    |------------------------|--------|--------|--------|--------|--------|
    | K-Base ≈ MiniMind2     | 104M   | 26.52  | 24.42  | 24.97  | 25.27  |
    | K-Small ≈ MiniMind2-Small | 26M | 26.37  | 24.97  | 25.39  | 24.63  |
    | K-MoE ≈ MiniMind2-MoE  | 145M   | 26.60  | 25.01  | 24.83  | 25.01  |

  这一档小模型整体接近「随机 25%」略高一些，主要适合作为结构/训练流程教学与对比实验用基线。

### 部署与长上下文

- **RoPE 长度外推（YaRN）**：`eval_llm.py --inference_rope_scaling`；Transformers 模型可在 `config.json` 里配置 `rope_scaling`。
- **模型转换**：`scripts/convert_model.py` 支持 PyTorch ↔ Transformers，并可导出带 `rope_scaling` 的 LlamaConfig。
- **API 与推理**：`scripts/serve_openai_api.py` 提供 OpenAI 兼容接口；vLLM、llama.cpp、Ollama、MNN 等更详细用法见 [Doc/评测与部署.md](Doc/评测与部署.md)。

---

## 许可证

本项目采用 [Apache 2.0](LICENSE) 许可证。

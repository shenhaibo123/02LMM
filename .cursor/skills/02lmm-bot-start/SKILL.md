---
name: 02lmm-bot-start
description: 一键式本地启动与自测流程。用于在用户本机上“真的跑一遍”02LMM：检查环境、创建/激活虚拟环境或 Conda、安装依赖、下载示例数据、跑最小预训练/微调/评测链路，而不是只给文字说明。不会上传数据集与模型权重到 GitHub。
---

# 02LMM Bot Start Skill

## 什么时候使用这个 skill

当用户提出类似需求时应自动考虑使用本 skill，例如：

- “一键跑通这个项目” / “帮我完整启动 02LMM”
- “帮我准备好环境 + 数据 + 训练并验证一下”
- “在本机从 0 到能训练/评测，实际跑一遍，不要光说”
- 用户刚 clone 本仓库，希望快速验证项目是否可用

如果当前对话中已经在手动做同样的事情（例如已经按本 skill 步骤执行），可以跳过重复步骤。

> 重要：本 skill 面向**本机实操**，默认只在本地跑命令，不会把数据/模型推到 GitHub。

## 总体目标

1. **环境就绪**：本地已有可用 Python 环境（推荐 Conda，根据脚本自动创建/激活）。
2. **依赖安装**：按 `requirements.txt` 安装项目依赖，可选安装 `lm_eval` / `fastapi` 等评测与服务端依赖。
3. **数据准备**：在 `dataset/` 下自动下载一份可用于预训练/SFT/DPO 的最小数据集（来自 `jingyaogong/minimind_dataset`）。
4. **最小训练链路打通**：
   - 跑一次最小预训练（小 hidden_size、少 layer、1 epoch）；
   - 可选再跑一次最小 SFT 或 LoRA；
   - 可选跑一次 `eval_model_benchmark.py --smoke` 做冒烟评测。
5. **状态回报**：每个阶段给出简短的命令输出总结与下一步说明，方便用户知道“现在跑到哪一步了、是否成功”。

## 详细步骤

### 1. 确认仓库根目录

1. 使用用户提供的路径或当前工作目录，确保已经在 02LMM 根目录（包含 `README.md`、`trainer/`、`dataset/` 等）。
2. 如有疑问，可通过列出目录结构（`ls` 或工具）确认，避免在错误目录执行脚本。

### 2. 准备 Python 环境与依赖

优先使用项目已有的自动化脚本：`Doc/进阶实践/prepare/setup_env_and_data.sh`。

1. 检查 `conda` 是否可用：
   - 如果不可用：
     - 明确提示用户需要先安装 Miniconda/Anaconda；
     - 在此之前可以退回到 `README.md` 中的 venv 步骤，仅做最小验证。
2. 若 `conda` 可用：
   - 执行（在仓库根目录）：

     ```bash
     bash Doc/进阶实践/prepare/setup_env_and_data.sh
     ```

   - 若用户在对话中提供了 `ENV_NAME` / `PYTHON_VERSION` / `DATA_ROOT` 等环境变量，按用户参数组装命令：

     ```bash
     ENV_NAME=... PYTHON_VERSION=... DATA_ROOT=... bash Doc/进阶实践/prepare/setup_env_and_data.sh
     ```

3. 脚本执行完成后，进行快速验证：
   - 在激活的环境中执行：

     ```bash
     python -c "import torch, transformers, trl, peft; print('ok')"
     ```

   - 若失败，尝试根据报错修复后重试；必要时将错误信息简要反馈给用户。

> 若用户明确要求使用 `python -m venv` 而非 Conda，则按 README 中的 venv 流程创建 `.venv` 并安装 `requirements.txt`，但仍复用本 skill 其它步骤。

### 3. 数据下载与检查

1. 确认 `dataset/` 或用户指定的 `DATA_ROOT` 已存在。若不存在，则创建目录。
2. 如果刚才已运行 `setup_env_and_data.sh`，默认会从 HuggingFace 下载：
   - `pretrain_hq.jsonl`
   - `sft_mini_512.jsonl`
   - `dpo.jsonl`
3. 如未运行脚本或用户只想下载部分数据，可根据 `Doc/进阶实践/prepare/环境与数据准备_从零跑通MiniMind.md` 中说明，按需设置：

   ```bash
   DOWNLOAD_PRETRAIN=0 DOWNLOAD_SFT_MINI=1 DOWNLOAD_DPO=0 bash Doc/进阶实践/prepare/setup_env_and_data.sh
   ```

4. 检查关键文件是否存在（根据本次需要选择其一或多个）：
   - `dataset/pretrain_hq.jsonl`
   - `dataset/sft_mini_512.jsonl`
   - `dataset/dpo.jsonl`

若缺失，则提示用户数据未准备完整，并提供重新执行脚本或手动放置数据的建议。

### 4. 跑通最小预训练

1. 在已激活的环境下，执行 README 中推荐的最小预训练命令之一，例如：

   ```bash
   python trainer/train_pretrain.py      --device cpu      --batch_size 4      --max_seq_len 128      --epochs 1      --hidden_size 64      --num_hidden_layers 2      --num_workers 0
   ```

2. 观察训练是否能正常开始并完成 1 个 epoch：
   - 若 OOM 或显存不足，可自动建议用户：
     - 换用 `--device cpu` 或 `--device mps`；
     - 进一步减小 `batch_size` / `max_seq_len`。
3. 训练完成后，确认：
   - `out/` 下是否生成了权重文件；
   - `logs/` 下是否有对应的 JSONL 或曲线数据（若脚本已集成）。

### 5. （可选）SFT / LoRA & DPO 快速验证

如用户希望一次性打通完整链路，可在环境与数据就绪后，追加以下步骤：

1. 使用 `trainer/train_full_sft.py` 或 `trainer/train_lora.py`，基于 `dataset/sft_mini_512.jsonl` 跑一个小规模 SFT/LoRA 实验（1 epoch 即可）。
2. 使用 `trainer/train_dpo.py`，基于 `dataset/dpo.jsonl` 跑几步 DPO 训练，验证偏好数据与 loss 计算是否正常。

这些步骤不强制，但若执行，应同样监控是否有报错，并将关键信息反馈给用户。

### 6. （可选）评测冒烟测试

在完成至少一次预训练或 SFT 后，可运行 `scripts/eval_model_benchmark.py` 做一个小规模冒烟评测：

```bash
python scripts/eval_model_benchmark.py --backend hf --model-path ./model --smoke --device cpu --limit 10
```

或按 README「评测与部署」一节中的示例命令，指定合适的 `--tasks` 和 `--limit`。

### 7. 收尾与报告

1. 最后再次运行 `git status`，确认此次 Bot Start 仅进行了本地环境/数据/训练相关操作，没有意外修改代码（若有，应提醒用户是否需要提交）。
2. 向用户简要总结：
   - 环境是否创建/激活成功；
   - 哪些数据文件已下载；
   - 预训练 / SFT / DPO / 评测分别成功执行到了哪一步；
   - 后续推荐操作（例如继续调参、接 VeOmni、多卡训练等）。

> 本 skill 不负责将数据集或模型权重推送到 GitHub；训练产生的 `out/`、`logs/` 等目录已经在 `.gitignore` 中忽略，保持仓库干净。

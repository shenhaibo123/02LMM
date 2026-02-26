---
name: 02lmm-dev-workflow
description: 02LMM 项目开发规范与工作流。涵盖代码解耦原则、模块自测要求、GPU/macOS 双平台兼容、训练观测指标集成、训练曲线绘图。当用户在本项目中编写训练脚本、添加新模块、修改模型代码、或进行训练/评测时自动生效。
---

# 02LMM 开发规范

## 1. 解耦原则

- **模块边界清晰**：`model/`（模型定义）、`metrics/`（训练监控）、`trainer/`（训练逻辑）、`dataset/`（数据处理）、`scripts/`（工具脚本）各自独立。
- **单向依赖**：trainer → model、trainer → metrics、trainer → dataset。metrics 不依赖 trainer，model 不依赖 trainer。
- **零侵入式集成**：新增功能（如指标监控）应通过外部调用方式集成，不修改被集成模块的内部代码。
  
  ```python
  # 正确：在训练循环中调用探针
  from metrics.probes import ModelProbe
  probe = ModelProbe()
  probe.attach(model, layer_names)
  
  # 错误：在 model_minimind.py 内部 import metrics
  ```

- **新建模块时**必须在文件底部提供 `if __name__ == "__main__":` 自测块，用合成数据验证核心功能。

## 2. 模块自测

每个 Python 模块文件应包含 `__main__` 自测，满足：

1. 不依赖外部数据文件或 GPU
2. 用合成/随机数据验证核心接口
3. 打印关键输出，以 `=== 自测通过 ===` 结尾
4. 可通过 `python <module_path>` 直接运行

验证流程：
```bash
python metrics/probes.py      # 探针模块自测
python metrics/tracker.py     # 跟踪器模块自测
python metrics/visualize.py --help  # 绘图工具帮助
```

## 3. GPU + macOS 双平台兼容

本项目在 macOS (Apple Silicon MPS) 做小规模实验，GPU (CUDA) 做大规模训练。

### 设备选择
```python
from trainer.trainer_utils import get_best_device, get_device_type

device = get_best_device()       # 自动: cuda:0 > mps > cpu
device_type = get_device_type(device)  # "cuda" / "mps" / "cpu"
```

### 编码规范

| 场景 | CUDA | MPS/CPU |
|------|------|---------|
| 混合精度 | `torch.cuda.amp.autocast` | `nullcontext()` 或 `torch.amp.autocast("cpu")` |
| GradScaler | `enabled=True` | `enabled=False` |
| num_workers | 8+ | 0 |
| pin_memory | True | False |
| 随机种子 | `torch.cuda.manual_seed_all()` | 跳过 |
| 显存清理 | `torch.cuda.empty_cache()` | 跳过 |
| DDP | nccl 后端 | 不可用 |

### 关键模式
```python
# 条件式 CUDA 调用
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.empty_cache()

# 混合精度上下文
if device_type == "cuda":
    autocast_ctx = torch.cuda.amp.autocast(dtype=dtype)
else:
    autocast_ctx = nullcontext()

# DataLoader
num_workers = 0 if device_type == "mps" else args.num_workers
pin_memory = (device_type == "cuda")
```

## 4. 训练观测指标

所有训练脚本（pretrain / SFT / RL 等）应集成观测指标：

### 必选指标（每步记录）
- Loss（当前值 + 滑动平均）
- Perplexity
- 学习率

### 推荐指标（定期记录）
- 梯度裁剪率 + 平均裁剪比例
- 各层梯度 L2 范数
- 训练速度（steps/sec）

### 探针指标（低频采样）
- 输出分布：entropy、top1_acc、top5_acc、confidence_gap、topk_mass、p_true_mean
- 表征多样性：cosine_sim_intra、cosine_sim_inter、participation_ratio

### 集成方式
```python
from metrics.probes import ModelProbe
from metrics.tracker import TrainingTracker

tracker = TrainingTracker(log_dir="logs/run_name")
probe = ModelProbe()
probe.attach(model, [f"model.layers.{i}" for i in [0, n//2, n-1]])

# 训练循环中
tracker.update_loss(loss_val, "train")
tracker.update_ppl(ppl_val, "train")
tracker.update_lr(lr)
tracker.log_to_tensorboard()

# 每 N 步
probe.activate()
dist_metrics = probe.compute_output_distribution(logits, labels)
repr_metrics = probe.compute_representation_diversity()
tracker.log_probe_metrics(dist_metrics, repr_metrics)
probe.deactivate()
```

## 5. 训练曲线绘图

训练完成后绘制曲线：
```bash
python metrics/visualize.py --log-dir logs/pretrain --output plots/pretrain.png
```

如果未安装 TensorBoard，也可使用 JSONL 日志：
```bash
python metrics/visualize.py --json-log logs/metrics.jsonl --output plots/run.png
```

## 6. 评测流程

训练后使用标准评测脚本验证模型质量：
```bash
# 转为 HF 格式（如需）
python scripts/convert_model.py

# 评测
python scripts/eval_model_benchmark.py --backend hf --model-path ./out/model_hf --tasks hellaswag arc_easy mmlu gsm8k
```

冒烟测试（快速验证）：
```bash
python scripts/eval_model_benchmark.py --smoke --device cpu --limit 10
```

## 7. 提交前检查

1. 所有新增模块通过 `python <module>` 自测
2. 在 CPU/MPS 上跑通最小配置训练（hidden_size=64, num_layers=2, epochs=1）
3. 评测脚本能正常输出结果表格
4. 无硬编码绝对路径
5. 无 CUDA-only 代码（条件判断包裹）

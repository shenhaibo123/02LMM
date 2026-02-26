"""
训练跟踪器（TrainingTracker）—— 解耦的训练过程指标汇聚与本地记录模块。

参考 QiChat (https://github.com/Morton-Li/QiChat) 的 TrainingTracker 设计，
将 Loss / PPL / 梯度裁剪 / 学习率 / 速度等指标的采集封装为独立模块。
指标写入本地 JSONL 文件，并支持训练结束时直接画图（不依赖 TensorBoard）。

典型用法：
    tracker = TrainingTracker(log_dir="logs/pretrain")

    for step, (loss, ppl) in enumerate(train_loop):
        tracker.update_loss(loss, split="train")
        tracker.update_ppl(ppl, split="train")
        tracker.update_lr(optimizer.param_groups[0]["lr"])
        tracker.log_step(extra={...})   # 写入 metrics.jsonl

    tracker.close(plot=True)   # 关闭并生成 curves.png
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Union, Any

import torch


class _SlidingMeter:
    """带多窗口滑动平均的标量跟踪器。"""

    def __init__(self, windows: Optional[List[int]] = None):
        self.windows = windows or []
        self._max_window = max(self.windows) if self.windows else 0
        self._buffer: Deque[float] = deque(maxlen=self._max_window if self._max_window > 0 else None)
        self._total: float = 0.0
        self._count: int = 0
        self._min: float = float("inf")
        self._max: float = float("-inf")
        self._current: Optional[float] = None

    def step(self, value: float) -> None:
        self._current = value
        self._total += value
        self._count += 1
        if self.windows:
            self._buffer.append(value)
        if value < self._min:
            self._min = value
        if value > self._max:
            self._max = value

    @property
    def current(self) -> Optional[float]:
        return self._current

    @property
    def avg(self) -> float:
        return self._total / max(self._count, 1)

    @property
    def min(self) -> float:
        return self._min

    @property
    def max(self) -> float:
        return self._max

    def window_avg(self) -> Dict[str, float]:
        if not self.windows or not self._buffer:
            return {}
        vals = list(self._buffer)
        n = len(vals)
        result = {}
        for w in self.windows:
            label = f"{w // 1000}k" if w % 1000 == 0 else f"{w}"
            result[label] = sum(vals[-w:]) / min(n, w)
        return result

    def reset(self) -> None:
        self._buffer.clear()
        self._total = 0.0
        self._count = 0
        self._min = float("inf")
        self._max = float("-inf")
        self._current = None


class _GradClipMeter:
    """梯度裁剪率跟踪器。"""

    def __init__(self, window: int = 1000):
        self._is_clipped: Deque[float] = deque(maxlen=window)
        self._clip_scale: Deque[float] = deque(maxlen=window)

    def step(self, is_clipped: bool, clip_scale: float = 1.0) -> None:
        self._is_clipped.append(1.0 if is_clipped else 0.0)
        self._clip_scale.append(clip_scale if is_clipped else 0.0)

    @property
    def clip_rate(self) -> float:
        if not self._is_clipped:
            return 0.0
        return sum(self._is_clipped) / len(self._is_clipped)

    @property
    def avg_clip_scale(self) -> float:
        clipped = sum(self._is_clipped)
        if clipped == 0:
            return 1.0
        return sum(self._clip_scale) / clipped


class TrainingTracker:
    """
    训练过程指标汇聚器，支持本地 JSONL 记录与直接画图。

    所有方法均为纯 Python（无 CUDA 依赖），同时兼容 GPU 和 macOS 环境。
    """

    def __init__(self, log_dir: Optional[Union[str, Path]] = None, windows: Optional[List[int]] = None):
        self.loss = {
            "train": _SlidingMeter(windows or [1000, 3000, 5000]),
            "valid": _SlidingMeter(),
        }
        self.ppl = {
            "train": _SlidingMeter(windows or [1000, 3000, 5000]),
            "valid": _SlidingMeter(),
        }
        self.grad_clip = _GradClipMeter()
        self.global_step: int = 0

        self._lr: float = 0.0
        self._speed_timer: Optional[float] = None
        self._speed_steps: int = 0
        self._steps_per_sec: float = 0.0

        self._log_dir: Optional[Path] = None
        self._jsonl_file = None
        self._jsonl_path: Optional[Path] = None
        self._last_probe: Dict[str, float] = {}
        if log_dir:
            self._log_dir = Path(log_dir)
            self._log_dir.mkdir(parents=True, exist_ok=True)
            self._jsonl_path = self._log_dir / "metrics.jsonl"
            self._jsonl_file = open(self._jsonl_path, "a", encoding="utf-8")

    # ── 更新接口 ──────────────────────────────────────

    def update_loss(self, loss: float, split: str = "train") -> None:
        self.loss[split].step(loss)

    def update_ppl(self, ppl: float, split: str = "train") -> None:
        self.ppl[split].step(ppl)

    def update_lr(self, lr: float) -> None:
        self._lr = lr

    def update_grad_clip(self, total_norm: float, max_norm: float) -> None:
        is_clipped = total_norm > max_norm
        scale = max_norm / total_norm if is_clipped else 1.0
        self.grad_clip.step(is_clipped, scale)

    def tick_speed(self) -> None:
        """每个优化器步调用一次，用于计算训练速度。"""
        now = time.time()
        self._speed_steps += 1
        if self._speed_timer is not None:
            elapsed = now - self._speed_timer
            if elapsed > 0:
                self._steps_per_sec = 1.0 / elapsed
        self._speed_timer = now

    def step(self) -> None:
        self.global_step += 1

    # ── 本地 JSONL 记录与画图 ─────────────────────────

    def log_step(self, global_step: Optional[int] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        """将当前指标（及上次 log_probe_metrics 的探针结果）写入一行 JSONL。"""
        if self._jsonl_file is None:
            return
        gs = global_step if global_step is not None else self.global_step
        row: Dict[str, Any] = {
            "step": gs,
            "Train/Loss": self.loss["train"].current if self.loss["train"].current is not None else None,
            "Train/Loss_Min": self.loss["train"].min,
            "Train/PPL": self.ppl["train"].current if self.ppl["train"].current is not None else None,
            "Train/LearningRate": self._lr,
            "Train/GradClipRate": self.grad_clip.clip_rate,
            "Train/GradClipScale": self.grad_clip.avg_clip_scale,
            "Train/Speed_steps_per_sec": self._steps_per_sec,
        }
        for k, v in self.loss["train"].window_avg().items():
            row[f"Train/Loss_Avg_{k}"] = v
        for k, v in self.ppl["train"].window_avg().items():
            row[f"Train/PPL_Avg_{k}"] = v
        if self._last_probe:
            for k, v in self._last_probe.items():
                if not (isinstance(v, float) and math.isnan(v)):
                    row[k] = v
            self._last_probe = {}
        if extra:
            for k, v in extra.items():
                if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                    row[k] = v
        self._jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._jsonl_file.flush()

    def log_probe_metrics(
        self,
        dist_metrics: Dict[str, float],
        repr_metrics: Dict[str, Dict[str, float]],
        global_step: Optional[int] = None,
        prefix: str = "Train",
    ) -> None:
        """缓存 ModelProbe 结果，供下一次 log_step 一并写入 JSONL。"""
        for k, v in dist_metrics.items():
            if not math.isnan(v):
                self._last_probe[f"{prefix}/Probe/{k}"] = v
        for layer_name, metrics in repr_metrics.items():
            for k, v in metrics.items():
                if not math.isnan(v):
                    self._last_probe[f"{prefix}/Probe/{layer_name}/{k}"] = v

    def log_grad_norms(
        self,
        model: torch.nn.Module,
        global_step: Optional[int] = None,
        prefix: str = "Train",
    ) -> Dict[str, float]:
        """计算各层梯度 L2 范数并返回（可选在 extra 中传入 log_step 写入）。"""
        norms: Dict[str, float] = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                norm = param.grad.data.norm(2).item()
                short = name.replace("model.", "").replace(".weight", "")
                norms[f"{prefix}/GradNorm/{short}"] = norm
        return norms

    def plot_and_save(self, output_path: Optional[Union[str, Path]] = None) -> Optional[str]:
        """从当前 run 的 metrics.jsonl 读取数据并绘制曲线图保存。"""
        if self._jsonl_path is None or not self._jsonl_path.exists():
            return None
        try:
            from .visualize import _read_json_log, plot_training_curves
        except ImportError:
            from metrics.visualize import _read_json_log, plot_training_curves
        data = _read_json_log(str(self._jsonl_path))
        if not data:
            return None
        out = output_path or (self._log_dir / "curves.png")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        plot_training_curves(data, output_path=str(out), title="Training Curves")
        return str(out)

    def close(self, plot: bool = True) -> None:
        """关闭 JSONL 文件；若 plot=True 则根据 metrics.jsonl 生成曲线图。"""
        if self._jsonl_file is not None:
            self._jsonl_file.close()
            self._jsonl_file = None
        if plot and self._log_dir is not None:
            self.plot_and_save()

    # ── 汇总 ─────────────────────────────────────────

    def summary(self) -> Dict[str, float]:
        """返回当前训练状态的快照字典。"""
        d = {
            "global_step": self.global_step,
            "train_loss": self.loss["train"].current or 0.0,
            "train_loss_avg": self.loss["train"].avg,
            "train_loss_min": self.loss["train"].min,
            "train_ppl": self.ppl["train"].current or 0.0,
            "lr": self._lr,
            "grad_clip_rate": self.grad_clip.clip_rate,
            "steps_per_sec": self._steps_per_sec,
        }
        return d


# ── 模块自测 ────────────────────────────────────────

if __name__ == "__main__":
    import random
    import tempfile

    print("=== TrainingTracker 模块自测 ===\n")

    with tempfile.TemporaryDirectory() as tmp:
        tracker = TrainingTracker(log_dir=tmp)
        for i in range(100):
            loss = 5.0 * math.exp(-i / 30) + random.gauss(0, 0.1)
            ppl = math.exp(loss)
            tracker.update_loss(loss, "train")
            tracker.update_ppl(ppl, "train")
            tracker.update_lr(5e-4 * (1 - i / 100))
            tracker.update_grad_clip(
                total_norm=random.uniform(0.5, 3.0),
                max_norm=1.0,
            )
            tracker.tick_speed()
            tracker.step()
            if i % 10 == 0:
                tracker.log_step()

        s = tracker.summary()
        print("训练摘要:")
        for k, v in s.items():
            print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
        print(f"\nLoss 滑动平均: {tracker.loss['train'].window_avg()}")
        print(f"PPL 滑动平均:  {tracker.ppl['train'].window_avg()}")
        print(f"梯度裁剪率:     {tracker.grad_clip.clip_rate:.4f}")
        tracker.close(plot=True)
        assert (Path(tmp) / "metrics.jsonl").exists(), "metrics.jsonl 未生成"
        print(f"\n已写入 {tmp}/metrics.jsonl 并生成曲线图")
    print("=== 自测通过 ===")

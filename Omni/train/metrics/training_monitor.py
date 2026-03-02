"""
训练监控器（TrainingMonitor）—— Omni 全模态训练的指标采集与日志记录。

参考项目 metrics/tracker.py 的设计风格，扩展以下能力：
    - 模态特定 loss 分别统计（lm_loss / ctc_loss / dpo_loss）
    - GPU 显存与利用率监控
    - 训练速度（tokens/s）统计
    - 输出 JSONL 日志，兼容 visualize 模块

典型用法：
    monitor = TrainingMonitor(log_dir="logs/stage1")

    for step, batch in enumerate(dataloader):
        loss_dict = model(batch)
        monitor.log_step(
            step=step,
            lm_loss=loss_dict["lm_loss"].item(),
            ctc_loss=loss_dict.get("ctc_loss", 0.0),
            lr=scheduler.get_last_lr()[0],
            grad_norm=total_norm,
            num_tokens=batch["num_tokens"],
        )

    monitor.close(plot=True)
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Union


class _SlidingMeter:
    """带滑动窗口的标量跟踪器（与 metrics/tracker.py 保持一致）。"""

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

    def window_avg(self) -> Dict[str, float]:
        if not self.windows or not self._buffer:
            return {}
        vals = list(self._buffer)
        n = len(vals)
        result = {}
        for w in self.windows:
            label = f"{w // 1000}k" if w >= 1000 and w % 1000 == 0 else f"{w}"
            result[label] = sum(vals[-w:]) / min(n, w)
        return result

    def reset(self) -> None:
        self._buffer.clear()
        self._total = 0.0
        self._count = 0
        self._min = float("inf")
        self._max = float("-inf")
        self._current = None


class TrainingMonitor:
    """
    Omni 训练监控器，支持多模态 loss 追踪、GPU 监控和 JSONL 日志。

    所有方法均为纯 Python，兼容 GPU 和 macOS 环境。
    """

    def __init__(
        self,
        log_dir: Optional[Union[str, Path]] = None,
        windows: Optional[List[int]] = None,
        stage: int = 1,
    ):
        _windows = windows or [500, 1000, 3000]

        # ── 多模态 Loss 追踪 ─────────────────────────
        self.loss_total = _SlidingMeter(_windows)       # 总 loss
        self.loss_lm = _SlidingMeter(_windows)          # 语言模型 loss
        self.loss_ctc = _SlidingMeter(_windows)         # CTC 语音 loss
        self.loss_dpo = _SlidingMeter(_windows)         # DPO loss

        # ── PPL 追踪 ─────────────────────────────────
        self.ppl = _SlidingMeter(_windows)

        # ── 梯度监控 ─────────────────────────────────
        self._grad_norms: Deque[float] = deque(maxlen=1000)
        self._grad_clipped: Deque[float] = deque(maxlen=1000)

        # ── 学习率 ───────────────────────────────────
        self._lr: float = 0.0

        # ── GPU 监控 ─────────────────────────────────
        self._gpu_mem_used: float = 0.0         # GB
        self._gpu_mem_total: float = 0.0        # GB
        self._gpu_utilization: float = 0.0      # 百分比

        # ── 训练速度 ─────────────────────────────────
        self._speed_timer: Optional[float] = None
        self._token_counter: int = 0
        self._tokens_per_sec: float = 0.0
        self._step_timer: Optional[float] = None
        self._steps_per_sec: float = 0.0

        # ── 全局状态 ─────────────────────────────────
        self.global_step: int = 0
        self.current_epoch: int = 0
        self.stage: int = stage

        # ── JSONL 日志 ───────────────────────────────
        self._log_dir: Optional[Path] = None
        self._jsonl_file = None
        self._jsonl_path: Optional[Path] = None
        if log_dir:
            self._log_dir = Path(log_dir)
            self._log_dir.mkdir(parents=True, exist_ok=True)
            self._jsonl_path = self._log_dir / f"metrics_stage{stage}.jsonl"
            self._jsonl_file = open(self._jsonl_path, "a", encoding="utf-8")

    # ── 指标更新 ──────────────────────────────────────

    def _update_gpu_stats(self) -> None:
        """采集 GPU 显存和利用率（仅 CUDA 可用时）。"""
        try:
            import torch
            if torch.cuda.is_available():
                self._gpu_mem_used = torch.cuda.memory_allocated() / (1024 ** 3)
                self._gpu_mem_total = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
                # 利用率需要 pynvml，这里用显存占用率近似
                self._gpu_utilization = (
                    self._gpu_mem_used / self._gpu_mem_total * 100
                    if self._gpu_mem_total > 0 else 0.0
                )
        except Exception:
            pass

    def _update_speed(self, num_tokens: int = 0) -> None:
        """更新训练速度统计。"""
        now = time.time()

        # tokens/s
        if num_tokens > 0:
            self._token_counter += num_tokens
            if self._speed_timer is not None:
                elapsed = now - self._speed_timer
                if elapsed > 0:
                    self._tokens_per_sec = num_tokens / elapsed
            self._speed_timer = now

        # steps/s
        if self._step_timer is not None:
            elapsed = now - self._step_timer
            if elapsed > 0:
                self._steps_per_sec = 1.0 / elapsed
        self._step_timer = now

    # ── 核心日志方法 ──────────────────────────────────

    def log_step(
        self,
        step: Optional[int] = None,
        lm_loss: float = 0.0,
        ctc_loss: float = 0.0,
        dpo_loss: float = 0.0,
        lr: float = 0.0,
        grad_norm: float = 0.0,
        max_grad_norm: float = 1.0,
        num_tokens: int = 0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        记录单步训练指标并写入 JSONL。

        Args:
            step:          全局步数（若为 None 则自动递增）
            lm_loss:       语言模型 loss
            ctc_loss:      CTC 语音 loss
            dpo_loss:      DPO loss
            lr:            当前学习率
            grad_norm:     梯度 L2 范数
            max_grad_norm: 梯度裁剪阈值
            num_tokens:    本 step 处理的 token 数
            extra:         额外指标字典
        """
        if step is not None:
            self.global_step = step
        else:
            self.global_step += 1

        # 更新各 loss 跟踪器
        total_loss = lm_loss + ctc_loss + dpo_loss
        self.loss_total.step(total_loss)
        if lm_loss > 0:
            self.loss_lm.step(lm_loss)
        if ctc_loss > 0:
            self.loss_ctc.step(ctc_loss)
        if dpo_loss > 0:
            self.loss_dpo.step(dpo_loss)

        # PPL（仅基于 lm_loss）
        if lm_loss > 0:
            ppl_val = math.exp(min(lm_loss, 20.0))  # 防止溢出
            self.ppl.step(ppl_val)

        # 学习率
        self._lr = lr

        # 梯度
        self._grad_norms.append(grad_norm)
        is_clipped = grad_norm > max_grad_norm
        self._grad_clipped.append(1.0 if is_clipped else 0.0)

        # 速度
        self._update_speed(num_tokens)

        # GPU 监控
        self._update_gpu_stats()

        # 写入 JSONL
        self._write_jsonl(extra)

    def log_epoch(self, epoch: int, val_loss: Optional[float] = None) -> None:
        """
        记录 epoch 级别的指标。

        Args:
            epoch:    当前 epoch 编号
            val_loss: 验证集 loss（可选）
        """
        self.current_epoch = epoch
        extra = {"epoch": epoch}
        if val_loss is not None:
            extra["val_loss"] = val_loss
            extra["val_ppl"] = math.exp(min(val_loss, 20.0))
        self._write_jsonl(extra)

    def _write_jsonl(self, extra: Optional[Dict[str, Any]] = None) -> None:
        """将当前状态写入 JSONL 文件。"""
        if self._jsonl_file is None:
            return

        row: Dict[str, Any] = {
            "step": self.global_step,
            "stage": self.stage,
            "epoch": self.current_epoch,
            "Loss/Total": self.loss_total.current,
            "Loss/Total_Min": self.loss_total.min if self.loss_total.min != float("inf") else None,
            "Loss/LM": self.loss_lm.current,
            "Loss/CTC": self.loss_ctc.current,
            "Loss/DPO": self.loss_dpo.current,
            "PPL": self.ppl.current,
            "LearningRate": self._lr,
            "Grad/Norm": self._grad_norms[-1] if self._grad_norms else None,
            "Grad/ClipRate": (
                sum(self._grad_clipped) / len(self._grad_clipped)
                if self._grad_clipped else 0.0
            ),
            "Speed/TokensPerSec": self._tokens_per_sec,
            "Speed/StepsPerSec": self._steps_per_sec,
            "GPU/MemUsedGB": self._gpu_mem_used,
            "GPU/MemTotalGB": self._gpu_mem_total,
            "GPU/Utilization": self._gpu_utilization,
        }

        # 滑动平均
        for k, v in self.loss_total.window_avg().items():
            row[f"Loss/Total_Avg_{k}"] = v
        for k, v in self.loss_lm.window_avg().items():
            row[f"Loss/LM_Avg_{k}"] = v

        # 额外指标
        if extra:
            for k, v in extra.items():
                if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                    row[k] = v

        # 过滤 None 值
        row = {k: v for k, v in row.items() if v is not None}

        self._jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._jsonl_file.flush()

    # ── 汇总 ─────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """返回当前训练状态的摘要字典。"""
        return {
            "global_step": self.global_step,
            "stage": self.stage,
            "epoch": self.current_epoch,
            "total_loss": self.loss_total.current or 0.0,
            "total_loss_avg": self.loss_total.avg,
            "total_loss_min": self.loss_total.min if self.loss_total.min != float("inf") else 0.0,
            "lm_loss": self.loss_lm.current or 0.0,
            "ctc_loss": self.loss_ctc.current or 0.0,
            "dpo_loss": self.loss_dpo.current or 0.0,
            "ppl": self.ppl.current or 0.0,
            "lr": self._lr,
            "grad_clip_rate": (
                sum(self._grad_clipped) / len(self._grad_clipped)
                if self._grad_clipped else 0.0
            ),
            "tokens_per_sec": self._tokens_per_sec,
            "steps_per_sec": self._steps_per_sec,
            "gpu_mem_used_gb": self._gpu_mem_used,
        }

    # ── 生命周期 ──────────────────────────────────────

    def close(self, plot: bool = True) -> None:
        """关闭日志文件，可选生成训练曲线。"""
        if self._jsonl_file is not None:
            self._jsonl_file.close()
            self._jsonl_file = None
        if plot and self._log_dir is not None:
            try:
                from .visualize import plot_omni_curves
                plot_omni_curves(
                    jsonl_path=str(self._jsonl_path),
                    output_path=str(self._log_dir / f"curves_stage{self.stage}.png"),
                )
            except Exception as e:
                print(f"绘制训练曲线失败: {e}")


# ── 模块自测 ────────────────────────────────────────────
if __name__ == "__main__":
    import random
    import tempfile

    print("=== TrainingMonitor 模块自测 ===\n")

    with tempfile.TemporaryDirectory() as tmp:
        monitor = TrainingMonitor(log_dir=tmp, stage=3)

        for i in range(200):
            lm_loss = 5.0 * math.exp(-i / 50) + random.gauss(0, 0.1)
            ctc_loss = 3.0 * math.exp(-i / 80) + random.gauss(0, 0.05) if i > 50 else 0.0
            monitor.log_step(
                lm_loss=max(lm_loss, 0.01),
                ctc_loss=max(ctc_loss, 0.0),
                lr=2e-5 * (1 - i / 200),
                grad_norm=random.uniform(0.3, 2.5),
                max_grad_norm=1.0,
                num_tokens=random.randint(1000, 4000),
            )

        monitor.log_epoch(epoch=1, val_loss=1.5)

        s = monitor.summary()
        print("训练摘要:")
        for k, v in s.items():
            fmt = f"{v:.6f}" if isinstance(v, float) else str(v)
            print(f"  {k}: {fmt}")

        monitor.close(plot=False)
        assert (Path(tmp) / "metrics_stage3.jsonl").exists(), "JSONL 文件未生成"
        print(f"\n已写入 {tmp}/metrics_stage3.jsonl")

    print("\n=== 自测通过 ===")

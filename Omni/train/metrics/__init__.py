"""
Omni 训练指标模块 —— 训练监控、可视化与评测。

导出：
    TrainingMonitor  — 训练过程指标采集与 JSONL 日志记录
    plot_omni_curves — 从 JSONL 日志绘制训练曲线
    compute_wer, compute_cer, compute_bleu, evaluate_model — 评测指标
"""

from .training_monitor import TrainingMonitor
from .visualize import plot_omni_curves
from .eval_metrics import compute_wer, compute_cer, compute_bleu, evaluate_model

__all__ = [
    "TrainingMonitor",
    "plot_omni_curves",
    "compute_wer",
    "compute_cer",
    "compute_bleu",
    "evaluate_model",
]

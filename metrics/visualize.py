"""
训练曲线绘图工具 —— 从 TensorBoard 事件文件或 JSON 日志生成训练曲线图。

用法：
    python metrics/visualize.py --log-dir logs/pretrain --output plots/pretrain.png
    python metrics/visualize.py --json-log logs/metrics.json --output plots/run.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _try_read_tb_events(log_dir: str) -> Dict[str, List[Tuple[int, float]]]:
    """尝试从 TensorBoard 事件文件读取标量数据。"""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("tensorboard 未安装，无法读取事件文件。请 pip install tensorboard")
        return {}

    acc = EventAccumulator(log_dir)
    acc.Reload()

    data: Dict[str, List[Tuple[int, float]]] = {}
    for tag in acc.Tags().get("scalars", []):
        events = acc.Scalars(tag)
        data[tag] = [(e.step, e.value) for e in events]
    return data


def _read_json_log(json_path: str) -> Dict[str, List[Tuple[int, float]]]:
    """从 JSONL 日志读取指标（每行一个 JSON 对象，含 step 字段）。"""
    data: Dict[str, List[Tuple[int, float]]] = {}
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            step = record.get("step", record.get("global_step", 0))
            for k, v in record.items():
                if k in ("step", "global_step"):
                    continue
                if isinstance(v, (int, float)) and not math.isnan(v):
                    data.setdefault(k, []).append((step, v))
    return data


def plot_training_curves(
    data: Dict[str, List[Tuple[int, float]]],
    output_path: str = "training_curves.png",
    title: str = "Training Curves",
    figsize: Tuple[int, int] = (16, 10),
    tag_groups: Optional[Dict[str, List[str]]] = None,
) -> None:
    """
    将指标数据绘制成多子图。

    Args:
        data: tag -> [(step, value), ...]
        output_path: 输出图片路径
        tag_groups: 子图分组，如 {"Loss": ["Train/Loss", "Train/Loss_Min"], ...}
                    若为 None 则自动按前缀分组
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装，无法绘图。请 pip install matplotlib")
        return

    if tag_groups is None:
        groups: Dict[str, List[str]] = {}
        for tag in sorted(data.keys()):
            prefix = tag.split("/")[0] if "/" in tag else "Other"
            groups.setdefault(prefix, []).append(tag)
        tag_groups = groups

    n_groups = len(tag_groups)
    if n_groups == 0:
        print("没有可绘制的数据。")
        return

    cols = min(3, n_groups)
    rows = math.ceil(n_groups / cols)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for idx, (group_name, tags) in enumerate(tag_groups.items()):
        ax = axes[idx // cols][idx % cols]
        for tag in tags:
            if tag not in data:
                continue
            steps, values = zip(*data[tag])
            label = tag.split("/")[-1] if "/" in tag else tag
            ax.plot(steps, values, label=label, alpha=0.8, linewidth=1.0)
        ax.set_title(group_name, fontsize=11)
        ax.set_xlabel("Step")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for idx in range(n_groups, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"训练曲线已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="训练曲线绘图工具")
    parser.add_argument("--log-dir", type=str, default=None, help="TensorBoard 日志目录")
    parser.add_argument("--json-log", type=str, default=None, help="JSONL 指标日志文件")
    parser.add_argument("--output", type=str, default="plots/training_curves.png", help="输出图片路径")
    parser.add_argument("--title", type=str, default="Training Curves", help="图片标题")
    args = parser.parse_args()

    data: Dict[str, List[Tuple[int, float]]] = {}
    if args.log_dir:
        data.update(_try_read_tb_events(args.log_dir))
    if args.json_log:
        data.update(_read_json_log(args.json_log))

    if not data:
        print("未找到任何指标数据。请指定 --log-dir 或 --json-log")
        return

    print(f"共加载 {len(data)} 个指标标签，总数据点: {sum(len(v) for v in data.values())}")
    plot_training_curves(data, output_path=args.output, title=args.title)


if __name__ == "__main__":
    main()

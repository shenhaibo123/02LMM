"""
Omni 训练曲线绘图工具 —— 从 JSONL 日志生成训练曲线图。

参考项目 metrics/visualize.py 的设计风格，扩展以下能力：
    - 支持多模态 loss 分别绘制（LM / CTC / DPO）
    - 支持多阶段对比绘制
    - 支持 GPU 使用曲线

用法：
    python -m Omni.train.metrics.visualize --jsonl logs/metrics_stage1.jsonl --output plots/stage1.png
    python -m Omni.train.metrics.visualize --jsonl-dir logs/ --output plots/all_stages.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _read_jsonl(jsonl_path: str) -> Dict[str, List[Tuple[int, float]]]:
    """从 JSONL 日志读取指标（每行一个 JSON 对象，含 step 字段）。"""
    data: Dict[str, List[Tuple[int, float]]] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            step = record.get("step", 0)
            for k, v in record.items():
                if k in ("step", "stage", "epoch"):
                    continue
                if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                    data.setdefault(k, []).append((step, float(v)))
    return data


def _read_multi_stage_jsonl(jsonl_dir: str) -> Dict[str, Dict[str, List[Tuple[int, float]]]]:
    """从目录中读取多阶段 JSONL 日志，返回 {stage_name: {tag: [(step, val)]}}。"""
    result = {}
    jsonl_dir_path = Path(jsonl_dir)
    for jsonl_file in sorted(jsonl_dir_path.glob("metrics_stage*.jsonl")):
        stage_name = jsonl_file.stem.replace("metrics_", "")
        result[stage_name] = _read_jsonl(str(jsonl_file))
    return result


# 默认指标分组
OMNI_TAG_GROUPS: Dict[str, List[str]] = {
    "Loss (Total)": [
        "Loss/Total",
        "Loss/Total_Min",
        "Loss/Total_Avg_500",
        "Loss/Total_Avg_1k",
        "Loss/Total_Avg_3k",
    ],
    "Loss (Modal)": [
        "Loss/LM",
        "Loss/CTC",
        "Loss/DPO",
    ],
    "PPL": ["PPL"],
    "Learning Rate": ["LearningRate"],
    "Gradient": ["Grad/Norm", "Grad/ClipRate"],
    "Speed": ["Speed/TokensPerSec", "Speed/StepsPerSec"],
    "GPU": ["GPU/MemUsedGB", "GPU/Utilization"],
}


def plot_omni_curves(
    jsonl_path: Optional[str] = None,
    data: Optional[Dict[str, List[Tuple[int, float]]]] = None,
    output_path: str = "training_curves.png",
    title: str = "Omni Training Curves",
    figsize: Tuple[int, int] = (18, 12),
    tag_groups: Optional[Dict[str, List[str]]] = None,
) -> None:
    """
    绘制 Omni 训练曲线。

    Args:
        jsonl_path:  JSONL 日志文件路径（与 data 二选一）
        data:        已解析的指标数据 {tag: [(step, value)]}
        output_path: 输出图片路径
        title:       图片标题
        figsize:     图片尺寸
        tag_groups:  子图分组
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装，无法绘图。请 pip install matplotlib")
        return

    # 读取数据
    if data is None:
        if jsonl_path is None:
            print("需要提供 jsonl_path 或 data 参数")
            return
        data = _read_jsonl(jsonl_path)

    if not data:
        print("没有可绘制的数据。")
        return

    # 确定分组
    if tag_groups is None:
        tag_groups = {}
        for group_name, tags in OMNI_TAG_GROUPS.items():
            existing = [t for t in tags if t in data]
            if existing:
                tag_groups[group_name] = existing
        # 未分组的 tag 归入 Other
        assigned = {t for tags in tag_groups.values() for t in tags}
        other = [t for t in sorted(data.keys()) if t not in assigned]
        if other:
            tag_groups["Other"] = other

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

    # 隐藏空白子图
    for idx in range(n_groups, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"训练曲线已保存到: {output_path}")


def plot_multi_stage_comparison(
    jsonl_dir: str,
    output_path: str = "multi_stage_curves.png",
    metric: str = "Loss/Total",
    title: str = "Multi-Stage Loss Comparison",
    figsize: Tuple[int, int] = (12, 6),
) -> None:
    """
    绘制多阶段指标对比图。

    Args:
        jsonl_dir:   包含多阶段 JSONL 文件的目录
        output_path: 输出图片路径
        metric:      要对比的指标名
        title:       图片标题
        figsize:     图片尺寸
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装，无法绘图。请 pip install matplotlib")
        return

    stage_data = _read_multi_stage_jsonl(jsonl_dir)
    if not stage_data:
        print(f"在 {jsonl_dir} 中未找到阶段日志文件。")
        return

    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for stage_name, data in stage_data.items():
        if metric in data:
            steps, values = zip(*data[metric])
            ax.plot(steps, values, label=stage_name, alpha=0.8, linewidth=1.2)

    ax.set_xlabel("Step")
    ax.set_ylabel(metric)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"多阶段对比图已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Omni 训练曲线绘图工具")
    parser.add_argument("--jsonl", type=str, default=None, help="单个 JSONL 日志文件路径")
    parser.add_argument("--jsonl-dir", type=str, default=None, help="多阶段 JSONL 日志目录")
    parser.add_argument("--output", type=str, default="plots/omni_curves.png", help="输出图片路径")
    parser.add_argument("--title", type=str, default="Omni Training Curves", help="图片标题")
    parser.add_argument("--metric", type=str, default="Loss/Total", help="多阶段对比指标名")
    args = parser.parse_args()

    if args.jsonl:
        data = _read_jsonl(args.jsonl)
        print(f"共加载 {len(data)} 个指标标签，总数据点: {sum(len(v) for v in data.values())}")
        plot_omni_curves(data=data, output_path=args.output, title=args.title)
    elif args.jsonl_dir:
        plot_multi_stage_comparison(
            jsonl_dir=args.jsonl_dir,
            output_path=args.output,
            metric=args.metric,
            title=args.title,
        )
    else:
        print("请指定 --jsonl 或 --jsonl-dir 参数")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
最小可用的 LLM 评测脚本，基于 lm-evaluation-harness。

支持两种评测模式：
  1. 本地 HuggingFace 权重（--backend hf）
  2. OpenAI 兼容 API（--backend api），对接 serve_openai_api.py

用法示例见文件底部或运行 --help。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

try:
    import lm_eval
except ImportError:
    print(
        "缺少 lm_eval，请先安装：\n"
        "  pip install lm_eval          # 评测框架\n"
        "  pip install transformers torch datasets  # 常用本地模型后端依赖\n"
        "详见 https://github.com/EleutherAI/lm-evaluation-harness",
        file=sys.stderr,
    )
    sys.exit(1)


# ── 默认评测任务 ──────────────────────────────────────────────
# 覆盖常用维度：常识/知识 · 推理 · 代码 · 中文
DEFAULT_TASKS = [
    "hellaswag",          # 常识推理（4选1，acc_norm）
    "arc_easy",           # 科学常识（简单）
    "arc_challenge",      # 科学常识（困难）
    "piqa",               # 物理直觉（2选1）
    "winogrande",         # 常识共指消解
    "mmlu",               # 多学科知识（57子任务）
    "gsm8k",              # 小学数学推理
    "truthfulqa_mc2",     # 真实性多选
]

# 如果只想快速冒烟测试，可用 SMOKE_TASKS
SMOKE_TASKS = [
    "hellaswag",
    "arc_easy",
    "piqa",
    "gsm8k",
]


def build_hf_args(model_path: str, dtype: str, device: str) -> str:
    parts = [f"pretrained={model_path}"]
    if dtype:
        parts.append(f"dtype={dtype}")
    if device and device != "auto":
        parts.append(f"device={device}")
    parts.append("trust_remote_code=True")
    return ",".join(parts)


def build_api_args(
    model_name: str, base_url: str, tokenizer_path: str
) -> str:
    parts = [
        f"model={model_name}",
        f"base_url={base_url}",
        "num_concurrent=1",
        "max_retries=3",
    ]
    if tokenizer_path:
        parts.append(f"tokenizer={tokenizer_path}")
    return ",".join(parts)


def run_eval(
    model_type: str,
    model_args: str,
    tasks: list[str],
    num_fewshot: int | None,
    batch_size: int | str,
    limit: int | None,
    output_path: str | None,
) -> dict:
    """调用 lm_eval.simple_evaluate 并返回结果字典。"""
    kwargs = dict(
        model=model_type,
        model_args=model_args,
        tasks=tasks,
        batch_size=batch_size,
        log_samples=False,
    )
    if num_fewshot is not None:
        kwargs["num_fewshot"] = num_fewshot
    if limit is not None:
        kwargs["limit"] = limit

    results = lm_eval.simple_evaluate(**kwargs)

    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        dumped = lm_eval.utils.make_table(results)
        p.write_text(dumped, encoding="utf-8")
        json_path = p.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                results.get("results", {}), f, ensure_ascii=False, indent=2
            )
        print(f"\n结果已保存到：\n  表格 -> {p}\n  JSON -> {json_path}")

    return results


def print_summary(results: dict) -> None:
    """用 lm_eval 内置表格漂亮地打印结果。"""
    try:
        table = lm_eval.utils.make_table(results)
        print("\n" + table)
    except Exception:
        for task, metrics in results.get("results", {}).items():
            print(f"\n[{task}]")
            for k, v in sorted(metrics.items()):
                if k.startswith("alias"):
                    continue
                print(f"  {k}: {v}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="基于 lm-evaluation-harness 的最小评测脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：

  # 1) 本地 HF 权重评测（smoke test，只跑 5 条样本）
  python scripts/eval_model_benchmark.py \\
      --backend hf \\
      --model-path ./model \\
      --tasks hellaswag arc_easy \\
      --limit 5

  # 2) 对接 serve_openai_api.py（先启动服务 python scripts/serve_openai_api.py）
  python scripts/eval_model_benchmark.py \\
      --backend api \\
      --base-url http://localhost:8998/v1/chat/completions \\
      --tokenizer-path ./model \\
      --tasks hellaswag gsm8k

  # 3) 评测远程模型（如 Qwen3-0.6B）
  python scripts/eval_model_benchmark.py \\
      --backend hf \\
      --model-path Qwen/Qwen3-0.6B \\
      --tasks hellaswag arc_easy piqa mmlu \\
      --limit 100

  # 4) 完整评测并保存结果
  python scripts/eval_model_benchmark.py \\
      --backend hf \\
      --model-path ./model \\
      --output eval_results/k.txt
""",
    )

    parser.add_argument(
        "--backend",
        choices=["hf", "api"],
        default="hf",
        help="评测后端：hf=本地 HuggingFace 权重，api=OpenAI 兼容 API（默认 hf）",
    )

    # HF 后端参数
    hf_group = parser.add_argument_group("HF 后端参数")
    hf_group.add_argument(
        "--model-path",
        default="./model",
        help="模型路径或 HuggingFace model id（默认 ./model）",
    )
    hf_group.add_argument(
        "--dtype",
        default="float32",
        help="模型精度（float32 / float16 / bfloat16，默认 float32）",
    )
    hf_group.add_argument(
        "--device",
        default="auto",
        help="设备映射（auto / cpu / cuda:0，默认 auto）",
    )

    # API 后端参数
    api_group = parser.add_argument_group("API 后端参数")
    api_group.add_argument(
        "--base-url",
        default="http://localhost:8998/v1/chat/completions",
        help="OpenAI 兼容 API 端点（默认 http://localhost:8998/v1/chat/completions）",
    )
    api_group.add_argument(
        "--model-name",
        default="K",
        help="API 请求中的 model 字段（默认 K）",
    )
    api_group.add_argument(
        "--tokenizer-path",
        default="",
        help="本地 tokenizer 路径，用于 API 模式下估算 token 数",
    )

    # 评测控制参数
    eval_group = parser.add_argument_group("评测控制")
    eval_group.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help=f"评测任务列表（空则使用默认：{' '.join(DEFAULT_TASKS)}）",
    )
    eval_group.add_argument(
        "--smoke",
        action="store_true",
        help=f"快速冒烟测试（只跑 {SMOKE_TASKS}，每任务 limit=20）",
    )
    eval_group.add_argument(
        "--num-fewshot",
        type=int,
        default=None,
        help="few-shot 数量（None=使用任务默认值）",
    )
    eval_group.add_argument(
        "--batch-size",
        default="auto",
        help="批次大小（数字或 auto，默认 auto）",
    )
    eval_group.add_argument(
        "--limit",
        type=int,
        default=None,
        help="每任务最多评测多少条样本（None=全部，调试时可设 5~50）",
    )
    eval_group.add_argument(
        "--output",
        default=None,
        help="结果保存路径（如 eval_results/run.txt），同时生成 .json",
    )

    args = parser.parse_args()

    # 确定任务列表
    if args.smoke:
        tasks = SMOKE_TASKS
        limit = args.limit or 20
    else:
        tasks = args.tasks or DEFAULT_TASKS
        limit = args.limit

    # 构建模型参数
    if args.backend == "hf":
        model_type = "hf"
        model_args = build_hf_args(args.model_path, args.dtype, args.device)
    else:
        model_type = "local-chat-completions"
        model_args = build_api_args(
            args.model_name, args.base_url, args.tokenizer_path
        )

    print("=" * 60)
    print(f"评测后端:   {args.backend}")
    print(f"模型参数:   {model_args}")
    print(f"评测任务:   {', '.join(tasks)}")
    print(f"few-shot:   {args.num_fewshot or '任务默认'}")
    print(f"batch_size: {args.batch_size}")
    print(f"limit:      {limit or '全部'}")
    print("=" * 60)

    try:
        batch_size = (
            int(args.batch_size)
            if args.batch_size.isdigit()
            else args.batch_size
        )
    except (ValueError, AttributeError):
        batch_size = args.batch_size

    results = run_eval(
        model_type=model_type,
        model_args=model_args,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=batch_size,
        limit=limit,
        output_path=args.output,
    )

    print_summary(results)


if __name__ == "__main__":
    main()

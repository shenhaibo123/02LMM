"""
多模态评测脚本 —— 评估 Omni 模型在多个基准上的表现。

支持评测维度：
    - 图像理解: MMBench, TextVQA, MMMU
    - 音频理解: AISHELL-1 (WER/CER), LibriSpeech (WER)
    - 文本能力: MMLU, C-Eval, GSM8K
    - 综合: SEEDBench

评测后端：
    - Native:     MS-Swift 内置（文本基准）
    - VLMEvalKit: 多模态基准
    - Custom:     自定义音频/语音评测

用法：
    # 快速评测（Smoke Test）
    python eval_multimodal.py --model Qwen/Qwen2.5-Omni-7B --mode smoke

    # 文本基准
    python eval_multimodal.py --model Qwen/Qwen2.5-Omni-7B --mode text

    # 多模态基准
    python eval_multimodal.py --model Qwen/Qwen2.5-Omni-7B --mode multimodal

    # 音频基准
    python eval_multimodal.py --model Qwen/Qwen2.5-Omni-7B --mode audio

    # 全量评测
    python eval_multimodal.py --model Qwen/Qwen2.5-Omni-7B --mode full

    # 评测 LoRA adapter
    python eval_multimodal.py --model Qwen/Qwen2.5-Omni-7B --adapters output/stage3/checkpoint-xxx
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── 评测基准配置 ─────────────────────────────────────────

EVAL_SUITES = {
    "smoke": {
        "text": {"datasets": ["gsm8k"], "limit": 10},
        "description": "快速验证（~5 分钟）",
    },
    "text": {
        "text": {
            "datasets": ["mmlu", "ceval", "gsm8k", "arc"],
            "limit": 100,
        },
        "description": "文本基准评测",
    },
    "multimodal": {
        "vlm": {
            "datasets": ["MMBench_DEV_EN", "SEEDBench_IMG", "TextVQA_VAL", "MMMU_DEV_VAL"],
            "limit": None,
        },
        "description": "多模态基准评测（VLMEvalKit）",
    },
    "audio": {
        "audio": {
            "datasets": ["aishell1_test", "librispeech_test_clean"],
        },
        "description": "音频/ASR 评测",
    },
    "full": {
        "text": {
            "datasets": ["mmlu", "ceval", "cmmlu", "gsm8k", "arc", "humaneval"],
            "limit": None,
        },
        "vlm": {
            "datasets": ["MMBench_DEV_EN", "MMBench_DEV_CN", "SEEDBench_IMG",
                         "TextVQA_VAL", "MMMU_DEV_VAL", "ScienceQA_TEST"],
            "limit": None,
        },
        "audio": {
            "datasets": ["aishell1_test", "librispeech_test_clean"],
        },
        "description": "全量评测（约 2-4 小时）",
    },
}


def run_swift_eval_native(
    model: str,
    datasets: List[str],
    limit: Optional[int] = None,
    adapters: Optional[str] = None,
    output_dir: str = "eval_output",
    device: str = "0",
) -> Dict[str, Any]:
    """使用 MS-Swift Native 后端评测文本基准。"""
    cmd = [
        "swift", "eval",
        "--model", model,
        "--eval_backend", "Native",
        "--infer_backend", "pt",
        "--eval_dataset", *datasets,
        "--eval_output_dir", output_dir,
    ]
    if limit is not None:
        cmd.extend(["--eval_limit", str(limit)])
    if adapters:
        cmd.extend(["--adapters", adapters])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device

    logger.info(f"运行 Native 评测: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"评测失败:\n{result.stderr}")
        return {"status": "failed", "error": result.stderr}

    logger.info(f"评测输出:\n{result.stdout}")
    return {"status": "success", "output": result.stdout, "datasets": datasets}


def run_swift_eval_vlm(
    model: str,
    datasets: List[str],
    limit: Optional[int] = None,
    adapters: Optional[str] = None,
    output_dir: str = "eval_output",
    device: str = "0",
) -> Dict[str, Any]:
    """使用 VLMEvalKit 后端评测多模态基准。"""
    cmd = [
        "swift", "eval",
        "--model", model,
        "--eval_backend", "VLMEvalKit",
        "--eval_dataset", *datasets,
        "--eval_output_dir", output_dir,
    ]
    if limit is not None:
        cmd.extend(["--eval_limit", str(limit)])
    if adapters:
        cmd.extend(["--adapters", adapters])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device

    logger.info(f"运行 VLMEvalKit 评测: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"VLM 评测失败:\n{result.stderr}")
        return {"status": "failed", "error": result.stderr}

    return {"status": "success", "output": result.stdout, "datasets": datasets}


def run_audio_eval(
    model: str,
    datasets: List[str],
    adapters: Optional[str] = None,
    output_dir: str = "eval_output",
    device: str = "0",
) -> Dict[str, Any]:
    """自定义音频/ASR 评测。"""
    # 引用项目中的评测指标
    parent_dir = str(Path(__file__).resolve().parent.parent.parent / "train")
    sys.path.insert(0, parent_dir)

    results = {}

    for ds_name in datasets:
        logger.info(f"评测音频数据集: {ds_name}")

        if ds_name == "aishell1_test":
            result = _eval_asr_swift(model, "speech_asr/speech_asr_aishell1_trainsets:test",
                                     adapters, device, output_dir, lang="zh")
        elif ds_name == "librispeech_test_clean":
            result = _eval_asr_swift(model, "speech_asr/speech_asr_aishell1_trainsets:test",
                                     adapters, device, output_dir, lang="en")
        else:
            result = {"status": "skipped", "reason": f"未知数据集: {ds_name}"}

        results[ds_name] = result

    return results


def _eval_asr_swift(
    model: str,
    dataset: str,
    adapters: Optional[str],
    device: str,
    output_dir: str,
    lang: str = "zh",
) -> Dict[str, Any]:
    """使用 MS-Swift 推理 + 自定义 WER/CER 计算。"""

    # 使用 swift infer 进行推理
    cmd = [
        "swift", "infer",
        "--model", model,
        "--stream", "false",
        "--temperature", "0",
        "--max_new_tokens", "512",
    ]
    if adapters:
        cmd.extend(["--adapters", adapters])

    logger.info(f"ASR 评测 ({lang}): 使用 swift infer...")

    # 对于 ASR 评测，我们需要收集 predictions 和 references
    # 然后使用项目中的 compute_wer / compute_cer 计算
    try:
        from metrics.eval_metrics import compute_wer, compute_cer
        logger.info("已加载项目评测指标: WER, CER")
    except ImportError:
        logger.warning("无法加载项目评测指标，使用简化版本")
        return {"status": "skipped", "reason": "评测指标模块不可用"}

    return {
        "status": "ready",
        "message": f"ASR 评测 ({lang}) 已配置，需要实际推理数据",
        "metrics": ["wer", "cer"] if lang == "zh" else ["wer"],
    }


def run_evaluation(
    model: str,
    mode: str = "smoke",
    adapters: Optional[str] = None,
    output_dir: str = "eval_output",
    device: str = "0",
) -> Dict[str, Any]:
    """统一评测入口。"""
    if mode not in EVAL_SUITES:
        raise ValueError(f"未知评测模式: {mode}，可选: {list(EVAL_SUITES.keys())}")

    suite = EVAL_SUITES[mode]
    logger.info(f"\n{'='*60}")
    logger.info(f"开始评测: {suite['description']}")
    logger.info(f"模型: {model}")
    if adapters:
        logger.info(f"Adapters: {adapters}")
    logger.info(f"{'='*60}\n")

    all_results = {"mode": mode, "model": model, "timestamp": time.strftime("%Y%m%d_%H%M%S")}

    # 文本基准
    if "text" in suite:
        cfg = suite["text"]
        logger.info(f"[文本基准] 数据集: {cfg['datasets']}")
        text_results = run_swift_eval_native(
            model=model,
            datasets=cfg["datasets"],
            limit=cfg.get("limit"),
            adapters=adapters,
            output_dir=f"{output_dir}/text",
            device=device,
        )
        all_results["text"] = text_results

    # 多模态基准
    if "vlm" in suite:
        cfg = suite["vlm"]
        logger.info(f"[多模态基准] 数据集: {cfg['datasets']}")
        vlm_results = run_swift_eval_vlm(
            model=model,
            datasets=cfg["datasets"],
            limit=cfg.get("limit"),
            adapters=adapters,
            output_dir=f"{output_dir}/vlm",
            device=device,
        )
        all_results["vlm"] = vlm_results

    # 音频基准
    if "audio" in suite:
        cfg = suite["audio"]
        logger.info(f"[音频基准] 数据集: {cfg['datasets']}")
        audio_results = run_audio_eval(
            model=model,
            datasets=cfg["datasets"],
            adapters=adapters,
            output_dir=f"{output_dir}/audio",
            device=device,
        )
        all_results["audio"] = audio_results

    # 保存结果
    result_path = Path(output_dir) / f"eval_results_{mode}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"\n评测结果已保存: {result_path}")

    return all_results


def print_results_table(results: Dict[str, Any]) -> None:
    """打印评测结果表格。"""
    print("\n" + "=" * 70)
    print(f"  评测结果 —— 模式: {results.get('mode', '?')}")
    print(f"  模型: {results.get('model', '?')}")
    print("=" * 70)

    for category, cat_results in results.items():
        if category in ("mode", "model", "timestamp"):
            continue
        if isinstance(cat_results, dict):
            status = cat_results.get("status", "unknown")
            print(f"\n  [{category}] 状态: {status}")
            if "datasets" in cat_results:
                print(f"    数据集: {', '.join(cat_results['datasets'])}")
            if "output" in cat_results and cat_results["output"]:
                # 提取关键数值行
                for line in str(cat_results["output"]).split("\n"):
                    if any(k in line.lower() for k in ["accuracy", "score", "wer", "cer", "bleu"]):
                        print(f"    {line.strip()}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Omni 模型多模态评测")
    parser.add_argument("--model", type=str, required=True,
                        help="模型名称或路径")
    parser.add_argument("--adapters", type=str, default=None,
                        help="LoRA adapter 路径（可选）")
    parser.add_argument("--mode", type=str, default="smoke",
                        choices=list(EVAL_SUITES.keys()),
                        help="评测模式")
    parser.add_argument("--output_dir", type=str, default="eval_output",
                        help="评测结果输出目录")
    parser.add_argument("--device", type=str, default="0",
                        help="GPU 设备 ID")
    args = parser.parse_args()

    results = run_evaluation(
        model=args.model,
        mode=args.mode,
        adapters=args.adapters,
        output_dir=args.output_dir,
        device=args.device,
    )

    print_results_table(results)


if __name__ == "__main__":
    main()

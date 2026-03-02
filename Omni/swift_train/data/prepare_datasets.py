"""
数据下载与预处理 —— 各阶段训练数据的下载、验证和基础预处理。

用法：
    # 下载全部阶段数据
    python prepare_datasets.py --stages all --data_dir ./datasets

    # 仅下载 Stage 1 数据
    python prepare_datasets.py --stages 1 --data_dir ./datasets

    # Smoke test（仅下载少量样本）
    python prepare_datasets.py --stages all --data_dir ./datasets --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── 数据集注册表 ─────────────────────────────────────────

DATASET_REGISTRY: Dict[int, List[Dict]] = {
    1: [
        {
            "name": "LLaVA-CC3M-Pretrain-595K",
            "hf_id": "liuhaotian/LLaVA-CC3M-Pretrain-595K",
            "type": "image_text",
            "split": "train",
            "description": "595K 图像-文本对齐预训练数据",
        },
    ],
    2: [
        {
            "name": "AISHELL-1",
            "swift_id": "speech_asr/speech_asr_aishell1_trainsets",
            "type": "audio_text",
            "split": "train",
            "description": "178h 中文语音识别数据",
        },
        {
            "name": "LibriSpeech-Clean",
            "hf_id": "openslr/librispeech_asr",
            "subset": "clean",
            "type": "audio_text",
            "split": "train.clean.100",
            "description": "100h 英文语音识别数据",
        },
    ],
    3: [
        {
            "name": "LLaVA-Instruct-150K",
            "hf_id": "liuhaotian/LLaVA-Instruct-150k",
            "type": "image_text_sft",
            "split": "train",
            "description": "150K 图像指令 SFT",
        },
        {
            "name": "Alpaca-GPT4-ZH",
            "swift_id": "AI-ModelScope/alpaca-gpt4-data-zh",
            "type": "text_sft",
            "split": "train",
            "description": "中文文本指令 SFT",
        },
    ],
    4: [
        {
            "name": "LibriTTS",
            "hf_id": "cdminix/libritts-r-aligned",
            "type": "tts",
            "split": "train",
            "description": "英文语音合成训练数据",
        },
    ],
    5: [
        {
            "name": "UltraFeedback",
            "hf_id": "openbmb/UltraFeedback",
            "type": "preference",
            "split": "train",
            "description": "64K 文本偏好对",
        },
    ],
}


def download_hf_dataset(
    hf_id: str,
    output_dir: Path,
    split: str = "train",
    subset: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Path:
    """从 HuggingFace 下载数据集并保存为 JSONL。"""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("请安装 datasets: pip install datasets")
        raise

    logger.info(f"下载数据集: {hf_id} (split={split}, subset={subset})")

    load_kwargs = {"split": split, "trust_remote_code": True}
    if subset:
        load_kwargs["name"] = subset

    try:
        ds = load_dataset(hf_id, **load_kwargs)
    except Exception as e:
        logger.warning(f"下载失败 {hf_id}: {e}")
        logger.info("创建占位数据文件...")
        return _create_placeholder(output_dir, hf_id, max_samples or 100)

    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))

    # 保存为 JSONL
    safe_name = hf_id.replace("/", "_")
    output_path = output_dir / f"{safe_name}.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for item in ds:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"已保存 {len(ds)} 条数据到 {output_path}")
    return output_path


def _create_placeholder(output_dir: Path, name: str, n: int = 100) -> Path:
    """创建占位数据（用于无法下载时的 smoke test）。"""
    safe_name = name.replace("/", "_")
    output_path = output_dir / f"{safe_name}_placeholder.jsonl"

    samples = []
    for i in range(n):
        samples.append({
            "messages": [
                {"role": "user", "content": f"测试问题 {i}"},
                {"role": "assistant", "content": f"测试回答 {i}"},
            ]
        })

    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    logger.info(f"已创建占位数据 {output_path} ({n} 条)")
    return output_path


def prepare_stage(
    stage: int,
    data_dir: Path,
    smoke: bool = False,
    max_samples: Optional[int] = None,
) -> List[Path]:
    """准备指定阶段的全部数据集。"""
    if stage not in DATASET_REGISTRY:
        logger.warning(f"阶段 {stage} 无注册数据集")
        return []

    stage_dir = data_dir / f"stage{stage}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    effective_max = max_samples or (200 if smoke else None)

    downloaded = []
    for ds_info in DATASET_REGISTRY[stage]:
        hf_id = ds_info.get("hf_id") or ds_info.get("swift_id", "unknown")
        try:
            path = download_hf_dataset(
                hf_id=hf_id,
                output_dir=stage_dir,
                split=ds_info.get("split", "train"),
                subset=ds_info.get("subset"),
                max_samples=effective_max,
            )
            downloaded.append(path)
        except Exception as e:
            logger.error(f"处理 {ds_info['name']} 失败: {e}")
            # 创建占位数据以确保流程不中断
            path = _create_placeholder(stage_dir, ds_info["name"], effective_max or 100)
            downloaded.append(path)

    return downloaded


def prepare_all(data_dir: Path, smoke: bool = False) -> Dict[int, List[Path]]:
    """准备全部阶段数据。"""
    results = {}
    for stage in sorted(DATASET_REGISTRY.keys()):
        logger.info(f"\n{'='*60}")
        logger.info(f"准备 Stage {stage} 数据...")
        logger.info(f"{'='*60}")
        results[stage] = prepare_stage(stage, data_dir, smoke=smoke)
    return results


def main():
    parser = argparse.ArgumentParser(description="Omni 训练数据准备")
    parser.add_argument("--stages", type=str, default="all",
                        help="要准备的阶段，用逗号分隔（如 1,2,3）或 'all'")
    parser.add_argument("--data_dir", type=str, default="./datasets",
                        help="数据保存目录")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test 模式：每个数据集仅下载少量样本")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="每个数据集的最大样本数")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.stages == "all":
        results = prepare_all(data_dir, smoke=args.smoke)
    else:
        stages = [int(s.strip()) for s in args.stages.split(",")]
        results = {}
        for stage in stages:
            results[stage] = prepare_stage(
                stage, data_dir,
                smoke=args.smoke,
                max_samples=args.max_samples,
            )

    # 打印摘要
    print("\n" + "=" * 60)
    print("数据准备完成 - 摘要")
    print("=" * 60)
    for stage, paths in sorted(results.items()):
        print(f"\nStage {stage}:")
        for p in paths:
            size = p.stat().st_size / (1024 * 1024) if p.exists() else 0
            print(f"  {p.name}: {size:.1f} MB")


if __name__ == "__main__":
    main()

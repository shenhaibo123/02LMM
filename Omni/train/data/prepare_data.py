"""
数据下载与预处理 —— 各阶段训练数据的下载、格式转换与验证。

五阶段数据规划:
    Stage 1 (音频-语言对齐): LibriSpeech-clean-100 + AISHELL-1
    Stage 2 (图像-文本预训练): LLaVA-CC3M-Pretrain-595K + ShareGPT4V-100K 子集
    Stage 3 (全模态联合 SFT): Alpaca-GPT4-ZH + LLaVA-Instruct-150K + AISHELL-1 子集
    Stage 4 (语音生成): LibriTTS-R (aligned subset)
    Stage 5 (DPO 偏好对齐): UltraFeedback (择优/择劣对)

用法:
    python prepare_data.py --stages all --data_dir ./datasets
    python prepare_data.py --stages 1,2 --data_dir ./datasets --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── 数据集注册表 ──────────────────────────────────────────
# 每个数据集定义: name, hf_id, type, split, description, smoke_samples, full_samples
DATASET_REGISTRY: Dict[int, List[Dict[str, Any]]] = {
    1: [
        {
            "name": "LibriSpeech-clean-100",
            "hf_id": "openslr/librispeech_asr",
            "subset": "clean",
            "type": "audio_text",
            "split": "train.clean.100",
            "description": "100h 英文语音识别 (LibriSpeech train-clean-100)",
            "smoke_samples": 500,
        },
        {
            "name": "AISHELL-1",
            "hf_id": "speech_asr/speech_asr_aishell1_trainsets",
            "type": "audio_text",
            "split": "train",
            "description": "178h 中文语音识别 (AISHELL-1 训练集)",
            "smoke_samples": 500,
        },
    ],
    2: [
        {
            "name": "LLaVA-CC3M-Pretrain-595K",
            "hf_id": "liuhaotian/LLaVA-CC3M-Pretrain-595K",
            "type": "image_text",
            "split": "train",
            "description": "595K 图像-文本对齐预训练数据",
            "smoke_samples": 2000,
        },
    ],
    3: [
        {
            "name": "Alpaca-GPT4-ZH",
            "hf_id": "silk-road/alpaca-data-gpt4-chinese",
            "type": "text_sft",
            "split": "train",
            "description": "中文指令 SFT (GPT-4 生成回答)",
            "smoke_samples": 1000,
        },
        {
            "name": "LLaVA-Instruct-150K",
            "hf_id": "liuhaotian/LLaVA-Instruct-150k",
            "type": "image_text_sft",
            "split": "train",
            "description": "150K 图像指令微调数据",
            "smoke_samples": 1000,
        },
    ],
    4: [
        {
            "name": "LibriTTS-R",
            "hf_id": "cdminix/libritts-r-aligned",
            "type": "tts",
            "split": "train",
            "description": "英文语音合成训练 (LibriTTS-R aligned)",
            "smoke_samples": 500,
        },
    ],
    5: [
        {
            "name": "UltraFeedback",
            "hf_id": "openbmb/UltraFeedback",
            "type": "preference",
            "split": "train",
            "description": "64K 偏好对数据 (多模型多角度评分)",
            "smoke_samples": 200,
        },
    ],
}


# ── 数据下载 ─────────────────────────────────────────────

def download_hf_dataset(
    hf_id: str,
    output_dir: Path,
    split: str = "train",
    subset: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Path:
    """从 HuggingFace Hub 下载数据集并保存为 JSONL。"""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("请安装 datasets 库: pip install datasets")
        raise

    logger.info(f"下载数据集: {hf_id} (split={split}, subset={subset}, max={max_samples})")

    load_kwargs: Dict[str, Any] = {"split": split, "trust_remote_code": True}
    if subset:
        load_kwargs["name"] = subset

    try:
        ds = load_dataset(hf_id, **load_kwargs)
    except Exception as e:
        logger.warning(f"下载失败 {hf_id}: {e}")
        logger.info("创建占位数据文件...")
        return _create_placeholder(output_dir, hf_id, max_samples or 100)

    if max_samples and len(ds) > max_samples:
        indices = random.sample(range(len(ds)), max_samples)
        ds = ds.select(indices)

    safe_name = hf_id.replace("/", "_")
    output_path = output_dir / f"{safe_name}.jsonl"

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for item in ds:
            row = _serialize_row(item)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    logger.info(f"已保存 {count} 条数据到 {output_path}")
    return output_path


def _serialize_row(item: dict) -> dict:
    """将 HuggingFace 数据行序列化为可 JSON 存储的格式。

    音频数据中的 numpy array 需要特殊处理。
    """
    row = {}
    for k, v in item.items():
        if isinstance(v, dict) and "array" in v:
            # HuggingFace Audio feature: {"array": np.ndarray, "sampling_rate": int, "path": str}
            row[k] = {
                "path": v.get("path", ""),
                "sampling_rate": v.get("sampling_rate", 16000),
                # 不保存原始 array（太大），仅保存路径和采样率
            }
        elif hasattr(v, "tolist"):
            # numpy array → list (for JSON)
            row[k] = v.tolist()
        else:
            row[k] = v
    return row


def _create_placeholder(output_dir: Path, name: str, n: int = 100) -> Path:
    """创建占位数据（用于无法下载时的 smoke test）。"""
    safe_name = name.replace("/", "_")
    output_path = output_dir / f"{safe_name}_placeholder.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(n):
            sample = {
                "messages": [
                    {"role": "user", "content": f"测试问题 {i}: 请解释这个概念。"},
                    {"role": "assistant", "content": f"测试回答 {i}: 这是一个关于 AI 的概念解释。"},
                ],
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info(f"已创建占位数据 {output_path} ({n} 条)")
    return output_path


# ── 格式转换 ─────────────────────────────────────────────

SYSTEM_PROMPTS = [
    "你是一个有帮助的多模态助手，可以理解图像、音频和视频。",
    "You are a helpful multimodal assistant.",
    "你是一个全能的 AI 助手，擅长处理各种模态的信息。",
    "请仔细分析输入的内容并给出详细回答。",
]


def _random_system() -> Optional[Dict[str, str]]:
    """以 30% 概率添加随机系统提示（参考 Baichuan-Omni 多样化策略）。"""
    if random.random() < 0.3:
        return {"role": "system", "content": random.choice(SYSTEM_PROMPTS)}
    return None


def convert_to_swift_format(
    input_path: Path,
    output_path: Path,
    data_type: str,
    max_samples: Optional[int] = None,
) -> int:
    """将原始 JSONL 转换为 MS-Swift 多模态 JSONL 格式。

    MS-Swift 格式:
      文本: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
      图像: {"messages": [...], "images": ["path/to/img.jpg"]}
      音频: {"messages": [...], "audios": ["path/to/audio.wav"]}
      DPO:  {"messages": [...], "chosen": {...}, "rejected": {...}}
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    converter = _CONVERTERS.get(data_type)
    if converter is None:
        raise ValueError(f"未知数据类型: {data_type}，可选: {list(_CONVERTERS.keys())}")

    count = 0
    skipped = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if max_samples and count >= max_samples:
                break
            try:
                item = json.loads(line.strip())
            except json.JSONDecodeError:
                skipped += 1
                continue
            result = converter(item)
            if result is not None:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                count += 1
            else:
                skipped += 1

    logger.info(f"[{data_type}] 转换完成: {count} 条成功, {skipped} 条跳过 → {output_path}")
    return count


def _convert_audio_text(item: dict) -> Optional[dict]:
    """音频-文本对 → MS-Swift 格式。"""
    messages = []
    sys_msg = _random_system()
    if sys_msg:
        messages.append(sys_msg)

    text = item.get("text") or item.get("sentence") or item.get("transcription", "")
    if not text:
        return None

    audio_path = ""
    audio_info = item.get("audio", {})
    if isinstance(audio_info, dict):
        audio_path = audio_info.get("path", "")
    elif isinstance(audio_info, str):
        audio_path = audio_info

    messages.append({"role": "user", "content": "<audio>请将这段语音转写为文字。"})
    messages.append({"role": "assistant", "content": text})

    result: Dict[str, Any] = {"messages": messages}
    if audio_path:
        result["audios"] = [audio_path]
    return result


def _convert_image_text(item: dict) -> Optional[dict]:
    """图像-文本对 → MS-Swift 格式 (LLaVA 风格)。"""
    messages = []
    sys_msg = _random_system()
    if sys_msg:
        messages.append(sys_msg)

    image_path = item.get("image", "")
    convs = item.get("conversations", [])

    if convs:
        for conv in convs:
            role = "user" if conv.get("from") in ("human", "user") else "assistant"
            content = conv.get("value", "")
            if role == "user" and "<image>" not in content:
                has_image_tag = any(
                    "<image>" in m.get("content", "")
                    for m in messages if m.get("role") == "user"
                )
                if not has_image_tag:
                    content = f"<image>{content}"
            messages.append({"role": role, "content": content})
    else:
        text = item.get("text") or item.get("caption") or item.get("blip_caption", "")
        if not text:
            return None
        messages.append({"role": "user", "content": "<image>描述这张图片的内容。"})
        messages.append({"role": "assistant", "content": text})

    result: Dict[str, Any] = {"messages": messages}
    if image_path:
        result["images"] = [image_path]
    return result


def _convert_image_text_sft(item: dict) -> Optional[dict]:
    """图像指令 SFT → MS-Swift 格式 (同 image_text)。"""
    return _convert_image_text(item)


def _convert_text_sft(item: dict) -> Optional[dict]:
    """纯文本 SFT → MS-Swift 格式 (Alpaca 风格)。"""
    messages = []
    sys_msg = _random_system()
    if sys_msg:
        messages.append(sys_msg)

    if "conversations" in item:
        for conv in item["conversations"]:
            role = "user" if conv.get("from") in ("human", "user") else "assistant"
            messages.append({"role": role, "content": conv.get("value", "")})
    elif "instruction" in item:
        query = item["instruction"]
        if item.get("input"):
            query += f"\n{item['input']}"
        response = item.get("output", "")
        if not response:
            return None
        messages.append({"role": "user", "content": query})
        messages.append({"role": "assistant", "content": response})
    elif "messages" in item:
        return item
    else:
        return None

    return {"messages": messages}


def _convert_tts(item: dict) -> Optional[dict]:
    """TTS 数据 → MS-Swift 格式。"""
    text = item.get("text") or item.get("original_text") or item.get("normalized_text", "")
    if not text:
        return None

    audio_path = ""
    audio_info = item.get("audio", {})
    if isinstance(audio_info, dict):
        audio_path = audio_info.get("path", "")
    elif isinstance(audio_info, str):
        audio_path = audio_info

    messages = [
        {"role": "user", "content": f"请将以下文本转为语音：{text}"},
        {"role": "assistant", "content": "[speech_output]"},
    ]

    result: Dict[str, Any] = {"messages": messages}
    if audio_path:
        result["audios"] = [audio_path]
    return result


def _convert_preference(item: dict) -> Optional[dict]:
    """偏好对 → MS-Swift DPO 格式。"""
    prompt = item.get("instruction") or item.get("prompt") or item.get("query", "")
    chosen = item.get("chosen") or item.get("chosen_response", "")
    rejected = item.get("rejected") or item.get("rejected_response", "")

    # UltraFeedback 格式: completions 列表按 overall_score 排序
    if not chosen and "completions" in item:
        completions = item["completions"]
        if len(completions) >= 2:
            sorted_c = sorted(
                completions,
                key=lambda x: x.get("overall_score", 0),
                reverse=True,
            )
            chosen = sorted_c[0].get("response", "")
            rejected = sorted_c[-1].get("response", "")
            if not prompt:
                prompt = item.get("instruction", "")

    if not prompt or not chosen or not rejected:
        return None

    # chosen/rejected 可能是字符串或消息列表
    if isinstance(chosen, list):
        chosen = chosen[-1].get("content", "") if chosen else ""
    if isinstance(rejected, list):
        rejected = rejected[-1].get("content", "") if rejected else ""

    return {
        "messages": [{"role": "user", "content": prompt}],
        "chosen": {"role": "assistant", "content": chosen},
        "rejected": {"role": "assistant", "content": rejected},
    }


_CONVERTERS = {
    "audio_text": _convert_audio_text,
    "image_text": _convert_image_text,
    "image_text_sft": _convert_image_text_sft,
    "text_sft": _convert_text_sft,
    "tts": _convert_tts,
    "preference": _convert_preference,
}


# ── 混合数据集（Stage 3 用） ─────────────────────────────

def create_mixed_dataset(
    input_files: List[Path],
    output_file: Path,
    ratios: Optional[List[float]] = None,
    max_total: Optional[int] = None,
    shuffle: bool = True,
) -> int:
    """混合多个 JSONL 文件为一个数据集（用于 Stage 3 联合 SFT）。"""
    if ratios is None:
        ratios = [1.0 / len(input_files)] * len(input_files)

    if len(ratios) != len(input_files):
        raise ValueError(f"比例数 ({len(ratios)}) 与文件数 ({len(input_files)}) 不一致")

    all_samples = []
    for fpath, ratio in zip(input_files, ratios):
        if not fpath.exists():
            logger.warning(f"文件不存在，跳过: {fpath}")
            continue
        samples = []
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    samples.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        n_select = int(len(samples) * ratio) if max_total is None else int(max_total * ratio)
        n_select = min(n_select, len(samples))
        if n_select > 0:
            selected = random.sample(samples, n_select) if n_select < len(samples) else samples
            all_samples.extend(selected)
            logger.info(f"  {fpath.name}: {len(selected)}/{len(samples)} 条 (ratio={ratio:.2f})")

    if shuffle:
        random.shuffle(all_samples)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    logger.info(f"混合数据集: {len(all_samples)} 条 → {output_file}")
    return len(all_samples)


# ── 阶段级调度 ───────────────────────────────────────────

def prepare_stage(
    stage: int,
    data_dir: Path,
    smoke: bool = False,
    max_samples: Optional[int] = None,
) -> List[Path]:
    """准备指定阶段的全部数据：下载 → 转换 → 保存。"""
    if stage not in DATASET_REGISTRY:
        logger.warning(f"阶段 {stage} 无注册数据集")
        return []

    stage_dir = data_dir / f"stage{stage}"
    swift_dir = stage_dir / "swift_format"
    stage_dir.mkdir(parents=True, exist_ok=True)
    swift_dir.mkdir(parents=True, exist_ok=True)

    result_paths = []
    for ds_info in DATASET_REGISTRY[stage]:
        hf_id = ds_info.get("hf_id", "unknown")
        effective_max = max_samples or (ds_info.get("smoke_samples", 200) if smoke else None)

        # 步骤 1: 下载
        try:
            raw_path = download_hf_dataset(
                hf_id=hf_id,
                output_dir=stage_dir,
                split=ds_info.get("split", "train"),
                subset=ds_info.get("subset"),
                max_samples=effective_max,
            )
        except Exception as e:
            logger.error(f"下载 {ds_info['name']} 失败: {e}")
            raw_path = _create_placeholder(stage_dir, ds_info["name"], effective_max or 100)

        # 步骤 2: 转换为 MS-Swift 格式
        swift_path = swift_dir / f"{raw_path.stem}_swift.jsonl"
        try:
            convert_to_swift_format(
                input_path=raw_path,
                output_path=swift_path,
                data_type=ds_info["type"],
                max_samples=effective_max,
            )
            result_paths.append(swift_path)
        except Exception as e:
            logger.error(f"转换 {ds_info['name']} 失败: {e}")
            result_paths.append(raw_path)

    return result_paths


def prepare_all(
    data_dir: Path,
    smoke: bool = False,
    max_samples: Optional[int] = None,
) -> Dict[int, List[Path]]:
    """准备全部阶段数据。"""
    results = {}
    for stage in sorted(DATASET_REGISTRY.keys()):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"准备 Stage {stage} 数据...")
        logger.info(f"{'=' * 60}")
        results[stage] = prepare_stage(stage, data_dir, smoke=smoke, max_samples=max_samples)
    return results


# ── CLI 入口 ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Omni 训练数据准备")
    parser.add_argument(
        "--stages", type=str, default="all",
        help="要准备的阶段，逗号分隔（如 1,2,3）或 'all' (默认: all)",
    )
    parser.add_argument(
        "--data_dir", type=str, default="./datasets",
        help="数据保存目录 (默认: ./datasets)",
    )
    parser.add_argument("--smoke", action="store_true", help="Smoke test 模式")
    parser.add_argument("--max_samples", type=int, default=None, help="每个数据集最大样本数")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.stages == "all":
        results = prepare_all(data_dir, smoke=args.smoke, max_samples=args.max_samples)
    else:
        stages = [int(s.strip()) for s in args.stages.split(",")]
        results = {}
        for stage in stages:
            results[stage] = prepare_stage(
                stage, data_dir, smoke=args.smoke, max_samples=args.max_samples,
            )

    # 摘要
    print(f"\n{'=' * 60}")
    print("数据准备完成 - 摘要")
    print(f"{'=' * 60}")
    for stage, paths in sorted(results.items()):
        print(f"\nStage {stage}:")
        for p in paths:
            if p.exists():
                size_mb = p.stat().st_size / (1024 * 1024)
                lines = sum(1 for _ in open(p, "r", encoding="utf-8"))
                print(f"  {p.name}: {lines} 条, {size_mb:.1f} MB")
            else:
                print(f"  {p.name}: (文件不存在)")


if __name__ == "__main__":
    main()

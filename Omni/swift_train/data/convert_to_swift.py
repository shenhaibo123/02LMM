"""
数据格式转换 —— 将各类原始数据转换为 MS-Swift 的 JSONL 格式。

MS-Swift 支持的多模态 JSONL 格式：
- 文本:  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
- 图像:  {"messages": [...], "images": ["path/to/img.jpg"]}
- 音频:  {"messages": [...], "audios": ["path/to/audio.wav"]}
- 视频:  {"messages": [...], "videos": ["path/to/video.mp4"]}
- DPO:   {"messages": [{"role": "user", "content": "..."}],
          "chosen": {"role": "assistant", "content": "..."},
          "rejected": {"role": "assistant", "content": "..."}}

用法：
    python convert_to_swift.py \
        --input_file raw_data.jsonl \
        --output_file swift_data.jsonl \
        --data_type image_text \
        --image_dir /path/to/images
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── 随机系统提示词（参考 Baichuan-Omni 多样化策略）───────
SYSTEM_PROMPTS = [
    "你是一个有帮助的多模态助手，可以理解图像、音频和视频。",
    "You are a helpful multimodal assistant.",
    "你是一个全能的 AI 助手，擅长处理各种模态的信息。",
    "请仔细分析输入的内容并给出详细回答。",
]


def _random_system() -> Optional[Dict[str, str]]:
    """以 30% 概率添加随机系统提示。"""
    if random.random() < 0.3:
        return {"role": "system", "content": random.choice(SYSTEM_PROMPTS)}
    return None


# ── 转换器：各类数据 → MS-Swift JSONL ──────────────────

def convert_image_text(item: Dict, image_dir: Optional[str] = None) -> Optional[Dict]:
    """图像-文本对 → MS-Swift 格式。

    输入格式（LLaVA 风格）:
        {"id": "...", "image": "xxx.jpg", "conversations": [{"from": "human", "value": "..."},
         {"from": "gpt", "value": "..."}]}
    """
    messages = []
    sys_msg = _random_system()
    if sys_msg:
        messages.append(sys_msg)

    image_path = item.get("image", "")
    if image_dir and image_path:
        image_path = str(Path(image_dir) / image_path)

    convs = item.get("conversations", [])
    if not convs:
        # 简单的 caption 格式
        text = item.get("text") or item.get("caption", "")
        if not text:
            return None
        messages.append({"role": "user", "content": "<image>描述这张图片的内容。"})
        messages.append({"role": "assistant", "content": text})
    else:
        for conv in convs:
            role = "user" if conv.get("from") in ("human", "user") else "assistant"
            content = conv.get("value", "")
            # 第一个 user 消息中加入 <image> 标签
            if role == "user" and "<image>" not in content and not any(
                m.get("content", "").startswith("<image>") for m in messages if m["role"] == "user"
            ):
                content = f"<image>{content}"
            messages.append({"role": role, "content": content})

    result = {"messages": messages}
    if image_path:
        result["images"] = [image_path]
    return result


def convert_audio_text(item: Dict, audio_dir: Optional[str] = None) -> Optional[Dict]:
    """音频-文本对 → MS-Swift 格式。

    输入格式:
        {"audio": "xxx.wav", "text": "转写文本", "task": "asr"}
    或:
        {"audio_filepath": "...", "text": "...", "duration": 5.2}
    """
    messages = []
    sys_msg = _random_system()
    if sys_msg:
        messages.append(sys_msg)

    audio_path = item.get("audio") or item.get("audio_filepath", "")
    if audio_dir and audio_path:
        audio_path = str(Path(audio_dir) / audio_path)

    text = item.get("text") or item.get("sentence") or item.get("transcription", "")
    if not text:
        return None

    task = item.get("task", "asr")
    if task == "asr":
        messages.append({"role": "user", "content": "<audio>请将这段语音转写为文字。"})
    elif task == "caption":
        messages.append({"role": "user", "content": "<audio>描述这段音频的内容。"})
    else:
        messages.append({"role": "user", "content": f"<audio>{task}"})

    messages.append({"role": "assistant", "content": text})

    result = {"messages": messages}
    if audio_path:
        result["audios"] = [audio_path]
    return result


def convert_video_text(item: Dict, video_dir: Optional[str] = None) -> Optional[Dict]:
    """视频-文本对 → MS-Swift 格式。"""
    messages = []
    sys_msg = _random_system()
    if sys_msg:
        messages.append(sys_msg)

    video_path = item.get("video") or item.get("video_path", "")
    if video_dir and video_path:
        video_path = str(Path(video_dir) / video_path)

    question = item.get("question") or item.get("query", "描述这段视频的内容。")
    answer = item.get("answer") or item.get("response", "")
    if not answer:
        return None

    messages.append({"role": "user", "content": f"<video>{question}"})
    messages.append({"role": "assistant", "content": answer})

    result = {"messages": messages}
    if video_path:
        result["videos"] = [video_path]
    return result


def convert_text_sft(item: Dict) -> Optional[Dict]:
    """纯文本 SFT → MS-Swift 格式。

    输入格式（Alpaca 风格）:
        {"instruction": "...", "input": "...", "output": "..."}
    或对话格式:
        {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
    """
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
        return item  # 已经是目标格式
    else:
        return None

    return {"messages": messages}


def convert_dpo(item: Dict) -> Optional[Dict]:
    """DPO 偏好对 → MS-Swift 格式。

    输入格式:
        {"prompt": "...", "chosen": "...", "rejected": "..."}
    或:
        {"instruction": "...", "chosen_response": "...", "rejected_response": "..."}
    """
    prompt = item.get("prompt") or item.get("instruction") or item.get("query", "")
    chosen = item.get("chosen") or item.get("chosen_response", "")
    rejected = item.get("rejected") or item.get("rejected_response", "")

    if not prompt or not chosen or not rejected:
        # 尝试 UltraFeedback 格式
        if "completions" in item:
            completions = item["completions"]
            if len(completions) >= 2:
                sorted_c = sorted(completions, key=lambda x: x.get("overall_score", 0), reverse=True)
                chosen = sorted_c[0].get("response", "")
                rejected = sorted_c[-1].get("response", "")
                prompt = item.get("instruction", "")
            else:
                return None
        else:
            return None

    if isinstance(chosen, list):
        chosen = chosen[-1].get("content", "") if chosen else ""
    if isinstance(rejected, list):
        rejected = rejected[-1].get("content", "") if rejected else ""

    return {
        "messages": [{"role": "user", "content": prompt}],
        "chosen": {"role": "assistant", "content": chosen},
        "rejected": {"role": "assistant", "content": rejected},
    }


# ── 转换器注册表 ─────────────────────────────────────────

CONVERTERS: Dict[str, Callable] = {
    "image_text": convert_image_text,
    "audio_text": convert_audio_text,
    "video_text": convert_video_text,
    "text_sft": convert_text_sft,
    "dpo": convert_dpo,
}


def convert_file(
    input_file: Path,
    output_file: Path,
    data_type: str,
    media_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> int:
    """
    将原始 JSONL 文件转换为 MS-Swift 格式。

    Args:
        input_file:  输入 JSONL 路径
        output_file: 输出 JSONL 路径
        data_type:   数据类型（image_text/audio_text/video_text/text_sft/dpo）
        media_dir:   媒体文件根目录
        max_samples: 最大转换样本数

    Returns:
        成功转换的样本数
    """
    if data_type not in CONVERTERS:
        raise ValueError(f"未知数据类型: {data_type}，可选: {list(CONVERTERS.keys())}")

    converter = CONVERTERS[data_type]
    count = 0
    skipped = 0

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            if max_samples and count >= max_samples:
                break
            try:
                item = json.loads(line.strip())
            except json.JSONDecodeError:
                skipped += 1
                continue

            # 根据数据类型决定是否传入 media_dir
            if data_type in ("image_text",):
                result = converter(item, image_dir=media_dir)
            elif data_type in ("audio_text",):
                result = converter(item, audio_dir=media_dir)
            elif data_type in ("video_text",):
                result = converter(item, video_dir=media_dir)
            else:
                result = converter(item)

            if result is not None:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                count += 1
            else:
                skipped += 1

    logger.info(f"转换完成: {count} 条成功, {skipped} 条跳过 → {output_file}")
    return count


def create_mixed_dataset(
    input_files: List[Path],
    output_file: Path,
    ratios: Optional[List[float]] = None,
    max_total: Optional[int] = None,
    shuffle: bool = True,
) -> int:
    """
    混合多个数据文件（用于 Stage 3 联合 SFT）。

    Args:
        input_files: 输入 JSONL 文件列表
        output_file: 输出混合 JSONL 文件
        ratios:      各文件的采样比例（默认均匀）
        max_total:   最大总样本数
        shuffle:     是否打乱

    Returns:
        混合后的总样本数
    """
    if ratios is None:
        ratios = [1.0 / len(input_files)] * len(input_files)

    assert len(ratios) == len(input_files), "比例数量需与文件数量一致"

    all_samples = []
    for fpath, ratio in zip(input_files, ratios):
        if not fpath.exists():
            logger.warning(f"文件不存在: {fpath}")
            continue

        samples = []
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    samples.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

        # 按比例采样
        n_select = int(len(samples) * ratio) if max_total is None else int(max_total * ratio)
        n_select = min(n_select, len(samples))
        selected = random.sample(samples, n_select) if n_select < len(samples) else samples
        all_samples.extend(selected)
        logger.info(f"  {fpath.name}: {len(selected)}/{len(samples)} 条")

    if shuffle:
        random.shuffle(all_samples)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    logger.info(f"混合数据集: {len(all_samples)} 条 → {output_file}")
    return len(all_samples)


def main():
    parser = argparse.ArgumentParser(description="数据格式转换为 MS-Swift JSONL")
    parser.add_argument("--input_file", type=str, required=True, help="输入 JSONL 文件路径")
    parser.add_argument("--output_file", type=str, required=True, help="输出 JSONL 文件路径")
    parser.add_argument("--data_type", type=str, required=True,
                        choices=list(CONVERTERS.keys()),
                        help="数据类型")
    parser.add_argument("--media_dir", type=str, default=None,
                        help="媒体文件（图片/音频/视频）根目录")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大转换样本数")
    args = parser.parse_args()

    convert_file(
        input_file=Path(args.input_file),
        output_file=Path(args.output_file),
        data_type=args.data_type,
        media_dir=args.media_dir,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()

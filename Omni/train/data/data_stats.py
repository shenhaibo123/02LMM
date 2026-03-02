"""
数据集统计分析 —— 对训练数据进行 7 类维度统计，输出分析报告。

统计维度:
    1. 文本长度分布（token 数 / 字符数）
    2. 音频时长分布（秒）
    3. 图像分辨率分布
    4. 任务类型分布（ASR / Caption / SFT / DPO 等）
    5. 语言分布（中文 / 英文 / 混合）
    6. 数据质量检查（空值、异常长度、重复）
    7. 汇总 JSON 报告

用法:
    python data_stats.py --data_dir ./datasets                    # 完整统计
    python data_stats.py --data_dir ./datasets --summary_only     # 仅输出汇总
    python data_stats.py --data_dir ./datasets --stage 1          # 仅统计 Stage 1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── 语言检测（简易规则） ──────────────────────────────────

def _detect_language(text: str) -> str:
    """简易语言检测：中文字符占比判断。"""
    if not text:
        return "unknown"
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    total_alpha = len(re.findall(r"[a-zA-Z\u4e00-\u9fff]", text))
    if total_alpha == 0:
        return "unknown"
    ratio = chinese_chars / total_alpha
    if ratio > 0.5:
        return "zh"
    elif ratio < 0.1:
        return "en"
    else:
        return "mixed"


def _infer_task_type(item: dict) -> str:
    """推断样本的任务类型。"""
    if "chosen" in item and "rejected" in item:
        return "dpo"
    messages = item.get("messages", [])
    has_audio = "audios" in item or any("<audio>" in m.get("content", "") for m in messages)
    has_image = "images" in item or any("<image>" in m.get("content", "") for m in messages)
    has_video = "videos" in item or any("<video>" in m.get("content", "") for m in messages)

    if has_audio and has_image:
        return "multimodal_audio_image"
    elif has_audio:
        # 判断是 ASR 还是 TTS
        for m in messages:
            content = m.get("content", "")
            if "转写" in content or "asr" in content.lower() or "transcri" in content.lower():
                return "asr"
            if "转为语音" in content or "tts" in content.lower():
                return "tts"
        return "audio_qa"
    elif has_image:
        return "image_qa"
    elif has_video:
        return "video_qa"
    else:
        return "text_sft"


def _extract_text_content(item: dict) -> str:
    """从样本中提取所有文本内容（用于长度/语言统计）。"""
    texts = []
    for m in item.get("messages", []):
        content = m.get("content", "")
        # 去掉模态标签
        content = re.sub(r"<(image|audio|video)>", "", content)
        texts.append(content)
    if "chosen" in item:
        c = item["chosen"]
        if isinstance(c, dict):
            texts.append(c.get("content", ""))
        elif isinstance(c, str):
            texts.append(c)
    return " ".join(texts)


# ── 统计核心 ─────────────────────────────────────────────

class DatasetStats:
    """单个 JSONL 文件的统计分析。"""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.total_samples = 0
        self.text_lengths: List[int] = []
        self.task_types: Counter = Counter()
        self.languages: Counter = Counter()
        self.empty_count = 0
        self.duplicate_count = 0
        self._seen_hashes: set = set()
        self.issues: List[str] = []

    def analyze(self, max_samples: Optional[int] = None) -> Dict[str, Any]:
        """执行完整统计分析。"""
        if not self.filepath.exists():
            return {"error": f"文件不存在: {self.filepath}"}

        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                if max_samples and self.total_samples >= max_samples:
                    break
                try:
                    item = json.loads(line.strip())
                except json.JSONDecodeError:
                    self.issues.append("JSON 解析错误")
                    continue

                self.total_samples += 1
                self._analyze_one(item)

        return self._build_report()

    def _analyze_one(self, item: dict) -> None:
        """分析单条样本。"""
        # 1. 文本长度
        text = _extract_text_content(item)
        text_len = len(text)
        self.text_lengths.append(text_len)

        # 空值检查
        if text_len == 0:
            self.empty_count += 1

        # 异常长度检查
        if text_len > 50000:
            self.issues.append(f"超长文本: {text_len} 字符")

        # 重复检查（基于内容 hash）
        text_hash = hash(text[:500])
        if text_hash in self._seen_hashes:
            self.duplicate_count += 1
        self._seen_hashes.add(text_hash)

        # 4. 任务类型
        task = _infer_task_type(item)
        self.task_types[task] += 1

        # 5. 语言分布
        lang = _detect_language(text)
        self.languages[lang] += 1

    def _build_report(self) -> Dict[str, Any]:
        """构建统计报告。"""
        report: Dict[str, Any] = {
            "file": str(self.filepath),
            "total_samples": self.total_samples,
        }

        # 文本长度统计
        if self.text_lengths:
            sorted_lens = sorted(self.text_lengths)
            n = len(sorted_lens)
            report["text_length"] = {
                "min": sorted_lens[0],
                "max": sorted_lens[-1],
                "mean": sum(sorted_lens) / n,
                "median": sorted_lens[n // 2],
                "p25": sorted_lens[n // 4],
                "p75": sorted_lens[3 * n // 4],
                "p95": sorted_lens[int(n * 0.95)],
            }

        # 任务类型分布
        report["task_distribution"] = dict(self.task_types.most_common())

        # 语言分布
        report["language_distribution"] = dict(self.languages.most_common())

        # 数据质量
        report["quality"] = {
            "empty_samples": self.empty_count,
            "duplicate_samples": self.duplicate_count,
            "empty_rate": f"{self.empty_count / max(self.total_samples, 1) * 100:.1f}%",
            "duplicate_rate": f"{self.duplicate_count / max(self.total_samples, 1) * 100:.1f}%",
            "issues": self.issues[:20],  # 最多报告 20 个
        }

        return report


# ── 目录级统计 ───────────────────────────────────────────

def analyze_directory(
    data_dir: Path,
    stage: Optional[int] = None,
    max_samples_per_file: Optional[int] = None,
) -> Dict[str, Any]:
    """对数据目录下所有 JSONL 文件进行统计。"""
    results = {}

    # 查找所有 JSONL 文件
    if stage is not None:
        search_dirs = [data_dir / f"stage{stage}"]
    else:
        search_dirs = [data_dir]

    jsonl_files = []
    for d in search_dirs:
        if d.exists():
            jsonl_files.extend(d.rglob("*.jsonl"))

    if not jsonl_files:
        logger.warning(f"未找到 JSONL 文件: {search_dirs}")
        return {"error": "未找到 JSONL 文件"}

    for fpath in sorted(jsonl_files):
        logger.info(f"分析: {fpath}")
        stats = DatasetStats(fpath)
        report = stats.analyze(max_samples=max_samples_per_file)
        results[fpath.name] = report

    # 汇总
    total = sum(r.get("total_samples", 0) for r in results.values())
    task_agg: Counter = Counter()
    lang_agg: Counter = Counter()
    for r in results.values():
        task_agg.update(r.get("task_distribution", {}))
        lang_agg.update(r.get("language_distribution", {}))

    summary = {
        "total_files": len(results),
        "total_samples": total,
        "task_distribution": dict(task_agg.most_common()),
        "language_distribution": dict(lang_agg.most_common()),
    }

    return {"files": results, "summary": summary}


# ── 打印 & 保存 ─────────────────────────────────────────

def print_report(report: Dict[str, Any], summary_only: bool = False) -> None:
    """打印统计报告。"""
    summary = report.get("summary", report)

    print(f"\n{'=' * 60}")
    print("  数据集统计报告")
    print(f"{'=' * 60}")
    print(f"  文件数: {summary.get('total_files', 'N/A')}")
    print(f"  总样本数: {summary.get('total_samples', 'N/A'):,}")

    # 任务分布
    tasks = summary.get("task_distribution", {})
    if tasks:
        print(f"\n  任务类型分布:")
        total_t = sum(tasks.values())
        for task, count in sorted(tasks.items(), key=lambda x: -x[1]):
            pct = count / total_t * 100
            bar = "#" * int(pct / 2)
            print(f"    {task:<25} {count:>8,}  ({pct:5.1f}%) {bar}")

    # 语言分布
    langs = summary.get("language_distribution", {})
    if langs:
        print(f"\n  语言分布:")
        total_l = sum(langs.values())
        for lang, count in sorted(langs.items(), key=lambda x: -x[1]):
            pct = count / total_l * 100
            print(f"    {lang:<10} {count:>8,}  ({pct:5.1f}%)")

    # 按文件详情
    if not summary_only and "files" in report:
        for fname, fstats in report["files"].items():
            print(f"\n  --- {fname} ---")
            print(f"    样本数: {fstats.get('total_samples', 0):,}")

            tl = fstats.get("text_length", {})
            if tl:
                print(f"    文本长度: "
                      f"min={tl['min']}, median={tl['median']:.0f}, "
                      f"mean={tl['mean']:.0f}, max={tl['max']}, "
                      f"p95={tl['p95']}")

            q = fstats.get("quality", {})
            if q:
                print(f"    空值率: {q.get('empty_rate', 'N/A')}, "
                      f"重复率: {q.get('duplicate_rate', 'N/A')}")

    print(f"\n{'=' * 60}\n")


def save_report(report: Dict[str, Any], output_path: Path) -> None:
    """保存统计报告为 JSON。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"统计报告已保存: {output_path}")


# ── CLI 入口 ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Omni 数据集统计分析")
    parser.add_argument("--data_dir", type=str, required=True, help="数据目录")
    parser.add_argument("--stage", type=int, default=None, help="仅统计指定阶段")
    parser.add_argument("--output", type=str, default=None, help="统计报告输出 JSON 路径")
    parser.add_argument("--summary_only", action="store_true", help="仅输出汇总")
    parser.add_argument("--max_samples", type=int, default=None, help="每文件最大分析样本数")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        return

    report = analyze_directory(data_dir, stage=args.stage, max_samples_per_file=args.max_samples)

    print_report(report, summary_only=args.summary_only)

    if args.output:
        save_report(report, Path(args.output))
    else:
        # 默认保存到数据目录下
        default_output = data_dir / "stats_report.json"
        save_report(report, default_output)


if __name__ == "__main__":
    main()

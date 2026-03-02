"""
PyTorch Dataset 类 —— 支持 5 阶段不同数据格式的统一加载。

每个 Dataset 从 MS-Swift 格式 JSONL 加载，适配 OmniModel.forward() 的输入接口:
    - input_ids, attention_mask, labels  (文本)
    - images                             (视觉)
    - audio_features                     (音频)
    - speech_target_tokens               (语音目标，Stage 4)
    - chosen/rejected pairs              (DPO，Stage 5)

用法:
    from data.dataset import build_dataloader

    loader = build_dataloader(
        jsonl_path="./datasets/stage1/swift_format/data_swift.jsonl",
        stage=1,
        tokenizer=tokenizer,
        batch_size=8,
        max_seq_len=2048,
    )
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class OmniDataset(Dataset):
    """统一的多模态数据集，从 MS-Swift 格式 JSONL 加载。

    支持的数据格式:
        - 纯文本 SFT:  {"messages": [...]}
        - 图像+文本:    {"messages": [...], "images": [...]}
        - 音频+文本:    {"messages": [...], "audios": [...]}
        - TTS:          {"messages": [...], "audios": [...]}  (Stage 4)
        - DPO:          {"messages": [...], "chosen": {...}, "rejected": {...}}
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: Any,
        stage: int = 3,
        max_seq_len: int = 2048,
        max_samples: Optional[int] = None,
    ):
        self.jsonl_path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.stage = stage
        self.max_seq_len = max_seq_len
        self.samples: List[Dict] = []

        self._load(max_samples)
        logger.info(
            f"OmniDataset(stage={stage}): {len(self.samples)} 条样本, "
            f"max_seq_len={max_seq_len}, 来源={self.jsonl_path.name}"
        )

    def _load(self, max_samples: Optional[int]) -> None:
        """从 JSONL 加载数据。"""
        if not self.jsonl_path.exists():
            logger.warning(f"数据文件不存在: {self.jsonl_path}")
            return

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    self.samples.append(item)
                except json.JSONDecodeError:
                    continue
                if max_samples and len(self.samples) >= max_samples:
                    break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]

        if self.stage == 5 and "chosen" in item:
            return self._process_dpo(item)
        else:
            return self._process_sft(item)

    def _process_sft(self, item: dict) -> Dict[str, Any]:
        """处理 SFT/对齐数据（Stage 1-4）。"""
        messages = item.get("messages", [])

        # 拼接对话为单一文本
        text_parts = []
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "system":
                text_parts.append(f"<|system|>{content}")
            elif role == "user":
                text_parts.append(f"<|user|>{content}")
            elif role == "assistant":
                text_parts.append(f"<|assistant|>{content}")
        full_text = "\n".join(text_parts)

        # Tokenize
        encoded = self.tokenizer(
            full_text,
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Labels: 与 input_ids 相同，但 padding 位置设为 -100
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        result: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # 多模态标记（仅记录路径，实际编码由 collator 或 trainer 处理）
        if "images" in item:
            result["image_paths"] = item["images"]
        if "audios" in item:
            result["audio_paths"] = item["audios"]

        return result

    def _process_dpo(self, item: dict) -> Dict[str, Any]:
        """处理 DPO 偏好对数据（Stage 5）。"""
        messages = item.get("messages", [])
        chosen = item.get("chosen", {})
        rejected = item.get("rejected", {})

        # 构建 prompt
        prompt_parts = []
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "user":
                prompt_parts.append(f"<|user|>{content}")
            elif role == "system":
                prompt_parts.append(f"<|system|>{content}")
        prompt_text = "\n".join(prompt_parts)

        chosen_text = chosen.get("content", "") if isinstance(chosen, dict) else str(chosen)
        rejected_text = rejected.get("content", "") if isinstance(rejected, dict) else str(rejected)

        chosen_full = f"{prompt_text}\n<|assistant|>{chosen_text}"
        rejected_full = f"{prompt_text}\n<|assistant|>{rejected_text}"

        chosen_enc = self.tokenizer(
            chosen_full, max_length=self.max_seq_len,
            truncation=True, padding="max_length", return_tensors="pt",
        )
        rejected_enc = self.tokenizer(
            rejected_full, max_length=self.max_seq_len,
            truncation=True, padding="max_length", return_tensors="pt",
        )

        return {
            "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
        }


class MultiSourceDataset(Dataset):
    """混合多个 JSONL 数据源的 Dataset（用于 Stage 3 联合训练）。

    按权重从各源采样，实现跨模态均衡训练。
    """

    def __init__(
        self,
        jsonl_paths: List[str],
        tokenizer: Any,
        stage: int = 3,
        max_seq_len: int = 2048,
        weights: Optional[List[float]] = None,
        max_samples: Optional[int] = None,
    ):
        self.datasets = [
            OmniDataset(p, tokenizer, stage, max_seq_len, max_samples)
            for p in jsonl_paths
        ]
        self.weights = weights or [1.0 / len(jsonl_paths)] * len(jsonl_paths)

        # 按权重构建索引映射
        self._indices: List[tuple] = []  # (dataset_idx, sample_idx)
        total_per_source = [len(ds) for ds in self.datasets]
        total = sum(total_per_source)
        if max_samples:
            total = min(total, max_samples)

        for ds_idx, (ds, w) in enumerate(zip(self.datasets, self.weights)):
            n = min(int(total * w), len(ds))
            indices = random.sample(range(len(ds)), n) if n < len(ds) else list(range(len(ds)))
            self._indices.extend((ds_idx, i) for i in indices)

        random.shuffle(self._indices)
        logger.info(
            f"MultiSourceDataset: {len(self._indices)} 条样本 "
            f"from {len(self.datasets)} 个数据源"
        )

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ds_idx, sample_idx = self._indices[idx]
        return self.datasets[ds_idx][sample_idx]


def build_dataloader(
    jsonl_path: str,
    stage: int,
    tokenizer: Any,
    batch_size: int = 8,
    max_seq_len: int = 2048,
    max_samples: Optional[int] = None,
    num_workers: int = 2,
    shuffle: bool = True,
) -> DataLoader:
    """构建 DataLoader 的便捷函数。"""
    dataset = OmniDataset(
        jsonl_path=jsonl_path,
        tokenizer=tokenizer,
        stage=stage,
        max_seq_len=max_seq_len,
        max_samples=max_samples,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


# ── 模块自测 ────────────────────────────────────────────
if __name__ == "__main__":
    """快速验证 Dataset 的基本功能（使用 mock tokenizer）。"""
    import tempfile
    import os

    print("=== OmniDataset 模块自测 ===\n")

    # 创建临时测试数据
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        # SFT 样本
        for i in range(10):
            sample = {
                "messages": [
                    {"role": "user", "content": f"测试问题 {i}"},
                    {"role": "assistant", "content": f"测试回答 {i}，这是第 {i} 条。"},
                ],
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        # DPO 样本
        for i in range(5):
            sample = {
                "messages": [{"role": "user", "content": f"DPO 问题 {i}"}],
                "chosen": {"role": "assistant", "content": f"好回答 {i}"},
                "rejected": {"role": "assistant", "content": f"差回答 {i}"},
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        tmp_path = f.name

    # Mock tokenizer
    class MockTokenizer:
        def __call__(self, text, max_length=128, truncation=True,
                     padding="max_length", return_tensors="pt"):
            # 简单模拟：按字符切分
            ids = [ord(c) % 1000 for c in text[:max_length]]
            ids = ids + [0] * (max_length - len(ids))
            mask = [1] * min(len(text), max_length) + [0] * (max_length - min(len(text), max_length))
            return {
                "input_ids": torch.tensor([ids]),
                "attention_mask": torch.tensor([mask]),
            }

    tokenizer = MockTokenizer()

    # 测试 SFT Dataset
    ds_sft = OmniDataset(tmp_path, tokenizer, stage=3, max_seq_len=64, max_samples=10)
    print(f"SFT 数据集: {len(ds_sft)} 条")
    item = ds_sft[0]
    print(f"  input_ids shape: {item['input_ids'].shape}")
    print(f"  labels shape: {item['labels'].shape}")

    # 测试 DPO Dataset
    ds_dpo = OmniDataset(tmp_path, tokenizer, stage=5, max_seq_len=64)
    print(f"\nDPO 数据集: {len(ds_dpo)} 条")
    # 查找一条 DPO 样本
    for i in range(len(ds_dpo)):
        sample = ds_dpo[i]
        if "chosen_input_ids" in sample:
            print(f"  chosen_input_ids shape: {sample['chosen_input_ids'].shape}")
            print(f"  rejected_input_ids shape: {sample['rejected_input_ids'].shape}")
            break

    # 测试 DataLoader
    loader = build_dataloader(
        tmp_path, stage=3, tokenizer=tokenizer,
        batch_size=4, max_seq_len=64, max_samples=8,
    )
    batch = next(iter(loader))
    print(f"\nDataLoader batch:")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  attention_mask: {batch['attention_mask'].shape}")
    print(f"  labels: {batch['labels'].shape}")

    # 清理
    os.unlink(tmp_path)

    print("\n=== 自测通过 ===")

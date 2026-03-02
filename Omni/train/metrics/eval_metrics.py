"""
评测指标计算 —— 提供 WER、CER、BLEU 等评测函数和统一评测入口。

支持的指标：
    compute_wer  — Word Error Rate（词错误率）
    compute_cer  — Character Error Rate（字符错误率）
    compute_bleu — BLEU score（机器翻译评测指标）
    evaluate_model — 统一评测入口，对数据集批量计算指定指标
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)


# ── 编辑距离（WER/CER 底层） ────────────────────────────

def _edit_distance(ref: Sequence, hyp: Sequence) -> Tuple[int, int, int, int]:
    """
    计算编辑距离，返回 (替换数, 删除数, 插入数, 参考长度)。

    使用动态规划算法，时间复杂度 O(n*m)。
    """
    n, m = len(ref), len(hyp)
    # dp[i][j] = (substitutions, deletions, insertions)
    dp = [[(0, 0, 0)] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = (0, i, 0)
    for j in range(1, m + 1):
        dp[0][j] = (0, 0, j)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # 替换
                sub = dp[i - 1][j - 1]
                sub_cost = sum(sub) + 1
                # 删除
                dele = dp[i - 1][j]
                del_cost = sum(dele) + 1
                # 插入
                ins = dp[i][j - 1]
                ins_cost = sum(ins) + 1

                min_cost = min(sub_cost, del_cost, ins_cost)
                if min_cost == sub_cost:
                    dp[i][j] = (sub[0] + 1, sub[1], sub[2])
                elif min_cost == del_cost:
                    dp[i][j] = (dele[0], dele[1] + 1, dele[2])
                else:
                    dp[i][j] = (ins[0], ins[1], ins[2] + 1)

    s, d, ins = dp[n][m]
    return s, d, ins, n


# ── WER ─────────────────────────────────────────────────

def compute_wer(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    计算 Word Error Rate（词错误率）。

    WER = (替换 + 删除 + 插入) / 参考词数

    Args:
        predictions: 模型预测文本列表
        references:  参考文本列表

    Returns:
        {"wer": float, "substitutions": int, "deletions": int,
         "insertions": int, "ref_words": int, "num_samples": int}
    """
    assert len(predictions) == len(references), "预测和参考文本数量不一致"

    total_sub, total_del, total_ins, total_ref = 0, 0, 0, 0

    for pred, ref in zip(predictions, references):
        pred_words = pred.strip().split()
        ref_words = ref.strip().split()
        s, d, i, n = _edit_distance(ref_words, pred_words)
        total_sub += s
        total_del += d
        total_ins += i
        total_ref += n

    wer = (total_sub + total_del + total_ins) / max(total_ref, 1)

    return {
        "wer": wer,
        "substitutions": total_sub,
        "deletions": total_del,
        "insertions": total_ins,
        "ref_words": total_ref,
        "num_samples": len(predictions),
    }


# ── CER ─────────────────────────────────────────────────

def compute_cer(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    计算 Character Error Rate（字符错误率）。

    CER = (替换 + 删除 + 插入) / 参考字符数

    Args:
        predictions: 模型预测文本列表
        references:  参考文本列表

    Returns:
        {"cer": float, "substitutions": int, "deletions": int,
         "insertions": int, "ref_chars": int, "num_samples": int}
    """
    assert len(predictions) == len(references), "预测和参考文本数量不一致"

    total_sub, total_del, total_ins, total_ref = 0, 0, 0, 0

    for pred, ref in zip(predictions, references):
        # 按字符切分（去除空格后逐字符比较）
        pred_chars = list(pred.strip().replace(" ", ""))
        ref_chars = list(ref.strip().replace(" ", ""))
        s, d, i, n = _edit_distance(ref_chars, pred_chars)
        total_sub += s
        total_del += d
        total_ins += i
        total_ref += n

    cer = (total_sub + total_del + total_ins) / max(total_ref, 1)

    return {
        "cer": cer,
        "substitutions": total_sub,
        "deletions": total_del,
        "insertions": total_ins,
        "ref_chars": total_ref,
        "num_samples": len(predictions),
    }


# ── BLEU ────────────────────────────────────────────────

def _count_ngrams(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
    """统计 n-gram 频次。"""
    counts: Dict[Tuple[str, ...], int] = defaultdict(int)
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        counts[ngram] += 1
    return counts


def compute_bleu(
    predictions: List[str],
    references: List[str],
    max_n: int = 4,
    smooth: bool = True,
) -> Dict[str, float]:
    """
    计算 BLEU score。

    实现 BLEU-1 到 BLEU-N 以及几何平均（含 brevity penalty）。

    Args:
        predictions: 模型预测文本列表
        references:  参考文本列表
        max_n:       最大 n-gram 阶数（默认 4，即 BLEU-4）
        smooth:      是否使用 +1 平滑（避免零分）

    Returns:
        {"bleu": float, "bleu_1": float, ..., "bleu_N": float,
         "brevity_penalty": float, "num_samples": int}
    """
    import math

    assert len(predictions) == len(references), "预测和参考文本数量不一致"

    # 各阶 n-gram 的匹配数和候选数
    match_counts = [0] * max_n
    candidate_counts = [0] * max_n
    ref_length_total = 0
    pred_length_total = 0

    for pred, ref in zip(predictions, references):
        pred_tokens = pred.strip().split()
        ref_tokens = ref.strip().split()
        pred_length_total += len(pred_tokens)
        ref_length_total += len(ref_tokens)

        for n in range(1, max_n + 1):
            pred_ngrams = _count_ngrams(pred_tokens, n)
            ref_ngrams = _count_ngrams(ref_tokens, n)

            # 裁剪匹配：每个 n-gram 的匹配次数不超过参考中的出现次数
            for ngram, count in pred_ngrams.items():
                match_counts[n - 1] += min(count, ref_ngrams.get(ngram, 0))
            candidate_counts[n - 1] += max(len(pred_tokens) - n + 1, 0)

    # 各阶精度
    precisions = []
    result = {"num_samples": len(predictions)}
    for n in range(max_n):
        if smooth:
            p = (match_counts[n] + 1) / (candidate_counts[n] + 1)
        else:
            p = match_counts[n] / max(candidate_counts[n], 1)
        precisions.append(p)
        result[f"bleu_{n + 1}"] = p

    # Brevity Penalty
    if pred_length_total == 0:
        bp = 0.0
    elif pred_length_total >= ref_length_total:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_length_total / pred_length_total)
    result["brevity_penalty"] = bp

    # 几何平均 BLEU
    if all(p > 0 for p in precisions):
        log_avg = sum(math.log(p) for p in precisions) / max_n
        bleu = bp * math.exp(log_avg)
    else:
        bleu = 0.0
    result["bleu"] = bleu

    return result


# ── 统一评测入口 ────────────────────────────────────────

# 内置指标注册表
METRIC_REGISTRY: Dict[str, Callable] = {
    "wer": compute_wer,
    "cer": compute_cer,
    "bleu": compute_bleu,
}


def evaluate_model(
    model: Any,
    dataset: Any,
    metrics: List[str],
    generate_fn: Optional[Callable] = None,
    batch_size: int = 16,
    max_samples: Optional[int] = None,
    device: str = "cpu",
) -> Dict[str, Dict[str, float]]:
    """
    统一评测入口：对数据集批量推理并计算指定指标。

    Args:
        model:       模型对象（需支持 generate 或传入 generate_fn）
        dataset:     数据集，每个样本需包含 "input" 和 "reference" 字段
        metrics:     指标名列表，如 ["wer", "cer", "bleu"]
        generate_fn: 自定义生成函数 fn(model, inputs) -> predictions
                     若为 None，则调用 model.generate()
        batch_size:  推理批大小
        max_samples: 最大评测样本数（None 表示全部）
        device:      推理设备

    Returns:
        {metric_name: {metric_key: value}} 各指标的详细结果
    """
    # 验证指标名
    for m in metrics:
        if m not in METRIC_REGISTRY:
            raise ValueError(f"未知指标: {m}，可选: {list(METRIC_REGISTRY.keys())}")

    # 收集样本
    samples = list(dataset)
    if max_samples is not None:
        samples = samples[:max_samples]

    if not samples:
        logger.warning("评测数据集为空")
        return {m: {} for m in metrics}

    # 提取输入和参考
    inputs = [s["input"] if isinstance(s, dict) else s[0] for s in samples]
    references = [s["reference"] if isinstance(s, dict) else s[1] for s in samples]

    # 批量推理
    predictions = []
    for i in range(0, len(inputs), batch_size):
        batch_inputs = inputs[i:i + batch_size]
        if generate_fn is not None:
            batch_preds = generate_fn(model, batch_inputs)
        elif hasattr(model, "generate"):
            batch_preds = model.generate(batch_inputs)
        else:
            raise ValueError("模型不支持 generate 方法，请提供 generate_fn 参数")

        if isinstance(batch_preds, str):
            batch_preds = [batch_preds]
        predictions.extend(batch_preds)

    logger.info(f"推理完成: {len(predictions)} 个样本")

    # 计算各指标
    results = {}
    for m in metrics:
        metric_fn = METRIC_REGISTRY[m]
        result = metric_fn(predictions, references)
        results[m] = result
        # 打印主要指标
        main_key = m  # wer/cer/bleu
        if main_key in result:
            logger.info(f"{m.upper()}: {result[main_key]:.4f}")

    return results


# ── 模块自测 ────────────────────────────────────────────
if __name__ == "__main__":
    print("=== 评测指标模块自测 ===\n")

    # 测试 WER
    preds = ["hello world how are you", "this is a test"]
    refs = ["hello world how are you", "this is the test"]
    wer_result = compute_wer(preds, refs)
    print(f"WER 测试: {wer_result}")
    assert wer_result["wer"] >= 0

    # 测试 CER
    preds_cn = ["你好世界", "测试一下"]
    refs_cn = ["你好世界", "测试以下"]
    cer_result = compute_cer(preds_cn, refs_cn)
    print(f"CER 测试: {cer_result}")
    assert cer_result["cer"] >= 0

    # 测试 BLEU
    preds_bleu = ["the cat sat on the mat", "there is a cat on the mat"]
    refs_bleu = ["the cat is on the mat", "there is a cat on the mat"]
    bleu_result = compute_bleu(preds_bleu, refs_bleu)
    print(f"BLEU 测试: bleu={bleu_result['bleu']:.4f}, bleu_1={bleu_result['bleu_1']:.4f}")
    assert 0 <= bleu_result["bleu"] <= 1

    # 测试完全匹配
    perfect_preds = ["hello world"]
    perfect_refs = ["hello world"]
    perfect_wer = compute_wer(perfect_preds, perfect_refs)
    print(f"完全匹配 WER: {perfect_wer['wer']:.4f}")
    assert perfect_wer["wer"] == 0.0

    # 测试统一评测入口
    class DummyModel:
        def generate(self, inputs):
            return inputs  # 直接返回输入作为预测

    dummy_dataset = [
        {"input": "hello world", "reference": "hello world"},
        {"input": "good morning", "reference": "good evening"},
    ]
    eval_results = evaluate_model(
        model=DummyModel(),
        dataset=dummy_dataset,
        metrics=["wer", "bleu"],
    )
    print(f"\n统一评测结果:")
    for metric_name, result in eval_results.items():
        main_val = result.get(metric_name, "N/A")
        print(f"  {metric_name}: {main_val}")

    print("\n=== 自测通过 ===")

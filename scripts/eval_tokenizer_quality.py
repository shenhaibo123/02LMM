import argparse
from collections import Counter
from typing import Dict, List, Optional

from transformers import AutoTokenizer


def read_corpus_lines(path: Optional[str], max_lines: Optional[int] = None) -> List[str]:
    if not path:
        return []
    lines: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        # 按「非空行」计数，确保 max_lines 对应的是有效文本行数，而不是包含空行在内的物理行数。
        count = 0
        for line in f:
            if max_lines is not None and count >= max_lines:
                break
            text = line.rstrip("\n")
            if not text:
                continue
            lines.append(text)
            count += 1
    return lines


def eval_tokenizer_on_corpus(name: str, tok, corpus: List[str]) -> None:
    print(f"\n=== [{name}] 在语料上的整体统计 ===")
    if not corpus:
        print("未提供语料文件，跳过整体统计。")
        return

    total_tokens = 0
    total_chars = 0
    total_lines = 0
    unk_id = tok.unk_token_id
    unk_count = 0
    freq: Counter = Counter()

    for text in corpus:
        total_lines += 1
        total_chars += len(text)
        encoded = tok(text, add_special_tokens=False)
        ids = encoded["input_ids"]
        total_tokens += len(ids)
        freq.update(ids)
        if unk_id is not None:
            unk_count += sum(1 for i in ids if i == unk_id)

    vocab_size = len(tok)
    used_token_count = len(freq)
    avg_tokens_per_line = total_tokens / max(total_lines, 1)
    avg_tokens_per_char = total_tokens / max(total_chars, 1)
    used_ratio = used_token_count / max(vocab_size, 1)

    print(f"总行数: {total_lines}")
    print(f"总字符数: {total_chars}")
    print(f"总 token 数: {total_tokens}")
    print(f"平均每行 token 数: {avg_tokens_per_line:.2f}")
    print(f"平均每个字符 token 数: {avg_tokens_per_char:.4f}")
    print(f"词表大小: {vocab_size}")
    print(f"在该语料中真正用到的 token 数: {used_token_count} (占词表比例 {used_ratio:.4f})")
    if unk_id is not None:
        print(f"UNK token id: {unk_id}，在语料中的出现次数: {unk_count}")

    # 看看 Top-K 高频 token 占比，衡量「头部」是否过于集中
    for k in (10, 100, 1000):
        if not freq:
            break
        most_common = freq.most_common(k)
        topk_sum = sum(c for _, c in most_common)
        ratio = topk_sum / max(total_tokens, 1)
        print(f"Top-{k} 高频 token 覆盖的比例: {ratio:.4f}")


def eval_phrases(name: str, tok, phrases: List[str]) -> None:
    print(f"\n=== [{name}] 常见词/短语的表示效率 ===")
    for s in phrases:
        ids = tok(s, add_special_tokens=False)["input_ids"]
        print(f"「{s}」 -> token 数: {len(ids)}，ids: {ids}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="对比两个 tokenizer 在语料与常见词上的表示效率与利用情况。"
    )
    parser.add_argument(
        "--id-a",
        type=str,
        default="./model",
        help="第一个 tokenizer 的模型名或本地路径。",
    )
    parser.add_argument(
        "--id-b",
        type=str,
        default="openbmb/MiniCPM4-0.5B",
        help="第二个 tokenizer 的模型名或本地路径。",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="",
        help="用于评估的语料文件（utf-8，每行一条文本）。可为空。",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=5000,
        help="语料最多读取多少行（避免过大）。",
    )

    args = parser.parse_args()

    tok_a = AutoTokenizer.from_pretrained(args.id_a, trust_remote_code=True)
    tok_b = AutoTokenizer.from_pretrained(args.id_b, trust_remote_code=True)

    print(f"=== 加载 tokenizer 成功 ===")
    print(f"A: {args.id_a}, vocab size = {len(tok_a)}")
    print(f"B: {args.id_b}, vocab size = {len(tok_b)}")

    corpus_lines = read_corpus_lines(args.corpus, max_lines=args.max_lines)
    if corpus_lines:
        print(f"\n从语料文件 {args.corpus} 读取到 {len(corpus_lines)} 行文本用于评估。")

    # 整体统计
    eval_tokenizer_on_corpus("A", tok_a, corpus_lines)
    eval_tokenizer_on_corpus("B", tok_b, corpus_lines)

    # 一组内置的常见词/短语示例，便于快速对比
    common_phrases = [
        "北京大学",
        "人工智能",
        "机器学习",
        "大语言模型",
        "中华人民共和国",
        "computer",
        "machine learning",
        "language model",
        "deep neural network",
        "natural language processing",
    ]

    eval_phrases("A", tok_a, common_phrases)
    eval_phrases("B", tok_b, common_phrases)


if __name__ == "__main__":
    main()


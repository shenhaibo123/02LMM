import argparse
import random
from typing import Dict, Set

from transformers import AutoTokenizer


def summarize_two_vocabs(
    id_a: str,
    id_b: str,
    trust_remote_code_a: bool = True,
    trust_remote_code_b: bool = True,
    top_k_examples: int = 20,
) -> None:
    # 加载两个 tokenizer 与各自 vocab
    tokenizer_a = AutoTokenizer.from_pretrained(
        id_a, trust_remote_code=trust_remote_code_a
    )
    tokenizer_b = AutoTokenizer.from_pretrained(
        id_b, trust_remote_code=trust_remote_code_b
    )

    vocab_a: Dict[str, int] = tokenizer_a.get_vocab()
    vocab_b: Dict[str, int] = tokenizer_b.get_vocab()

    va: Set[str] = set(vocab_a.keys())
    vb: Set[str] = set(vocab_b.keys())

    only_a = va - vb
    only_b = vb - va
    inter = va & vb

    print(f"=== Tokenizer A: {id_a} ===")
    print(f"  vocab size: {len(va)}")
    print()
    print(f"=== Tokenizer B: {id_b} ===")
    print(f"  vocab size: {len(vb)}")
    print()

    print("=== 集合关系统计（基于 token 字符串） ===")
    print(f"  A ∩ B size: {len(inter)}")
    print(f"  A - B size: {len(only_a)}")
    print(f"  B - A size: {len(only_b)}")

    union_size = len(va | vb)
    if union_size > 0:
        jaccard = len(inter) / union_size
        print(f"  Jaccard 相似度: {jaccard:.6f}")
    print()

    def show_examples(label: str, items: Set[str]) -> None:
        items_list = sorted(items)
        print(f"--- {label} 示例（最多 {top_k_examples} 个） ---")
        for t in items_list[:top_k_examples]:
            print(repr(t))
        print()

    show_examples("仅在 A 中的 token", only_a)
    show_examples("仅在 B 中的 token", only_b)

    # 随机抽取一部分差异 token，查看在对侧 tokenizer 中的表示
    def show_cross_tokenization():
        print("=== 随机差异 token 在对侧 tokenizer 中的表示 ===")
        rand = random.Random(0)  # 固定种子，方便复现

        only_a_list = list(only_a)
        only_b_list = list(only_b)

        sample_a = rand.sample(only_a_list, min(top_k_examples, len(only_a_list)))
        sample_b = rand.sample(only_b_list, min(top_k_examples, len(only_b_list)))

        print(f"\n--- 从 A - B 中随机选取 {len(sample_a)} 个，在 B 中的表示 ---")
        for s in sample_a:
            ids_in_b = tokenizer_b.encode(s, add_special_tokens=False)
            toks_in_b = tokenizer_b.convert_ids_to_tokens(ids_in_b)
            print(f"A-only token: {repr(s)}")
            print(f"  B ids: {ids_in_b}")
            print(f"  B tokens: {[repr(t) for t in toks_in_b]}")
            print()

        print(f"--- 从 B - A 中随机选取 {len(sample_b)} 个，在 A 中的表示 ---")
        for s in sample_b:
            ids_in_a = tokenizer_a.encode(s, add_special_tokens=False)
            toks_in_a = tokenizer_a.convert_ids_to_tokens(ids_in_a)
            print(f"B-only token: {repr(s)}")
            print(f"  A ids: {ids_in_a}")
            print(f"  A tokens: {[repr(t) for t in toks_in_a]}")
            print()

    show_cross_tokenization()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "对比两个 tokenizer 的词表差异（基于公开模型或本地目录）。"
        )
    )
    parser.add_argument(
        "--id-a",
        type=str,
        default="openbmb/MiniCPM4-0.5B",
        help="第一个 tokenizer 的模型名或本地路径，例如 openbmb/MiniCPM4-0.5B。",
    )
    parser.add_argument(
        "--id-b",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="第二个 tokenizer 的模型名或本地路径，例如 Qwen/Qwen3-0.6B。",
    )
    parser.add_argument(
        "--no-trust-remote-code-a",
        action="store_true",
        help="对 A 关闭 trust_remote_code。",
    )
    parser.add_argument(
        "--no-trust-remote-code-b",
        action="store_true",
        help="对 B 关闭 trust_remote_code。",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="每侧仅显示多少个差异 token 示例。",
    )

    args = parser.parse_args()

    summarize_two_vocabs(
        id_a=args.id_a,
        id_b=args.id_b,
        trust_remote_code_a=not args.no_trust_remote_code_a,
        trust_remote_code_b=not args.no_trust_remote_code_b,
        top_k_examples=args.top_k,
    )


if __name__ == "__main__":
    main()


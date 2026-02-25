# 03 分词器与 Tokenizer 基础

> 本文档由 Vibe Writing 大模型生成初稿，并结合本仓库实践与个人学习需求进行整理与校订。

## 1. 本篇目标
 
- 理解「token」与「字符 / 词」的区别。
- 掌握常见分词算法（BPE、WordPiece、Unigram）的直观概念。
- 了解如何从 0 训练一个自定义 Tokenizer，并在 MiniMind 中使用。
 
 ## 2. 为什么需要分词器
 
 - 模型只能处理有限大小的词表。
 - 直接按「字 / 词」建表的问题：稀疏、冗余、OOV。
 - 使用 subword 的折中方案。
 
 ## 3. 常见分词算法概览
 
- BPE（Byte Pair Encoding）
  - 思想：从最小单元（字符或字节）出发，迭代合并最高频相邻对子，直至达到目标词表大小。
  - Byte-level BPE：以字节为基础单元（256 个），避免 OOV，常见于多语言/跨域模型（如 GPT-2、Qwen）。
  - 优点：实现简单、控制词表规模；缺点：强依赖语料统计，对于极端长尾 token 化不一定最优。
- WordPiece
  - 与 BPE 类似，但合并准则改为最大化训练数据似然（近似互信息）。
  - 在 BERT 系列中常见，词表更偏稳健、合并序列更“语言学”。
- Unigram（常与 SentencePiece 配合）
  - 思想：从一个较大的候选词表出发，迭代删除“贡献最小”的子词，直到达到目标大小。
  - 优点：搜索空间更全局、对多语言友好；缺点：实现与训练过程相对复杂。
- 中文与多语言要点
  - 建议启用 byte-level 或保留中文字符全集，减少未知字符回退。
  - 适度加入常见词缀/数字/标点规范化规则，兼顾覆盖与序列长度。
 
 ## 4. 训练自定义 Tokenizer 的流程
 
 - 准备原始语料与清洗。
 - 使用工具（如 `tokenizers` 库）训练词表。
 - 保存并加载：`tokenizer.json` 与 `tokenizer_config.json`。

结合本仓库的 `trainer/train_tokenizer.py`，可以把「怎么分词、用的什么规则」说具体一点：

- **语料读取：怎么把文本喂给分词器**
  - 对应函数：`get_texts(data_path)`。
  - 规则：按行读取 `jsonl` 文件，每行是一个 JSON 对象，从中取出 `data['text']` 作为训练样本。示例路径：`../dataset/pretrain_hq.jsonl`。
  - 代码核心：
    - `for line in f: data = json.loads(line); yield data['text']`
    - 脚本里用 `i >= 10000` 做了一个「只取前 1 万行」的小规模实验限制。

- **子词模型与预分词：token 是「怎么被切出来」的**
  - 对应代码：
    - `tokenizer = Tokenizer(models.BPE())`
    - `tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)`
    - `tokenizer.decoder = decoders.ByteLevel()`
  - 关键点：
    - `models.BPE()`：选用 **BPE（Byte Pair Encoding）** 作为子词模型。
    - `ByteLevel` 预分词 + decoder：
      - **规则**：先把输入编码为 UTF-8 字节序列，再在字节级上做 BPE 合并。
      - 实际行为：任何字符最终都能拆成若干字节 token，不会出现「未知词」（和 Qwen 的 byte-level BPE 思路一致）。
    - `add_prefix_space=False`：不自动在句首加空格，行为更接近 Qwen / GPT 的默认习惯。

- **训练规则：通过哪些函数、什么参数来「学」这些 token**
  - 对应代码：
    - `trainer = trainers.BpeTrainer(...)`
    - `tokenizer.train_from_iterator(texts, trainer=trainer)`
  - 主要参数含义：
    - `vocab_size=vocab_size`：
      - 词表大小是**完全自定义的**，脚本里用的是 `VOCAB_SIZE = 6400`。
      - 你可以改成 8K、32K、50K……数值越大，子词越粗（更接近完整词），序列越短，但 Embedding/LM Head 越大。
    - `special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"]`：
      - 在训练开始前就「锁定」这 3 个特殊 token，确保它们一定出现在词表中。
    - `initial_alphabet=pre_tokenizers.ByteLevel.alphabet()`：
      - 把 256 个字节全部放进初始字母表，保证任何字节都能被覆盖，避免训练后有字符拆不出来。
  - 训练过程（内部逻辑）：
    - `train_from_iterator` 会多次遍历你提供的 `texts`，统计所有字节序列的共现频率。
    - 从「单字节」开始，不断合并最高频的相邻对子，直到达到 `vocab_size`。
    - 训练结束后，`tokenizer.json` 中就包含了 **合并规则（merges）和最终词表（vocab）**。

- **特殊 token ID 固定：为什么、怎么做**
  - 对应代码：
    - `assert tokenizer.token_to_id("<|endoftext|>") == 0`
    - `assert tokenizer.token_to_id("<|im_start|>") == 1`
    - `assert tokenizer.token_to_id("<|im_end|>") == 2`
  - 原因：
    - 模型代码与聊天模板都默认这 3 个 id 分别是 0/1/2（如 pad、BOS、EOS）。
    - 如果不固定，BPE 训练可能改变顺序，导致模型与分词器协议不一致。

- **保存与配置：让 Hugging Face 能直接加载使用**
  - 脚本会生成两类文件：
    - `tokenizer.json`：完整保存 BPE 模型（词表 + merges + 预分词配置）。
    - `tokenizer_config.json`：补充 Hugging Face 需要的元信息：
      - `bos_token`, `eos_token`, `pad_token`, `unk_token` 等；
      - `added_tokens_decoder` 中列出 3 个特殊 token 的属性；
      - `model_max_length`、`clean_up_tokenization_spaces` 等推理相关参数；
      - 一大段 `chat_template`（定义 system/user/assistant/tool 消息如何拼成模型实际看到的文本）。
  - 加载方式：
    - 在训练/推理代码中使用：`AutoTokenizer.from_pretrained(tokenizer_dir)`，即可获得与本脚本完全一致的分词器行为。

进一步实践要点：
- 归一化（Normalization）：统一全角/半角、大小写、标点与空白；按需保留大小写与数字形态。
- 预分词（Pre-tokenization）：英文可按空格/标点切分；中文多直接走字符级或字节级。
- 模型与目标大小：中文/多语推荐 50K～150K；更大词表减少分裂但会放大 Embedding/LM Head。
- 质量评估：统计 OOV 情况、平均 token/字数、跨域覆盖与常见词拆分率。
- 与模型对齐：确保 `vocab_size` 与 `model/model_minimind.py` 中配置一致，特殊 token 索引固定。
 
 ## 5. 与 MiniMind 项目的关系
 
 - 对应到 `model/tokenizer.json` 与 `train_tokenizer.py`。
 - 训练好的分词器如何与模型参数对齐。

以本项目默认的 MiniMind 小词表为例，可以回答几个工程上常见的问题：

- **词表大小是不是自定义的？**
  - 是的。`train_tokenizer.py` 里通过 `VOCAB_SIZE = 6400` 控制目标词表规模。
  - 如果你想实验更大的词表，只需要改这个数字，然后重新训练 tokenizer，并同步更新模型配置中的 `vocab_size`。

- **怎么「看」词表里到底有哪些 token？**
  - 使用 Hugging Face 接口：
    - `from transformers import AutoTokenizer`
    - `tok = AutoTokenizer.from_pretrained("./model")`
    - `vocab = tok.get_vocab()` → 一个 `token -> id` 的字典。
  - 你可以打印前几条，或按 id 排序看低频 / 高频 token（频次需要另算，见下）。

- **如何评估一个词表的「好坏」？常见维度与权衡**
  - 可以从三个互相牵制的方向来理解：
    - **表达能力（coverage & segmentation）**：
      - 是否几乎不出现 OOV / unk token（对非 byte-level 模型尤其重要）。
      - 对常见词、领域术语、中文短语的切分是否「自然」——既不过碎，也不过于粗糙。
    - **效率（efficiency）**：
      - 在同一批文本上，平均每行 / 每字符的 token 数是多少（影响序列长度与推理开销）。
      - 高频常用词/短语是否用较少 token 表示（例如 1–2 个 token 而不是 5–6 个）。
    - **利用率（vocab utilization）**：
      - 实际语料中有多少词表里的 token 被真正使用到。
      - 是否存在大量几乎从不出现的长尾 token（可以考虑裁剪）。
  - 一个「好的」词表，通常是在上述三点之间做平衡：既能覆盖目标语料，又不过度浪费参数，还能在关键场景下给出高效的切分。

- **在代码中如何做这些评估？**
  - 典型做法是选一批代表性文本（比如验证集 / 领域语料），然后对每个 tokenizer 分别统计：
    - **覆盖率**：
      - 对每条文本调用 `tok(text)`，检查 `input_ids` 中是否出现 unk token（本项目中 unk 也映射为 `<|endoftext|>`，可以统计该 id 的出现次数）。
      - 对 byte-level BPE 来说，理论上不会产生真正「无法拆解」的字符，但对于非 byte-level 方案，unk 比例是重要指标。
    - **利用率 / 频次分布**：
      - 对所有文本做一次分词，把所有 token id 计数，得到 `id -> count`。
      - 再按 count 排序，可以看到：
        - 高频 token 是否主要覆盖单双字、常见词片和标点；
        - 是否存在大量**几乎不出现的长尾 token**（利用率低，意味着词表可以优化/裁剪）。
      - 可以进一步计算「Top-10 / Top-100 / Top-1000 高频 token 占总 token 的比例」，观察头部是否过于集中。
    - **常见词/短语的表示效率**：
      - 选一批你关心的常见词/短语，比如：
        - 高频中文词：如「北京大学」「人工智能」「机器学习」「中华人民共和国」等；
        - 高频英文词：如 `machine`, `learning`, `language model`, `natural language processing` 等。
      - 对每个词调用 `tok(word, add_special_tokens=False)`，记录 `len(input_ids)`：
        - 长度 1 → 该词是单 token，高效；
        - 长度 2–3 → 被拆成少量子词，通常可以接受；
        - 长度很多 → 表示效率不高，说明该 tokenizer 对这个词所在领域不够「友好」。
      - 将本项目 tokenizer 与 MiniCPM4、Qwen3 等并排比较，可以直观看出「小词表 vs 大词表」「不同训练策略」在这些关键词上的实际差异。

在本仓库中，我们提供了两个脚本来辅助做这类对比评估：

- **脚本 1：`scripts/compare_tokenizers.py`（侧重「词表集合关系 + 差异 token 样例」）**
  - 主要看：
    - 两个 tokenizer 的 vocab 大小；
    - 集合级关系（交集、A-B、B-A、大致 Jaccard 相似度）；
    - 以及差异 token 在对侧 tokenizer 中是如何被拆解的。
  - 示例（Qwen3 vs MiniCPM4、MiniCPM4 vs 本项目小词表）见前文小节。

- **脚本 2：`scripts/eval_tokenizer_quality.py`（侧重「实际语料上的效率与利用率」）**
  - 用法示例（比较本项目小词表 vs MiniCPM4-0.5B，在一份自选语料上）：
    - 准备一个简单语料文件，例如 `dataset/sample_corpus.txt`，每行一条文本（utf-8）。
    - 然后运行：
      ```bash
      python3 scripts/eval_tokenizer_quality.py \
        --id-a ./model \
        --id-b openbmb/MiniCPM4-0.5B \
        --corpus dataset/sample_corpus.txt \
        --max-lines 5000
      ```
  - 脚本会输出：
    - 对每个 tokenizer：
      - 总行数、总字符数、总 token 数；
      - 平均每行 token 数、平均每个字符 token 数（可以近似理解为「序列长度膨胀程度」）；
      - 词表大小、在该语料中真正用到的 token 数及其占比（大概的「利用率」）；
      - 若有 unk token，会统计其出现次数（非 byte-level 模型时用于衡量覆盖率）。
    - 同时还会用一组内置的中英常见短语（如「北京大学」「人工智能」「机器学习」「computer」「machine learning」「natural language processing」等），对比它们在两个 tokenizer 下：
      - 具体被拆成多少个 token；
      - 对应的 `input_ids` 长度。

通过这些工具和统计，你可以系统地比较 MiniCPM4 与本项目小词表在「覆盖率、利用率、序列长度、常见词表示效率」这些维度上的差别，并据此调节：

- `VOCAB_SIZE`（词表规模）；
- 归一化与预分词规则（如是否使用 byte-level、是否保留大小写、如何处理空格与标点）；
- 训练语料的构成（是否需要加入更多目标领域文本）。

这也是 MiniMind 文档中「模型与目标大小」「质量评估」「与模型对齐」几条建议在实际工程里的具体落地方式。

在本仓库：
- 分词器资产位于：[tokenizer.json](file:///Users/shenhaibo/Desktop/02LLM/model/tokenizer.json)、[tokenizer_config.json](file:///Users/shenhaibo/Desktop/02LLM/model/tokenizer_config.json)
- 训练脚本入口参考：[train_tokenizer.py](file:///Users/shenhaibo/Desktop/02LLM/trainer/train_tokenizer.py)
- 与模型对齐时需同步修改 `vocab_size` 等配置字段（见模型配置类与加载逻辑）。

## 6. Qwen 与「面壁」Tokenizer 说明

Qwen（通义千问）：
- 官方说明：采用基于 UTF-8 字节的 BPE（tiktoken 风格），保证无未知词（无 UNK），遇到罕见字符回退到单字节切分。
- 词表规模：约 151,643 个常规 token + 约 200 个控制 token，总计约 151,851 个索引位（不同版本略有微差）。参考：
  - Qwen 官方 tokenization 说明（GitHub）：
    https://github.com/QwenLM/Qwen/blob/main/tokenization_note.md
  - Qwen 文档「核心概念」页面：
    https://qwen.readthedocs.io/zh-cn/latest/getting_started/concepts.html
- 特性要点：支持注入防护（allowed_special 控制）、支持扩展词表（extra vocab）等。

「面壁」（MiniCPM 系列）：
- 官方仓库与技术博客未统一公布所有版本的具体词表规模与构建细节，不同代次/型号可能存在差异（例如多模态版本、长上下文版本等）。
- 业界报道与工程实践中常见到“优化词表结构、剔除低频子词、在保持覆盖的同时缩减 Embedding/LM Head 参数”的做法。这类做法可能包括：
  - 从一个较大的通用词表裁剪（Prune）到目标规模（如 ~70K）；
  - 或者在特定数据分布上重新训练/蒸馏一个更小的词表（重新定义 merge 列表）。

关于词表关系的工程验证，可以直接基于公开模型做「集合级」对比，而不预设任何“谁是谁的子集”之类结论。

在本仓库中，我们提供了一个简单工具脚本 `scripts/compare_tokenizers.py`，默认对比：

- Tokenizer A：`openbmb/MiniCPM4-0.5B`（MiniCPM4 系列）
- Tokenizer B：`Qwen/Qwen3-0.6B`（Qwen3 系列）

运行方式（需安装 `transformers`）：

```bash
python3 scripts/compare_tokenizers.py \
  --id-a openbmb/MiniCPM4-0.5B \
  --id-b Qwen/Qwen3-0.6B
```

该脚本基于公开模型实际 tokenizer，只做集合统计和交叉 tokenization，输出示例（节选）：

- 词表规模（基于 `get_vocab()`）：
  - A：MiniCPM4-0.5B → vocab size = 73,448
  - B：Qwen3-0.6B   → vocab size = 151,669
- 集合关系（以 token 字符串为元素）：
  - 交集 \|A ∩ B\| = 13,260
  - 仅在 A 中 \|A - B\| = 60,188
  - 仅在 B 中 \|B - A\| = 138,409
  - Jaccard 相似度 ≈ 0.063

进一步对「差异 token」做交叉 tokenization，可以观察到两者的一些设计偏好（这里仅列现象，不做主观评价）：

- 对于 **A - B（MiniCPM4 独有）的一部分 token**：
  - 出现了大量 `▁word` 风格的英文 token（`▁them`、`▁Corp`、`▁everything` 等），以及中文词组（如“为例”“交警”“作品”“航海”“申请撤诉”）、emoji（如 `🛗`）和 `<reserved_xxx>` 这类保留位。
  - 这些在 Qwen3 中会被拆解为若干 byte-level BPE 子词，例如：
    - `'<reserved_807>'` → `'<', 'reserved', '_', '8', '0', '7', '>'`
    - `'▁everything'` → 若干表示空格/边界的前缀 token + `'everything'`
    - `🛗` → 两个字节级子词。

- 对于 **B - A（Qwen3 独有）的一部分 token**：
  - 出现了大量 `Ġword` 风格的英文 token（`Ġinhibitors`、`Ġability`、`ĠCafe` 等），以及多种多语言/符号组合。
  - 这些在 MiniCPM4 中通常会被拆解为 `▁`（词首空格）+ `'Ġ'` 字符本身 + 若干子词，例如：
    - `'Ġability'` → `['▁', 'Ġ', 'ability']`
    - `'Ġcommanders'` → `['▁', 'Ġ', 'command', 'ers']`

从工程角度看，两者都可以表示任意 UTF-8 文本，区别主要在于：哪些片段被视作「一个 token」还是「多个子词组合」，这会影响平均序列长度、Embedding/LM Head 参数量以及不同语料类型上的建模颗粒度。

如需对其它公开模型或本地 tokenizer 做同样分析，可以仿照上述脚本思路，更换 `--id-a` / `--id-b` 为目标模型名或本地路径，并统计 vocab 集合的大小、交集/差集规模、以及差异 token 在对侧 tokenizer 中的具体编码方式。

### 与本项目小词表的对比（MiniMind Tokenizer vs MiniCPM4）

在本项目中，我们还用同一脚本对比了本地小词表（`./model`）与 MiniCPM4-0.5B 的 tokenizer，仅基于公开模型和本仓库的 `tokenizer.json`：

- 词表规模：
  - A：本项目 MiniMind tokenizer → vocab size = 6,400
  - B：MiniCPM4-0.5B → vocab size = 73,448
- 集合关系（以 token 字符串为元素）：
  - 交集 \|A ∩ B\| = 1,311
  - 仅在本项目中 \|A - B\| = 5,089
  - 仅在 MiniCPM4 中 \|B - A\| = 72,137
  - Jaccard 相似度 ≈ 0.0167（重叠极小，基本是两套独立设计）

从「差异 token 的交叉表示」可以看出两者的风格差异：

- 本项目小词表独有的 token（A - B）：
  - 包含大量英文缩写/后缀（`'d`、`'ll`、`'re` 等）、标点组合（`----`、`..`、`00`、`100` 等），以及若干 `Ġword` 风格的英文子词和若干常见中文词组的编码。
  - 在 MiniCPM4 中，这些模式会被拆分为更基础的子词，例如：
    - `'Ġspecies'` → `['▁', 'Ġ', 'species']`
    - `'Ġcities'` → `['▁', 'Ġ', 'c', 'ities']`
    - 某些中文组合会拆成若干 `['▁é', '«', 'ĺ', 'æ', 'ķ', 'Ī']` 之类的子词。
- MiniCPM4 独有的 token（B - A）：
  - 典型为 `▁word` 风格的英文 token（`▁SPD`、`▁resp`、`▁trait`、`▁recruiting` 等），以及中文短语（如“再审申请人”“的距离”）和 emoji/生僻字符。
  - 在本项目 tokenizer 中，它们通常会被拆解为：
    - 若干表示空格/边界的前缀 token（如 `âĸ`, `ģ` 等 byte-level 前缀）；
    - 再加上一到数个子词，例如：
      - `▁resp` → `['â', 'ĸ', 'ģ', 'res', 'p']`
      - `再审申请人` → `['再', '审', '申', '请', '人']`（在实际编码中对应为 Unicode→UTF-8 后的若干子词）
      - `🗏` → 拆成 4 个与字节相关的子词。

综合来看：

- 本项目 MiniMind tokenizer 是一个**非常小的词表**，参数量小、实现简单，但同一段文本会被拆成更多 token，序列更长。
- MiniCPM4 采用更大的 `▁word` 风格词表，在英文和部分中文短语上提供了更多单 token 表达，能缩短序列长度，但需要更大的 Embedding/LM Head。
- 两者都可以完整覆盖 UTF-8 文本，区别主要体现在「哪些模式被视作单一 token」以及由此带来的序列长度、参数量和不同语料上的建模颗粒度差异。

## 7. 延伸阅读与思考
 
 - 推荐阅读：SentencePiece / Hugging Face Tokenizers 文档。
 - 思考题：如果词表太小 / 太大，会分别带来什么问题？

补充参考：
- Hugging Face Tokenizers 文档（Normalization / Pre-tokenization / Models / Post-processors）
  https://huggingface.co/docs/tokenizers
- Qwen 官方 Tokenization 说明（BPE 细节、special tokens、扩展词表实操）
  https://github.com/QwenLM/Qwen/blob/main/tokenization_note.md
- Qwen 文档「核心概念」页（词表规模、Byte-level BPE 原理概述）
  https://qwen.readthedocs.io/zh-cn/latest/getting_started/concepts.html
 

# 03 模型核心：Attention 与 Transformer

> 本文档由 Vibe Writing 大模型生成初稿，并结合本仓库实践与个人学习需求进行整理与校订。

如果说「预测下一个 token」定义了语言模型在做什么，那么 Transformer 基本上定义了「模型是如何做到的」。这一篇试着用尽量通俗的方式，把注意力机制和 Transformer 的关键结构串起来，并和本仓库里的实现对应起来。

## 1. 本篇目标

- 建立一个清晰的心智模型：Transformer 里都有哪些模块，它们怎么串起来。
- 用类比方式理解自注意力（Self-Attention）在做什么，而不是被公式吓退。
- 知道「为什么几乎所有主流 LLM 都是 Transformer 解码器」。
- 能在 `model/model_minimind.py` 里大致认出 Attention、FFN、LayerNorm 等模块的位置和功能。

## 2. Transformer 长什么样？

Transformer 这个词经常出现，但真正画出来，其实就是「多层堆叠的积木」：

- 每一层由若干标准模块组成（自注意力 + 前馈网络 + 正则化等）；
- 输入是一串 token 的向量表示，输出还是同样长度的一串向量；
- 多层堆叠之后，最后一层的输出再接一个线性层 + softmax，得到每个位置上「下一个 token」的概率分布。

从宏观结构看，常见有三种形态：

- Encoder-Decoder（原始 Transformer，用于机器翻译）；
- Encoder-only（BERT 一类）；
- Decoder-only（GPT、LLaMA、Qwen、MiniMind 等）。

本仓库里的模型属于 Decoder-only，也就是：

- 只保留解码器部分；
- 使用因果掩码（causal mask），保证当前位置只能看到「自己之前」的 token。

你可以在 `model/model_minimind.py` 中找到类似的结构：

- 嵌入层（token embedding + 位置编码）；
- 多层 `TransformerBlock`（或类似命名）；
- 最后的输出层（通常叫 `lm_head` 或类似）。

## 3. 自注意力：模型为什么要「自己看自己」

很多对 Transformer 的困惑来自一个词：Attention（注意力）。直觉可以先这样理解：

> 对于序列里的每个 token，它可以「回头」看前面所有 token，从中挑出当前预测最需要的信息，并给不同 token 分配不同的权重。

相比只能顺序读取的 RNN，自注意力有几个明显优势：

- 任意两个位置之间都可以直接建立连接；
- 计算可以高度并行化（对整个序列一起算）；
- 模型可以更容易捕捉长距离依赖。

### 3.1 Q / K / V 是什么？

在代码里，你会看到把输入向量线性变换出三份：

- Query（Q）
- Key（K）
- Value（V）

如果把每个 token 的向量想象成一句话的「自我介绍」，那么：

- Q：我现在在问的问题；
- K：每个 token 的「标签」，用来判断它是否与当前问题相关；
- V：真正要从这个 token 身上「拿走」的信息。

注意力计算可以概括为：

- 用当前 token 的 Q 去和所有 token 的 K 做相似度匹配；
- 把相似度通过 softmax 变成一组权重；
- 用这些权重对所有 V 做加权求和，得到当前 token 的新表示。

这套东西在代码里一般会对应到：

- 三个线性层（`q_proj` / `k_proj` / `v_proj` 或类似名字）；
- 一个矩阵乘法算注意力分数；
- 再乘上 V 得到输出。

## 4. 多头注意力：为什么不只用一组 Q/K/V？

现实中，我们不会只用单头注意力，而是：

> 把 embedding 维度切分为多份，每份单独做一套 Q/K/V + 注意力，再把结果拼回来。

直觉上可以类比为：

- 一组头负责「语法关系」；
- 一组头负责「实体与指代」；
- 一组头负责「长距离依赖」；
- ……

每个头看到的是同一个序列，但关注的「特征子空间」不同。多头注意力让模型可以更丰富、解耦地表达序列里各种关系。

在 MiniMind 的实现中，你通常会看到：

- 配置里有 `num_heads` 或类似字段；
- Attention 模块中，会先把输入 reshape 成 `(batch, seq_len, num_heads, head_dim)` 的形状，再做注意力计算。

## 5. 位置编码：让模型知道「谁在前谁在后」

自注意力有一个特点：如果不给它任何位置信息，它会把所有 token 看作是无序集合。

但语言是有顺序的：

- 「今天下雨所以我没去跑步」和「我没去跑步所以今天下雨」含义完全不同。

因此，需要给模型额外输入位置信息：

- 早期 Transformer 使用绝对位置编码（sin/cos）；
- 后来出现了各种相对位置编码、旋转位置编码（RoPE）等。

MiniMind 中使用的是 RoPE，可以更好地支持长上下文外推。在代码里，你会看到类似：

- 在 Attention 里对 Q/K 做一个旋转变换；
- 或有一个单独的 RoPE 模块，在计算注意力前插入。

当你在配置里调整 `max_position_embeddings` 或相关参数时，本质上就是在控制模型「记忆的最长序列长度」。

## 6. Transformer 中的其他关键模块

一个完整的 Transformer Block 通常不只有注意力，还有：

### 6.1 前馈网络（Feed-Forward Network, FFN）

结构一般类似：

> 线性层 → 激活函数（如 GELU） → 线性层

它的作用可以粗糙地理解为：

- 在「每个 token 的内部」做非线性变换；
- 把注意力层聚合来的信息进一步混合和抽象。

### 6.2 残差连接（Residual Connection）

典型写法：

> output = x + sublayer(x)

残差连接让训练深层网络变得容易得多：

- 它为梯度提供了一条「捷径」，减轻梯度消失问题；
- 也让模型在必要时可以保持「接近恒等映射」。

### 6.3 归一化层（LayerNorm）

在很多实现中你会看到：

- Pre-Norm：先 LayerNorm，再进 Attention/FFN；
- Post-Norm：先 Attention/FFN，再 LayerNorm。

不同做法在稳定性上略有差异，但它们的目的都是：

- 控制不同层之间激活分布的尺度；
- 让训练过程更稳定、更容易收敛。

### 6.4 Dropout 等正则化

- 在注意力权重或 FFN 中加入 Dropout；
- 帮助模型不过度依赖某些特定特征，提升泛化能力。

### 6.5 注意力变体与长上下文扩展（前沿速览）

- GQA / MQA：为推理效率优化 KV-Cache
  - MQA（Multi-Query Attention）：所有查询头共享一组 K/V 头，显著降低 KV-Cache 占用与访存压力，但在部分场景下会有轻微质量退化。
  - GQA（Grouped-Query Attention）：介于 MHA 与 MQA 之间，若干查询头共享同一组 K/V 头，在接近 MQA 吞吐/显存的同时更稳健。LLaMA、Qwen 等系列已大规模采用（Ainslie 等，EMNLP 2023）。
- 稀疏/线性/混合注意力：为超长上下文而生
  - 稀疏注意力：只对关键位置计算注意力，兼顾质量与开销。
  - 线性注意力：用核技巧或快速权重将复杂度从 O(N²) 降为 O(N)，适合百万级上下文；常与稀疏层组合以兼顾全局与精确召回。
  - 混合方案示例：社区近期探索如线性+稀疏的混合架构（SALA 路线），结合稳定化与位置编码改造，使端侧长上下文可行。
- RoPE 长上下文扩展
  - Position Interpolation（PI）：简单插值扩展频率，易用但大倍率下保真度有限。
  - NTK-aware/NTK-by-parts：分频段处理以缓解高频损失。
  - YaRN：在 NTK-by-parts 基础上引入注意力“温度”缩放与动态策略，配合少量数据微调，可将原生上下文扩展至更长窗口（ICLR 2024、Eleuther 资料）。

### 6.6 计算内核与系统优化（简述）

- FlashAttention v2/v3：IO 感知与异步/低精度流水，在 H100 等硬件上进一步提升注意力吞吐，官方报告显示可达 1.5–2.0× 加速（NeurIPS 2024 Spotlight/相关技术报告）。
- 系统层加速（详见第 07 篇）：
  - PagedAttention：分页管理 KV-Cache，近零碎片并支持前缀共享，是 vLLM 高吞吐的关键（SOSP 2023）。
  - 推测式解码（Speculative Decoding）：小模型起草、大模型并行校验，多 token 一跳确认，在低/中 QPS 下显著降延迟，vLLM 支持 EAGLE、草稿模型、MLP、n-gram 等多种路线。

MiniMind 中的 `TransformerBlock` 大概率会包含上述所有组件，你可以试着在 `model/model_minimind.py` 中对照阅读：

- 找到 Attention 模块的实现；
- 找到 FFN、LayerNorm、Dropout 的组合顺序；
- 对比和论文《Attention is All You Need》中的结构有何异同。

## 7. 与本仓库代码的关联

结合这一篇内容，可以这样浏览代码：

1. 打开 `model/model_minimind.py`：
   - 找到模型的 `__init__` 和 `forward`；
   - 找到 embedding、位置编码、堆叠的 block 列表，以及最后的输出层。
2. 在每个 Block 里：
   - 找出自注意力模块：一般会看到 Q/K/V 的线性层、注意力计算、softmax；
   - 找出 FFN 模块：两层线性层 + 激活函数；
   - 注意残差与 LayerNorm 的调用顺序。
3. 再结合 `trainer/train_pretrain.py`：
   - 看看一个 batch 的 token 是如何被送进模型；
   - 模型输出如何参与 loss 计算和反向传播。

把这一套串起来，你就从「看不懂魔法」迈向「知道魔法棒里面有什么」了。

## 8. 延伸阅读与思考

- 推荐阅读：
  - 原始论文《Attention is All You Need》；
  - 任意一个开源 LLM 的模型定义文件（比如 Llama 3 的 `modeling_llama.py`），对比 MiniMind 的实现差异。
- 思考题：
  - 如果去掉注意力，只保留 FFN 堆叠，模型还能学到什么样的模式？会失去哪些能力？
  - 如果把 RoPE 换成最简单的绝对位置编码，在长文本场景下会有什么影响？
  - 对于一个小模型（比如本仓库中的 MiniMind），堆叠层数 vs 隐藏维度，应该优先加哪个？

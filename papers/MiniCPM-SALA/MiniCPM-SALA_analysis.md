# MiniCPM-SALA 论文精读分析报告

**论文标题**：MiniCPM-SALA: Hybridizing Sparse and Linear Attention for Efficient Long-Context Modeling  
**作者**：MiniCPM Team（Wenhao An, Yingfa Chen, Yewei Fang 等 40+ 人）  
**机构**：面壁智能 / 清华大学  
**发布日期**：2026-02-12  
**来源链接**：
- arXiv: https://arxiv.org/abs/2602.11761
- HTML 全文: https://arxiv.org/html/2602.11761v1
- PDF: https://arxiv.org/pdf/2602.11761

---

## 逐句翻译（中英对照）

### Abstract（摘要）

- **EN**: The evolution of large language models (LLMs) towards applications with ultra-long contexts faces challenges posed by the high computational and memory costs of the Transformer architecture.
- **中**: 大语言模型（LLMs）向超长上下文应用演进时，面临 Transformer 架构带来的高计算和内存开销挑战。

- **EN**: While existing sparse and linear attention mechanisms attempt to mitigate these issues, they typically involve a trade-off between memory efficiency and model performance.
- **中**: 虽然现有的稀疏注意力和线性注意力机制试图缓解这些问题，但它们通常在内存效率和模型性能之间存在权衡。

- **EN**: This paper introduces MiniCPM-SALA, a 9B-parameter hybrid architecture that integrates the high-fidelity long-context modeling of sparse attention (InfLLM-V2) with the global efficiency of linear attention (Lightning Attention).
- **中**: 本文介绍 MiniCPM-SALA——一种 9B 参数的混合架构，将稀疏注意力（InfLLM-V2）的高保真长上下文建模与线性注意力（Lightning Attention）的全局效率相结合。

- **EN**: By employing a layer selection algorithm to integrate these mechanisms in a 1:3 ratio and utilizing a hybrid positional encoding (HyPE), the model maintains efficiency and performance for long-context tasks.
- **中**: 通过使用层选择算法以 1:3 的比例集成这两种机制，并利用混合位置编码（HyPE），模型在长上下文任务上保持了效率和性能的平衡。

- **EN**: Furthermore, we introduce a cost-effective continual training framework that transforms pre-trained Transformer-based models into hybrid models, which reduces training costs by approximately 75% compared to training from scratch.
- **中**: 此外，我们引入了一种低成本的持续训练框架，将预训练的 Transformer 模型转换为混合模型，相比从头训练降低约 75% 的训练成本。

- **EN**: Extensive experiments show that MiniCPM-SALA maintains general capabilities comparable to full-attention models while offering improved efficiency.
- **中**: 大量实验表明，MiniCPM-SALA 在保持与全注意力模型相当的通用能力的同时，提供了更高的效率。

- **EN**: On a single NVIDIA A6000D GPU, the model achieves up to 3.5× the inference speed of the full-attention model at the sequence length of 256K tokens and supports context lengths of up to 1M tokens, a scale where traditional full-attention 8B models fail because of memory constraints.
- **中**: 在单张 NVIDIA A6000D GPU 上，该模型在 256K token 序列长度下实现了全注意力模型 3.5 倍的推理速度，并支持最长 1M token 的上下文长度——这一规模下传统全注意力 8B 模型因内存限制而无法运行。

### 1 Introduction（引言）

- **EN**: As large language models become increasingly effective, the application scenarios of LLMs are undergoing a profound paradigm shift, transitioning from simple question-answering to more advanced applications, such as deep understanding and generation of ultra-long contexts, repository-scale code engineering, and long-horizon agents for complex tasks.
- **中**: 随着大语言模型日益有效，LLM 的应用场景正经历深刻的范式转变，从简单的问答转向更高级的应用，如超长上下文的深度理解与生成、仓库级代码工程以及面向复杂任务的长程智能体。

- **EN**: For these advanced applications, models are no longer confined to processing fragmented information. Instead, they must demonstrate the capacity to handle ultra-long contexts, such as grasping entire technical manuals at once, analyzing comprehensive project dependency trees containing tens of thousands of lines of code, and maintaining coherent task states and memory over multi-day human-AI collaborations.
- **中**: 对于这些高级应用，模型不再局限于处理碎片化信息，而必须展现处理超长上下文的能力——如一次性理解完整的技术手册、分析包含数万行代码的项目依赖树，以及在多天的人机协作中维持连贯的任务状态和记忆。

- **EN**: However, the Transformer architecture, which is the foundation of modern LLMs, encounters severe computational bottlenecks when handling ultra-long contexts due to its core full-attention mechanism.
- **中**: 然而，作为现代 LLM 基础的 Transformer 架构，由于其核心的全注意力机制，在处理超长上下文时遇到严重的计算瓶颈。

- **EN**: This bottleneck manifests primarily in two dimensions: (1) the compute bottleneck of computational complexity: for the standard attention mechanism, the computational cost grows quadratically with the sequence length N, i.e., its complexity is O(N²); (2) the memory bottleneck of KV-Cache: during the auto-regressive generation process, the model must store the key and value states of all historical contextual tokens.
- **中**: 该瓶颈主要表现在两个维度：(1) 计算复杂度的计算瓶颈：标准注意力机制的计算成本随序列长度 N 二次增长，即复杂度为 O(N²)；(2) KV-Cache 的内存瓶颈：在自回归生成过程中，模型必须存储所有历史上下文 token 的键值状态。

- **EN**: To address the aforementioned challenges, existing solutions have developed two primary paradigms: Sparse Attention and Linear Attention.
- **中**: 为应对上述挑战，现有解决方案发展了两种主要范式：稀疏注意力和线性注意力。

- **EN**: Sparse attention methods attempt to break the compute bottleneck by computing only the most salient portions of the attention matrix. However, these methods are hindered by a "sparse computation, dense storage" limitation.
- **中**: 稀疏注意力方法试图通过仅计算注意力矩阵中最显著的部分来突破计算瓶颈。然而，这些方法受到"稀疏计算、密集存储"局限性的制约。

- **EN**: Linear attention utilizes recurrent formulations to successfully reduce computational complexity to O(N). Nevertheless, this extreme efficiency is achieved by the lossy compression of contextual information and inevitably results in performance degradation.
- **中**: 线性注意力利用循环公式成功将计算复杂度降至 O(N)。然而，这种极端效率是通过对上下文信息的有损压缩实现的，不可避免地导致性能下降。

- **EN**: MiniCPM-SALA employs a hybrid architecture of sparse and linear attention, specifically designed to achieve efficient ultra-long sequence modeling.
- **中**: MiniCPM-SALA 采用稀疏注意力和线性注意力的混合架构，专门设计用于实现高效的超长序列建模。

- **EN**: We introduce a Sparse-Linear hybrid attention mechanism integrating 25% InfLLM-V2 and 75% Lightning Attention to strike a balance between throughput and precision.
- **中**: 我们引入了一种稀疏-线性混合注意力机制，将 25% InfLLM-V2 和 75% Lightning Attention 集成，以在吞吐量和精度之间取得平衡。

- **EN**: We demonstrate that the Transformer-to-hybrid paradigm is a highly effective strategy for building strong hybrid models, reducing the total training budget to approximately 25% relative to training from scratch.
- **中**: 我们证明了 Transformer-to-hybrid 范式是构建强混合模型的高效策略，将总训练预算降低至从头训练的约 25%。

- **EN**: We adopt HyPE (Hybrid Positional Encoding) to effectively harmonize the performance across both short and long contexts.
- **中**: 我们采用 HyPE（混合位置编码）来有效协调短上下文和长上下文的性能。

- **EN**: MiniCPM-SALA demonstrates substantial resource savings and speed advantages in long-context scenarios. On the NVIDIA A6000D GPU, MiniCPM-SALA achieves up to 3.5× the inference speed of Qwen3-8B at a sequence length of 256K tokens.
- **中**: MiniCPM-SALA 在长上下文场景中展现了显著的资源节省和速度优势。在 NVIDIA A6000D GPU 上，MiniCPM-SALA 在 256K token 序列长度下实现了 Qwen3-8B 3.5 倍的推理速度。

### 2 Model Development（模型开发）

- **EN**: MiniCPM-SALA adopts a hybrid architecture that interleaves sparse attention layers and linear attention layers.
- **中**: MiniCPM-SALA 采用交错排列稀疏注意力层和线性注意力层的混合架构。

- **EN**: We retain the Feed-Forward Network (FFN) block after each attention block in the Transformer architecture to ensure high-capacity knowledge representation.
- **中**: 我们在 Transformer 架构中每个注意力块之后保留前馈网络（FFN）块，以确保高容量的知识表示。

- **EN**: We employ a 1:3 mixing ratio: 25% of the layers adopt sparse attention while the remaining 75% employ linear attention.
- **中**: 我们采用 1:3 的混合比例：25% 的层采用稀疏注意力，其余 75% 采用线性注意力。

- **EN**: This hybrid configuration leverages the complementary strengths of both attention mechanisms. Linear attention layers have constant computational and memory complexities with respect to sequence length, facilitating efficient processing of long contexts. On the other hand, sparse attention layers facilitate effective modeling of long-range dependencies.
- **中**: 这种混合配置利用了两种注意力机制的互补优势。线性注意力层的计算和内存复杂度相对序列长度为常数，便于高效处理长上下文。另一方面，稀疏注意力层有助于有效建模长程依赖关系。

- **EN**: Rather than naively uniformly interleaving the two attention variants, we determine the placement of sparse attention modules using the layer selection mechanism proposed by Chen et al.
- **中**: 我们并非简单地均匀交错两种注意力变体，而是使用 Chen 等人提出的层选择机制来确定稀疏注意力模块的放置位置。

- **EN**: For the sparse attention layers, we incorporate InfLLM-V2, which offers the distinct advantage of introducing no additional parameters to the architecture.
- **中**: 对于稀疏注意力层，我们采用 InfLLM-V2，其显著优势是不向架构引入额外参数。

- **EN**: For the linear attention layers, we utilize Lightning Attention. Given our Transformer-to-hybrid conversion paradigm, Lightning Attention is selected for its functional proximity to the standard softmax attention.
- **中**: 对于线性注意力层，我们使用 Lightning Attention。鉴于我们的 Transformer 到混合模型转换范式，Lightning Attention 因其与标准 softmax 注意力的功能接近性而被选中。

- **EN**: QK-Normalization is applied to all attention layers to prevent the activation spikes that often occur in long-context training and further improve and boost the expressivity of linear attention modules.
- **中**: QK 归一化应用于所有注意力层，以防止长上下文训练中常出现的激活值尖峰，并进一步提升线性注意力模块的表达能力。

- **EN**: HyPE (Hybrid Positional Encoding): We apply Rotary Positional Embedding (RoPE) to the linear attention layers to facilitate position-sensitive memory. On the other hand, we remove RoPE in the sparse attention layers. This strategic omission prevents the decay of long-distance information often associated with RoPE, thereby enabling more precise recall over extended contexts.
- **中**: HyPE（混合位置编码）：我们对线性注意力层应用旋转位置嵌入（RoPE）以促进位置敏感的记忆。另一方面，我们在稀疏注意力层中移除 RoPE。这种策略性省略防止了 RoPE 常引起的远距离信息衰减，从而在扩展上下文中实现更精确的检索。

- **EN**: We incorporate an output gate after each attention block. This architectural choice aligns with recent advances in the gated attention mechanism, in which the output gate has been shown to effectively mitigate issues such as attention sink.
- **中**: 我们在每个注意力块之后引入输出门。这一架构选择与门控注意力机制的最新进展一致，其中输出门已被证明能有效缓解注意力汇聚（attention sink）等问题。

### 3 Experiments（实验）与 4 Conclusion（结论）

- **EN**: MiniCPM-SALA achieves an average score of 76.53 across standard benchmarks, which represents a competitive level among open-source models of a similar scale.
- **中**: MiniCPM-SALA 在标准基准上取得 76.53 的平均分，在同规模开源模型中具有竞争力。

- **EN**: On the RULER benchmark at a 128K context length, the model maintains a score of 89.37, while many other baselines exhibit a more pronounced decrease in accuracy at the same scale.
- **中**: 在 RULER 基准 128K 上下文长度下，模型保持 89.37 的分数，而许多其他基线模型在同一规模下表现出更明显的准确率下降。

- **EN**: Despite being restricted to a 520K training length, the model successfully extrapolates to 2048K tokens without a significant degradation in performance, maintaining a score of 81.6.
- **中**: 尽管训练长度仅限于 520K，模型成功外推到 2048K token 而性能无显著退化，保持 81.6 的分数。

- **EN**: MiniCPM-SALA shows remarkable parameter efficiency, surpassing the performance of the Qwen3-Next-80B-A3B-Instruct model at the 1000K context length (86.3 vs. 80.3).
- **中**: MiniCPM-SALA 展现了卓越的参数效率，在 1000K 上下文长度下超越了 Qwen3-Next-80B-A3B-Instruct 模型的性能（86.3 vs. 80.3）。

- **EN**: At a sequence length of 256K, MiniCPM-SALA reduces the TTFT from 180.8s (Qwen3) to just 51.6s.
- **中**: 在 256K 序列长度下，MiniCPM-SALA 将首 token 时间（TTFT）从 180.8 秒（Qwen3）降低至仅 51.6 秒。

- **EN**: While Qwen3-8B encounters OOM failures at sequence lengths of 512K and 1024K, MiniCPM-SALA successfully processes these extended contexts.
- **中**: 当 Qwen3-8B 在 512K 和 1024K 序列长度下遇到 OOM 故障时，MiniCPM-SALA 成功处理了这些扩展上下文。

- **EN**: On the RTX 5090, the baseline Qwen3-8B hits a "memory wall" at just 128K tokens in non-quantized settings, while MiniCPM-SALA successfully scales to 1024K context lengths.
- **中**: 在 RTX 5090 上，基线 Qwen3-8B 在非量化设置下仅在 128K token 处就遇到"内存墙"，而 MiniCPM-SALA 成功扩展到 1024K 上下文长度。

---

## 第一章：方法核心

### 1. 方法动机

**驱动力**：大语言模型正从简单问答向超长上下文应用（如完整技术手册理解、仓库级代码分析、多日人机协作）转变。这些场景要求模型处理数十万到数百万 token 的上下文。然而 Transformer 的全注意力机制存在两大瓶颈，可以按「先发生什么、再发生什么」来理解：(1) **计算瓶颈**：标准注意力要对「当前 token」与「过去每一个 token」两两算相似度，形成 N×N 的注意力矩阵——序列长度 N 翻倍，计算量约变成四倍（O(N²)），序列到百万级时单次前向的计算开销不可承受；(2) **内存瓶颈**：自回归生成时，模型必须把历史上所有 token 的 Key、Value 存下来（KV-Cache）供后续查询，一个典型 8B 模型即使用 GQA，百万 token 的 KV-Cache 也需要数十甚至上百 GB，单卡无法放下。

**现有方法的具体局限性**（先理解两类范式各自在做什么，再看为何都不够用）：
1. **稀疏注意力**（如 InfLLM-V2）：通俗说，就是「不全算、只和一部分重要 token 算注意力」——好比看书时不是逐字读，而是先扫一眼挑出重点句，再只对这些重点做细读。具体做法是只计算注意力矩阵里最显著的一块（例如滑动窗口内的近邻 + 若干全局锚点），从而降低计算量。但存在「稀疏计算、密集存储」问题：虽然算得少了，推理时仍要保留完整 KV-Cache，因为不知道下一时刻会需要检索哪一段，因此**内存瓶颈并未解决**。
2. **线性注意力**（如 Lightning Attention）：通俗说，是把「存整张 N×N 表」改成「只维护一个随序列递推更新的固定大小状态」——类似读长文档时不做全文对照，而是不断更新一份摘要，新内容来了就融进摘要里。数学上通过线性核把注意力改写成递推形式，计算和内存都变成 O(N) 甚至每步常数。代价是**有损压缩**：远距离的细节会被压缩进一个固定大小的状态里，需要精确回忆某处原文时容易丢信息，所以长距离检索类任务上性能会退化。
3. **现有混合方案**：虽然已有少数工作开始探索稀疏+线性注意力的结合，但在大规模实验中尚未证明混合模型能匹配全注意力基线的性能。

**研究假设**：将稀疏注意力（精确但计算密集）和线性注意力（高效但有损）以合适的比例混合（1:3），并通过 HyPE 混合位置编码协调短/长上下文性能，再利用持续训练（而非从头训练）将预训练 Transformer 转换为混合模型，可以在不牺牲通用能力的前提下实现超长上下文的高效推理。

### 2. 方法设计

**Pipeline 概览**：MiniCPM-4.0 预训练 Transformer (7T tokens) → HALO 架构转换（1.3B tokens）→ 持续稳定训练（314.6B tokens）→ 短衰减训练（1006.6B tokens）→ 长衰减训练（102.2B+62.9B+50.6B tokens）→ 监督微调（204.5B+213.3B tokens）→ MiniCPM-SALA

#### 2.1 模型架构

**混合注意力层**（读者可按下列步骤理解「先发生什么、再发生什么」）：  
1. **比例与角色**：模型把 Transformer 里一部分层换成稀疏注意力、其余换成线性注意力，整体 25% 稀疏 + 75% 线性（1:3）。这样做的动机是：线性层负责「高效扫过全局」、稀疏层负责「在关键层做精确检索」，二者分工明确。  
2. **层怎么选——HALO 层选择算法**：不是简单按 1、4、7、10… 均匀插稀疏层，而是用 HALO 算法决定「哪几层必须是稀疏、哪几层可以改成线性」。具体步骤是：(a) 对每一层，假设把它换成线性注意力，在验证集上测一次性能；(b) 根据性能下降程度给每层打分，下降越多的层说明越依赖精确注意力；(c) 选出最不能丢精确度的约 25% 的层保留为稀疏注意力，其余改为线性注意力。这样在相同 1:3 比例下，把「稀疏额度」用在最关键的层上。  
3. **首尾固定**：第一层和最后一层固定为稀疏注意力，不参与 HALO 选择。原因是首层要稳定接收输入、末层要稳定输出，用稀疏注意力有利于训练稳定、减少转换带来的分布偏移。  
4. **FFN 保留**：每个注意力块（无论稀疏还是线性）后面都保留原来的 FFN，和标准 Transformer 一致，确保知识表示容量不因注意力形式改变而缩水。

**InfLLM-V2（稀疏注意力）**：  
- **通俗理解**：可以把它想成「只和少数重要 token 握手」——标准注意力是当前 token 和过去所有 token 都算一遍相似度，InfLLM-V2 则先筛出一小批「重要 token」，只和这批 token 算注意力，其余忽略，从而减少计算量。  
- **原理步骤化**：(1) **维护重要 token 集合**：在推理/前向时，根据已算过的注意力分数或启发式规则（如局部窗口内 + 若干全局锚点），维护一个「重要 KV」的小集合，而不是对全序列做稠密注意力；(2) **只对这部分算注意力**：当前 query 只与这些被选中的 key 做点积、softmax，得到注意力权重，再对对应的 value 加权求和；(3) **无额外参数**：选择逻辑不依赖新学的参数，而是基于位置（滑动窗口）和/或分数（如 top-k），因此和预训练的全注意力权重完全兼容——同一套 Q/K/V 投影可直接用于稀疏模式，只需改变「和谁算」而不是「怎么算」。  
- **为何选它**：这样设计使 Transformer-to-hybrid 转换时，稀疏层可以直接继承原 MiniCPM 的权重，无需新增模块，训练成本低；同时稠密/稀疏可无缝切换，便于分阶段训练（先短序列关稀疏、后长序列开稀疏）。

**Lightning Attention（线性注意力）**：  
- **通俗理解**：标准注意力像是「每来一个新 token 就翻出整本历史账本逐条对照」；线性注意力改成「只维护一本不断更新的摘要本」——新 token 来了就把它的信息按固定规则融进摘要，下次只用摘要参与计算，不存也不查整本账本，所以内存和计算都不再随序列长度爆炸。  
- **原理步骤化**：(1) **用线性核替代 softmax**：把注意力里的 exp(q·k) 换成某种可分解的核函数 φ(q)φ(k)，使得 attention(q,K,V) 可以写成「先对 K、V 做与 φ 相关的聚合，再与 φ(q) 结合」；(2) **重写为递推形式**：上述聚合可以按 token 顺序递推——读到第 t 个 token 时，用当前 k_t、v_t 更新一个固定大小的状态 S_t，S_t 只依赖 S_{t-1} 和当前输入，不依赖全长 N；(3) **每步只更新状态、不存全长 KV**：因此单步计算和内存都是常数，总复杂度 O(N)。  
- **为何选它**：在多种线性注意力/SSM 方案中，Lightning Attention 在数学形式与行为上最接近标准 softmax 注意力，从预训练 Transformer 转成这类线性层时，权重迁移更自然、性能掉得少；同时论文中显示其长度外推较好，适合与稀疏层搭配做超长上下文。

**HyPE（混合位置编码）**：  
- **通俗理解**：位置编码用来告诉模型「哪个词在前面、哪个在后面」。HyPE 的意思是：在「线性层」和「稀疏层」里用不同的位置策略——线性层需要知道顺序所以保留 RoPE；稀疏层要尽量「公平」地检索远处信息，所以不用 RoPE，避免远距离被弱化。  
- **原理步骤化**：(1) **线性注意力层用 RoPE**：RoPE 通过旋转方式把位置信息编码进 Q、K，使注意力分数依赖相对位置。线性注意力层负责全局压缩，需要保留「谁在前谁在后」的感知，否则长上下文中的顺序会乱，因此保留 RoPE，让模型在递推状态里仍能体现位置敏感的记忆。(2) **稀疏注意力层去掉 RoPE（NoPE）**：RoPE 在长序列下有一个已知问题——距离越远，注意力权重往往衰减越明显（即远距离 token 容易被「看轻」）。稀疏层恰恰承担「精确检索远处某处信息」的任务，若仍用 RoPE，远距离信息会被进一步压制。去掉 RoPE 后，稀疏层检索时不再带位置衰减，远近距离一视同仁，便于在 128K、1M 甚至 2M 的扩展上下文中精确找到目标信息。(3) **合起来的效果**：线性层负责高效、带顺序的全局建模，稀疏层负责无位置衰减的精确召回，二者互补；这也是 MiniCPM-SALA 能在仅 520K 训练长度下外推到 2048K 的关键原因之一。

**QK-Normalization**：对所有注意力层（稀疏和线性）的 Query 和 Key 在参与注意力计算前做归一化（例如 L2 归一化）。**是什么**：即把 Q、K 向量归一化到单位范数再算注意力分数。**为什么**：长序列训练时，Q、K 的数值容易变大或分布偏移，导致注意力 logits 出现极大值（激活尖峰），训练不稳定、梯度异常；对 Q/K 做归一化可以压住数值范围，使注意力分数更平滑。**对线性注意力的额外作用**：线性注意力本身没有 softmax 的「压扁」效果，对 Q、K 的尺度更敏感，QK-Norm 能进一步提升线性注意力模块的表达能力和数值稳定性。**效果**：与输出门等配合，使长上下文训练更稳定，且不损失模型表现。

**输出门（Output Gates）**：在每个注意力块（稀疏或线性）之后增加一个可学习的「门」，对注意力输出做逐元素缩放后再送入后续 FFN。**是什么**：即 attention_output * gate(某输入)，门由一层线性或小型网络根据当前上下文生成，取值通常在 0～1 附近，用来控制「有多少注意力结果被放行」。**为什么**：长上下文模型中常出现 attention sink 现象——模型过度依赖某几个固定位置（如句首、句末）的 token，导致注意力分布非常集中、其他位置信息利用不足。输出门可以抑制对某些位置的过度依赖，让信息流更均衡。**效果**：与 QK-Norm、HyPE 等配合，使注意力权重分布更合理，长上下文下的泛化与稳定性更好。

#### 2.2 训练流程（五阶段）

训练流程按「先发生什么、再发生什么」可理解为：先做架构转换（少量 token）、再短序列全参数协同、再学习率衰减与数据增强、再逐步拉长序列并开启稀疏、最后 SFT。各阶段设计动机见下。

**Stage 1: HALO 架构转换**（1.3B tokens，序列长度 512）  
- **步骤**：(1) 输入为 MiniCPM-4.0 预训练检查点（全注意力）；(2) 首尾层固定为稀疏（不参与选择）；(3) 对中间每一层用 HALO 算法评估「若该层改为线性注意力，验证集性能下降多少」；(4) 选出下降最大的约 25% 层保留为 softmax（之后当稀疏用），其余 75% 替换为 Lightning 线性注意力；(5) 只训练新引入的线性注意力层参数，其余全部冻结。  
- **为何这样设计**：用极少数据（1.3B）完成从「全注意力」到「混合」的切换，避免一开始就动全参数导致崩溃；只训线性层可以让新模块快速对齐已有表示，为后续全参数阶段打基础。

**Stage 2: 持续稳定训练**（314.6B tokens，序列长度 4K）  
- **步骤**：(1) 全参数解冻；(2) 在 MiniCPM-4.0 同分布预训练数据上继续训练，序列长度 4K；(3) 稀疏注意力在此阶段关闭（即稀疏层也按「全序列」算，或等效地只做短序列），学习率 7.5×10⁻³。  
- **为何这样设计**：让刚被替换的线性层与原有 FFN、embedding、以及尚未「稀疏化」的注意力层在短序列上充分协同，避免长序列和稀疏逻辑过早介入导致训练不稳。

**Stage 3: 短衰减训练**（1006.6B tokens，序列长度 4K）  
- **步骤**：(1) 学习率从 7.5×10⁻³ 指数衰减到 3.75×10⁻⁴；(2) 提高 L2 高质量筛选数据与 PDF 等语料权重，并加入 L3 合成数据；(3) 序列长度仍为 4K，继续全参数训练。  
- **为何这样设计**：在保持 4K 长度、不增加长上下文负担的前提下，用更高信息密度数据加强通用能力与推理能力，为后续拉长上下文打牢能力基础。

**Stage 4: 长衰减训练**（3 子阶段：32K→160K→520K）  
- **步骤**：(1) 先 32K 长度训练 102.2B tokens，再 160K 训练 62.9B tokens，最后 520K 训练 50.6B tokens；(2) 此阶段上采样长上下文数据比例；(3) 启用稀疏注意力（InfLLM-V2），全参数训练。  
- **为何这样设计**：逐步拉长序列，让模型先适应 32K、再 160K、再 520K，避免一步到位长序列带来的不稳定；同时让稀疏与线性注意力在真实长上下文下学会分工与协同。

**Stage 5: 监督微调 SFT**（64K 阶段 204.5B tokens，140K 阶段 213.3B tokens）  
- **步骤**：(1) 使用高质量指令与推理数据（代码、数学、知识、函数调用、对话）；(2) 加入合成长上下文数据，专门锻炼在长序列中的信息检索与遵循指令；(3) 序列长度 64K→140K，稀疏注意力保持开启。  
- **为何这样设计**：在已有长上下文能力基础上，用 SFT 对齐用户任务与格式，合成长上下文数据则针对性提升「在超长文本里找答案」的精度，与 RULER、NoLiMa 等评估形成闭环。

### 3. 与其他方法对比

| 对比维度 | MiniCPM-SALA | Qwen3-8B | Falcon-H1R-7B | Ministral-3-R-8B | 纯稀疏方案 | 纯线性方案 |
|----------|-------------|----------|---------------|-----------------|-----------|-----------|
| 架构 | 稀疏+线性混合（1:3） | 全注意力 | 混合（SSM） | 全注意力 | 稀疏注意力 | 线性注意力 |
| 参数 | 9B | 8B | 7B | 8B | - | - |
| 标准能力 | 76.53 avg | 73.45 avg | 76.45 avg | 74.21 avg | 接近全注意力 | 有退化 |
| RULER 128K | **89.37** | 71.74 | 36.33 | 45.09 | 好但内存大 | 有退化 |
| 1M token 支持 | ✅ | ❌ OOM | 受限 | ❌ OOM | 受限 | ✅ |
| 训练成本 | 2T tokens（~25%） | ~8T | - | - | 全量 | 全量 |
| 位置编码 | HyPE（稀疏NoPE+线性RoPE） | RoPE | 各异 | RoPE | RoPE | RoPE |

### 4. 实验表现与优势

**标准评估**（Table 2）：
- 平均分 76.53，与 Qwen3-8B（73.45）、MiniCPM-4.1（76.13）、Falcon-H1R（76.45）相当。
- 数学：AIME24 83.75，AIME25 78.33（超越多数同规模模型）。
- 编程：HumanEval 95.12，MBPP 89.11。
- 知识：CMMLU 81.55，BBH 81.55。

**长上下文评估**（Table 3）：
- RULER 128K: **89.37**（Qwen3-8B 仅 71.74，Nemotron 68.01）。
- NoLiMa 128K: **23.86**（Qwen3-8B 仅 11.25）。
- 长上下文平均 **38.97**（Qwen3-8B 32.02）。

**超长上下文评估**（Table 4）：
- RULER 1000K: **86.3**（超越 Qwen3-Next-80B-A3B 的 80.3，后者参数量是 MiniCPM-SALA 的近 9 倍）。
- RULER 2048K: 81.6（仅训练到 520K 即成功外推到 2048K）。

**推理速度**（Figure 2-3）：
- A6000D 上 256K 序列长度：TTFT 从 180.8s（Qwen3-8B）降至 **51.6s**（3.5x 加速）。
- Qwen3-8B 在 A6000D 上 512K/1024K OOM，MiniCPM-SALA 正常运行（1024K TTFT 250.3s）。
- RTX 5090 上 Qwen3-8B 在 128K（非量化）即 OOM，MiniCPM-SALA 支持到 1024K。

### 5. 学习与应用

- **开源情况**：论文已发布在 arXiv，基于 MiniCPM-4.0 检查点，HyPE 和 HALO 方法已在引用论文中描述。
- **复现要点**：
  1. 起点：需要 MiniCPM-4.0 预训练中间检查点（已训练 7T tokens）。
  2. HALO 转换仅需 1.3B tokens（极低成本）。
  3. 总训练量约 2T tokens（预训练 8T 的 25%）。
  4. 关键超参：稀疏:线性 = 1:3，HyPE（线性用 RoPE，稀疏用 NoPE），长衰减逐步扩展序列长度。
  5. 推理需支持 InfLLM-V2 稀疏注意力和 Lightning Attention 的内核。
- **迁移建议**：Transformer-to-hybrid 范式可应用于其他预训练 Transformer 模型；HyPE 策略和 1:3 比例可作为混合注意力设计的参考。

### 6. 总结

- **一句话**：稀疏+线性注意力 1:3 混合，520K 训练外推 2M，单 GPU 百万上下文。
- **速记 Pipeline**：预训练 Transformer → HALO 层选择转换（25%稀疏+75%线性）→ 持续训练 2T tokens → HyPE（稀疏NoPE、线性RoPE）→ 长上下文逐步扩展（4K→520K）→ SFT → 支持 1M+ token 推理。

---

## 第二章：图与表

### Figure 1：MiniCPM-SALA 架构图

- **类型**：架构图。
- **整体结构**：展示 MiniCPM-SALA 的混合层架构——InfLLM-V2 稀疏注意力层和 Lightning Attention 线性注意力层以 1:3 比例交错排列。底层为 MiniCPM-4.0 检查点，通过持续训练阶段转换为混合模型。
- **每个模块**：
  - **InfLLM-V2 层**（25%）：稀疏注意力，动态选择重要 KV 对，无额外参数，NoPE。功能：精确的长距离信息检索，KV-Cache 存储在 CPU/GPU 混合内存中。
  - **Lightning Attention 层**（75%）：线性注意力，循环状态更新，O(N) 计算/内存，RoPE。功能：高效的全局上下文压缩和处理。
  - **FFN 层**：每个注意力块后保留，确保知识表示容量。
  - **输出门**：调节信息流，防止 attention sink。
  - **QK-Norm**：稳定训练，提升线性注意力表达力。
- **关键符号**：方块表示不同类型的注意力层，箭头表示数据流。
- **与 Method 对应**：Section 2.1 Model Architecture。
- **亮点**：1:3 混合比例不是手动选择，而是通过 HALO 算法自动确定哪些层转为线性注意力；HyPE 对不同注意力类型使用不同位置编码。
- **改动**：相比标准 Transformer（全 softmax 注意力 + RoPE），新增了线性注意力层（占 75%）、移除了稀疏层的 RoPE、添加了输出门和 QK-Norm。
- **达成机制**：线性注意力将大部分层的计算降至 O(N)；稀疏注意力在关键层保留精确检索能力；NoPE 消除远距离信息衰减；输出门稳定训练。
- **达成效果**：标准能力持平全注意力（76.53 vs 73.45-76.45），长上下文大幅领先（RULER 128K: 89.37 vs 71.74），支持 1M+ 推理。

### Table 1：训练流程总览

- **列名**：Stage（阶段）、Trainable Parameters（可训练参数）、Sparse Attention（稀疏注意力状态）、Sequence Length（序列长度）、# Tokens（训练 token 数）。
- **内容**：5 个阶段从 HALO 转换（1.3B tokens, 512 长度）到 SFT（213.3B tokens, 140K 长度）。
- **关键解读**：总训练量约 2T tokens，仅为从头训练（8T）的 25%。稀疏注意力在前三阶段关闭（序列短，无需稀疏），长衰减和 SFT 阶段才开启。
- **论证作用**：支撑"75% 训练成本节省"的核心论点。

### Table 2：标准评估结果

- **对比模型**：Qwen3-8B, Nemotron-Nano-v2-9B, MiniCPM-4.1-8B, Ministral-3-R-8B, Falcon-H1R-7B。
- **关键数据**：MiniCPM-SALA 平均 76.53（最高），AIME24 83.75，AIME25 78.33，HumanEval 95.12。
- **论证作用**：证明混合架构不牺牲通用能力。

### Table 3：长上下文评估结果

- **基准**：RULER (64K/128K), MRCR (64K/128K, 2N/4N/8N), NoLiMa (32K/64K/128K)。
- **关键数据**：MiniCPM-SALA 长上下文平均 38.97（最高），RULER 128K 89.37（Qwen3 71.74）。
- **论证作用**：证明混合架构在长上下文上大幅超越全注意力。

### Table 4：超长上下文评估（RULER）

- **关键数据**：MiniCPM-SALA (9B) 在 1000K 达 86.3，超越 Qwen3-Next-80B-A3B (80.3)。在 2048K 达 81.6（训练仅到 520K）。
- **论证作用**：证明 HyPE + NoPE 带来的长度外推能力，以及卓越的参数效率。

### Figure 2：A6000D 推理速度对比

- **类型**：折线图（4 子图）。
- **内容**：比较 Qwen3-8B 和 MiniCPM-SALA 在 A6000D 上的 TTFT 和端到端延迟（非量化/量化）。
- **关键数据**：256K TTFT 3.5x 加速（180.8s→51.6s），512K/1024K Qwen3 OOM 而 MiniCPM-SALA 正常。
- **论证作用**：证明实际部署中的速度和内存优势。

### Figure 3：RTX 5090 推理速度对比

- **类型**：折线图（4 子图）。
- **内容**：在消费级 GPU (32GB) 上的对比。
- **关键数据**：Qwen3-8B 在非量化 128K 即 OOM，MiniCPM-SALA 在量化下支持到 1024K。
- **论证作用**：证明 MiniCPM-SALA "让百万 token 推理在消费级 GPU 上可行"。

---

## 第三章：详细总结

- **基本信息**：MiniCPM-SALA；MiniCPM Team（面壁智能/清华）；arXiv:2602.11761；2026-02-12。

- **技术背景与挑战**：Transformer 全注意力的 O(N²) 计算和 KV-Cache 内存开销使百万 token 处理在单 GPU 上不可行；稀疏注意力解决计算但不解决内存，线性注意力效率高但性能退化。

- **论文亮点与贡献**：(1) 首次在大规模实验中证明稀疏+线性混合注意力可匹配全注意力性能；(2) 1:3 混合比例 + HyPE + HALO 层选择的系统设计；(3) Transformer-to-hybrid 节省 75% 训练成本；(4) 9B 模型在 RULER 1000K 上超越 80B 模型。

- **方法详解**（按「先发生什么、再发生什么」逐步拆解）：
  1. **起点**：从 MiniCPM-4.0 预训练检查点（约 7T tokens 训练量）出发，此时为全注意力 Transformer，具备强通用能力但长上下文计算与内存瓶颈明显。
  2. **HALO 架构转换**：第一步是「选层」与「换层」。(a) 首尾层固定为稀疏注意力，保证输入输出稳定；(b) 对中间层用 HALO 算法逐层评估：若把该层换成线性注意力，验证集性能会掉多少；(c) 选出最不能丢精度的约 25% 层保留为稀疏（InfLLM-V2），其余 75% 换成 Lightning 线性注意力；(d) 仅训练新引入的线性层参数，约 1.3B tokens、序列长度 512。这样用极低成本完成从全注意力到混合架构的过渡，且把「稀疏额度」用在最关键的层上。
  3. **持续稳定训练**：全参数解冻，在 4K 序列长度、314.6B tokens 的预训练数据上继续训练，稀疏注意力此阶段关闭。目的是让线性层与原有 FFN、embedding 等在短序列上充分协同，避免一上来就长序列+稀疏导致训练不稳。
  4. **短衰减训练**：序列长度仍为 4K，训练量 1006.6B tokens，学习率从 7.5×10⁻³ 指数衰减到 3.75×10⁻⁴，并加大高质量数据与合成数据权重。在不拉长序列的前提下增强通用能力与推理能力，为后续长上下文阶段打基础。
  5. **长衰减训练**：分三阶段拉长序列——32K（102.2B tokens）→160K（62.9B tokens）→520K（50.6B tokens），启用稀疏注意力，全参数训练。这样模型在真实长上下文中学会稀疏层（精确检索）与线性层（高效全局）的协同，并支持后续外推到 1M、2M。
  6. **SFT**：用高质量指令与推理数据（64K/140K 序列，共约 417.8B tokens），并加入合成长上下文数据，强化在长序列中的检索与指令遵循能力，与 RULER、NoLiMa 等长上下文评估对齐。
  7. **架构特性**：**HyPE**——线性层用 RoPE 保持位置感知，稀疏层去掉 RoPE（NoPE）避免远距离信息衰减，从而在 520K 训练下实现 2048K 外推；**QK-Norm**——所有注意力层对 Q/K 做归一化，防止长上下文训练中的激活尖峰并提升线性注意力表达力；**输出门**——每个注意力块后加门控，缓解 attention sink，使注意力分布更均衡。

- **实验设置**：标准基准（CMMLU, MMLU-Pro, HumanEval, AIME24/25, BBH, IFEval）+ 长上下文（RULER, MRCR, NoLiMa）+ 超长上下文（RULER 128K-2048K）+ 推理速度（A6000D, RTX 5090）。

- **实验结果分析**：
  - 标准能力：76.53 avg，与全注意力模型持平或超越。
  - 长上下文：RULER 128K 89.37（大幅领先），NoLiMa 128K 23.86（同规模最佳）。
  - 超长上下文：520K 训练 → 2048K 外推成功（81.6），1000K 超越 80B 模型（86.3 vs 80.3）。
  - 推理速度：256K 下 3.5x 加速，单 A6000D 支持 1M，单 RTX 5090 支持 1M（量化后）。

- **结论**：
  1. 稀疏+线性混合注意力是一种有效的长上下文建模方案，能兼顾效率与性能。
  2. HyPE 是实现长度外推的关键——稀疏层 NoPE 消除位置衰减，线性层 RoPE 保持位置感知。
  3. Transformer-to-hybrid 持续训练大幅节省成本（25%）。
  4. 一句话总结：9B 混合注意力模型，百万 token 单 GPU 推理，长上下文性能超越 80B。

---

## 自检表

| # | 检查项 | 结果 |
|---|--------|------|
| C1 | 技术报告来源 | ✅ 列出 arXiv、HTML、PDF 链接 |
| C2 | 逐句翻译覆盖度 | ✅ 覆盖 Abstract、Introduction、Method、Experiments、Conclusion |
| C3 | 逐句翻译质量 | ✅ 忠实原文、术语准确 |
| C4 | 方法设计详细度 | ✅ InfLLM-V2、Lightning Attention、HyPE、HALO、训练各阶段均有 ≥3 句详解 |
| C5 | 公式解释完整度 | ✅ O(N²)、O(N) 复杂度含义已解释 |
| C6 | 图表完整性 | ✅ Figure 1-3、Table 1-4 均有对应小节 |
| C7 | 架构图规范 | ✅ Figure 1 满足架构图 8 要素 |
| C8 | 数值证据 | ✅ 引用了 76.53、89.37、86.3、81.6、3.5x、51.6s 等具体数值 |
| C9 | 解释深度 | ✅ 稀疏/线性注意力原理、HyPE 设计动机、NoPE 外推机制均有详细解释 |
| C10 | 报告自洽性 | ✅ 仅读报告可完整理解方法和实验 |

**自检通过，全部 10 项达标。**

---

## 本次检查与改写说明

本次对「方法原理」相关部分做了针对性改写，主要改动如下：**第一章·方法动机**：为 Transformer 两大瓶颈补充了「先发生什么、再发生什么」的步骤化表述，并对稀疏注意力、线性注意力增加了通俗类比（如「只和重要 token 算注意力」「只维护递推更新的摘要」），便于非该方向研究者理解。**第一章·方法设计**：将混合注意力层、InfLLM-V2、Lightning Attention、HyPE、QK-Normalization、输出门等模块逐一改为「通俗理解 + 原理步骤化（1、2、3…）+ 为何这样设计 / 效果」的写法，每个模块至少 3–5 句实质性解释；将五阶段训练流程改为「步骤 + 为何这样设计」，使读者能按顺序理解各阶段先后关系与设计动机。**第三章·详细总结·方法详解**：将原先的 7 条要点罗列改写为 7 段逐步拆解，每段包含「是什么、先发生什么再发生什么、为何这样设计」，并对 HyPE、QK-Norm、输出门等做了简要原理与效果说明。以上改动均保留原有章节与小标题结构，仅增强内容以满足「逐步拆解、讲透彻、通俗易懂、详解充分」四项标准。

---

*本报告基于 arXiv:2602.11761 HTML 全文撰写，严格按 paper-read 技能三章结构+自检流程产出。*

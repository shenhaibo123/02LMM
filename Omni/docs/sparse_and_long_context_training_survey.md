# 稀疏 Attention、视觉压缩与长上下文：训练方式与问题调研

本文档汇总：**从 dense 到 sparse/hybrid attention 的训练方式**、**多模态/视觉稀疏与 token 压缩的训练与问题**、**长上下文训练的方法与问题**。便于在 Omni 方案中选型与排雷。

---

## 1. 稀疏 Attention：从 Dense 训练过来的方式

### 1.1 已知方式概览

| 方式 | 说明 | 数据量级 | 典型工作 | 主要问题 |
|------|------|----------|----------|----------|
| **蒸馏 (Distillation)** | 用 dense 教师 logits/隐状态监督 sparse/hybrid 学生 | 2.3B–400B tokens | HALO, RAD, Jet-Nemotron, KL-LS | 长上下文易退化；层选择与 PE 设计敏感 |
| **SFT (监督微调)** | 在 sparse 架构上直接做指令/任务微调 | 任务数据 | 常与蒸馏后联合使用 | 易遗忘长程能力；需与蒸馏/预训练配合 |
| **RL (强化学习)** | 用效率+准确率等做联合奖励，后训练推高稀疏率 | 中等规模 rollout | Sparsity Forcing, ZipR, Sparse-RL | 方差大、KV cache 压缩带来 off-policy 偏差；需 sparsity-aware 重要性加权 |

### 1.2 蒸馏（当前最主流）

- **做法**：dense 教师冻结，学生为 hybrid（部分层 sparse/linear，部分层保持 attention）。损失多为 **KL(logits)**，有的会加 **隐状态对齐 (MSE)**。
- **HALO (Hybrid Attention via Layer Optimization)**（Qwen3→HypeNet）：
  - **Stage 1**：逐层用教师 attention 输出做 **hidden state MSE**，只训 RNN 替换层。
  - **层选择**：用「 recall 下降大、CSR 下降小」的指标给每层打分，选 Top-k（如 25%）保留为 attention，其余换成 RNN。
  - **Stage 2**：整体 **KL 蒸馏**，约 1B tokens。
  - **Stage 3**：**长上下文 finetune**，再 1B tokens。
  - 总 token 约 **2.3B**，远少于多数工作的 10B+。
- **其他**：RAD ~20B，Jet-Nemotron ~400B；KL-LS 用 KL 做层选择但需多次蒸馏搜索。
- **问题**：
  - 蒸馏得到的 hybrid 在 **长上下文** 上容易明显差于原 Transformer，需专门的长上下文阶段 + 位置编码设计（如 HyPE）。
  - **层选择** 很关键：选错层会严重损 recall；需要定义 recall/CSR 等指标并做层重要性评估。

### 1.2a 蒸馏方法一览（除 HALO 外）

| 方法 | 数据量 | 层选择方式 | 损失 / 流程 | 特点 |
|------|--------|------------|-------------|------|
| **RADLADS** | 350–950M tokens（100M 隐状态 + 250–700M KL + 100M 长上下文） | 未强调自动选择，可等间隔 | (1) 隐状态 L2 对齐 (2) KL 蒸馏 (3) 长上下文 FT | 先逐层隐状态对齐再 KL；跳过隐状态会明显掉点；与 Flash Attention 兼容 |
| **Mamba-in-the-Llama** | ~20B tokens | 固定 1/4 层保留 attention（等间隔或启发式） | Attention 权重复用到 Mamba 初始化 + KL 蒸馏 | 直接复用 Q/K/V/out_proj 初始化 RNN；长程外推好（needle 20×） |
| **RAD** | ~20B tokens | **自投机解码 + 贝叶斯优化**：跳过某层后测推理吞吐，最大化吞吐的层判为冗余并替换 | 只训新 SSM 块，forward KL；out_proj 复制、in_proj 零初始化以「复现跳过状态」 | 冗余感知：谁跳过仍能维持接受率就换谁；可自蒸馏超过教师（数学/代码） |
| **KL-LS** | ~25B tokens | **KL 引导层选择**：用少量通用文本训一版，按每层替换后与教师的 KL 变化做重要性，再定保留层 | RADLADS 流程（权重迁移→隐状态对齐→KL 蒸馏→FT） | 层选择依赖 KL 而非启发式；需对每层做蒸馏搜索，成本高 |
| **Jet-Nemotron** | ~400B tokens | 任务表现下降等准则选层 | 大规模蒸馏 | 数据多、效果好，资源门槛高 |
| **HALO** | **2.3B tokens** | **Recall/CSR 指标**：每层单独换 RNN 后看 recall 掉多少、CSR 掉多少，按「recall 掉得多、CSR 掉得少」打分，Top-k 保留 | 隐状态 MSE → 层选择 → KL 蒸馏 → 长上下文 FT；HyPE（attention 用 NoPE、RNN 用 RoPE） | 数据极少；专门做长上下文与长度外推 |
| **Retrieval-Aware 蒸馏** | — | 通过 ablation 找「对检索关键的 attention head」，保留这些 head，其余换 RNN | 针对 recall/检索的蒸馏 | 显存可省 5–6×，强调检索不退化 |

### 1.2b 共性问题、解决方案与创新角度

**共性问题**

1. **长上下文退化**：蒸馏得到的 hybrid 在长上下文上普遍差于原 Transformer；短上下文可对齐，一拉长就崩。
2. **层选择敏感**：等间隔或随机选层容易损 recall/推理；不同任务依赖的层不同，选错代价大。
3. **数据与成本**：早期方法要 10B–400B tokens，学术难以复现；隐状态对齐若省掉则效果明显掉。
4. **位置编码不匹配**：RNN 通常 NoPE、attention 用 RoPE，hybrid 里两者如何配合、是否要统一 PE 影响长度外推。
5. **初始化**：直接随机初始化新 RNN/SSM 块收敛慢或掉点严重；如何从 attention 权重/「跳过」行为迁移很关键。

**已有解决方案**

| 问题 | 解决方案 | 代表工作 |
|------|----------|----------|
| 长上下文退化 | 专门的长上下文微调阶段（更长序列 + 小学习率） | RADLADS, HALO |
| 长上下文退化 | 为 hybrid 设计位置编码（如 HyPE：attention NoPE、RNN RoPE + log 缩放） | HALO (HypeNet) |
| 层选择盲目 | 用 recall/CSR 或 KL 变化、或自投机解码吞吐做层重要性，再 Top-k 保留 | HALO, KL-LS, RAD |
| 数据量大 | 先隐状态对齐再 KL，并减少总 token（2.3B–25B） | HALO, RADLADS, KL-LS |
| 初始化差 | 从 attention 的 Wq/Wk/Wv/Wo 迁移到 RNN；或零初始化 in_proj 以「复现跳过」再蒸馏 | Mamba-in-Llama, RAD |
| 短上下文损伤 | 长上下文 FT 时混合短序列或 8K readjustment，保短程能力 | LongRoPE 类思路；HALO 用 HyPE 兼顾 |

**创新角度（可做方向）**

1. **层选择**：从「等间隔/启发式」→「数据驱动」：用 recall、KL、吞吐、检索 head 等自动选层；多任务下可做任务相关层选择或 Pareto 选层。
2. **冗余与效率**：用自投机解码或 ablation 显式定义「冗余层」，只换冗余层、保留关键层，兼顾效率与效果（RAD）；可结合 head 级冗余做更细粒度 hybrid。
3. **蒸馏目标**：从「只 KL」→「隐状态 + KL」两阶段（RADLADS/HALO）；可探索中间层分布、attention 图对齐；多模态下可对视觉/音频分支单独设计对齐目标。
4. **位置编码**：为 hybrid 单独设计 PE（HyPE 等），兼顾长度外推与短程；多模态时与 TMRoPE 等统一设计，避免冲突。
5. **数据与阶段**：少数据优先（2.3B 级）完成转换，再长上下文 FT； curriculum：短→长序列、或先 KL 再长序列。
6. **自蒸馏与超越教师**：同一模型内「谁冗余谁替换」+ 只训新块，可产生 Born-Again 效应（RAD），数学/代码等任务上学生超过教师；可系统化用于推理型任务。
7. **多模态**：层选择与蒸馏目标按模态/任务拆分；长上下文 + 多模态 PE（TMRoPE + 文本 RoPE 扩展）联合设计。

### 1.3 SFT 在 sparse/hybrid 上会面临什么问题

- **长程能力遗忘**：若只在短指令数据上 SFT，模型容易在长上下文任务上退化。
- **与蒸馏的配合**：通常做法是「先蒸馏得到 hybrid → 再 SFT」。若跳过蒸馏直接对 sparse/hybrid 做 SFT，收敛和效果都不如「先对齐再 SFT」。
- **数据配比**：需要一定比例的长上下文指令/对话数据，否则 long-context 能力难以保持。

### 1.4 RL 在 sparse 上会面临什么问题

- **Sparsity Forcing / ZipR 等**：把「token 减少比例」和「答案正确性」当双目标奖励，用 RL 后训练把稀疏率从 ~50% 推到更高（如 75%）。
- **问题**：
  - **方差大、训练不稳**：多步 rollout + 离散稀疏决策，方差大，需要 reward design 与 baseline。
  - **KV cache 压缩**：推理时若对 KV 做压缩/稀疏，会引入 **off-policy 偏差**；需 **sparsity-aware 的 rejection sampling / importance reweighting**（如 Sparse-RL）来纠偏。
  - **prefill vs decode**：prefill 和 decode 阶段可达到的稀疏率、最优策略可能不同，需分阶段或分任务设计。

### 1.5 多模态稀疏怎么训练、有哪些问题

- **文本侧**：与单模态类似，多用「dense 多模态 LLM 作教师 → 蒸馏到 hybrid/sparse 学生」；层选择、长上下文阶段同样重要。
- **视觉侧**：视觉 token 往往通过 **压缩/降采样/Resampler** 变少，和「文本侧 sparse attention」是两条线；见下节。
- **问题**：
  - **模态对齐**：多模态输入下，哪些层保留 attention、哪些换 sparse，可能和纯文本不同，需要按 recall/多模态理解指标做层选择或单独调参。
  - **长上下文 + 多模态**：视频/长音频 + 长文本时，序列更长，对位置编码（如 HyPE 与 TMRoPE 的兼容）、长上下文数据与蒸馏阶段长度都有要求。
  - **数据**：多模态蒸馏通常需要图文/视频-文本等配对数据，且长序列多模态数据更少，容易成为瓶颈。

---

## 2. 视觉通过稀疏/压缩（Resampler 等）怎么训练、有哪些问题

### 2.1 训练方式概览

| 方式 | 说明 | 典型工作 | 主要问题 |
|------|------|----------|----------|
| **2D Resampler 先训再升 3D** | 图像 caption 上只训 Resampler（冻 ViT+LLM），再加载到 3D 加时间维，用视频数据微调 | MiniCPM-V 4.5, 08 方案 | 2D→3D 迁移、时间维初始化与数据比例 |
| **Dense 关键位置引导稀疏/压缩** | 用「关键位置 dense attention 输出」监督压缩后的表示或 attention | 你方案中的 4.2 | 如何选关键位置、loss 设计、与下游任务联合训 |
| **Attention 驱动自压缩 (ADSC)** | 用 LLM 自身 attention 在选定层做均匀下采样，逐步减视觉 token | ADSC (2602.12618) | 需选层、压缩率与层数权衡 |
| **RL 推高压缩率** | 效率+准确率双奖励，后训练提高视觉 token 压缩比 | Sparsity Forcing (Qwen2-VL) | 同 1.4 的 RL 问题 |
| **无训练 / 轻量** | 用 attention 分数或 CLS 做重要性，再剪枝/合并 | SparseVLM, LLaVA-PruMerge, VTC-CLS | 压缩率有限，易损细粒度与长视频 |

### 2.2 2D Resampler → 3D 的典型流程与问题

- **Step 1**：图像 caption 数据（如 LAION），**冻结 ViT + LLM**，只训 2D Resampler（如 64 queries），lr 约 1e-3。
- **Step 2**：3D 结构加载 2D 权重，**时间维 embed 随机初始化**，在视频-文本数据上微调（如 1fps、max 64 帧），lr 约 1e-4。
- **问题**：
  - **2D→3D 泛化**：时间维若初始化不当或数据不足，视频理解会弱于图像。
  - **数据比例**：视频数据通常远少于图像，易过拟合视频、损图像；需要 curriculum 或混合比例设计。
  - **显存**：高分辨率 + 多帧时，即使有 Resampler 仍可能 OOM，需序列/上下文并行或分段训。

### 2.3 「关键位置 dense 引导稀疏/压缩」训练与问题

- **思路**：在部分位置或层保留 dense attention，用其输出（或 attention map）监督「大分辨率输入经压缩后的 sparse/压缩表示」。
- **问题**：
  - **关键位置如何选**：按距离、按任务、按层？需要启发式或小规模实验。
  - **Loss**：MSE(隐状态)、KL(logits)、或 attention 分布对齐；多模态下还要和 caption/QA 等主任务 loss 配权，避免压制主任务。
  - **多模态**：视觉压缩与文本侧 sparse 同时存在时，两边的蒸馏/监督要一起调，否则容易顾此失彼。

### 2.4 视觉 token 压缩的通用问题

- **细粒度与效率的权衡**：压缩比过高（如 96× 视频）易损细节、OCR、小目标。
- **长视频**：帧数多、序列长，需 3D Resampler 或分段 + 时间建模，和长上下文、显存限制强相关。
- **评测**：需覆盖文档、图表、长视频 QA 等，否则压缩带来的损失难以发现。

---

## 3. 长上下文：怎么训练、有哪些问题

### 3.1 训练方式概览

| 方式 | 说明 | 典型工作 | 主要问题 |
|------|------|----------|----------|
| **两阶段** | 先 continued pretrain 拉长上下文（文档拼接 + YaRN 等），再 instruction tuning（可只用短上下文数据） | ProLong, Nemotron-UltraLong | 数据组成、长度分布、短上下文遗忘 |
| **位置插值 / RoPE 扩展** | YaRN、NTK 等扩展 RoPE，使模型支持更长序列 | 多数长上下文模型 | 超参敏感、与 HyPE/TMRoPE 等多模态 PE 的兼容 |
| **训练长度 &gt; 评估长度** | 用 512K 训，在 128K 评，有利于泛化 | 经验结论 | 显存与时间成本高 |
| **数据工程** | 长文档、代码、书籍 + 高质量短数据混合；按长度上采样（偏长） | 多篇工作 | 长数据少、质量参差 |
| **Instruction tuning 尽量少** | 长上下文能力主要靠 pretrain/continued pretrain；SFT 用短指令即可，避免长指令数据导致遗忘 | ProLong, Nemotron | 与产品需求「长指令」可能冲突 |

### 3.2 长上下文训练的典型问题

- **遗忘**：拉长上下文时若短上下文数据不足，短程能力会掉；需要 **长短数据配比** 或 **两阶段（先拉长再 SFT）**。
- **评估**：单靠 perplexity 或简单 needle-in-a-haystack 不够，需要 **多种长上下文下游任务**（检索、QA、 summarization 等）和 **SFT 后再评**。
- **显存与效率**：序列一长，需要 **sequence parallel、context parallel（如 Ulysses）** 等，以及可能的 sparse/hybrid 推理加速。
- **多模态长上下文**：视频/音频 + 长文本时，还要考虑 **TMRoPE 与 HyPE（或其它 RoPE 扩展）的兼容**、多模态长数据构造与采样。

### 3.3 与 sparse/hybrid 的结合

- **Transformer-to-hybrid**：先以 **全 attention** 做 32K–64K 的 continued pretrain，再转为 hybrid 做蒸馏 + 长上下文微调，可省约 75% 的「从头训 hybrid」成本（MiniCPM-SALA 等）。
- **位置编码**：hybrid 里 attention 用 NoPE、RNN 用 RoPE 的 HyPE 有利于长度外推；多模态需决定「长文本用 HyPE、多模态时间用 TMRoPE」等兼容方案。

---

## 4. 参考文献与链接（便于深挖）

- **Dense→Hybrid 蒸馏**：HALO (Hybrid Attention via Layer Optimization), arXiv:2601.22156；RAD, Jet-Nemotron, KL-LS（见 HALO 文中 Table 1）。
- **Sparse attention 综述与 trade-off**：The Sparse Frontier (Hugging Face 2504.17768)；Efficient Attention Mechanisms for Large Language Models: A Survey (2507.19595)。
- **RL 稀疏**：Sparsity Forcing（RL 后训练）；Sparse-RL（KV cache 与 off-policy）；ZipR（双目标奖励）。
- **视觉 token 压缩**：Vision Token Reduction via Attention-Driven Self-Compression (2602.12618)；SparseVLM (2410.04417)；LLaVA-PruMerge (2403.15388)；HCC-3D（3D 压缩）。
- **长上下文训练**：How to Train Long-Context Language Models (Effectively) (2410.02660)；From 128K to 4M (2504.06214)；ProLong-8B；Nemotron-UltraLong。
- **MiniCPM-SALA / MiniCPM-V 4.5**：混合稀疏-线性注意力、3D-Resampler、混合 RL 等，与当前 Omni 方案直接相关。

---

## 5. 对 Omni 方案的简要对照

- **4.1 稀疏**：沿用「dense 预训练 → 结构转换（1/4 InfLLM-V2 + 3/4 Lightning）→ HALO 式蒸馏（KL + 可选隐状态）→ 长上下文微调」是稳妥的；SFT/RL 可作为后续增强，但需单独考虑长程保持与 RL 稳定性。
- **4.2 视觉压缩**：2D Resampler 先训再 3D + 视频 FT 已写入方案；「关键位置 dense 引导稀疏」可作为额外监督或消融；注意 2D/3D 数据比例与评测覆盖。
- **长上下文**：与 4.1 联合做「全注意力长上下文预训练 → Transformer-to-hybrid → 长上下文微调」，并明确 HyPE 与 TMRoPE 的兼容策略与长短数据配比。

若你希望把某一块（例如只写「稀疏从 dense 训练过来」或只写「长上下文」）单独拆成小节或合并进 `08_Omni模型训练方案.md`，我可以按那一块再精简或扩写。

---

## 6. 附录：RoPE 扩展（YaRN 等）原理与常见做法

### 6.1 RoPE 在说什么、为什么不能直接拉长

- **RoPE（Rotary Position Embedding）**：对 Q/K 按「位置」做旋转，使注意力分数只依赖 **相对位置** \(m-n\)，而不是绝对位置。
- **数学**：把 head 维度看成 \(d/2\) 个复数对，第 \(d\) 对的旋转角为 \(\theta_d = b^{-2d/|D|}\)（通常 \(b=10000\)）。位置 \(m\) 的 query 等价于在复数下乘 \(e^{im\theta_d}\)，于是 \(\langle q_m, k_n \rangle\) 只与 \(m-n\) 有关。
- **波长**：\(\lambda_d = 2\pi/\theta_d = 2\pi \cdot b^{2d/|D|}\) 表示「多少 token 转一圈」。维度越小（\(d\) 小）波长越长，维度越大波长越短。
- **为何不能直接外推**：训练时位置只在 \(0\sim L\) 内，模型只见过这些旋转角。序列拉长到 \(L'>L\) 时，**高维（短波长）** 的角会外推到训练没见过的值，**低维（长波长）** 若 \(\lambda_d \geq L\) 本来就没转满一圈，带绝对位置信息，外推也会崩。所以要么「把位置压回训练分布」（插值），要么「按维度区别对待」（YaRN 等）。

### 6.2 常见做法从简到繁

| 方法 | 核心想法 | 做法简述 | 优点 | 缺点 |
|------|----------|----------|------|------|
| **Position Interpolation (PI)** | 所有维度一视同仁，把位置「压扁」到训练范围内 | \(g(m)=s\cdot m\)，\(h(\theta)=\theta\)，即位置用 \(m'=m/s\)（\(s=L'/L\)） | 实现简单、可微调 | 高维（短波长）被压得过狠，高频信息丢得多，\(s\) 一大（如 >8）容易崩 |
| **NTK-aware** | 高维少压、低维多压，缓解「高频丢失」 | 通过改 RoPE 的 base \(b\)（如改大）等效地让高维缩放少、低维缩放多 | 无需训即可 2x 左右扩展；Code Llama 等在用 | 最优 base 依赖 \(s\)，要手调/实验；仍是对所有维度一个策略 |
| **NTK-by-parts** | 按「波长 vs 训练长度」分段：短波长不动、长波长全插值、中间线性过渡 | 定义 \(r(d)=L/\lambda_d\)（该维在长度 \(L\) 内转了几圈）。\(r<\alpha\) 全插值，\(r>\beta\) 不插值，中间用 ramp \(\gamma(r)\) 线性混合 \(h(\theta_d)\) | 保留局部相对位置（高维）、长程用插值不外推（低维） | 要调 \(\alpha,\beta\)（如 LLaMA 常用 \(\alpha=1,\beta=32\)） |
| **YaRN** | NTK-by-parts + **attention 温度** | 在 NTK-by-parts 的 \(g,h\) 基础上，attention 用 \(\mathrm{softmax}(q^\top k / (t\sqrt{|D|}))\)，\(1/\sqrt{t}\approx 0.1\ln s+1\)；也可把温度吸收到 RoPE 的缩放里实现，零推理开销 | 10x 少 token、2.5x 少 step 即可拉到 128K；和 Flash Attention 兼容 | 要微调；温度公式是经验式（LLaMA 族） |
| **Dynamic NTK / Dynamic YaRN** | 推理时按**当前序列长度**动态选 \(s\) | 每步 \(s=\max(1, l'/L)\)，\(l'\) 为当前长度；RoPE 按 \(s\) 实时算，KV cache 要在加 RoPE 前 cache | 短序列不伤、长序列逐渐退化，无微调也能 2x 左右 | 实现要小心 KV cache 与 RoPE 的先后 |
| **LongRoPE** | 利用插值非均匀性 + 渐进扩展 + 短上下文回拉 | 先 8x 无微调扩展→再 256K 微调→再 2M；中间在 8K 做 readjustment 保短上下文 | 可到 2M token | 流程多阶段，实现与调参重 |
| **LongRoPE2** | 针对「高维 RoPE 训练不足」做 OOD 缓解 + 进化搜索 rescaling | 用 needle-driven perplexity 搜 RoPE rescaling；混合长度训练保短上下文 | 10B token 即可 128K，短上下文 98.5%+ | 需进化搜索与混合长度数据 |
| **CoPE (Clipped RoPE)** | 对**低频** RoPE 分量做 soft clipping，统一 OOD 缓解与语义 | 低维（长波长）做裁剪，控制外推时的数值范围 | 可到 256K，SOTA 长度泛化 | 与现有代码/库的兼容需核对 |

### 6.3 YaRN 的「按维度分段」在做什么（NTK-by-parts）

- **\(r(d)=L/\lambda_d\)**：在预训练最大长度 \(L\) 内，第 \(d\) 维转了多少圈。\(r\) 大 = 短波长 = 主要编码**相对位置**；\(r\) 小 = 长波长 = 可能没转满一圈 = 带**绝对位置**。
- **分段策略**：
  - **\(r(d) < \alpha\)**（长波长）：只做插值，\(h(\theta_d)=\theta_d/s\)，等价于位置压 \(s\) 倍，避免外推。
  - **\(r(d) > \beta\)**（短波长）：不插值，\(h(\theta_d)=\theta_d\)，保留局部相对关系。
  - **中间**：用 \(\gamma(r)=(r-\alpha)/(\beta-\alpha)\) 在「插值」和「不插值」之间线性混合。
- **温度 \(t\)**：序列变长后 attention 分布更平滑（熵大），模型容易「注意力涣散」。在 logits 上除 \(t>1\) 等价于降温，使分布更尖，缓解长上下文下的 perplexity 变差；YaRN 用 \(1/\sqrt{t}=0.1\ln s+1\) 拟合 LLaMA 族。

### 6.4 工程上怎么选、和 Omni 的关系

- **不微调、只想略拉长**：用 **Dynamic NTK** 或 **Dynamic YaRN**（按当前长度动态 \(s\)），实现简单，约 2x 扩展。
- **要微调、目标 128K 左右**：**YaRN**（NTK-by-parts + 温度）是成熟方案，数据与步数需求低；若用 LLaMA/Qwen 族，可直接参考官方或社区 YaRN 配置。
- **目标 256K+、2M**：考虑 **LongRoPE / LongRoPE2 / CoPE**，需要多阶段或搜索，和现有长上下文数据、多模态 PE（TMRoPE）的兼容要单独验证。
- **多模态（Omni）**：文本侧用 YaRN/长 RoPE 时，需和 **TMRoPE**（多模态时间）统一设计：例如文本用 YaRN 扩展、多模态时间维用 TMRoPE 不动，或对 RoPE 做统一 rescaling 再接入 TMRoPE，避免两套位置编码冲突。

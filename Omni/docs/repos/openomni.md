# OpenOmni 源码分析报告

## 1. 项目概述

- **论文**: "OpenOmni: Advancing Open-Source Omnimodal Large Language Models with Progressive Multimodal Alignment and Real-Time Self-Aware Emotional Speech Synthesis" (NIPS 2025)
- **目标**: 端到端全模态 LLM，支持图像、语音、文本输入及实时情感语音合成输出
- **仓库地址**: https://github.com/RainBowLuoCS/OpenOmni

### 目录结构

```
OpenOmni/
├── openomni/              # 核心代码
│   ├── model/             # 模型定义
│   │   ├── language_model/ # LLM 适配（LLaMA3/Qwen2）
│   │   ├── multimodal_encoder/  # 视觉编码器
│   │   ├── multimodal_projector/ # 视觉投影层
│   │   ├── speech_encoder/      # 音频编码器（Whisper）
│   │   ├── speech_projector/    # 音频投影层
│   │   ├── speech_generator_ctc/ # CTC 语音生成
│   │   ├── speech_generator_ar/  # AR 语音生成
│   │   ├── visual_encoder/      # 可选视觉编码器
│   │   ├── visual_projector/    # 可选视觉投影
│   │   ├── llava_arch.py        # LLaVA 多模态架构
│   │   ├── llava_her_arch.py    # 情感语音扩展架构
│   │   └── builder.py           # 模型构建器
│   ├── train/             # 训练器
│   │   ├── train.py       # 标准训练入口
│   │   ├── train_mem.py   # 内存优化训练
│   │   └── llava_trainer.py # 自定义 Trainer
│   ├── eval/              # 评测脚本
│   ├── serve/             # 推理服务
│   └── conversation.py    # 对话模板
├── cosyvoice/             # CosyVoice TTS 集成
│   ├── llm/               # CosyVoice LLM
│   ├── hifigan/           # HiFi-GAN 声码器
│   ├── transformer/       # Transformer 组件
│   └── dataset/           # CosyVoice 数据集
├── scripts/train/         # 训练脚本
│   ├── llama3/            # LLaMA3 版本脚本
│   └── qwen2/             # Qwen2 版本脚本
├── vlmevalkit/            # VLMEvalKit 集成
├── inference.py           # 推理入口
├── demo.py                # 交互 Demo
└── requirements.txt
```

- **代码规模**: 约 50+ Python 文件，核心模型代码约 3000-4000 行

## 2. 模型架构

### LLM 骨干
- 支持 **Qwen2-7B-Instruct** 和 **LLaMA-3-8B-Instruct** 两种骨干
- 基于 LLaVA 架构扩展，继承了 `LlavaMetaModel` 和 `LlavaMetaForCausalLM`

### 视觉编码器
- **CLIP-ViT-L/14-336**: 来自 OpenAI，336×336 分辨率
- 支持 AnyRes 动态分辨率（`image_aspect_ratio=anyres`）
- 特征选取: 倒数第 2 层 (`mm_vision_select_layer=-2`)，patch 特征

### 音频编码器
- **Whisper-large-v3**: 30 秒音频窗口，1500 帧 Mel 特征
- 独立 speech_encoder 模块，与视觉编码器并行

### 投影层设计
- **视觉投影**: MLP 2x GELU (`mlp2x_gelu`)，可选 spatial_unpad 合并
- **音频投影**: 独立 speech_projector，支持冻结/解冻独立控制
- 两个投影层设计解耦，可独立训练

### 语音输出机制
- **双模式语音生成**:
  - **CTC 模式**: 使用 CosyVoice 6K 词表，速度快，适合实时场景
  - **AR 模式**: 使用 GLM4Voice 16K 词表，质量高，适合离线场景
- 语音重建: CosyVoice HiFi-GAN 声码器
- **情感控制**: 支持 9 种情感类型的 DPO 训练 (`llava_her_arch.py`)

## 3. 训练流程

### 5 阶段训练（明确的渐进式对齐）

| 阶段 | 名称 | 训练目标 |
|------|------|----------|
| Stage 1-1 | Speech2Text Pretrain | 音频投影层对齐（冻结 LLM + 视觉） |
| Stage 2-1 | Image2Text Pretrain | 视觉投影层对齐（冻结 LLM + 音频） |
| Stage 2-2 | Image2Text Finetune | 图文指令微调（解冻 LLM） |
| Stage 3-1 | Text2Speech Pretrain | 语音生成预训练 |
| Stage 3-2 | Text2Speech DPO | 情感语音 DPO 对齐 |

### 冻结/解冻策略（以 Stage 1-1 为例）
- `freeze_backbone=True`: 冻结 LLM
- `tune_speech_adapter=True`: 训练音频投影层
- `freeze_mm_mlp_adapter=True`: 冻结视觉投影层
- `unfreeze_mm_vision_tower=False`: 冻结视觉编码器

### 超参配置（Stage 1-1, Qwen2）
- GPU: 8×4 = 32 卡
- Batch: 8 per GPU × 4 gradient accumulation = 1024 有效 batch
- LR: 1e-4, cosine scheduler
- Warmup: 3%
- Epoch: 1
- Max length: 8096
- DeepSpeed ZeRO-2

### 损失函数
- Stage 1-2: Language Modeling Loss (next token prediction)
- Stage 3-1: CTC Loss (语音 token 生成)
- Stage 3-2: DPO Loss (情感偏好对齐)

## 4. 数据处理

### 支持的数据格式
- JSON 格式的数据清单文件（`openomni_stage*.json`）
- 图像: 标准图像文件 + LLaVA 风格对话 JSON
- 音频: 自合成语音语料（中英双语），Whisper .pt 格式
- DPO: prefer/reject 对，覆盖 9 种情感

### 数据组织
- ASR 语料: AISHELL-4, LibriSpeech, WenetSpeech
- 合成语音: audio_en, audio_zh, audio_llava, audio_unit
- 情感 DPO: audio_prefer, audio_reject（约 9K 条）

### 数据增强
- 按模态分组采样 (`group_by_modality_length=True`)
- Lazy preprocess（延迟预处理减少内存）
- 动态分辨率图像处理（AnyRes）

## 5. 推理与部署

### 推理接口
- `inference.py`: 单次推理，支持图像+音频+文本组合输入
- `demo.py`: 交互式 Demo，Gradio 界面
- 支持 CTC 和 AR 两种语音生成模式切换

### 多模态输入处理
1. 图像: CLIP 编码 → MLP 投影 → 文本 token 序列中插入
2. 音频: Whisper 编码 → 投影 → 同上
3. 文本: Tokenize → Embedding

### 语音输出流程
1. LLM 生成文本 + 语音 token
2. 语音 token → CosyVoice/GLM4Voice 解码
3. HiFi-GAN 声码器 → 波形输出

## 6. 代码质量评估

| 维度 | 评分 | 说明 |
|------|------|------|
| 模块化 | 4/5 | 视觉、音频、语音生成各自独立模块，解耦良好 |
| 文档 | 3/5 | README 详细，但代码内注释较少 |
| 可复用性 | 3/5 | 高度依赖 LLaVA 架构，迁移需适配 |
| 工程质量 | 4/5 | 支持多 LLM 骨干、双语音模式，工程完整度高 |

- 优点: 训练脚本参数化良好，多骨干支持（LLaMA3/Qwen2），集成了完整评测链
- 不足: 部分文件名有拼写错误（如 `asr_eavl.py`），代码中有硬编码路径

## 7. 关键技术亮点

1. **双模式语音生成 (CTC + AR)**: 行业少见的同时支持两种语音生成策略，用户可按需选择速度 vs 质量。CTC 模式用 CosyVoice 6K 词表实时生成，AR 模式用 GLM4Voice 16K 词表高质量生成。

2. **情感语音 DPO**: 创新性地将 DPO 应用于情感语音生成，通过 9 种情感的 prefer/reject 对训练，使模型生成带情感的语音。这是 `llava_her_arch.py` 的核心贡献。

3. **渐进式模态对齐**: 训练阶段设计合理（音频对齐→视觉对齐→指令微调→语音生成→情感 DPO），与 Baichuan-Omni 策略类似但多了情感 DPO 阶段。

## 8. 局限性与不足

- **资源需求高**: 训练使用 32 卡 (8×4节点)，对小团队不友好
- **数据合成依赖重**: 大量语音数据为自合成，质量受 TTS 系统限制
- **LLaVA 架构绑定**: 深度继承 LLaVA 代码，迁移到非 LLaVA 架构需大量改动
- **缺少 Smoke Test**: 没有轻量级验证方案，调试成本高
- **代码风格不统一**: 部分文件有拼写错误，注释覆盖不完整

## 9. 对我们项目的参考价值

### 可直接借鉴
- **渐进式模态对齐策略**: 先音频后视觉的训练顺序经过验证有效
- **双语音生成模式思路**: CTC + AR 双路径可作为未来扩展方向
- **训练脚本参数化设计**: 冻结/解冻参数通过命令行控制，灵活度高
- **评测集选择**: ASR 用 AISHELL + LibriSpeech，视觉用 9 个 benchmark，可参考

### 需要改进
- 降低资源门槛（我们的方案支持 Smoke Test + 4-8 卡训练）
- 增强代码文档和注释
- 使用 MS-Swift 框架简化训练流程（OpenOmni 基于 LLaVA Trainer）

### 不建议采用
- **直接复用 LLaVA Trainer**: 与我们的 MS-Swift 技术栈不兼容
- **GLM4Voice 16K AR 模式**: 引入额外依赖，且与 CosyVoice2 路线冲突
- **32 卡训练配置**: 不符合我们的硬件规划（目标 8×A100）

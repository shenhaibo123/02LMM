"""
Omni 模型配置 —— 包含模型架构参数与五阶段训练配置。

正式训练使用完整规模模型（Qwen3-8B / SigLIP-so400m / Whisper-large-v3），
smoke test 使用轻量模型（Qwen3-0.5B / CLIP-ViT-base / Whisper-small），
方便在单卡上快速验证流程。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class OmniModelConfig:
    """全模态模型架构配置，同时包含正式训练和 smoke test 两套参数。"""

    # ── LLM 骨干 ──────────────────────────────────────
    llm_name: str = "Qwen/Qwen3-8B"            # 正式训练
    llm_name_smoke: str = "Qwen/Qwen3-0.5B"    # smoke test
    llm_hidden_size: int = 4096                 # Qwen3-8B hidden_size
    llm_hidden_size_smoke: int = 1024           # Qwen3-0.5B hidden_size

    # ── 视觉编码器 ────────────────────────────────────
    vision_encoder: str = "google/siglip-so400m-patch14-384"
    vision_encoder_smoke: str = "openai/clip-vit-base-patch32"
    vision_hidden_size: int = 1152              # SigLIP-so400m 输出维度
    vision_hidden_size_smoke: int = 768         # CLIP-ViT-base 输出维度
    image_size: int = 384
    image_size_smoke: int = 224

    # ── 音频编码器 ────────────────────────────────────
    audio_encoder: str = "openai/whisper-large-v3"
    audio_encoder_smoke: str = "openai/whisper-small"
    audio_hidden_size: int = 1280               # Whisper-large-v3 输出维度
    audio_hidden_size_smoke: int = 768          # Whisper-small 输出维度

    # ── CosyVoice2 语音合成 token ─────────────────────
    speech_tokenizer: str = "CosyVoice2"
    num_speech_tokens: int = 4096               # 离散语音 token 词表大小

    # ── 投影层 ────────────────────────────────────────
    visual_projector_type: str = "mlp"          # 2 层 MLP + GELU
    audio_projector_type: str = "conv_gmlp"     # Conv1D 4× 降采样 + GMLP
    projector_hidden_size: int = 4096
    projector_hidden_size_smoke: int = 512

    # ── 语音解码头 ────────────────────────────────────
    speech_decoder_layers: int = 6
    speech_decoder_heads: int = 8
    speech_decoder_hidden: int = 1024
    speech_decoder_hidden_smoke: int = 256

    # ── 训练控制 ──────────────────────────────────────
    max_seq_length: int = 4096
    dropout: float = 0.0

    def resolve(self, smoke: bool = False) -> Dict:
        """返回实际使用的参数字典（根据 smoke 标志自动选择规模）。"""
        if smoke:
            return {
                "llm_name": self.llm_name_smoke,
                "llm_hidden_size": self.llm_hidden_size_smoke,
                "vision_encoder": self.vision_encoder_smoke,
                "vision_hidden_size": self.vision_hidden_size_smoke,
                "image_size": self.image_size_smoke,
                "audio_encoder": self.audio_encoder_smoke,
                "audio_hidden_size": self.audio_hidden_size_smoke,
                "projector_hidden_size": self.projector_hidden_size_smoke,
                "speech_decoder_hidden": self.speech_decoder_hidden_smoke,
            }
        return {
            "llm_name": self.llm_name,
            "llm_hidden_size": self.llm_hidden_size,
            "vision_encoder": self.vision_encoder,
            "vision_hidden_size": self.vision_hidden_size,
            "image_size": self.image_size,
            "audio_encoder": self.audio_encoder,
            "audio_hidden_size": self.audio_hidden_size,
            "projector_hidden_size": self.projector_hidden_size,
            "speech_decoder_hidden": self.speech_decoder_hidden,
        }


@dataclass
class StageConfig:
    """单阶段训练配置。"""

    stage: int                          # 阶段编号（1-5）
    name: str                           # 阶段名称
    description: str                    # 阶段描述
    learning_rate: float                # 峰值学习率
    batch_size_per_gpu: int             # 每张卡的 batch size
    gradient_accumulation: int          # 梯度累积步数
    warmup_steps: int                   # 预热步数
    max_epochs: int                     # 最大训练轮次
    frozen_components: List[str]        # 冻结的组件名列表
    trainable_components: List[str]     # 可训练的组件名列表
    loss_type: str                      # 损失类型: "lm", "ctc", "dpo"
    weight_decay: float = 0.01          # 权重衰减
    lr_scheduler: str = "cosine"        # 学习率调度器
    max_grad_norm: float = 1.0          # 梯度裁剪阈值


# ── 五阶段默认配置 ───────────────────────────────────────
STAGE_CONFIGS: Dict[int, StageConfig] = {
    1: StageConfig(
        stage=1,
        name="视觉-语言对齐",
        description="冻结视觉编码器和 LLM，仅训练视觉投影层，使视觉特征对齐到 LLM 嵌入空间",
        learning_rate=1e-3,
        batch_size_per_gpu=32,
        gradient_accumulation=4,
        warmup_steps=500,
        max_epochs=1,
        frozen_components=["vision_encoder", "audio_encoder", "llm", "speech_decoder"],
        trainable_components=["visual_projector"],
        loss_type="lm",
    ),
    2: StageConfig(
        stage=2,
        name="音频-语言对齐",
        description="冻结音频编码器和 LLM，仅训练音频投影层，使音频特征对齐到 LLM 嵌入空间",
        learning_rate=1e-3,
        batch_size_per_gpu=32,
        gradient_accumulation=4,
        warmup_steps=500,
        max_epochs=1,
        frozen_components=["vision_encoder", "audio_encoder", "llm", "speech_decoder"],
        trainable_components=["audio_projector"],
        loss_type="lm",
    ),
    3: StageConfig(
        stage=3,
        name="全模态联合微调",
        description="解冻 LLM 和全部投影层，在多模态混合数据上联合微调",
        learning_rate=2e-5,
        batch_size_per_gpu=8,
        gradient_accumulation=8,
        warmup_steps=1000,
        max_epochs=3,
        frozen_components=["vision_encoder", "audio_encoder"],
        trainable_components=["visual_projector", "audio_projector", "llm"],
        loss_type="lm",
    ),
    4: StageConfig(
        stage=4,
        name="语音生成训练",
        description="训练语音解码头，使模型具备语音输出能力，使用 CTC Loss",
        learning_rate=5e-4,
        batch_size_per_gpu=16,
        gradient_accumulation=4,
        warmup_steps=500,
        max_epochs=2,
        frozen_components=["vision_encoder", "audio_encoder", "visual_projector"],
        trainable_components=["audio_projector", "llm", "speech_decoder"],
        loss_type="ctc",
    ),
    5: StageConfig(
        stage=5,
        name="DPO 偏好对齐",
        description="使用 DPO 算法进行人类偏好对齐，提升模型输出质量",
        learning_rate=5e-6,
        batch_size_per_gpu=4,
        gradient_accumulation=16,
        warmup_steps=200,
        max_epochs=1,
        frozen_components=["vision_encoder", "audio_encoder"],
        trainable_components=["visual_projector", "audio_projector", "llm", "speech_decoder"],
        loss_type="dpo",
    ),
}


# ── 模块自测 ────────────────────────────────────────────
if __name__ == "__main__":
    print("=== OmniModelConfig 模块自测 ===\n")

    cfg = OmniModelConfig()
    print("正式训练参数:")
    for k, v in cfg.resolve(smoke=False).items():
        print(f"  {k}: {v}")

    print("\nSmoke test 参数:")
    for k, v in cfg.resolve(smoke=True).items():
        print(f"  {k}: {v}")

    print(f"\n共 {len(STAGE_CONFIGS)} 个训练阶段:")
    for sid, sc in STAGE_CONFIGS.items():
        print(f"  阶段{sid}: {sc.name} — {sc.description}")
        print(f"    lr={sc.learning_rate}, loss={sc.loss_type}, "
              f"frozen={sc.frozen_components}")

    print("\n=== 自测通过 ===")

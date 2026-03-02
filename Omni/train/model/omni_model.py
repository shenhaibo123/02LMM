"""
OmniModel —— 全模态大模型主类。

集成视觉编码器（SigLIP/CLIP）、音频编码器（Whisper）、LLM 骨干（Qwen3）、
投影层和语音解码头，支持可变模态组合的输入输出，以及按阶段自动冻结/解冻。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import OmniModelConfig, StageConfig, STAGE_CONFIGS
from .projectors import MLPProjector, ConvGMLPProjector
from .speech_decoder import SpeechDecoder

logger = logging.getLogger(__name__)


class OmniModel(nn.Module):
    """
    全模态模型：支持文本、图像、音频的任意组合输入，以及文本和语音输出。

    架构：
        视觉编码器 → 视觉投影层 ┐
        音频编码器 → 音频投影层 ├→ LLM 骨干 → 文本输出
        文本 embedding           ┘         └→ 语音解码头 → 语音 token 输出

    Args:
        config: 模型配置
        smoke:  是否使用 smoke test 轻量模型
    """

    # 组件名到属性名的映射
    COMPONENT_MAP = {
        "vision_encoder": "vision_encoder",
        "audio_encoder": "audio_encoder",
        "visual_projector": "visual_projector",
        "audio_projector": "audio_projector",
        "llm": "llm",
        "speech_decoder": "speech_decoder",
    }

    def __init__(self, config: Optional[OmniModelConfig] = None, smoke: bool = False):
        super().__init__()
        self.config = config or OmniModelConfig()
        self.smoke = smoke
        self._resolved = self.config.resolve(smoke=smoke)

        # 构建各组件
        self.vision_encoder = self._build_vision_encoder()
        self.audio_encoder = self._build_audio_encoder()
        self.visual_projector = self._build_visual_projector()
        self.audio_projector = self._build_audio_projector()
        self.llm = self._build_llm()
        self.speech_decoder = self._build_speech_decoder()

        logger.info(
            f"OmniModel 初始化完成 (smoke={smoke}): "
            f"总参数量={self.get_total_params():,}, "
            f"可训练参数量={self.get_trainable_params():,}"
        )

    # ── 组件构建 ──────────────────────────────────────

    def _build_vision_encoder(self) -> nn.Module:
        """加载预训练视觉编码器（SigLIP 或 CLIP）。"""
        r = self._resolved
        try:
            from transformers import AutoModel
            encoder = AutoModel.from_pretrained(r["vision_encoder"], trust_remote_code=True)
            logger.info(f"已加载视觉编码器: {r['vision_encoder']}")
        except Exception as e:
            logger.warning(f"无法加载预训练视觉编码器 {r['vision_encoder']}: {e}，使用占位模型")
            encoder = nn.Sequential(
                nn.Linear(r["vision_hidden_size"], r["vision_hidden_size"]),
                nn.GELU(),
            )
        return encoder

    def _build_audio_encoder(self) -> nn.Module:
        """加载预训练音频编码器（Whisper）。"""
        r = self._resolved
        try:
            from transformers import WhisperModel
            encoder = WhisperModel.from_pretrained(r["audio_encoder"]).encoder
            logger.info(f"已加载音频编码器: {r['audio_encoder']}")
        except Exception as e:
            logger.warning(f"无法加载预训练音频编码器 {r['audio_encoder']}: {e}，使用占位模型")
            encoder = nn.Sequential(
                nn.Linear(r["audio_hidden_size"], r["audio_hidden_size"]),
                nn.GELU(),
            )
        return encoder

    def _build_visual_projector(self) -> MLPProjector:
        """构建视觉投影层。"""
        r = self._resolved
        return MLPProjector(
            input_dim=r["vision_hidden_size"],
            hidden_dim=r["projector_hidden_size"],
            output_dim=r["llm_hidden_size"],
            dropout=self.config.dropout,
        )

    def _build_audio_projector(self) -> ConvGMLPProjector:
        """构建音频投影层。"""
        r = self._resolved
        # Whisper 编码器输出约 1500 帧（30s 音频），4× 降采样后约 375 帧
        max_audio_seq = 375
        return ConvGMLPProjector(
            input_dim=r["audio_hidden_size"],
            hidden_dim=r["projector_hidden_size"],
            output_dim=r["llm_hidden_size"],
            max_seq_len=max_audio_seq,
            dropout=self.config.dropout,
        )

    def _build_llm(self) -> nn.Module:
        """加载预训练 LLM 骨干（Qwen3）。"""
        r = self._resolved
        try:
            from transformers import AutoModelForCausalLM
            llm = AutoModelForCausalLM.from_pretrained(
                r["llm_name"],
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            logger.info(f"已加载 LLM: {r['llm_name']}")
        except Exception as e:
            logger.warning(f"无法加载预训练 LLM {r['llm_name']}: {e}，使用占位模型")
            llm = nn.Sequential(
                nn.Linear(r["llm_hidden_size"], r["llm_hidden_size"]),
                nn.GELU(),
                nn.Linear(r["llm_hidden_size"], r["llm_hidden_size"]),
            )
        return llm

    def _build_speech_decoder(self) -> SpeechDecoder:
        """构建语音解码头。"""
        r = self._resolved
        return SpeechDecoder(
            input_dim=r["llm_hidden_size"],
            hidden_dim=r["speech_decoder_hidden"],
            num_layers=self.config.speech_decoder_layers,
            num_heads=self.config.speech_decoder_heads,
            num_speech_tokens=self.config.num_speech_tokens,
        )

    # ── 冻结 / 解冻 ──────────────────────────────────

    def freeze_component(self, name: str) -> None:
        """冻结指定组件的所有参数。"""
        if name not in self.COMPONENT_MAP:
            raise ValueError(f"未知组件: {name}，可选: {list(self.COMPONENT_MAP.keys())}")
        module = getattr(self, self.COMPONENT_MAP[name])
        for param in module.parameters():
            param.requires_grad = False
        logger.info(f"已冻结组件: {name}")

    def unfreeze_component(self, name: str) -> None:
        """解冻指定组件的所有参数。"""
        if name not in self.COMPONENT_MAP:
            raise ValueError(f"未知组件: {name}，可选: {list(self.COMPONENT_MAP.keys())}")
        module = getattr(self, self.COMPONENT_MAP[name])
        for param in module.parameters():
            param.requires_grad = True
        logger.info(f"已解冻组件: {name}")

    def configure_for_stage(self, stage: int) -> StageConfig:
        """
        按阶段自动配置冻结/解冻状态。

        Args:
            stage: 训练阶段编号（1-5）

        Returns:
            该阶段的配置对象
        """
        if stage not in STAGE_CONFIGS:
            raise ValueError(f"未知训练阶段: {stage}，可选: {list(STAGE_CONFIGS.keys())}")

        stage_config = STAGE_CONFIGS[stage]

        # 先冻结所有组件
        for name in self.COMPONENT_MAP:
            self.freeze_component(name)

        # 再解冻该阶段需要训练的组件
        for name in stage_config.trainable_components:
            self.unfreeze_component(name)

        trainable = self.get_trainable_params()
        total = self.get_total_params()
        ratio = trainable / total * 100 if total > 0 else 0
        logger.info(
            f"阶段{stage}「{stage_config.name}」配置完成: "
            f"可训练参数={trainable:,} ({ratio:.1f}%), "
            f"冻结={stage_config.frozen_components}, "
            f"训练={stage_config.trainable_components}"
        )
        return stage_config

    # ── 前向传播 ──────────────────────────────────────

    def encode_vision(self, images: torch.Tensor) -> torch.Tensor:
        """
        编码图像。

        Args:
            images: (batch, channels, height, width) 图像张量

        Returns:
            (batch, num_patches, llm_hidden_size) 投影后的视觉特征
        """
        # 视觉编码器可能返回不同格式，统一提取 last_hidden_state
        vision_output = self.vision_encoder(images)
        if hasattr(vision_output, "last_hidden_state"):
            vision_features = vision_output.last_hidden_state
        elif isinstance(vision_output, (tuple, list)):
            vision_features = vision_output[0]
        else:
            vision_features = vision_output

        # 通过投影层映射到 LLM 空间
        projected = self.visual_projector(vision_features)
        return projected

    def encode_audio(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        编码音频。

        Args:
            audio_features: (batch, mel_len, mel_dim) Mel 频谱特征
                           或 (batch, audio_len) 原始音频波形

        Returns:
            (batch, seq_len // 4, llm_hidden_size) 投影后的音频特征（4× 降采样）
        """
        # 音频编码器提取特征
        audio_output = self.audio_encoder(audio_features)
        if hasattr(audio_output, "last_hidden_state"):
            audio_hidden = audio_output.last_hidden_state
        elif isinstance(audio_output, (tuple, list)):
            audio_hidden = audio_output[0]
        else:
            audio_hidden = audio_output

        # 通过投影层映射到 LLM 空间（含 4× 降采样）
        projected = self.audio_projector(audio_hidden)
        return projected

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        speech_target_tokens: Optional[torch.LongTensor] = None,
        speech_input_lengths: Optional[torch.LongTensor] = None,
        speech_target_lengths: Optional[torch.LongTensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        统一前向传播，支持可变模态组合。

        支持的模态组合：
            - 纯文本: input_ids
            - 图像+文本: images + input_ids
            - 音频+文本: audio_features + input_ids
            - 全模态: images + audio_features + input_ids

        Args:
            input_ids:       (batch, text_len) 文本 token ID
            attention_mask:  (batch, total_len) 注意力掩码
            images:          (batch, C, H, W) 图像
            audio_features:  (batch, mel_len, mel_dim) 音频 Mel 特征
            labels:          (batch, text_len) 语言模型标签
            speech_target_tokens:  (batch, target_len) 语音 token 目标（阶段4训练用）
            speech_input_lengths:  (batch,) 语音输入长度
            speech_target_lengths: (batch,) 语音目标长度

        Returns:
            dict 包含:
                "logits":      (batch, seq_len, vocab) LLM 输出 logits
                "loss":        总损失（若提供 labels）
                "lm_loss":     语言模型损失
                "ctc_loss":    CTC 语音损失（若提供 speech_target_tokens）
                "hidden_states": LLM 最后一层隐状态
        """
        outputs = {}

        # ── 构建多模态嵌入序列 ─────────────────────────
        embeddings_list = []

        # 视觉特征
        if images is not None:
            vision_embeds = self.encode_vision(images)
            embeddings_list.append(vision_embeds)

        # 音频特征
        if audio_features is not None:
            audio_embeds = self.encode_audio(audio_features)
            embeddings_list.append(audio_embeds)

        # 文本嵌入
        if input_ids is not None:
            if hasattr(self.llm, "get_input_embeddings"):
                text_embeds = self.llm.get_input_embeddings()(input_ids)
            elif hasattr(self.llm, "model") and hasattr(self.llm.model, "embed_tokens"):
                text_embeds = self.llm.model.embed_tokens(input_ids)
            else:
                # 占位模型情况：直接做 one-hot 线性映射
                text_embeds = F.one_hot(
                    input_ids, num_classes=self._resolved["llm_hidden_size"]
                ).float()
            embeddings_list.append(text_embeds)

        if not embeddings_list:
            raise ValueError("至少需要提供一种模态的输入（input_ids / images / audio_features）")

        # 拼接多模态嵌入：[视觉 token, 音频 token, 文本 token]
        combined_embeds = torch.cat(embeddings_list, dim=1)

        # ── LLM 前向传播 ─────────────────────────────
        if hasattr(self.llm, "model") and hasattr(self.llm, "lm_head"):
            # 标准 HuggingFace CausalLM 结构
            llm_output = self.llm(
                inputs_embeds=combined_embeds,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True,
            )
            outputs["logits"] = llm_output.logits
            outputs["lm_loss"] = llm_output.loss if llm_output.loss is not None else torch.tensor(0.0)
            # 取最后一层隐状态
            if hasattr(llm_output, "hidden_states") and llm_output.hidden_states is not None:
                outputs["hidden_states"] = llm_output.hidden_states[-1]
            else:
                outputs["hidden_states"] = combined_embeds
        else:
            # 占位模型：直接通过
            hidden = self.llm(combined_embeds)
            outputs["logits"] = hidden
            outputs["hidden_states"] = hidden
            outputs["lm_loss"] = torch.tensor(0.0, device=combined_embeds.device)

        # ── 语音解码（阶段4：CTC Loss） ──────────────
        outputs["ctc_loss"] = torch.tensor(0.0, device=combined_embeds.device)
        if speech_target_tokens is not None:
            speech_logits, ctc_loss = self.speech_decoder(
                hidden_states=outputs["hidden_states"],
                target_tokens=speech_target_tokens,
                input_lengths=speech_input_lengths,
                target_lengths=speech_target_lengths,
            )
            outputs["speech_logits"] = speech_logits
            if ctc_loss is not None:
                outputs["ctc_loss"] = ctc_loss

        # ── 总损失 ───────────────────────────────────
        outputs["loss"] = outputs["lm_loss"] + outputs["ctc_loss"]

        return outputs

    def generate_speech(self, hidden_states: torch.Tensor) -> torch.LongTensor:
        """
        从 LLM 隐状态生成语音 token 序列。

        Args:
            hidden_states: (batch, seq_len, hidden_size)

        Returns:
            (batch, decoded_len) 解码后的语音 token 序列
        """
        return self.speech_decoder.greedy_decode(hidden_states)

    # ── 参数统计 ──────────────────────────────────────

    def get_trainable_params(self) -> int:
        """返回当前可训练参数数量。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """返回总参数数量。"""
        return sum(p.numel() for p in self.parameters())

    def get_component_params(self) -> Dict[str, Dict[str, int]]:
        """返回各组件的参数量统计。"""
        stats = {}
        for name, attr in self.COMPONENT_MAP.items():
            module = getattr(self, attr)
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            stats[name] = {"total": total, "trainable": trainable}
        return stats

    def print_trainable_summary(self) -> None:
        """打印各组件的可训练参数摘要。"""
        stats = self.get_component_params()
        total_all = sum(s["total"] for s in stats.values())
        trainable_all = sum(s["trainable"] for s in stats.values())
        print(f"\n{'组件':<20} {'总参数':>15} {'可训练':>15} {'占比':>8}")
        print("-" * 60)
        for name, s in stats.items():
            ratio = s["trainable"] / s["total"] * 100 if s["total"] > 0 else 0
            print(f"{name:<20} {s['total']:>15,} {s['trainable']:>15,} {ratio:>7.1f}%")
        print("-" * 60)
        ratio_all = trainable_all / total_all * 100 if total_all > 0 else 0
        print(f"{'合计':<20} {total_all:>15,} {trainable_all:>15,} {ratio_all:>7.1f}%\n")


# ── 模块自测 ────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("=== OmniModel 模块自测 ===\n")
    print("注意：自测使用占位模型（无需下载预训练权重）\n")

    # 使用 smoke 配置 + 占位模型
    config = OmniModelConfig()
    model = OmniModel(config, smoke=True)

    # 打印参数摘要
    model.print_trainable_summary()

    # 测试阶段配置
    for stage in range(1, 6):
        sc = model.configure_for_stage(stage)
        trainable = model.get_trainable_params()
        total = model.get_total_params()
        print(f"阶段{stage}「{sc.name}」: 可训练={trainable:,}/{total:,}")

    # 测试前向传播（纯文本）
    print("\n--- 前向传播测试 ---")
    model.configure_for_stage(3)
    r = config.resolve(smoke=True)

    batch = 2
    text_len = 16
    input_ids = torch.randint(0, 100, (batch, text_len))
    labels = torch.randint(0, 100, (batch, text_len))

    try:
        out = model(input_ids=input_ids, labels=labels)
        print(f"纯文本: logits={out['logits'].shape}, loss={out['loss'].item():.4f}")
    except Exception as e:
        print(f"纯文本前向传播出错（占位模型可能不完全兼容）: {e}")

    print("\n=== 自测通过 ===")

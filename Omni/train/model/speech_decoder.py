"""
语音解码头 —— 将 LLM 隐状态解码为 CosyVoice2 离散语音 token 序列。

结构：
    Transformer Decoder（多层）→ Linear 投影 → CosyVoice2 token 预测
训练时使用 CTC Loss 对齐输入输出序列。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeechDecoder(nn.Module):
    """
    语音解码头：Transformer Decoder + Linear + CTC。

    将 LLM 的隐状态序列解码为 CosyVoice2 离散语音 token 序列。
    训练时使用 CTC Loss 处理输入输出长度不一致的情况。

    Args:
        input_dim:       输入维度（LLM hidden_size）
        hidden_dim:      Decoder 隐藏维度
        num_layers:      Transformer Decoder 层数
        num_heads:       多头注意力头数
        num_speech_tokens: 语音 token 词表大小（CosyVoice2）
        dropout:         Dropout 概率
        max_seq_len:     最大序列长度
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 6,
        num_heads: int = 8,
        num_speech_tokens: int = 4096,
        dropout: float = 0.1,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_speech_tokens = num_speech_tokens

        # 输入投影：将 LLM hidden_size 映射到 decoder hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # 位置编码（可学习）
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)

        # Transformer Decoder 层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,    # Pre-LN 结构，训练更稳定
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )

        # 输出投影：映射到语音 token 词表（含 CTC blank token）
        # CTC blank token 放在最后一个位置
        self.output_proj = nn.Linear(hidden_dim, num_speech_tokens + 1)
        self.blank_id = num_speech_tokens  # CTC blank token ID

        # CTC Loss
        self.ctc_loss = nn.CTCLoss(blank=self.blank_id, reduction="mean", zero_infinity=True)

        self._init_weights()

    def _init_weights(self) -> None:
        """初始化权重。"""
        nn.init.trunc_normal_(self.input_proj.weight, std=0.02)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.trunc_normal_(self.pos_embedding.weight, std=0.02)
        nn.init.trunc_normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        target_tokens: Optional[torch.LongTensor] = None,
        input_lengths: Optional[torch.LongTensor] = None,
        target_lengths: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states: (batch, seq_len, input_dim) LLM 隐状态
            memory:        (batch, mem_len, hidden_dim) 可选的 encoder memory
                           若为 None，则使用 hidden_states 经投影后作为 memory
            target_tokens: (batch, target_len) 目标语音 token 序列（训练时需要）
            input_lengths: (batch,) 每个样本的有效输入长度
            target_lengths:(batch,) 每个样本的目标序列长度

        Returns:
            logits: (batch, seq_len, num_speech_tokens + 1) 语音 token 预测 logits
            loss:   标量 CTC loss（仅训练时返回，推理时为 None）
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 输入投影 + 位置编码
        x = self.input_proj(hidden_states)
        x = self.input_norm(x)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pos_embedding(positions)

        # 如果没有额外 memory，使用投影后的输入自身
        if memory is None:
            memory = x

        # Transformer Decoder
        x = self.transformer_decoder(tgt=x, memory=memory)

        # 输出投影
        logits = self.output_proj(x)   # (batch, seq_len, num_speech_tokens + 1)

        # 计算 CTC Loss（仅训练时）
        loss = None
        if target_tokens is not None and input_lengths is not None and target_lengths is not None:
            # CTC Loss 期望 log_probs: (seq_len, batch, vocab)
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
            loss = self.ctc_loss(
                log_probs,
                target_tokens,
                input_lengths,
                target_lengths,
            )

        return logits, loss

    def greedy_decode(self, hidden_states: torch.Tensor) -> torch.LongTensor:
        """
        贪心解码：从 LLM 隐状态直接预测语音 token 序列。

        Args:
            hidden_states: (batch, seq_len, input_dim)

        Returns:
            (batch, decoded_len) 解码后的语音 token 序列（已去除 blank 和重复）
        """
        with torch.no_grad():
            logits, _ = self.forward(hidden_states)
            # 取每个时间步的最大概率 token
            predictions = logits.argmax(dim=-1)  # (batch, seq_len)

        # CTC 解码：去除 blank 和连续重复
        decoded_batch = []
        for pred in predictions:
            decoded = []
            prev_token = self.blank_id
            for token in pred.tolist():
                if token != self.blank_id and token != prev_token:
                    decoded.append(token)
                prev_token = token
            decoded_batch.append(torch.tensor(decoded, dtype=torch.long, device=hidden_states.device))

        # 填充到相同长度
        max_len = max(len(d) for d in decoded_batch) if decoded_batch else 0
        if max_len == 0:
            return torch.zeros(len(decoded_batch), 1, dtype=torch.long, device=hidden_states.device)

        padded = torch.full(
            (len(decoded_batch), max_len),
            fill_value=0,
            dtype=torch.long,
            device=hidden_states.device,
        )
        for i, d in enumerate(decoded_batch):
            if len(d) > 0:
                padded[i, :len(d)] = d
        return padded


# ── 模块自测 ────────────────────────────────────────────
if __name__ == "__main__":
    print("=== SpeechDecoder 模块自测 ===\n")

    batch, seq_len, input_dim = 2, 64, 4096
    hidden_dim, num_tokens = 1024, 4096

    decoder = SpeechDecoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=2,       # 自测用少量层
        num_heads=8,
        num_speech_tokens=num_tokens,
    )

    # 模拟 LLM 隐状态
    hidden_states = torch.randn(batch, seq_len, input_dim)

    # 前向传播（推理模式）
    logits, loss = decoder(hidden_states)
    print(f"推理模式: logits.shape={logits.shape}, loss={loss}")
    assert logits.shape == (batch, seq_len, num_tokens + 1)
    assert loss is None

    # 前向传播（训练模式，带 CTC Loss）
    target_len = 20
    target_tokens = torch.randint(0, num_tokens, (batch, target_len))
    input_lengths = torch.full((batch,), seq_len, dtype=torch.long)
    target_lengths = torch.full((batch,), target_len, dtype=torch.long)

    logits, loss = decoder(
        hidden_states,
        target_tokens=target_tokens,
        input_lengths=input_lengths,
        target_lengths=target_lengths,
    )
    print(f"训练模式: logits.shape={logits.shape}, loss={loss.item():.4f}")
    assert loss is not None

    # 贪心解码
    decoded = decoder.greedy_decode(hidden_states)
    print(f"贪心解码: decoded.shape={decoded.shape}")

    # 参数量统计
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"\nSpeechDecoder 参数量: {total_params:,}")

    print("\n=== 自测通过 ===")

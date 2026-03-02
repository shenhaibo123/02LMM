"""
模态投影层 —— 将视觉/音频编码器的输出映射到 LLM 的嵌入空间。

包含：
    MLPProjector      : 2 层 MLP + GELU，用于视觉特征投影
    ConvGMLPProjector : Conv1D(k=4, s=4) 4× 时序降采样 + GMLP，用于音频特征投影
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPProjector(nn.Module):
    """
    视觉投影层：2 层 MLP + GELU 激活。

    结构：input_dim → hidden_dim (GELU) → output_dim

    Args:
        input_dim:  输入特征维度（视觉编码器输出维度）
        hidden_dim: 中间隐藏维度
        output_dim: 输出维度（LLM 嵌入维度）
        dropout:    Dropout 概率
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """使用截断正态分布初始化权重。"""
        for linear in [self.linear1, self.linear2]:
            nn.init.trunc_normal_(linear.weight, std=0.02)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim) 视觉编码器输出

        Returns:
            (batch, seq_len, output_dim) 映射到 LLM 空间的特征
        """
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.layer_norm(x)
        return x


class _GMLPBlock(nn.Module):
    """
    gMLP 块：通过空间门控实现序列混合。

    结构：LayerNorm → Linear → GELU → SpatialGating → Linear
    """

    def __init__(self, dim: int, hidden_dim: int, seq_len: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj_in = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        # 空间门控：对序列维度做线性变换
        self.spatial_gate_norm = nn.LayerNorm(hidden_dim // 2)
        self.spatial_gate_proj = nn.Linear(seq_len, seq_len)
        self.proj_out = nn.Linear(hidden_dim // 2, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
        """
        residual = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = self.act(x)

        # 将特征分为两半：一半直通，一半做空间门控
        gate, value = x.chunk(2, dim=-1)
        gate = self.spatial_gate_norm(gate)
        # 转置后对序列维度做线性变换，再转回
        gate = self.spatial_gate_proj(gate.transpose(1, 2)).transpose(1, 2)
        x = gate * value

        x = self.proj_out(x)
        x = self.dropout(x)
        return x + residual


class ConvGMLPProjector(nn.Module):
    """
    音频投影层：Conv1D 4× 时序降采样 + GMLP 序列建模。

    流程：
        1. Conv1D(kernel_size=4, stride=4) 将时序长度压缩为 1/4
        2. GMLP 块进行序列级特征混合
        3. 线性层映射到 LLM 维度

    Args:
        input_dim:     输入特征维度（音频编码器输出维度）
        hidden_dim:    中间隐藏维度
        output_dim:    输出维度（LLM 嵌入维度）
        max_seq_len:   降采样后的最大序列长度（用于 GMLP 空间门控）
        num_gmlp_layers: GMLP 块数量
        dropout:       Dropout 概率
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        max_seq_len: int = 375,
        num_gmlp_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Conv1D 4× 降采样（在特征维度上操作，kernel_size=4, stride=4）
        self.conv_downsample = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=4,
            stride=4,
            padding=0,
        )
        self.conv_norm = nn.LayerNorm(hidden_dim)

        # GMLP 序列建模
        self.gmlp_layers = nn.ModuleList([
            _GMLPBlock(
                dim=hidden_dim,
                hidden_dim=hidden_dim * 2,
                seq_len=max_seq_len,
                dropout=dropout,
            )
            for _ in range(num_gmlp_layers)
        ])

        # 输出映射
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """初始化权重。"""
        nn.init.kaiming_normal_(self.conv_downsample.weight, mode="fan_out", nonlinearity="relu")
        if self.conv_downsample.bias is not None:
            nn.init.zeros_(self.conv_downsample.bias)
        nn.init.trunc_normal_(self.output_proj.weight, std=0.02)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim) 音频编码器输出

        Returns:
            (batch, seq_len // 4, output_dim) 降采样后映射到 LLM 空间的特征
        """
        # Conv1D 期望 (batch, channels, seq_len)
        x = x.transpose(1, 2)
        x = self.conv_downsample(x)
        x = x.transpose(1, 2)      # 回到 (batch, seq_len // 4, hidden_dim)
        x = self.conv_norm(x)

        # GMLP 序列建模
        for gmlp in self.gmlp_layers:
            x = gmlp(x)

        # 输出映射
        x = self.output_proj(x)
        x = self.output_norm(x)
        return x


# ── 模块自测 ────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Projectors 模块自测 ===\n")

    batch, vis_seq, vis_dim = 2, 729, 1152
    aud_seq, aud_dim = 1500, 1280
    llm_dim = 4096
    hidden = 4096

    # 测试视觉投影层
    mlp = MLPProjector(vis_dim, hidden, llm_dim)
    vis_input = torch.randn(batch, vis_seq, vis_dim)
    vis_output = mlp(vis_input)
    print(f"MLPProjector: {vis_input.shape} -> {vis_output.shape}")
    assert vis_output.shape == (batch, vis_seq, llm_dim)

    # 测试音频投影层
    conv_gmlp = ConvGMLPProjector(aud_dim, hidden, llm_dim, max_seq_len=aud_seq // 4)
    aud_input = torch.randn(batch, aud_seq, aud_dim)
    aud_output = conv_gmlp(aud_input)
    print(f"ConvGMLPProjector: {aud_input.shape} -> {aud_output.shape}")
    assert aud_output.shape == (batch, aud_seq // 4, llm_dim)

    # 参数量统计
    mlp_params = sum(p.numel() for p in mlp.parameters())
    conv_params = sum(p.numel() for p in conv_gmlp.parameters())
    print(f"\nMLPProjector 参数量: {mlp_params:,}")
    print(f"ConvGMLPProjector 参数量: {conv_params:,}")

    print("\n=== 自测通过 ===")

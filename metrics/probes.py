"""
模型探针（ModelProbe）—— 解耦的训练监控指标模块。

参考 QiChat (https://github.com/Morton-Li/QiChat) 的 ModelProbe 设计，
将输出分布指标和表征多样性指标封装为独立模块，可被任意训练脚本零侵入式调用。

指标一览：
  输出分布 (compute_output_distribution):
    logits_entropy, logits_std, top1_acc, top5_acc,
    confidence_gap, topk_mass, p_true_mean

  表征多样性 (compute_representation_diversity):
    cosine_sim_intra, cosine_sim_inter, participation_ratio
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle


class ModelProbe:
    """
    通过 forward hook 捕获指定层的隐状态，并计算输出分布 / 表征多样性指标。

    典型用法：
        probe = ModelProbe()
        probe.attach(model, ["model.layers.0", "model.layers.3", "model.layers.7"])
        probe.activate()

        # 训练 step
        output = model(input_ids, labels=labels)
        metrics = probe.compute_output_distribution(output.logits, labels)
        repr_metrics = probe.compute_representation_diversity()
        probe.deactivate()   # 释放缓存
    """

    def __init__(self):
        self._hooks: List[RemovableHandle] = []
        self._hidden_buffers: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._active: bool = False

    # ── 生命周期 ──────────────────────────────────────

    def attach(self, model: nn.Module, layer_names: List[str]) -> None:
        """在指定的子模块上注册 forward hook。"""
        self.detach()
        for name in layer_names:
            module = model.get_submodule(name)
            hook = module.register_forward_hook(self._make_hook(name))
            self._hooks.append(hook)

    def detach(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._hidden_buffers.clear()

    def activate(self) -> None:
        self._active = True

    def deactivate(self) -> None:
        self._active = False
        self._hidden_buffers.clear()

    @property
    def active(self) -> bool:
        return self._active

    # ── Hook ─────────────────────────────────────────

    def _make_hook(self, name: str) -> Callable:
        def hook(_module: nn.Module, _input, output):
            if not self._active:
                return
            tensor = None
            if torch.is_tensor(output) and output.dim() == 3:
                tensor = output
            elif isinstance(output, (tuple, list)):
                for item in output:
                    if torch.is_tensor(item) and item.dim() == 3:
                        tensor = item
                        break
            if tensor is not None:
                self._hidden_buffers[name] = tensor.detach()
        return hook

    # ── 输出分布指标 ─────────────────────────────────

    @torch.no_grad()
    def compute_output_distribution(
        self,
        logits: torch.Tensor,
        labels: torch.LongTensor,
        topk: int = 5,
        ignore_index: int = -100,
        max_tokens: Optional[int] = 256,
    ) -> Dict[str, float]:
        """
        计算输出分布指标。

        返回 dict:
            logits_entropy, logits_std, top1_acc, top5_acc,
            confidence_gap, topk_mass, p_true_mean
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        valid_mask = shift_labels != ignore_index
        if not valid_mask.any():
            return {k: float("nan") for k in
                    ["logits_entropy", "logits_std", "top1_acc", "top5_acc",
                     "confidence_gap", "topk_mass", "p_true_mean"]}

        flat_logits = shift_logits.reshape(-1, shift_logits.size(-1))[valid_mask.reshape(-1)]
        flat_labels = shift_labels.reshape(-1)[valid_mask.reshape(-1)]

        if max_tokens is not None and flat_logits.size(0) > max_tokens:
            idx = torch.randperm(flat_logits.size(0), device=flat_logits.device)[:max_tokens]
            flat_logits = flat_logits[idx]
            flat_labels = flat_labels[idx]

        flat_logits = flat_logits.float()
        probs = torch.softmax(flat_logits, dim=-1)

        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean().item()
        logits_std = flat_logits.std(dim=-1).mean().item()
        p_true_mean = probs.gather(1, flat_labels.unsqueeze(1)).mean().item()

        k_eval = min(max(topk, 5), probs.size(-1))
        topk_vals, topk_idx = torch.topk(probs, k=k_eval, dim=-1)

        top1_acc = (topk_idx[:, 0] == flat_labels).float().mean().item()
        top5_acc = (topk_idx[:, :5] == flat_labels.unsqueeze(-1)).any(dim=-1).float().mean().item()
        confidence_gap = (topk_vals[:, 0] - topk_vals[:, 1]).mean().item()
        topk_mass = topk_vals[:, :min(topk, topk_vals.size(1))].sum(dim=-1).mean().item()

        return {
            "logits_entropy": entropy,
            "logits_std": logits_std,
            "top1_acc": top1_acc,
            "top5_acc": top5_acc,
            "confidence_gap": confidence_gap,
            "topk_mass": topk_mass,
            "p_true_mean": p_true_mean,
        }

    # ── 表征多样性指标 ───────────────────────────────

    @torch.no_grad()
    def compute_representation_diversity(
        self,
        attention_mask: Optional[torch.BoolTensor] = None,
        max_tokens: Optional[int] = 256,
    ) -> Dict[str, Dict[str, float]]:
        """
        对每个已捕获的层计算表征多样性指标。

        返回 dict[layer_name] -> {
            cosine_sim_intra, cosine_sim_inter, participation_ratio
        }
        """
        results: Dict[str, Dict[str, float]] = {}

        for layer_name, hidden in self._hidden_buffers.items():
            batch_size, seq_len, hidden_size = hidden.size()
            hidden = hidden.float()

            use_mask = attention_mask is not None and torch.is_tensor(attention_mask)
            if use_mask:
                mask = attention_mask.to(hidden.device) if attention_mask.device != hidden.device else attention_mask
                if mask.dtype != torch.bool:
                    mask = mask.bool()
                flat = hidden[mask]
            else:
                flat = hidden.reshape(-1, hidden_size)

            if flat.numel() == 0 or flat.size(0) < 2:
                results[layer_name] = {
                    "cosine_sim_intra": float("nan"),
                    "cosine_sim_inter": float("nan"),
                    "participation_ratio": float("nan"),
                }
                continue

            # 采样
            n = flat.size(0)
            if max_tokens is not None and n > max_tokens:
                idx = torch.randperm(n, device=flat.device)[:max_tokens]
                flat = flat[idx]

            # --- 跨样本余弦相似度 ---
            normed = torch.nn.functional.normalize(flat, dim=-1)
            p1 = torch.randperm(normed.size(0), device=normed.device)
            p2 = torch.randperm(normed.size(0), device=normed.device)
            inter_cos = (normed[p1] * normed[p2]).sum(dim=-1).mean().item()

            # --- 样本内余弦相似度 ---
            intra_values = []
            tokens_per = max(max_tokens // batch_size, 2) if max_tokens else seq_len
            for i in range(batch_size):
                if use_mask:
                    h_i = hidden[i][mask[i]]
                else:
                    h_i = hidden[i]
                if h_i.size(0) < 2:
                    continue
                sample_idx = torch.randperm(h_i.size(0), device=h_i.device)[:tokens_per]
                h_i = h_i[sample_idx]
                h_normed = torch.nn.functional.normalize(h_i, dim=-1)
                r1 = torch.randperm(h_normed.size(0), device=h_normed.device)
                r2 = torch.randperm(h_normed.size(0), device=h_normed.device)
                cos = (h_normed[r1] * h_normed[r2]).sum(dim=-1).mean()
                intra_values.append(cos)

            intra_cos = torch.stack(intra_values).mean().item() if intra_values else float("nan")

            # --- 参与度比 ---
            centered = flat - flat.mean(dim=0, keepdim=True)
            cov = (centered @ centered.t()) / max(1, centered.size(0) - 1)
            try:
                eigvals = torch.linalg.eigvalsh(cov).clamp_min(0)
            except (NotImplementedError, RuntimeError):
                eigvals = torch.linalg.eigvalsh(cov.cpu()).clamp_min(0)
            pr = (eigvals.sum() ** 2 / eigvals.square().sum().clamp_min(1e-12)).item()

            results[layer_name] = {
                "cosine_sim_intra": intra_cos,
                "cosine_sim_inter": inter_cos,
                "participation_ratio": pr,
            }

        return results


# ── 模块自测 ────────────────────────────────────────

if __name__ == "__main__":
    print("=== ModelProbe 模块自测 ===\n")

    batch, seq, vocab, hidden = 2, 16, 100, 64
    num_layers = 3

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab, hidden)
            self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(num_layers)])
            self.head = nn.Linear(hidden, vocab, bias=False)

        def forward(self, input_ids):
            h = self.embed(input_ids)
            for layer in self.layers:
                h = torch.relu(layer(h))
            return self.head(h)

    model = TinyModel()
    probe = ModelProbe()

    layer_names = [f"layers.{i}" for i in range(num_layers)]
    probe.attach(model, layer_names)
    probe.activate()

    input_ids = torch.randint(0, vocab, (batch, seq))
    labels = torch.randint(0, vocab, (batch, seq))
    labels[:, :3] = -100

    logits = model(input_ids)

    print("1) 输出分布指标:")
    dist_metrics = probe.compute_output_distribution(logits, labels)
    for k, v in dist_metrics.items():
        print(f"   {k}: {v:.6f}")

    print("\n2) 表征多样性指标:")
    repr_metrics = probe.compute_representation_diversity()
    for layer, metrics in repr_metrics.items():
        print(f"   [{layer}]")
        for k, v in metrics.items():
            print(f"     {k}: {v:.6f}")

    probe.deactivate()
    probe.detach()
    print("\n=== 自测通过 ===")

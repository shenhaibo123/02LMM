# Omni / 全模态大模型与论文清单

本清单汇总主流 **Omni（全模态）** 模型、相关论文与优秀开源项目，便于整体掌握与追踪。  
**由 [omni-daily-collect](../../.cursor/skills/omni-daily-collect/SKILL.md) 技能每日可更新。**

- **最后更新日期**：2026-02-28

---

## 一、模型与论文总表

| 类型 | 名称 | 机构 | 论文 / 链接 | GitHub | 备注 |
|------|------|------|-------------|--------|------|
| 模型 | **MiniCPM-o 2.5 / 2.6 / 4.5** | 面壁智能 / OpenBMB | [博客](https://openbmb.github.io/) | [OpenBMB/MiniCPM-o](https://github.com/OpenBMB/MiniCPM-o) | 8B–9B 端侧全模态；**4.5**：全双工实时、主动智能、超拟人语音，视觉超 GPT-4o；HF: [MiniCPM-o-4_5](https://huggingface.co/openbmb/MiniCPM-o-4_5)、[MiniCPM-o-2_6](https://huggingface.co/openbmb/MiniCPM-o-2_6) |
| 模型 | **Qwen2.5-Omni** | 阿里 | [arXiv:2503.20215](https://arxiv.org/abs/2503.20215)；[博客](https://qwenlm.github.io/blog/qwen2.5-omni/) | [QwenLM/Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni) | Thinker-Talker，端到端全模态理解+流式语音生成，OmniBench SOTA |
| 模型 | **Qwen3-Omni** | 阿里 | [arXiv:2509.17765](https://arxiv.org/abs/2509.17765)；[官网](https://qwen3omni.org) | 见官网 / HF | MoE，SOTA 多模态 |
| 模型 | **OmniVinci** | Nvidia | [arXiv:2510.15870](https://arxiv.org/abs/2510.15870)；[项目页](https://nvlabs.github.io/OmniVinci/) | [nvidia/omnivinci](https://github.com/nvidia/omnivinci)（若开源） | 9B，OmniAlignNet 等；HF: nvidia/omnivinci |
| 模型 | **LongCat-Flash-Omni** | 美团 | [arXiv:2511.00279](https://arxiv.org/abs/2511.00279) | - | 560B MoE，27B 激活；HF: [meituan-longcat/LongCat-Flash-Omni](https://huggingface.co/meituan-longcat/LongCat-Flash-Omni) |
| 模型 | **Ming-Flash-Omni / 2.0** | 蚂蚁 | [arXiv:2510.24821](https://arxiv.org/abs/2510.24821) | - | 全模态生成与理解；HF: [inclusionAI/Ming-flash-omni-2.0](https://huggingface.co/inclusionAI/Ming-flash-omni-2.0) |
| 工作 | **OmniGAIA / OmniAtlas** | 人大/小红书 | [arXiv:2602.22897](https://arxiv.org/abs/2602.22897) | [RUC-NLPIR/OmniGAIA](https://github.com/RUC-NLPIR/OmniGAIA) | 原生 Omni 智能体基准（360 任务，多跳推理+工具使用）+ OmniAtlas 智能体 |
| 模型 | **MiniCPM-V 4.5** | 面壁智能 / OpenBMB | [arXiv:2509.18154](https://arxiv.org/abs/2509.18154)；[GitHub PDF](https://github.com/OpenBMB/MiniCPM-o/blob/main/docs/MiniCPM_V_4_5_Technical_Report.pdf) | [OpenBMB/MiniCPM-V](https://github.com/openbmb/MiniCPM-V) | 8B 高效 MLLM；3D-Resampler 96x 视频压缩；超越 GPT-4o |
| 模型 | **MiniCPM-SALA** | 面壁智能 / 清华 | [arXiv:2602.11761](https://arxiv.org/abs/2602.11761) | - | 9B 稀疏+线性混合注意力；支持 1M token；单 GPU 百万上下文 |
| 参考 | **Baichuan-Omni** | 百川 | [arXiv:2410.08565](https://arxiv.org/abs/2410.08565) | 百川官方仓库 | 7B 开源 |
| 参考 | **Ola** | - | [arXiv:2502.04328](https://arxiv.org/abs/2502.04328) | 待查 | 渐进式模态对齐 |
| 参考 | **Mini-Omni2** | - | [arXiv:2410.11190](https://arxiv.org/abs/2410.11190) | 待查 | 开源 GPT-4o 风格 |
| 参考 | **GPT-4o** | OpenAI | [系统卡 / 博客](https://openai.com) | - | 商业基线 |

---

## 二、优秀 GitHub / 开源项目（审核收录）

以下为与 Omni/全模态 相关的优质仓库，便于复现与二次开发。

| 项目 | 机构/作者 | 简介 | 链接 |
|------|-----------|------|------|
| **MiniCPM-o** | 面壁智能 / OpenBMB | 端侧全模态（文本/图像/音频等），支持 2.5/2.6/**4.5**；4.5 全双工实时、主动抢答、声音克隆 | [GitHub](https://github.com/OpenBMB/MiniCPM-o) |
| **Qwen2 / Qwen2.5** | 阿里 | 含 Qwen2.5-Omni 等多模态与 Omni 能力 | [GitHub](https://github.com/Qwen2/Qwen2.5) |
| **OmniVinci** | Nvidia | 9B 全模态，OmniAlignNet 等；HF/项目页见上表 | [NVLabs 项目页](https://nvlabs.github.io/OmniVinci/) |
| **VeOmni** | 社区/厂商 | 多模态训练框架，可与 LMM/Omni 训练流程结合 | 检索关键词：VeOmni omni multimodal |
| **LLaVA / LLaVA-NeXT** | 社区 | 图文多模态基线，与 Omni 技术路线可对照 | [LLaVA](https://github.com/haotian-liu/LLaVA) |

*更多仓库可由 omni-daily-collect 每日检索 GitHub「omni-modal」「omni model」等关键词后补充至本表或 changelog。*

---

## 三、检索与更新说明

- **检索关键词（中英）**：omni-modal, omni-modal LLM, “Omni” model, 全模态大模型, 全能模型。
- **更新方式**：运行项目技能「收集 Omni」/「检查 Omni 新工作」/「今日 Omni 简报」，或直接调用 `omni-daily-collect`，会更新本清单与 [changelog.md](./changelog.md)。
- **单篇精读**：若对某篇论文做深度分析，可在 `papers/{论文简称}/` 下建子目录，与 [paper-read](../../.cursor/skills/paper-read/SKILL.md) 约定一致。

---

## 四、已精读条目（paper-read 分析报告）

以下条目已按 [paper-read](../../.cursor/skills/paper-read/SKILL.md) 技能产出三章式精读报告，可直接查看对应目录下的 `{简称}_analysis.md`：

| 论文简称 | 精读报告路径 |
|----------|--------------|
| Qwen3-Omni | [papers/Qwen3-Omni/Qwen3-Omni_analysis.md](../Qwen3-Omni/Qwen3-Omni_analysis.md) |
| OmniVinci | [papers/OmniVinci/OmniVinci_analysis.md](../OmniVinci/OmniVinci_analysis.md) |
| LongCat-Flash-Omni | [papers/LongCat-Flash-Omni/LongCat-Flash-Omni_analysis.md](../LongCat-Flash-Omni/LongCat-Flash-Omni_analysis.md) |
| Ming-Flash-Omni | [papers/Ming-Flash-Omni/Ming-Flash-Omni_analysis.md](../Ming-Flash-Omni/Ming-Flash-Omni_analysis.md) |
| OmniGAIA | [papers/OmniGAIA/OmniGAIA_analysis.md](../OmniGAIA/OmniGAIA_analysis.md) |
| Baichuan-Omni | [papers/Baichuan-Omni/Baichuan-Omni_analysis.md](../Baichuan-Omni/Baichuan-Omni_analysis.md) |
| Ola | [papers/Ola/Ola_analysis.md](../Ola/Ola_analysis.md) |
| Mini-Omni2 | [papers/Mini-Omni2/Mini-Omni2_analysis.md](../Mini-Omni2/Mini-Omni2_analysis.md) |
| Qwen2.5-Omni | [papers/Qwen2.5-Omni/Qwen2.5-Omni_analysis.md](../Qwen2.5-Omni/Qwen2.5-Omni_analysis.md) |
| MiniCPM-V-4.5 | [papers/MiniCPM-V-4.5/MiniCPM-V-4.5_analysis.md](../MiniCPM-V-4.5/MiniCPM-V-4.5_analysis.md) |
| MiniCPM-SALA | [papers/MiniCPM-SALA/MiniCPM-SALA_analysis.md](../MiniCPM-SALA/MiniCPM-SALA_analysis.md) |

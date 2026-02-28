---
name: omni-daily-collect
description: 每日从 arXiv、Hugging Face、GitHub、OpenReview、厂商博客等渠道检查是否有新的 Omni/全模态 相关模型或论文，更新项目 papers/omni/ 下的清单与 changelog，并输出简短简报。适用于「收集 Omni」「检查 Omni 新工作」「今日 Omni 简报」等触发。
---

# Omni 每日收集技能

## 能力概览

| 维度 | 说明 |
|------|------|
| **目标** | 发现并汇总新的 Omni/全模态 模型、论文与优秀开源项目，保持 `papers/omni/` 清单与 changelog 更新。 |
| **触发** | 用户说「收集 Omni」「检查 Omni 新工作」「今日 Omni 简报」或每天调用一次时执行。 |
| **输出** | 更新 `papers/omni/README.md`（及可选 `changelog.md`），并在对话中给出「今日 Omni 新工作简报」。 |

---

## 检索关键词（中英）

- 英文：`omni-modal`, `omni-modal LLM`, `"Omni" model`, `omnimodal`, `unified multimodal`
- 中文：全模态大模型、全能模型、多模态统一
- 扩展：具体模型名如 `Qwen Omni`, `MiniCPM-o`, `OmniVinci`, `LongCat Omni`, `Ming-Flash-Omni` 等，用于查新版本或衍生工作。

---

## 平台与动作（执行顺序建议）

### 1. arXiv

- **范围**：cs.CL、cs.CV、cs.SD 等；近期 7 天或 30 天内提交/更新的论文。
- **动作**：用上述关键词在标题/摘要中检索，列出：标题、arXiv 链接、一句摘要。
- **注意**：仅收录与 Omni/全模态 直接相关的论文，避免泛多模态噪音。

### 2. Hugging Face

- **动作**：在 Models 搜索 `omni`、`omni-modal`、`omnimodal` 等，按 **Recently updated** 排序。
- **输出**：新出现或近期更新的模型卡片：模型名、组织/作者、链接、一句话描述（可从卡片摘要提取）。

### 3. GitHub

- **动作**：搜索 `omni-modal`、`omni model`、`omnimodal`、`全模态` 等，按 **Recently updated** 或 **Stars** 排序。
- **输出**：新/近仓库或高星优质项目：仓库名、作者/机构、链接、简短简介（README 首段或描述）；**重点收录有明确 Omni/全模态 定位且可复现的开源项目**。
- **注意**：与 [papers/omni/README.md](../../../papers/omni/README.md) 中「优秀 GitHub 项目」表对齐，避免重复；新发现的可补充进清单。

### 4. OpenReview（若可访问）

- **动作**：检索 NeurIPS/ICML/ICLR 等近期会议中标题/摘要含 omni、unified multimodal 的投稿。
- **输出**：标题、会议、链接、一句概括。

### 5. 厂商/博客

- **对象**：阿里（Qwen）、OpenBMB、Nvidia、美团、蚂蚁等已知发布过 Omni 相关模型或博客的机构。
- **动作**：通过 Web 检索或已知信息，汇总是否有新公告（新模型、新版本、技术博客）。
- **输出**：若有，标题、来源 URL、一句话说明。

---

## 工作流程（步骤）

1. **执行检索**：按上述平台依次进行（可使用 `web_search` 等）；优先保证 **arXiv + Hugging Face + GitHub** 三处覆盖。
2. **去重与筛选**：与 `papers/omni/README.md` 中已有条目对比，仅保留**新**或**有重要更新**的项（如新版本、新仓库、新论文）。
3. **更新清单**：
   - **README.md**：若有新模型/论文/项目，在总表或「优秀 GitHub 项目」表中新增一行，列明：类型、名称、机构、论文/链接、GitHub、备注；并更新文末「最后更新日期」。
   - **changelog.md**：在 `papers/omni/changelog.md` 中追加当日日期 +「本次新发现」列表（标题、来源、链接）。
4. **输出简报**：在对话中给出简短「今日 Omni 新工作简报」：列出新发现项（标题、来源、链接）；若本次无新发现，则明确说明「未发现新条目」。

---

## 输出路径与格式

| 产出 | 路径 | 说明 |
|------|------|------|
| Omni 清单 | `papers/omni/README.md` | 总表 + 优秀 GitHub 项目表；新增行时保持表格格式一致。 |
| 更新摘要 | `papers/omni/changelog.md` | 按日期追加「本次新发现」列表。 |
| 简报 | 对话内 | 简短列举新条目或声明无新发现。 |

---

## 注意事项

- **不编造链接**：仅写入经检索或已知可访问的 URL；若仅有名称无链接，可在备注中写「链接待补」。
- **注明来源**：changelog 与简报中写清来源（如 arXiv、HF、GitHub、某博客）。
- **依赖**：不依赖本地爬虫或 API；由大模型通过 Web 检索与现有知识汇总并写入文件。若需定时自动化，可由用户本机 cron/脚本每日调用 Cursor 或本技能入口。

---

## 与 paper-read 的约定

- 单篇论文深度精读仍使用 [paper-read](../paper-read/SKILL.md)，产出放在 `papers/{论文简称}/`。
- 本技能只维护 **清单与 changelog**；若某篇新论文需要精读，可在简报中提示用户可对 `papers/{论文简称}/` 使用 paper-read。

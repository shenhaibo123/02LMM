---
name: Omni 模型清单与每日收集技能
overview: 在项目 papers 目录下建立 Omni 模型/论文清单（含你已知及检索到的主流工作），并新增一个 Cursor Skill：每日从 arXiv、Hugging Face、GitHub、OpenReview 等平台检查并汇总新 Omni 相关工作，更新清单或输出简报。
todos: []
isProject: false
---

# Omni 模型清单与每日收集技能

## 一、目标

1. **清单**：在 `papers/` 下提供一份结构化的 **Omni 模型/论文清单**（含名称、机构、论文/链接、简要说明），便于你整体掌握。
2. **技能**：新增 **omni-daily-collect** 技能，供你每天调用一次，从多平台拉取/检查是否有新的 Omni 相关工作，并更新清单或输出简报。

---

## 二、Omni 清单内容范围（拟纳入）

基于检索与你的已知项，清单拟包含以下条目（均可在执行时写入 `papers/omni/` 下的 Markdown/表格中）：


| 类型  | 名称                    | 机构      | 论文/链接                                                              | 备注                                   |
| --- | --------------------- | ------- | ------------------------------------------------------------------ | ------------------------------------ |
| 模型  | MiniCPM-o 2.5 / 2.6   | OpenBMB | GitHub: OpenBMB/MiniCPM-o; HF: MiniCPM-o-2_6                       | 8B，端侧全模态，2.6 为当前最新                   |
| 模型  | Qwen2.5-Omni          | 阿里      | arXiv:2503.xxx; qwenlm.github.io/blog/qwen2.5-omni                 | Thinker-Talker，实时语音+多模态              |
| 模型  | Qwen3-Omni            | 阿里      | arXiv:2509.17765; qwen3omni.org                                    | MoE，SOTA 多模态                         |
| 模型  | OmniVinci             | Nvidia  | arXiv:2510.15870; nvlabs.github.io/OmniVinci; HF: nvidia/omnivinci | 9B，OmniAlignNet 等                    |
| 模型  | LongCat-Flash-Omni    | 美团      | arXiv:2511.00279; HF: meituan-longcat/LongCat-Flash-Omni           | 560B MoE，27B 激活                      |
| 模型  | Ming-Flash-Omni / 2.0 | 蚂蚁      | arXiv:2510.24821; HF: inclusionAI/Ming-flash-omni-2.0              | 全模态生成与理解                             |
| 工作  | OmniGAIA / OmniAtlas  | -       | arXiv:2602.22897                                                   | 原生 Omni 智能体 + 基准，OmniAtlas 为其中 agent |
| 参考  | Baichuan-Omni         | 百川      | arXiv:2410.08565                                                   | 7B 开源                                |
| 参考  | Ola                   | -       | arXiv:2502.04328                                                   | 渐进式模态对齐                              |
| 参考  | Mini-Omni2            | -       | arXiv:2410.11190                                                   | 开源 GPT-4o 风格                         |
| 参考  | GPT-4o                | OpenAI  | 系统卡 / 博客                                                           | 商业基线                                 |


清单文件建议路径：`**papers/omni/README.md`**（或 `papers/omni/omni_models_list.md`），内含表格 + 每项简短说明与链接；必要时在 `papers/omni/` 下为单篇论文建子目录（与现有 paper-read 的 `papers/{论文简称}/` 约定一致）。

---

## 三、`papers/` 目录结构（约定）

- `**papers/omni/`**  
  - `README.md`（或 `omni_models_list.md`）：Omni 模型与论文总清单（表格 + 链接 + 简短说明）。  
  - 可选：`papers/omni/changelog.md` 记录“每日收集”的更新摘要（由 skill 追加或覆盖）。
- 其他论文仍为 `**papers/{论文简称}/`**，与 [.cursor/skills/paper-read/SKILL.md](.cursor/skills/paper-read/SKILL.md) 中约定一致。

---

## 四、新技能：omni-daily-collect

- **名称**：`omni-daily-collect`  
- **位置**：项目级 [.cursor/skills/omni-daily-collect/SKILL.md](.cursor/skills/omni-daily-collect/SKILL.md)  
- **描述（建议）**：每日从 arXiv、Hugging Face、GitHub、OpenReview、厂商博客等渠道检查是否有新的 Omni/全模态 相关模型或论文，并更新项目 `papers/omni/` 下的清单或 changelog。  
- **触发**：用户说「收集 Omni」「检查 Omni 新工作」「今日 Omni 简报」或每天调用一次时执行。  
- **工作流程（建议）**：  
  1. **定义检索关键词**：omni-modal, omni-modal LLM, “Omni” model, 全模态大模型 等（中英兼顾）。
  2. **平台与动作**：
    - **arXiv**：检索 cs.CL / cs.CV / cs.SD 近期（如最近 7 天或 30 天）标题/摘要含上述关键词的论文，列出标题、链接、摘要一句。  
    - **Hugging Face**：在 Models 搜索 "omni" / "omni-modal" 等，按最近更新排序，列出新出现或近期更新的模型卡片与链接。  
    - **GitHub**：搜索 "omni-modal" 或 "omni model" 等，按最近更新排序，列出新/近仓库与简介。  
    - **OpenReview**：检索近期会议（如 NeurIPS/ICML/ICLR）中 omni / multimodal 相关投稿（若可访问）。  
    - **厂商/博客**：阿里、OpenBMB、Nvidia、美团、蚂蚁等官网或博客是否有新 Omni 相关公告（通过网页摘要或已知信息汇总）。
  3. **输出**：
    - 更新 `**papers/omni/README.md`**（或 `omni_models_list.md`）：若有新条目，在表格中新增一行并补充链接与说明。  
    - 可选：`**papers/omni/changelog.md`** 中追加当日日期 + “本次新发现”列表（标题、来源、链接）。  
    - 在对话中给出一份简短「今日 Omni 新工作简报」（无则说明“未发现新条目”）。
- **依赖**：不依赖本地爬虫或 API；由大模型根据 Web 检索（如 `web_search`）与现有知识汇总，并写入上述文件。若需自动化定时，可由用户本机 cron/脚本每日调用 Cursor 或该 skill 的入口。

---

## 五、实现步骤（执行顺序）

1. **创建目录与清单文件**
  - 创建 `papers/omni/`。  
  - 在 `papers/omni/README.md` 中写入结构化清单：表格列至少包含「名称、机构、论文/链接、备注」；上表内容作为初始数据填入，并注明“最后更新日期”和“由 omni-daily-collect 可更新”。
2. **创建 omni-daily-collect 技能**
  - 在 `.cursor/skills/omni-daily-collect/` 下新建 `SKILL.md`。  
  - 内容包含：技能 name/description；适用场景（每日收集、检查新 Omni 工作）；检索关键词与平台列表；具体步骤（arXiv / HF / GitHub / OpenReview / 博客）；输出路径（`papers/omni/README.md`、可选 `changelog.md`）与简报格式；注意事项（不编造链接、注明来源）。
3. **（可选）**.gitignore
  - 若希望 `papers/` 下论文正文与翻译不提交，仅保留清单与脚本，可在 `.gitignore` 中保留或调整对 `papers/` 的规则；清单文件 `papers/omni/README.md` 通常建议纳入版本控制。

---

## 六、输出物小结


| 产出           | 路径                                           |
| ------------ | -------------------------------------------- |
| Omni 模型/论文清单 | `papers/omni/README.md`                      |
| 每日收集更新摘要（可选） | `papers/omni/changelog.md`                   |
| 每日收集技能       | `.cursor/skills/omni-daily-collect/SKILL.md` |


执行完成后，你可直接打开 `papers/omni/README.md` 掌握整体框架；每日运行一次 omni-daily-collect（或说“收集 Omni”）即可更新清单并查看简报。
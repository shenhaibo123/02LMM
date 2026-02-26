# 深度探索目录

> 本目录用于记录围绕 MiniMind 做的各种「好奇心驱动」的小实验，偏向现象观察与直觉总结。初稿由 Vibe Writing 大模型生成，并结合后续实验结果逐步补充。

可以考虑在这里写的内容示例：

- **模型尺寸与能力**：在保持数据不变的情况下，改变层数 / hidden size / heads 数量，观察 loss 曲线与主观体验的变化。
- **位置编码与上下文长度**：对比不同 RoPE 配置、不同最大长度下的困惑度与长文本表现。
- **训练超参数扫描**：学习率、batch size、warmup 步数等对收敛速度和最终效果的影响。
- **推理策略实验**：贪心 / top-k / top-p / temperature 等组合对生成质量的影响，结合 `scripts/chat_openai_api.py` 做交互实验。
- **对齐方法对比**：在相同基座模型上，对比 DPO、PPO、GRPO、SPO 等方法带来的输出风格差异。
- **与上游 MiniMind / 其他开源模型对比**：在相同 prompt 下对比回答质量，并尝试分析差异来源。
- **VeOmni 与 Omni 模型**：基于 VeOmni 等成熟多模态训练框架，将本仓库的数据格式与训练阶段迁移到更大规模 LMM/Omni 训练，并记录配置与实战笔记。

写作时可以：

- 在正文中标注本次实验对应的代码位置（如 trainer 某个脚本、配置片段等）。
- 尽量保留失败实验和负面结果，它们往往更有参考价值。

---

## 部署与推理使用说明

### 模型转换

`scripts/convert_model.py` 可实现 **PyTorch 原生权重 ↔ Transformers** 的互相转换。如无特别说明，本仓库导出的模型均默认为 Transformers 格式；若从 `out/` 下的 `.pth`  checkpoint 出发，需先用该脚本做转换后再用下文方式部署。

### 基于 OpenAI 兼容 API 的服务

`scripts/serve_openai_api.py` 提供兼容 OpenAI API 的聊天接口，便于将模型接入 FastGPT、Open-WebUI、Dify 等第三方 UI。

**Transformers 格式模型目录示例**（转换后或从 HuggingFace 下载）：

```
<模型根目录>/
├── config.json
├── generation_config.json
├── model_minimind.py（可选）
├── pytorch_model.bin 或 model.safetensors
├── special_tokens_map.json
├── tokenizer_config.json
└── tokenizer.json
```

**启动服务与测试：**

```bash
# 启动聊天服务端（在项目根目录）
python scripts/serve_openai_api.py

# 测试接口
python scripts/chat_openai_api.py
```

**API 示例（兼容 OpenAI 格式）：**

```bash
curl http://<ip>:<port>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model-identifier",
    "messages": [ { "role": "user", "content": "世界上最高的山是什么？" } ],
    "temperature": 0.7,
    "max_tokens": 512,
    "stream": true
  }'
```

### vLLM

vLLM 支持高吞吐推理与显存优化。以 OpenAI 兼容方式启动本仓库转换后的 Transformers 模型：

```bash
vllm serve ./<模型路径> --model-impl transformers --served-model-name "minimind" --port 8998
```

### llama.cpp

llama.cpp 支持命令行多线程与 GPU 推理。建议与 02LMM 放在同级目录：

```
parent/
├── 02LLM/              # 本仓库
│   ├── <Transformers 模型目录>   # 需先用 convert_model.py 生成
│   ├── model/
│   ├── scripts/
│   └── ...
└── llama.cpp/
    ├── build/
    ├── convert_hf_to_gguf.py
    └── ...
```

1. 按 llama.cpp 官方步骤完成安装。
2. 在 `convert_hf_to_gguf.py` 的 `get_vocab_base_pre` 末尾为 MiniMind tokenizer 增加分支（若为 None 可设 `res = "qwen2"` 等）。
3. 转换：`python convert_hf_to_gguf.py ../02LLM/<模型路径>`，得到 `.gguf`。
4. 可选量化：`./build/bin/llama-quantize <输入.gguf> <输出.gguf> Q4_K_M`。
5. 推理：`./build/bin/llama-cli -m <模型.gguf> -sys "You are a helpful assistant"`。

### Ollama

1. 得到 gguf 后，在模型目录下新建 `minimind.modelfile`，写入：

```
FROM ./Q4-<模型名>.gguf
SYSTEM """You are a helpful assistant"""
TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ .Response }}<|im_end|>
"""
```

2. 创建并运行：`ollama create -f minimind.modelfile minimind-local`，`ollama run minimind-local`。
3. 也可使用社区镜像（若存在）：`ollama run jingyaogong/minimind2` 等。

### MNN

MNN 面向端侧推理。进入 MNN 官方仓库的 `MNN/transformers/llm/export`，使用其导出脚本将 Transformers 模型导出为 MNN 格式（如 4bit HQQ 量化），再在 Mac 或手机上用 `llm_demo` 或官方 APP 加载测试。具体参数与用法见 MNN 官方文档。

以上框架的详细用法请以各自官方文档为准。

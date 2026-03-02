# 数据源说明

## 各阶段数据来源

### Stage 1: 视觉-语言对齐

| 数据集 | 规模 | 语言 | 说明 | HuggingFace ID |
|--------|------|------|------|---------------|
| LLaVA-CC3M-Pretrain | 595K | EN | 图像-文本对齐预训练 | `liuhaotian/LLaVA-CC3M-Pretrain-595K` |
| ShareGPT4V-PT | 1.2M | EN | 高质量图文对 | `Lin-Chen/ShareGPT4V` |
| Wukong-100M (子集) | 自选 | ZH | 中文图文对 | `BAAI/Wukong` |

### Stage 2: 音频-语言对齐

| 数据集 | 规模 | 语言 | 说明 | 来源 |
|--------|------|------|------|------|
| AISHELL-1 | 178h/150K | ZH | 中文语音识别 | `speech_asr/speech_asr_aishell1_trainsets` |
| LibriSpeech | 960h | EN | 英文语音识别 | `openslr/librispeech_asr` |
| WenetSpeech (子集) | 自选 | ZH | 大规模中文语音 | `wenet-e2e/WenetSpeech` |
| AudioCaps | 50K | EN | 音频描述 | `AudioCaps` |

### Stage 3: 全模态联合 SFT

| 数据集 | 规模 | 类型 | 说明 | HuggingFace ID |
|--------|------|------|------|---------------|
| LLaVA-Instruct-150K | 150K | 图文 | 图像指令数据 | `liuhaotian/LLaVA-Instruct-150k` |
| ShareGPT4V-SFT | 100K | 图文 | 高质量图文 SFT | `Lin-Chen/ShareGPT4V` |
| AISHELL-SFT | 50K | 音频 | 语音理解指令 | 自建 |
| VideoChat-Instruct | 100K | 视频 | 视频理解指令 | `OpenGVLab/VideoChat2-IT` |
| Alpaca-GPT4 | 52K | 文本 | 通用文本指令 | `AI-ModelScope/alpaca-gpt4-data-zh` |

### Stage 4: 语音生成

| 数据集 | 规模 | 语言 | 说明 |
|--------|------|------|------|
| LibriTTS | 585h | EN | 英文语音合成 |
| AISHELL-3 | 85h/88K | ZH | 多说话人中文 TTS |
| 自建 TTS 对 | 自选 | ZH+EN | 文本→CosyVoice2 token 对 |

### Stage 5: DPO 偏好对齐

| 数据集 | 规模 | 类型 | 说明 | HuggingFace ID |
|--------|------|------|------|---------------|
| UltraFeedback | 64K | 文本偏好 | 通用偏好数据 | `openbmb/UltraFeedback` |
| 自建多模态偏好 | 10K+ | 多模态偏好 | chosen/rejected 对 | 自建 |

---

## MS-Swift 数据格式

MS-Swift 使用 JSONL 格式，支持以下字段：

### 文本 SFT 格式
```json
{"messages": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！有什么可以帮你的吗？"}]}
```

### 图像多模态格式
```json
{"messages": [{"role": "user", "content": "<image>描述这张图片"}, {"role": "assistant", "content": "这是一张..."}], "images": ["/path/to/image.jpg"]}
```

### 音频多模态格式
```json
{"messages": [{"role": "user", "content": "<audio>请将这段语音转写为文字"}, {"role": "assistant", "content": "你好世界"}], "audios": ["/path/to/audio.wav"]}
```

### 视频多模态格式
```json
{"messages": [{"role": "user", "content": "<video>描述这段视频的内容"}, {"role": "assistant", "content": "视频中..."}], "videos": ["/path/to/video.mp4"]}
```

### DPO 偏好对格式
```json
{"messages": [{"role": "user", "content": "问题"}], "chosen": {"role": "assistant", "content": "好的回答"}, "rejected": {"role": "assistant", "content": "差的回答"}}
```

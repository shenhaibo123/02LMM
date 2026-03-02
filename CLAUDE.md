# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**02LMM** (From Zero to Large Multimodal Model) — a hands-on LLM training project starting from a 25.8M-parameter model. Covers the full pipeline: pretrain → SFT/LoRA → RLHF (DPO/PPO/GRPO/SPO) → distillation → evaluation → deployment. All core algorithms are implemented in native PyTorch without third-party training framework abstractions.

## Common Commands

### Environment Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install lm_eval          # optional, for evaluation
pip install fastapi uvicorn   # optional, for API server
```

For Conda-based setup, use: `bash Doc/进阶实践/prepare/setup_env_and_data.sh`

### Training
```bash
# Minimal pretrain (macOS MPS)
python trainer/train_pretrain.py --device mps --batch_size 4 --max_seq_len 128 --epochs 1 --hidden_size 64 --num_hidden_layers 2

# Pretrain (CUDA)
python trainer/train_pretrain.py --device cuda:0 --batch_size 32 --max_seq_len 340 --epochs 1

# SFT / LoRA / DPO / PPO / GRPO / SPO — same pattern:
python trainer/train_<variant>.py --device <device> [args]
```

### Evaluation
```bash
# Smoke test
python scripts/eval_model_benchmark.py --backend hf --model-path ./model --smoke --device cpu --limit 10

# Full benchmark
python scripts/eval_model_benchmark.py --backend hf --model-path ./model --tasks hellaswag arc_easy mmlu --limit 100
```

### Serving & Inference
```bash
python scripts/serve_openai_api.py --device cuda:0   # OpenAI-compatible API
python eval_llm.py                                     # Interactive inference
python scripts/web_demo.py                             # Streamlit Web Demo
```

### Model Conversion
```bash
python scripts/convert_model.py   # PyTorch ↔ Transformers format
```

## Architecture

### Model (`model/`)
- `model_minimind.py` — Core model: `MiniMindConfig` (extends `PretrainedConfig`) + `MiniMindForCausalLM` (extends `PreTrainedModel`). Implements a decoder-only Transformer with GQA, RoPE, SwiGLU, and optional MoE (Mixture of Experts with shared experts + router).
- `model_lora.py` — LoRA adapter implementation.
- Tokenizer files (`tokenizer.json`, `tokenizer_config.json`) live alongside the model.

### Trainers (`trainer/`)
Each training stage is a standalone script (`train_pretrain.py`, `train_full_sft.py`, `train_lora.py`, `train_dpo.py`, `train_ppo.py`, `train_grpo.py`, `train_spo.py`, `train_distillation.py`, `train_reason.py`). They share utilities from `trainer_utils.py` which provides: model param counting, distributed training init (NCCL), cosine LR scheduler, seed setting, tokenizer loading, and a `Logger` helper that gates output on `is_main_process()`.

### Metrics (`metrics/`) — Decoupled monitoring module
- `probes.py` — Model probes: output distribution metrics (entropy, top-k accuracy, confidence gap) and representation diversity (cosine similarity, participation ratio).
- `tracker.py` — Training tracker: loss/PPL/gradient norm/resource usage with sliding averages.
- `visualize.py` — Generates training curve plots from JSONL logs.

### Dataset (`dataset/lm_dataset.py`)
Unified dataset module with classes for all stages: `PretrainDataset`, `SFTDataset`, `DPODataset`, `RLAIFDataset`. Uses HuggingFace `datasets` for loading JSONL files. Chat preprocessing adds random system prompts; post-processing handles `<think>` tags for reasoning models.

### Scripts (`scripts/`)
- `eval_model_benchmark.py` — Thin wrapper around `lm-evaluation-harness`; supports `--backend hf` (local) and `--backend api` (OpenAI-compatible).
- `serve_openai_api.py` — FastAPI server providing OpenAI Chat Completions API.
- `convert_model.py` — Bidirectional PyTorch ↔ Transformers weight conversion.

### Output Directories
- `out/` — Saved model weights after training.
- `logs/` — JSONL training logs and generated curve plots.
- `checkpoints/` — Training checkpoints.

## Key Design Decisions

- **Dual platform**: GPU (CUDA) for full training, macOS (MPS) for quick experiments with smaller configs (hidden_size=64, num_layers=2).
- **No training framework dependency**: All algorithms (DPO, PPO, GRPO, SPO, distillation) are implemented directly in PyTorch. `trl` and `peft` are available as optional dependencies.
- **Modules are decoupled**: metrics, data, and model definitions are independent — metrics can be used by any trainer without coupling.
- **DDP support**: Training scripts support `torchrun` for multi-GPU via `init_distributed_mode()` in `trainer_utils.py`.

## Language

This is a Chinese-oriented project. Documentation, comments, and commit messages are primarily in Chinese. Code identifiers and API interfaces are in English.

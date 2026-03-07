# General Fine-Tuning

LoRA fine-tuning scripts for large language models using [Unsloth](https://github.com/unslothai/unsloth), [HuggingFace TRL](https://github.com/huggingface/trl), and [PEFT](https://github.com/huggingface/peft).  All scripts target **8 × A100 80GB** GPUs and expect an Alpaca-style dataset.

---

## Repository Structure

```
Qwen-3.5/
├── Qwen3.5-4B.py            # Unsloth multi-GPU DDP (4B dense)
├── Qwen3.5-9B.py            # HuggingFace DDP (9B dense)
├── Qwen3.5-35B-MOE.py       # HuggingFace DDP (35B-A3B MoE, VLM wrapper)
├── train_dense_ddp.py       # Unsloth DDP (27B dense)
└── train_moe_single_gpu.py  # Unsloth single-GPU (35B-A3B MoE)
```

---

## Scripts

| Script | Model | Stack | Hardware |
|--------|-------|-------|----------|
| `Qwen3.5-4B.py` | Qwen3.5-4B | Unsloth + TRL | 8 × A100 80GB (DDP) |
| `Qwen3.5-9B.py` | Qwen3.5-9B | HuggingFace + TRL | 8 × A100 80GB (DDP) |
| `Qwen3.5-35B-MOE.py` | Qwen3.5-35B-A3B | HuggingFace + TRL | 8 × A100 80GB (DDP) |
| `train_dense_ddp.py` | Qwen3.5-27B | Unsloth + TRL | 8 × A100 80GB (DDP) |
| `train_moe_single_gpu.py` | Qwen3.5-35B-A3B | Unsloth FastModel | 1 × A100/H100 80GB |

---

## Quick Start

### Multi-GPU (DDP via torchrun)

```bash
# Install dependencies
pip install unsloth trl datasets peft transformers accelerate

# Run any DDP script
UNSLOTH_COMPILE_TRANSFORMERS=0 torchrun --nproc_per_node 8 Qwen-3.5/train_dense_ddp.py
torchrun --nproc_per_node 8 Qwen-3.5/Qwen3.5-9B.py
torchrun --nproc_per_node 8 Qwen-3.5/Qwen3.5-35B-MOE.py
```

### Single GPU (MoE)

```bash
pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth unsloth_zoo
pip install git+https://github.com/huggingface/transformers.git --no-deps
pip install --no-deps trl==0.22.2

python Qwen-3.5/train_moe_single_gpu.py
```

---

## Configuration

Each script has a `# ── config ──` section at the top.  Set at minimum:

```python
MODEL_NAME    = "Qwen/Qwen3.5-9B"       # HuggingFace model id
DATASET_NAME  = "username/dataset"       # HuggingFace dataset id
HF_MODEL_REPO = "username/model-name"    # destination for the merged model
```

Key hyperparameters (defaults tuned for 8 × A100 80GB):

| Parameter | Default | Notes |
|-----------|---------|-------|
| `MAX_SEQ_LENGTH` | 2048–4096 | Increase for longer reasoning traces |
| `LORA_RANK` | 16 | 16–64 typical; higher = more capacity |
| `BATCH_SIZE` | 1–16 | Per device; adjust to fill VRAM |
| `GRAD_ACCUM` | 4 | Effective batch = batch × accum × GPUs |
| `LR` | 2e-4 | Standard LoRA learning rate |
| `EPOCHS` | 1–3 | |

HuggingFace token (needed for Hub push):

```bash
export HF_TOKEN=hf_...
```

---

## Dataset Format (Alpaca-style)

All scripts expect a dataset with three columns:

| Column | Description |
|--------|-------------|
| `instruction` | The question or task |
| `input` | Optional extra context (can be empty string) |
| `output` | The answer |

### Chain-of-Thought Output Format

To train the model to reason before answering, structure outputs as:

```
<think>
Step-by-step reasoning here...
</think>
Final concise answer here.
```

Scripts that use `train_on_responses_only` set `response_part = "<|im_start|>assistant\n<think>"` so the model learns to produce reasoning from its very first output token.

---

## Chat Template

Qwen3.5 uses the [ChatML](https://huggingface.co/docs/transformers/main/chat_templating) format:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
<think>
The user is asking a basic geography question.
</think>
Paris.<|im_end|>
```

---

## Serving with vLLM

After training the merged 16-bit model can be served directly:

```bash
# Dense models
vllm serve <YOUR_HF_REPO> \
    --dtype bfloat16 \
    --tensor-parallel-size 8 \
    --max-model-len 32768 \
    --enable-prefix-caching

# Text-only serving (Qwen3.5-9B after vLLM patches)
vllm serve <YOUR_MERGED_DIR> \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 32768 \
    --language-model-only
```

---

## References

- [Unsloth Documentation](https://docs.unsloth.ai)
- [Qwen3.5 on HuggingFace](https://huggingface.co/Qwen)
- [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)
- [PEFT LoRA](https://huggingface.co/docs/peft/conceptual_guides/lora)

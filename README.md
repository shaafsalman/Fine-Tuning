# Qwen3.5 Fine-Tuning with Unsloth

LoRA fine-tuning scripts for Qwen3.5 language models using [Unsloth](https://github.com/unslothai/unsloth) and [TRL](https://github.com/huggingface/trl).

## Scripts

| Script | Model | Hardware | Description |
|--------|-------|----------|-------------|
| `train_dense_ddp.py` | Qwen3.5-27B (dense) | 8 × A100 80GB | Multi-GPU DDP via `torchrun` |
| `train_moe_single_gpu.py` | Qwen3.5-35B-A3B (MoE) | 1 × H100 80GB | Single-GPU BF16 LoRA |

## Quick Start

### Dense model (DDP)
```bash
pip install unsloth trl datasets
UNSLOTH_COMPILE_TRANSFORMERS=0 torchrun --nproc_per_node 8 train_dense_ddp.py
```

### MoE model (single GPU)
```bash
pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth unsloth_zoo
pip install git+https://github.com/huggingface/transformers.git --no-deps
pip install --no-deps trl==0.22.2

python train_moe_single_gpu.py
```

## Configuration

Edit the `# ── config ──` section at the top of each script:

```python
MODEL_NAME     = "unsloth/Qwen3.5-27B"   # or "unsloth/Qwen3.5-35B-A3B"
DATASET_NAME   = "<YOUR_DATASET>"         # HuggingFace dataset id
HF_REPO        = "<YOUR_HF_REPO>"         # where to push the merged model
MAX_SEQ_LENGTH = 2048                     # increase for longer reasoning traces
LORA_RANK      = 16                       # 16–64 typical range
```

## Dataset Format (Alpaca-style)

Both scripts expect an Alpaca-style dataset with three columns:

| Column | Description |
|--------|-------------|
| `instruction` | The question or task |
| `input` | Optional extra context (can be empty) |
| `output` | Answer — ideally contains `<think>...</think>` + final answer |

### Chain-of-Thought Output Format

To teach the model to reason before answering, structure outputs like:

```
<think>
Step-by-step reasoning here...
</think>
Final concise answer here.
```

The `train_dense_ddp.py` script uses `train_on_responses_only` with
`response_part = "<|im_start|>assistant\n<think>"` so the model is trained
to produce CoT reasoning from the very first token of its response.

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

## Serving with vLLM

After training, the merged 16-bit model can be served directly:

```bash
vllm serve <YOUR_HF_REPO> --dtype bfloat16 --max-model-len 4096
```

## References

- [Unsloth Documentation](https://docs.unsloth.ai)
- [Qwen3.5 on HuggingFace](https://huggingface.co/Qwen)
- [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)

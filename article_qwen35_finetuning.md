# Fine-Tuning Qwen3.5 with Unsloth: Chain-of-Thought Training on Custom Datasets

> A practical guide to LoRA fine-tuning Qwen3.5 — covering the model's hybrid architecture, thinking tokens, tokenizer mechanics, and two complete training setups for multi-GPU and single-GPU environments.

---

## What Is Qwen3.5?

Qwen3.5 is the latest generation of open-weight language models from Alibaba's Qwen team, released in February 2026. It is not a minor update — Qwen3.5 represents a fundamental rethinking of transformer architecture, combining **Gated Delta Networks** (linear attention) with either dense feed-forward layers or **sparse Mixture-of-Experts (MoE)** routing, resulting in models that are both computationally efficient at inference time and highly capable on reasoning, coding, and multilingual tasks.

The family ships in several sizes:

| Model | Total Params | Active Params | Architecture |
|---|---|---|---|
| Qwen3.5-27B | 27B | 27B | Dense (Gated DeltaNet + FFN) |
| Qwen3.5-35B-A3B | 35B | ~3B | MoE (Gated DeltaNet + Sparse MoE) |
| Qwen3.5-122B-A10B | 122B | ~10B | MoE |

The **A3B / A10B** suffix means "Activated 3B / 10B" — during any given forward pass, only that many parameters are active. This is the fundamental MoE trade-off: large model capacity, modest compute cost.

All variants support **262,144 tokens natively** (extendable to ~1 million via YaRN rope scaling) and cover **201 languages and dialects**. Licensing is Apache 2.0 across the board.

---

## Architecture Deep Dive

### The Hybrid Attention Design

Classical transformers use full quadratic self-attention in every layer. Qwen3.5 replaces most attention layers with **Gated DeltaNet** — a form of linear attention that processes sequences in O(n) rather than O(n²) time. The hidden layout pattern in Qwen3.5-27B looks like:

```
16 × [ 3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN) ]
```

That means for every 4 sublayers, only one uses traditional (quadratic) attention. The rest use linear attention. This dramatically reduces the memory and compute cost for long sequences while the occasional full-attention layer preserves global context modelling.

Key attention specs for Qwen3.5-27B:
- **Gated DeltaNet**: 48 value heads, 16 Q/K heads, head dim 128
- **Gated Attention**: 24 Q heads, 4 KV heads (grouped query), head dim 256, 64-dim RoPE

### MoE in 35B-A3B

The 35B-A3B variant layers sparse MoE on top of the DeltaNet backbone:

- **256 total experts** per MoE block
- **8 routed + 1 shared expert activated** per token
- Expert intermediate dimension: 512

The router selects 8 specialist sub-networks per token from a pool of 256, then blends their outputs. Because most experts stay dormant during any single forward pass, the effective FLOP count stays near the 3B range despite 35B parameters being stored in VRAM.

---

## The `<think>` Tag: Chain-of-Thought by Default

Qwen3.5 is the first generation in the series to ship **thinking mode on by default**. Every response starts with an internal scratchpad enclosed in `<think>` tags before the model commits to its final answer:

```
<think>
The user is asking about X. Let me reason through this step by step.
First, ...
Therefore, ...
</think>

Here is my answer: ...
```

This is not cosmetic. The model was post-trained — using large-scale reinforcement learning — to genuinely reason inside these tags rather than hallucinate a plausible-sounding scratchpad. The thinking tokens are part of the model's actual computation graph during generation; they influence the final output.

You can disable thinking at inference time:

```python
chat_response = client.chat.completions.create(
    model="Qwen/Qwen3.5-27B",
    messages=messages,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
```

When fine-tuning, whether you train on the `<think>` tokens or not has a large impact on the resulting model's reasoning quality — more on this below.

---

## Tokenizer

Qwen3.5 uses a **BPE tokenizer with a vocabulary of 248,320 tokens** (padded). This large vocabulary is what enables strong multilingual coverage across 201 languages without ballooning sequence lengths. The tokenizer is implemented via Hugging Face's `PreTrainedTokenizerFast` and is fully compatible with `tiktoken`-style BPE.

Special tokens you'll encounter constantly during fine-tuning:

| Token | Role |
|---|---|
| `<\|im_start\|>` | Begin a conversational turn |
| `<\|im_end\|>` | End a conversational turn |
| `<think>` | Begin chain-of-thought reasoning block |
| `</think>` | End chain-of-thought reasoning block |
| `<\|endoftext\|>` | End of sequence |

The chat template follows **ChatML format**:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the boiling point of water at high altitude?<|im_end|>
<|im_start|>assistant
<think>
At high altitude, atmospheric pressure is lower. Lower pressure means...
water boils at a lower temperature than 100°C at sea level.
</think>
At high altitude, water boils below 100°C — around 90°C at 3,000 metres.
<|im_end|>
```

When you call `tokenizer.apply_chat_template(messages, tokenize=False)`, this is exactly the string you get back. Every supervised fine-tuning (SFT) training run converts your raw dataset into strings that follow this template before tokenizing.

---

## Fine-Tuning Strategy: LoRA with Unsloth

Full fine-tuning a 27B or 35B model is impractical for most labs. **LoRA (Low-Rank Adaptation)** injects small, trainable rank-decomposition matrices into specific weight matrices and freezes everything else. Instead of updating all 27 billion parameters, you update a few hundred million — typically less than 1% of the total.

[**Unsloth**](https://github.com/unslothai/unsloth) is a library that implements highly optimised kernels for LoRA training. The key gains come from:

- **Custom Triton kernels** for the backward pass
- **Gradient checkpointing in "unsloth" mode**, which recomputes activations on the backward pass instead of storing them, cutting VRAM by ~30% for long sequences
- **4-bit quantised base weights** (optional) for further VRAM reduction

We use `lora_rank=16` and `lora_alpha=16` as a reasonable starting point. The alpha-to-rank ratio of 1:1 means the LoRA scale factor stays at 1.0, keeping training stable.

### Which modules to target

For Qwen3.5, we attach LoRA adapters to both attention and FFN projections:

```python
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # attention
    "gate_proj", "up_proj", "down_proj",       # dense FFN
    "gate_up_proj",                            # MoE fused projection (35B-A3B only)
]
```

Targeting only attention weights is common for smaller rank budgets, but including FFN projections consistently produces better downstream task performance for reasoning-heavy domains.

---

## Training on Responses Only — And Why `<think>` Matters

By default, SFT computes cross-entropy loss over the **entire** tokenized sequence, including the user prompt and system message. That means the model is penalised for not predicting the user's question, which is wasteful.

Unsloth's `train_on_responses_only` utility masks all prompt tokens and only backpropagates through the assistant turn:

```python
from unsloth.chat_templates import train_on_responses_only

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part    = "<|im_start|>assistant\n<think>",
)
```

The critical detail: `response_part` starts with `<|im_start|>assistant\n<think>`. This means the loss begins **at the `<think>` token**, not at the first word of the final answer. The model must learn to:

1. Open a `<think>` block
2. Reason step-by-step inside it
3. Close with `</think>`
4. Produce a clean final answer

If you set `response_part = "<|im_start|>assistant\n"` instead (skipping `<think>`), the model can learn to answer correctly but the thinking scaffold degrades — it has no gradient signal pushing it to reason inside the tags.

Your dataset outputs should therefore look like:

```
<think>
Step-by-step reasoning...
</think>
Final concise answer.
```

---

## Setup 1: Multi-GPU DDP — Qwen3.5-27B on 8 × A100 (80GB)

The full script is on GitHub: [**train_dense_ddp.py**](https://github.com/shaafsalman/qwen35-finetune/blob/main/train_dense_ddp.py)

### Run command

```bash
UNSLOTH_COMPILE_TRANSFORMERS=0 torchrun --nproc_per_node 8 train_dense_ddp.py
```

### Key design choices

**One GPU per process.** Each `torchrun` worker sets `CUDA_VISIBLE_DEVICES` to its own `LOCAL_RANK`. This is the simplest DDP pattern and avoids any multi-GPU visibility issues in Unsloth.

```python
local_rank = int(os.environ.get("LOCAL_RANK", 0))
os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
```

**BF16 full precision LoRA.** We load the base weights in BF16 (no 4-bit quantisation) for maximum numerical stability. 27B in BF16 fits comfortably across 8 × 80GB GPUs.

**Sequence packing.** Setting `packing=True` in `SFTConfig` bins multiple short sequences into a single training window, maximising GPU utilisation and reducing the effective number of wasted padding tokens.

**Save from rank 0 only.** After training, all DDP ranks must synchronise before the primary process saves:

```python
if dist.is_available() and dist.is_initialized():
    dist.barrier()

if local_rank == 0:
    model.save_pretrained_merged(MERGED_DIR, tokenizer, save_method="merged_16bit")
```

### Effective batch size

With `BATCH_SIZE=2` per GPU, 8 GPUs, and `GRAD_ACCUM=4`:

```
2 × 8 × 4 = 64 effective examples per optimizer step
```

---

## Setup 2: Single GPU — Qwen3.5-35B-A3B (MoE) on 1 × H100 (80GB)

The full script is on GitHub: [**train_moe_single_gpu.py**](https://github.com/shaafsalman/qwen35-finetune/blob/main/train_moe_single_gpu.py)

### Run command

```bash
python train_moe_single_gpu.py
```

### Why BF16 and not 4-bit for MoE?

Unsloth's documentation explicitly cautions against 4-bit QLoRA for MoE architectures. The expert routing mechanism introduces non-trivial numerical sensitivity; quantising the base weights degrades routing decisions and can cause training instability. BF16 LoRA on the 35B-A3B fits on a single H100 80GB because only ~3B parameters are active per forward pass — peak activation memory stays bounded.

### The `FastModel` API

For the 35B-A3B MoE, we use `FastModel` instead of `FastLanguageModel`:

```python
from unsloth import FastModel

model, processor = FastModel.from_pretrained(
    model_name    = "unsloth/Qwen3.5-35B-A3B",
    max_seq_length = 4096,
    load_in_4bit  = False,
    load_in_16bit = True,
)

# FastModel may return a multimodal processor — extract the text tokenizer
tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
```

`FastModel` is Unsloth's unified entry point for models that include vision encoders or non-standard architectures. Even if you only intend to fine-tune the text path, use `FastModel` for any Qwen3.5 variant to avoid dtype mismatches in the MoE routing layers.

### The `UNSLOTH_COMPILE_DISABLE=1` flag

MoE models with mixed-precision routing are sensitive to PyTorch's `torch.compile`. The compiled graph can introduce BF16/FP32 dtype mismatches that cause NaN losses. Setting this flag at the very top of the script — **before any imports** — prevents the compiler from running:

```python
import os
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

# NOW import unsloth and everything else
from unsloth import FastModel
```

Order matters. If any Unsloth module is imported before this flag is set, the compiler may already have been initialised.

---

## Dataset Formatting

Both scripts expect an Alpaca-style dataset with three columns: `instruction`, `input`, `output`. The `output` column should contain the full assistant response — ideally already structured with `<think>` and `</think>` tags if you want to train reasoning.

Formatting a row into ChatML:

```python
def format_to_chat(example):
    instruction = example["instruction"].strip()
    inp         = example["input"].strip()
    output      = example["output"].strip()

    user_content = f"{instruction}\n\n{inp}" if inp else instruction

    messages = [
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": output},
    ]
    return {"text": tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )}
```

The scripts also filter sequences that exceed `MAX_SEQ_LENGTH` before training to avoid silent truncation errors.

---

## After Training: Merging and Serving

LoRA adapters are small and fast to train, but they live as a separate set of weight deltas on top of the frozen base model. For production serving, you'll want to **merge the adapters into the base weights** to create a standalone model:

```python
# Merge and save locally
model.save_pretrained_merged("merged_model", tokenizer, save_method="merged_16bit")

# Or merge and push directly to HuggingFace
model.push_to_hub_merged("username/my-fine-tuned-qwen35", tokenizer, save_method="merged_16bit")
```

The merged model can then be served with **vLLM**:

```bash
vllm serve username/my-fine-tuned-qwen35 \
    --dtype bfloat16 \
    --max-model-len 4096
```

vLLM's speculative decoding and multi-token prediction (MTP) support aligns well with Qwen3.5's training recipe, which also uses MTP during pre-training.

---

## Quick Reference

| | DDP (27B Dense) | Single GPU (35B-A3B MoE) |
|---|---|---|
| **Script** | `train_dense_ddp.py` | `train_moe_single_gpu.py` |
| **Hardware** | 8 × A100 80GB | 1 × H100 80GB |
| **Precision** | BF16 LoRA | BF16 LoRA |
| **LoRA rank** | 16 | 16 |
| **Max seq len** | 2048 | 4096 |
| **Batch (effective)** | 64 | 4 |
| **Unsloth API** | `FastLanguageModel` | `FastModel` |
| **Key env flag** | `UNSLOTH_COMPILE_TRANSFORMERS=0` | `UNSLOTH_COMPILE_DISABLE=1` |
| **Response-only training** | ✅ with `<think>` | ❌ (full sequence loss) |

---

## GitHub Repository

Both scripts — cleaned and ready to run against your own dataset — are available here:

**[github.com/shaafsalman/qwen35-finetune](https://github.com/shaafsalman/qwen35-finetune)**

Clone, swap in your dataset name and HuggingFace repo, and you're off.

---

*Thanks for reading. If you found this useful, consider starring the repo or leaving a comment with questions.*

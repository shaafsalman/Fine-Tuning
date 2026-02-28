"""
train_dense_ddp.py — Qwen3.5-27B LoRA fine-tune on a medical reasoning dataset
Hardware : 8 × A100-SXM4-80GB (DDP via torchrun)
Run with : UNSLOTH_COMPILE_TRANSFORMERS=0 torchrun --nproc_per_node 8 train_dense_ddp.py

Key design decisions:
- FastLanguageModel (text-only, not vision)
- BF16 full precision LoRA (no 4-bit quantization)
- response_part includes "<think>" so thinking tokens are trained
- Unsloth gradient checkpointing ("unsloth" mode) for VRAM savings
- DDP: proper barrier placement, dist init check before barrier
- Save: only rank-0 saves
"""

import os, sys, glob, shutil
import torch
import torch.distributed as dist

# ── per-rank GPU assignment ────────────────────────────────────────────────────
local_rank = int(os.environ.get("LOCAL_RANK", 0))
os.environ["CUDA_VISIBLE_DEVICES"]          = str(local_rank)
os.environ["PYTORCH_CUDA_ALLOC_CONF"]       = "expandable_segments:True"
os.environ["TORCHDYNAMO_DISABLE"]           = "1"
os.environ["UNSLOTH_MOE_BACKEND"]           = "triton"
os.environ["UNSLOTH_DISABLE_AUTOTUNE"]      = "1"
os.environ["TRITON_CACHE_DIR"]              = f"/tmp/triton_cache_{local_rank}"
os.environ["UNSLOTH_COMPILE_TRANSFORMERS"]  = "0"

# ── config ─────────────────────────────────────────────────────────────────────
MODEL_NAME          = "unsloth/Qwen3.5-27B"   # use unsloth mirror (faster dl)
DATASET_NAME        = "<YOUR_DATASET>"         # replace with your dataset
DATASET_SPLIT       = "train"
MAX_SEQ_LENGTH      = 2048
LORA_RANK           = 16
LORA_ALPHA          = 16
NUM_TRAIN_EPOCHS    = 1
BATCH_SIZE          = 2     # 2 per GPU × 8 GPUs = 16 (before grad accum)
GRAD_ACCUM          = 4     # effective batch = 64
WARMUP_STEPS        = 5
LEARNING_RATE       = 2e-4
SEED                = 3407
LOGGING_STEPS       = 10
DATASET_NUM_PROC    = 4
OUTPUT_DIR          = "outputs_qwen35"
MERGED_DIR          = "qwen35_merged"
HF_REPO             = "<YOUR_HF_REPO>"         # e.g. "username/model-name"
HF_TOKEN            = os.environ.get("HF_TOKEN", "")

# ── clear stale compiled cache (rank 0 only, before dist init) ────────────────
if local_rank == 0:
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unsloth_compiled_cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"Cleared stale compiled cache: {cache_dir}")

# ── unsloth imports ────────────────────────────────────────────────────────────
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

if local_rank == 0:
    print(f"GPU {local_rank}: {torch.cuda.get_device_name(0)}")

# ── load model ─────────────────────────────────────────────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = MODEL_NAME,
    max_seq_length  = MAX_SEQ_LENGTH,
    load_in_4bit    = False,
    load_in_8bit    = False,   # BF16 full precision
    full_finetuning = False,   # LoRA
)

# ── LoRA adapters ──────────────────────────────────────────────────────────────
model = FastLanguageModel.get_peft_model(
    model,
    r                       = LORA_RANK,
    lora_alpha              = LORA_ALPHA,
    lora_dropout            = 0,
    bias                    = "none",
    target_modules          = [
        "q_proj", "k_proj", "v_proj", "o_proj",   # attention
        "gate_proj", "up_proj", "down_proj",        # FFN
        "out_proj",
    ],
    use_gradient_checkpointing = "unsloth",  # ~30% less VRAM on long seqs
    random_state            = SEED,
    use_rslora              = False,
    loftq_config            = None,
)
model.config.use_cache = False

# ── dataset formatting ─────────────────────────────────────────────────────────
# Qwen3.5 uses ChatML format: <|im_start|>role\ncontent<|im_end|>
# The <think>...</think> block in the output trains the model's CoT reasoning.
PROMPT_TEMPLATE = """\
<|im_start|>system
{instruction}<|im_end|>
<|im_start|>user
{input}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""

def format_dataset(examples):
    texts = []
    for instruction, inp, output in zip(
        examples["instruction"],
        examples["input"],
        examples["output"],
    ):
        texts.append(PROMPT_TEMPLATE.format(
            instruction = instruction.strip(),
            input       = inp.strip(),
            output      = output.strip(),
        ))
    return {"text": texts}

if local_rank == 0:
    print(f"Loading dataset: {DATASET_NAME} ...")

dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
dataset = dataset.map(
    format_dataset,
    batched         = True,
    remove_columns  = dataset.column_names,
    num_proc        = DATASET_NUM_PROC,
)

if local_rank == 0:
    print(f"Dataset size: {len(dataset):,} examples")

# ── trainer ────────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model           = model,
    tokenizer       = tokenizer,
    train_dataset   = dataset,
    args = SFTConfig(
        dataset_text_field          = "text",
        max_seq_length              = MAX_SEQ_LENGTH,
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        num_train_epochs            = NUM_TRAIN_EPOCHS,
        warmup_steps                = WARMUP_STEPS,
        learning_rate               = LEARNING_RATE,
        lr_scheduler_type           = "cosine",
        weight_decay                = 0.01,
        bf16                        = True,
        logging_steps               = LOGGING_STEPS,
        save_steps                  = 200,
        save_total_limit            = 3,
        output_dir                  = OUTPUT_DIR,
        optim                       = "adamw_8bit",
        seed                        = SEED,
        dataset_num_proc            = DATASET_NUM_PROC,
        packing                     = True,
        dataloader_num_workers      = 4,
        dataloader_pin_memory       = True,
        report_to                   = "none",
        ddp_find_unused_parameters  = False,
    ),
)

# ── train on responses only ────────────────────────────────────────────────────
# instruction_part / response_part tell the trainer which tokens to compute loss on.
# Including "<think>" in response_part means the model learns to produce
# chain-of-thought reasoning before the final answer.
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part    = "<|im_start|>assistant\n<think>",
)

# ── resume from checkpoint if exists ──────────────────────────────────────────
ckpts  = sorted(glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*")))
resume = ckpts[-1] if ckpts else False
if local_rank == 0:
    print(f"Resuming from: {resume}" if resume else "Starting fresh")

# ── train ──────────────────────────────────────────────────────────────────────
trainer_stats = trainer.train(resume_from_checkpoint=resume)

# ── save (rank 0 only) ─────────────────────────────────────────────────────────
if dist.is_available() and dist.is_initialized():
    dist.barrier()  # wait for all ranks to finish before saving

if local_rank == 0:
    print("Training complete. Saving LoRA adapters...")
    model.save_pretrained(OUTPUT_DIR + "_lora")
    tokenizer.save_pretrained(OUTPUT_DIR + "_lora")

    print("Merging to 16-bit...")
    model.save_pretrained_merged(MERGED_DIR, tokenizer, save_method="merged_16bit")
    print(f"Saved merged model to: {MERGED_DIR}")

    if HF_TOKEN:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        api.create_repo(repo_id=HF_REPO, exist_ok=True, private=False)
        print(f"Uploading to {HF_REPO} ...")
        api.upload_folder(
            folder_path = MERGED_DIR,
            repo_id     = HF_REPO,
            repo_type   = "model",
        )
        print("Upload complete!")
    else:
        print("HF_TOKEN not set — skipping upload.")

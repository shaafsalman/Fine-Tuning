import unsloth  # MUST be first import

import os
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel
from accelerate import PartialState

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
MODEL_NAME        = ""
DATASET_NAME      = ""
HF_PUSH_REPO      = ""

MAX_SEQ_LENGTH    = 4096
LOAD_IN_4BIT      = False
LOAD_IN_16BIT     = True

LORA_R            = 32
LORA_ALPHA        = 32
LORA_DROPOUT      = 0.0

# With 80GB per GPU and only ~14GB used at batch=2, push to 8
PER_DEVICE_BATCH  = 16      # was 2 → 4x more throughput per GPU
GRAD_ACCUM        = 1     # effective batch = 8 x 2 x 8 GPUs = 128
MAX_STEPS         = -1
NUM_EPOCHS        = 1
LR                = 2e-4
WARMUP_STEPS      = 100    # replaces deprecated warmup_ratio
LR_SCHEDULER      = "cosine"
WEIGHT_DECAY      = 0.01
SEED              = 3407

OUTPUT_DIR        = "./outputs/qwen35-4b-medreasoning"
MERGED_DIR        = "./outputs/qwen35-4b-medreasoning-merged"

# ─────────────────────────────────────────
# 1. Load model + tokenizer
# ─────────────────────────────────────────
print("Loading model …")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name             = MODEL_NAME,
    max_seq_length         = MAX_SEQ_LENGTH,
    load_in_4bit           = LOAD_IN_4BIT,
    load_in_16bit          = LOAD_IN_16BIT,
    full_finetuning        = False,
)

# ─────────────────────────────────────────
# 2. LoRA adapters
# ─────────────────────────────────────────
model = FastLanguageModel.get_peft_model(
    model,
    r                          = LORA_R,
    target_modules             = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha                 = LORA_ALPHA,
    lora_dropout               = LORA_DROPOUT,
    bias                       = "none",
    use_gradient_checkpointing = "unsloth",
    random_state               = SEED,
    use_rslora                 = False,
    loftq_config               = None,
)

# ─────────────────────────────────────────
# 3. Load & format dataset
#    main_process_first: only rank-0 downloads/processes, others wait
#    avoids 8 processes hitting HF hub simultaneously
# ─────────────────────────────────────────
state = PartialState()

with state.main_process_first():
    print("Loading dataset …")
    dataset = load_dataset(DATASET_NAME, split="train")
    print("Dataset columns:", dataset.column_names)

    def format_example(example):
        user_content = example["instruction"]
        if example.get("input", "").strip():
            user_content += "\n\n" + example["input"]
        # output already contains <think>...</think>\boxed{...}
        assistant_content = example["output"]
        messages = [
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize              = False,
            add_generation_prompt = False,
        )
        return {"text": text}

    dataset = dataset.map(
        format_example,
        remove_columns = dataset.column_names,
        num_proc       = 8,
    )

# ─────────────────────────────────────────
# 4. Trainer
# ─────────────────────────────────────────
trainer = SFTTrainer(
    model               = model,
    tokenizer           = tokenizer,
    train_dataset       = dataset,
    max_seq_length      = MAX_SEQ_LENGTH,
    args = SFTConfig(
        dataset_text_field          = "text",
        packing                     = True,   # fills sequences to MAX_SEQ_LENGTH — major speedup
        per_device_train_batch_size = PER_DEVICE_BATCH,
        gradient_accumulation_steps = GRAD_ACCUM,
        num_train_epochs            = NUM_EPOCHS,
        max_steps                   = MAX_STEPS,
        learning_rate               = LR,
        warmup_steps                = WARMUP_STEPS,
        lr_scheduler_type           = LR_SCHEDULER,
        weight_decay                = WEIGHT_DECAY,
        fp16                        = not torch.cuda.is_bf16_supported(),
        bf16                        = torch.cuda.is_bf16_supported(),
        logging_steps               = 10,
        save_steps                  = 200,
        save_total_limit            = 3,
        output_dir                  = OUTPUT_DIR,
        seed                        = SEED,
        report_to                   = "none",
        dataset_num_proc            = 8,
        ddp_find_unused_parameters  = False,
        dataloader_num_workers      = 4,
        dataloader_pin_memory       = True,
        neftune_noise_alpha         = 5,      # mild noise regularization for instruction tuning
        optim                       = "adamw_8bit",  # 8-bit Adam — saves ~2GB VRAM per GPU
    ),
)

# ─────────────────────────────────────────
# 5. Train
# ─────────────────────────────────────────
print("Starting training …")
trainer_stats = trainer.train()
print(f"Training complete. Steps: {trainer_stats.global_step}")
print(f"Peak VRAM per GPU: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")

# ─────────────────────────────────────────
# 6. Save LoRA adapter
# ─────────────────────────────────────────
if int(os.environ.get("LOCAL_RANK", 0)) == 0:
    adapter_dir = os.path.join(OUTPUT_DIR, "final-adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"LoRA adapter saved → {adapter_dir}")

# ─────────────────────────────────────────
# 7. Merge for vLLM
# ─────────────────────────────────────────
if int(os.environ.get("LOCAL_RANK", 0)) == 0:
    print("Merging LoRA into base model for vLLM …")
    model.save_pretrained_merged(
        MERGED_DIR,
        tokenizer,
        save_method = "merged_16bit",
    )
    print(f"Merged model saved → {MERGED_DIR}")

# ─────────────────────────────────────────
# 8. Push to HF Hub
# ─────────────────────────────────────────
if int(os.environ.get("LOCAL_RANK", 0)) == 0:
    print(f"Pushing to HF Hub: {HF_PUSH_REPO} …")
    model.push_to_hub_merged(
        HF_PUSH_REPO,
        tokenizer,
        save_method = "merged_16bit",
    )
    print(f"✅ Done → https://huggingface.co/{HF_PUSH_REPO}")

# ─────────────────────────────────────────
# 9. vLLM serve command
# ─────────────────────────────────────────
if int(os.environ.get("LOCAL_RANK", 0)) == 0:
    print(f"""
To serve with vLLM:
  vllm serve {HF_PUSH_REPO} \\
      --tensor-parallel-size 8 \\
      --max-model-len 32768 \\
      --reasoning-parser qwen3 \\
      --enable-prefix-caching
""")

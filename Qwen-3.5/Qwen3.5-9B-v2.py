from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset
from huggingface_hub import snapshot_download
from trl import SFTTrainer, SFTConfig
import torch
import os
import json
import shutil

# ── config ────────────────────────────────────────────────────────────────────
MODEL_NAME  = "unsloth/Qwen3.5-9B"
BASE_MODEL  = "Qwen/Qwen3.5-9B"          # original HF model for config files
HF_REPO     = "repo"
HF_TOKEN    = ""
MAX_SEQ_LEN = 4096
LORA_DIR    = "qwen35_9b_med_lora"
MERGED_DIR  = "qwen35_9b_med_merged"

# ── model ─────────────────────────────────────────────────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MODEL_NAME,
    max_seq_length = MAX_SEQ_LEN,
    load_in_4bit   = False,   # NOT recommended for Qwen3.5
    load_in_8bit   = False,
    load_in_16bit  = True,    # bf16 — 22GB VRAM for 9B
)

model = FastLanguageModel.get_peft_model(
    model,
    r                         = 16,
    lora_alpha                = 16,   # keep equal to r per Unsloth docs
    target_modules            = ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"],
    lora_dropout              = 0,
    bias                      = "none",
    use_gradient_checkpointing= "unsloth",
    random_state              = 3407,
)

# ── dataset ───────────────────────────────────────────────────────────────────
ds = load_dataset("", split="train")
print(f"Dataset size: {len(ds)}")

def format_dataset(examples):
    texts = []
    for instruction, inp, output in zip(
        examples["instruction"],
        examples["input"],
        examples["output"],
    ):
        convo = [
            {"role": "system",    "content": instruction.strip()},
            {"role": "user",      "content": inp.strip()},
            {"role": "assistant", "content": output.strip()},
        ]
        # do NOT pass enable_thinking here:
        # - output field already has <think>...</think> baked in
        # - enable_thinking is inference-time only (add_generation_prompt=True)
        # - passing it here would cause duplicate <think> tags
        texts.append(
            tokenizer.apply_chat_template(
                convo,
                tokenize              = False,
                add_generation_prompt = False,
            )
        )
    return {"text": texts}

ds = ds.map(format_dataset, batched=True, num_proc=8, remove_columns=ds.column_names)

# verify format — must show <think> inside assistant turn
sample = ds[0]["text"]
assert "<|im_start|>assistant" in sample
assert "<think>"   in sample
assert "</think>"  in sample
assert sample.index("<think>") > sample.index("<|im_start|>assistant")
print("FORMAT CHECK PASSED")
print(sample[:1000])
print()

# ── trainer ───────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model         = model,
    tokenizer     = tokenizer,
    train_dataset = ds,
    args          = SFTConfig(
        output_dir                  = "qwen35_9b_med_checkpoints",
        dataset_text_field          = "text",
        max_length                  = MAX_SEQ_LEN,
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 2,
        num_train_epochs            = 1,
        warmup_steps                = 10,
        learning_rate               = 2e-4,
        lr_scheduler_type           = "cosine",
        weight_decay                = 0.01,
        bf16                        = True,
        logging_steps               = 10,
        save_steps                  = 200,
        save_total_limit            = 3,
        optim                       = "adamw_8bit",
        seed                        = 3407,
        dataset_num_proc            = 8,
        packing                     = True,
        report_to                   = "none",
    ),
)

# full assistant turn — NOT just <think> — so model learns to generate after </think>
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part    = "<|im_start|>assistant\n",
)

# verify masking — only assistant content should be visible
decoded = tokenizer.decode(
    [tokenizer.pad_token_id if x == -100 else x
     for x in trainer.train_dataset[0]["labels"]]
).replace(tokenizer.pad_token, " ")
assert "<|im_start|>user" not in decoded, "user turn is NOT masked — fix response_part"
assert "<think>" in decoded, "assistant content missing from labels"
print("MASKING CHECK PASSED")
print(decoded[:600])
print()

# ── train ─────────────────────────────────────────────────────────────────────
trainer.train()

# ── save lora ─────────────────────────────────────────────────────────────────
model.save_pretrained(LORA_DIR)
tokenizer.save_pretrained(LORA_DIR)
print(f"LoRA adapter saved to {LORA_DIR}")

# ── merge to 16bit ────────────────────────────────────────────────────────────
model.save_pretrained_merged(
    MERGED_DIR,
    tokenizer,
    save_method = "merged_16bit",
)
print(f"Merged model saved to {MERGED_DIR}")

# ── restore original base model config files ──────────────────────────────────
# Unsloth merged model may have altered config.json/tokenizer_config.json
# We download the original base model files and restore them to ensure
# tokenizer, chat template, processor config are identical to base model
print("Restoring original base model config files...")

base_cache = snapshot_download(
    repo_id   = BASE_MODEL,
    token     = HF_TOKEN,
    ignore_patterns = ["*.safetensors", "*.bin", "*.pt"],  # only config files
)

# files to restore from base model
CONFIG_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "chat_template.json",
]

for fname in CONFIG_FILES:
    src = os.path.join(base_cache, fname)
    dst = os.path.join(MERGED_DIR, fname)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  restored {fname}")
    else:
        print(f"  skipped  {fname} (not in base model)")

print("Config restore complete.")

# ── push to hub ───────────────────────────────────────────────────────────────
if HF_TOKEN and HF_REPO:
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=HF_REPO, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path    = MERGED_DIR,
        repo_id        = HF_REPO,
        repo_type      = "model",
        commit_message = "Qwen3.5-9B MedReasoning fine-tune merged 16bit",
    )
    print(f"Pushed → https://huggingface.co/{HF_REPO}")

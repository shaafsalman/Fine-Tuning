import os
import glob
import json
import shutil
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import SFTTrainer, SFTConfig
from huggingface_hub import HfApi

DATASET_NAME  = "Dataset_name"
MODEL_NAME    = "Qwen/Qwen3.5-35B-A3B"
HF_MODEL_REPO = "output_repo"
OUTPUT_DIR    = "medreasoning_checkpoints"
LORA_DIR      = OUTPUT_DIR + "_lora"
MERGED_DIR    = OUTPUT_DIR + "_merged"
MAX_SEQ_LEN   = 2048
LORA_RANK     = 16
LORA_ALPHA    = 32
BATCH_SIZE    = 1
GRAD_ACCUM    = 4
EPOCHS        = 3
LR            = 2e-4
WARMUP_STEPS  = 100
SEED          = 3407
HF_TOKEN      = os.environ.get("HF_TOKEN", "")

PROMPT_TEMPLATE = (
    "<|im_start|>system\n{instruction}<|im_end|>\n"
    "<|im_start|>user\n{input}<|im_end|>\n"
    "<|im_start|>assistant\n{output}<|im_end|>"
)


def format_medreasoning(examples):
    texts = []
    for instruction, inp, output in zip(
        examples["instruction"], examples["input"], examples["output"]
    ):
        texts.append(PROMPT_TEMPLATE.format(
            instruction=instruction.strip(),
            input=inp.strip(),
            output=output.strip(),
        ))
    return {"text": texts}


def train():
    dataset = load_dataset(DATASET_NAME, split="train")
    dataset = dataset.map(
        format_medreasoning,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=8,
    )

    # FIX: use AutoProcessor (not AutoTokenizer) so all processor files are
    # saved alongside the adapter — prevents missing preprocessor_config.json
    # and video_preprocessor_config.json during merge/push
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right",
    )
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # FIX: use AutoModelForImageTextToText (not AutoModelForCausalLM) so the
    # full VLM is loaded including vision encoder weights. Using CausalLM
    # produces a text-only checkpoint that causes 400+ missing visual.*
    # weight errors when vLLM tries to load Qwen3_5MoeForConditionalGeneration
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        # FIX: target_modules must use fully-qualified paths matching the VLM
        # model's parameter names (language_model.layers.X.self_attn.*_proj).
        # Using bare names like "q_proj" causes PEFT to match nothing under the
        # VLM wrapper, resulting in all LoRA keys being "missing" at merge time.
        target_modules=[
            "language_model.layers.*.self_attn.q_proj",
            "language_model.layers.*.self_attn.k_proj",
            "language_model.layers.*.self_attn.v_proj",
            "language_model.layers.*.self_attn.o_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    ckpts = sorted(glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*")))
    resume = ckpts[-1] if ckpts else False

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        max_length=MAX_SEQ_LEN,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        bf16=True,
        max_grad_norm=1.0,
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        optim="adamw_torch_fused",
        seed=SEED,
        dataset_num_proc=8,
        packing=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )
    trainer.train(resume_from_checkpoint=resume)

    if trainer.is_world_process_zero():
        print("Saving LoRA adapter...")
        model.save_pretrained(LORA_DIR)
        # FIX: save full processor (not just tokenizer) so
        # preprocessor_config.json, video_preprocessor_config.json, and
        # processor_config.json are all present for vLLM
        processor.save_pretrained(LORA_DIR)
        print(f"Adapter saved to {LORA_DIR}")


def merge_and_push():
    print("Loading base model for merge...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, LORA_DIR)
    model = model.merge_and_unload(progressbar=True)

    print(f"Saving merged model to {MERGED_DIR}...")
    model.save_pretrained(MERGED_DIR, safe_serialization=True, max_shard_size="4GB")

    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    processor.save_pretrained(MERGED_DIR)

    # FIX: patch tokenizer_class — PEFT adapter save writes "TokenizersBackend"
    # which vLLM rejects. Must be "Qwen2TokenizerFast" to match the processor
    # class Qwen3VLProcessor expects (Qwen2Tokenizer / Qwen2TokenizerFast)
    tok_cfg_path = os.path.join(MERGED_DIR, "tokenizer_config.json")
    with open(tok_cfg_path) as f:
        tok_cfg = json.load(f)
    tok_cfg["tokenizer_class"] = "Qwen2TokenizerFast"
    with open(tok_cfg_path, "w") as f:
        json.dump(tok_cfg, f, indent=2)
    print("Patched tokenizer_class -> Qwen2TokenizerFast")

    # FIX: ensure config.json has the VLM architecture, not the text-only
    # variant (qwen3_5_moe_text / Qwen3_5MoeForCausalLM) that CausalLM saves.
    # vLLM resolves "Qwen3_5MoeForConditionalGeneration" but not the text variant.
    cfg_path = os.path.join(MERGED_DIR, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["model_type"] = "qwen3_5_moe"
    cfg["architectures"] = ["Qwen3_5MoeForConditionalGeneration"]
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print("Patched config.json architecture")

    if not HF_TOKEN:
        print("No HF_TOKEN set, skipping push.")
        return

    print(f"Pushing to {HF_MODEL_REPO}...")
    # FIX: use HfApi.upload_folder instead of model.push_to_hub to avoid
    # deprecated safe_serialization kwarg issues with large sharded models
    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=HF_MODEL_REPO, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=MERGED_DIR,
        repo_id=HF_MODEL_REPO,
        repo_type="model",
        commit_message="Add merged Qwen3.5-35B-A3B MedReasoning model",
    )
    print(f"Done — https://huggingface.co/{HF_MODEL_REPO}")


def main():
    train()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        merge_and_push()


if __name__ == "__main__":
    main()

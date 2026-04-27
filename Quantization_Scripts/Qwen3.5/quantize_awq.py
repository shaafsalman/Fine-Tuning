"""
AWQ W8A16 Quantization for Qwen3.5-9B (VL Architecture)

Method: Activation-aware Weight Quantization (AWQ) with calibration data
Output: ~13 GB  (vs ~19 GB BF16 original)
Time  : ~35-45 minutes on A100 40GB

Architecture notes (Qwen3.5-9B):
  - Unified VLM: vision encoder is fused, not a separate tower
  - Hybrid layers: 8x full-attention + 24x GDN (linear_attn) blocks
  - GDN layers ignored — same as Qwen's official FP8/GPTQ models
  - offload_device=cpu required to prevent OOM during AWQ scale search
"""

import os
import torch
from datasets import load_dataset, Dataset
from transformers import AutoProcessor, AutoTokenizer, AutoModelForImageTextToText
from llmcompressor import oneshot
from llmcompressor.modifiers.transform.awq import AWQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from huggingface_hub import HfApi
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_id",       default="YOUR_ORG/YOUR-FINETUNED-MODEL")
parser.add_argument("--dataset_id",     default="YOUR_ORG/YOUR-CALIBRATION-DATASET")
parser.add_argument("--save_path",      default="/tmp/YOUR-MODEL-AWQ-W8A16")
parser.add_argument("--hf_repo",        default="YOUR_HF_USERNAME/YOUR-MODEL-AWQ-W8A16")
parser.add_argument("--num_samples",    type=int, default=256)
parser.add_argument("--max_seq_len",    type=int, default=2048)
parser.add_argument("--no_push",        action="store_true")
args = parser.parse_args()

os.makedirs(args.save_path, exist_ok=True)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print("=" * 62)
print(f"  Method   : AWQ W8A16 (calibration-based)")
print(f"  Model    : {args.model_id}")
print(f"  Dataset  : {args.dataset_id} ({args.num_samples} samples)")
print(f"  Save to  : {args.save_path}")
print(f"  HF repo  : {args.hf_repo}")
print("=" * 62)

print("\n[1/5] Loading tokenizer + processor...")
tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

print("[2/5] Loading full VL model in BF16...")
model = AutoModelForImageTextToText.from_pretrained(
    args.model_id,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
print(f"      Architecture : {type(model).__name__}")

print(f"\n[3/5] Building calibration dataset from {args.dataset_id}...")
raw = load_dataset(args.dataset_id, split="train")
raw = raw.shuffle(seed=42).select(range(args.num_samples))

calib_texts = []
for sample in raw:
    # Adjust field names to match your dataset schema
    instruction  = sample.get("instruction", "").strip()
    user_input   = sample.get("input", "").strip()
    model_output = sample.get("output", "").strip()
    user_content = f"{instruction}\n\n{user_input}" if user_input else instruction
    messages = [
        {"role": "system",    "content": "You are a helpful assistant. Think step by step before answering."},
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": model_output},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    calib_texts.append(text.strip())

calib_dataset = Dataset.from_dict({"text": calib_texts})
print(f"      Prepared {len(calib_texts)} samples.")

IGNORE_LAYERS = [
    "lm_head",
    "re:visual.*",
    "re:merger.*",
    "re:.*linear_attn.*",
]

print("\n[4/5] Running AWQ W8A16 quantization (~35-45 min on A100)...")
print("      INT8   : full-attention + MLP layers")
print("      BF16   : lm_head | visual.* | merger.* | linear_attn.*")
print("      OOM fix: offload_device=cpu")

recipe = [
    AWQModifier(
        duo_scaling="both",
        offload_device=torch.device("cpu"),
    ),
    QuantizationModifier(
        targets=["Linear"],
        scheme="W8A16",
        ignore=IGNORE_LAYERS,
    ),
]

oneshot(
    model=model,
    tokenizer=tokenizer,
    dataset=calib_dataset,
    recipe=recipe,
    max_seq_length=args.max_seq_len,
    num_calibration_samples=args.num_samples,
    output_dir=args.save_path,
)

processor.save_pretrained(args.save_path)
size = os.popen(f"du -sh {args.save_path}").read().strip()
print(f"\n      Saved : {args.save_path}")
print(f"      Size  : {size}")

if not args.no_push:
    print(f"\n[5/5] Pushing to HuggingFace: {args.hf_repo}...")
    api = HfApi()
    api.create_repo(repo_id=args.hf_repo, exist_ok=True)
    api.upload_folder(folder_path=args.save_path, repo_id=args.hf_repo, repo_type="model")
    print(f"\n  Done. https://huggingface.co/{args.hf_repo}")
else:
    print(f"\n[5/5] Skipped push. Model at {args.save_path}")

print("""
================================================================
  Serve with vLLM:
================================================================
  VLLM_USE_DEEP_GEMM=0 vllm serve {path} \\
    --port 8000 \\
    --tensor-parallel-size 1 \\
    --max-model-len 32000 \\
    --gpu-memory-utilization 0.95 \\
    --reasoning-parser deepseek_r1 \\
    --gdn-prefill-backend triton \\
    --default-chat-template-kwargs '{{"enable_thinking": true}}' \\
    --served-model-name YOUR-MODEL-NAME \\
    --api-key YOUR-API-KEY \\
    --trust-remote-code
================================================================
""".format(path=args.save_path))

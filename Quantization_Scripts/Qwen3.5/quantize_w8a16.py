"""
W8A16 Data-Free Quantization for Qwen3.5-9B (VL Architecture)

Method: Round-to-Nearest (RTN) — no calibration data required
Output: ~11 GB  (vs ~19 GB BF16 original)
Time  : ~3 minutes on any GPU

Architecture notes (Qwen3.5-9B):
  - Unified VLM: vision encoder is fused, not a separate tower
  - Hybrid layers: 8x full-attention + 24x GDN (linear_attn) blocks
  - Vision encoder + merger + lm_head stay in BF16
"""

import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from huggingface_hub import HfApi
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_id",   default="YOUR_ORG/YOUR-FINETUNED-MODEL")
parser.add_argument("--save_path",  default="/tmp/YOUR-MODEL-W8A16")
parser.add_argument("--hf_repo",    default="YOUR_HF_USERNAME/YOUR-MODEL-W8A16")
parser.add_argument("--no_push",    action="store_true")
args = parser.parse_args()

os.makedirs(args.save_path, exist_ok=True)

print("=" * 62)
print(f"  Method   : W8A16 RTN (data-free)")
print(f"  Model    : {args.model_id}")
print(f"  Save to  : {args.save_path}")
print(f"  HF repo  : {args.hf_repo}")
print("=" * 62)

print("\n[1/4] Loading processor...")
processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

print("[2/4] Loading full VL model in BF16...")
model = AutoModelForImageTextToText.from_pretrained(
    args.model_id,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
print(f"      Architecture : {type(model).__name__}")

IGNORE_LAYERS = [
    "lm_head",
    "re:visual.*",
    "re:merger.*",
]

recipe = [
    QuantizationModifier(
        targets=["Linear"],
        scheme="W8A16",
        ignore=IGNORE_LAYERS,
    ),
]

print("\n[3/4] Quantizing (data-free, ~3 min)...")
print("      INT8   : full-attention + MLP + linear_attn layers")
print("      BF16   : lm_head | visual.* | merger.*")
oneshot(model=model, recipe=recipe, output_dir=args.save_path)

processor.save_pretrained(args.save_path)
size = os.popen(f"du -sh {args.save_path}").read().strip()
print(f"\n      Saved : {args.save_path}")
print(f"      Size  : {size}")

if not args.no_push:
    print(f"\n[4/4] Pushing to HuggingFace: {args.hf_repo}...")
    api = HfApi()
    api.create_repo(repo_id=args.hf_repo, exist_ok=True)
    api.upload_folder(folder_path=args.save_path, repo_id=args.hf_repo, repo_type="model")
    print(f"\n  Done. https://huggingface.co/{args.hf_repo}")
else:
    print(f"\n[4/4] Skipped push. Model at {args.save_path}")

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

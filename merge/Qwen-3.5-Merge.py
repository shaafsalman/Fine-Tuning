#!/usr/bin/env python3
"""
Custom SLERP + DARE-TIES merge for Qwen3.5-9B hybrid architecture.
Uses huggingface_hub snapshot_download to handle all model formats.
"""

import torch
import json
import os
import shutil
import glob
from pathlib import Path
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from tqdm import tqdm

BASE_MODEL       = "Qwen/Qwen3.5-9B"
FINETUNED_MODEL  = ""
BASE_CACHE       = "./cache_base"
FT_CACHE         = "./cache_ft"
OUTPUT_SLERP     = "./merged-slerp"
OUTPUT_DARE_TIES = "./merged-dare-ties"
DEVICE           = "cpu"

# ── Merge ratio ───────────────────────────────────────────────────────────────
# FT_RATIO = how much of the finetuned model to blend in (0.0 = pure base, 1.0 = pure ft)
# 0.30 → 70% base / 30% finetuned
FT_RATIO         = 0.30
# DARE-TIES: fraction of delta weights kept before scaling
DARE_DENSITY     = 0.70
# ─────────────────────────────────────────────────────────────────────────────


def download_model(repo_id: str, cache_dir: str) -> str:
    print(f"  Downloading {repo_id} → {cache_dir}")
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=cache_dir,
        ignore_patterns=["*.bin", "*.pt", "original/*"],
    )
    return path


def load_sharded_model(model_path: str) -> dict:
    path = Path(model_path)
    tensors = {}

    index_file = path / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
        print(f"  Found index file → loading {len(shard_files)} shards...")
        for shard in tqdm(shard_files, desc="  Shards"):
            tensors.update(load_file(str(path / shard), device=DEVICE))
        return tensors

    shard_files = sorted(glob.glob(str(path / "*.safetensors")))
    if shard_files:
        print(f"  Found {len(shard_files)} safetensors file(s) (no index)...")
        for shard in tqdm(shard_files, desc="  Shards"):
            tensors.update(load_file(shard, device=DEVICE))
        return tensors

    raise FileNotFoundError(f"No safetensors files found in {model_path}")


def slerp(t: float, v0: torch.Tensor, v1: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    orig_shape = v0.shape
    v0_flat = v0.reshape(-1).float()
    v1_flat = v1.reshape(-1).float()

    v0_norm = torch.nn.functional.normalize(v0_flat, dim=0)
    v1_norm = torch.nn.functional.normalize(v1_flat, dim=0)
    dot = torch.clamp((v0_norm * v1_norm).sum(), -1.0, 1.0)

    if abs(dot.item()) > 1.0 - eps:
        return (v0.float() + t * (v1.float() - v0.float())).to(v0.dtype).reshape(orig_shape)

    theta     = torch.acos(dot)
    sin_theta = torch.sin(theta)
    s0 = torch.sin((1.0 - t) * theta) / sin_theta
    s1 = torch.sin(t * theta) / sin_theta

    return (s0 * v0_flat + s1 * v1_flat).to(v0.dtype).reshape(orig_shape)


def dare_ties_merge(
    base: torch.Tensor,
    ft: torch.Tensor,
    density: float = DARE_DENSITY,
    weight: float = FT_RATIO,
) -> torch.Tensor:
    delta        = ft.float() - base.float()
    mask         = (torch.rand_like(delta) < density).float()
    pruned_delta = delta * mask / (density + 1e-8)
    merged       = base.float() + weight * pruned_delta
    return merged.to(base.dtype)


def save_sharded(tensors: dict, output_dir: str, shard_size_gb: float = 4.0):
    os.makedirs(output_dir, exist_ok=True)
    shard_size_bytes = int(shard_size_gb * 1024 ** 3)

    weight_map    = {}
    shards        = []
    current_shard = {}
    current_size  = 0

    for key, tensor in tensors.items():
        tensor_size = tensor.nbytes
        if current_size + tensor_size > shard_size_bytes and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size  = 0
        current_shard[key]  = tensor
        current_size       += tensor_size
    if current_shard:
        shards.append(current_shard)

    total = len(shards)
    print(f"  Writing {total} shard(s)...")
    for i, shard in enumerate(tqdm(shards, desc="  Saving"), 1):
        fname = f"model-{i:05d}-of-{total:05d}.safetensors"
        save_file(shard, os.path.join(output_dir, fname))
        for key in shard:
            weight_map[key] = fname

    index = {
        "metadata":   {"total_size": sum(t.nbytes for t in tensors.values())},
        "weight_map": weight_map,
    }
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)


def copy_config_files(src: str, dst: str):
    os.makedirs(dst, exist_ok=True)
    skip_exts = {".safetensors", ".bin", ".pt"}
    for f in Path(src).iterdir():
        if f.is_file() and f.suffix not in skip_exts:
            shutil.copy2(str(f), dst)
            print(f"  Copied config: {f.name}")


def merge_tensors_slerp(base: dict, ft: dict) -> dict:
    common  = set(base.keys()) & set(ft.keys())
    only_ft = set(ft.keys()) - set(base.keys())
    result  = {}

    for key in tqdm(common, desc="  SLERP"):
        b, f = base[key], ft[key]
        if b.shape != f.shape:
            result[key] = f
        elif b.dim() == 0 or b.numel() == 1:
            result[key] = ((1 - FT_RATIO) * b.float() + FT_RATIO * f.float()).to(b.dtype)
        else:
            result[key] = slerp(FT_RATIO, b, f)

    for key in only_ft:
        result[key] = ft[key]
    return result


def merge_tensors_dare_ties(base: dict, ft: dict) -> dict:
    common  = set(base.keys()) & set(ft.keys())
    only_ft = set(ft.keys()) - set(base.keys())
    result  = {}

    for key in tqdm(common, desc="  DARE-TIES"):
        b, f = base[key], ft[key]
        if b.shape != f.shape:
            result[key] = f
        elif b.dim() == 0 or b.numel() == 1:
            result[key] = f
        else:
            result[key] = dare_ties_merge(b, f)

    for key in only_ft:
        result[key] = ft[key]
    return result


def main():
    print("=" * 60)
    print("  Qwen3.5-9B Custom Model Merger")
    print(f"  Base:      {BASE_MODEL}")
    print(f"  Finetuned: {FINETUNED_MODEL}")
    print(f"  Ratio:     {int((1 - FT_RATIO) * 100)}% base / {int(FT_RATIO * 100)}% ft")
    print(f"  DARE density: {int(DARE_DENSITY * 100)}%")
    print("=" * 60)

    if not os.path.exists(BASE_CACHE) or not any(Path(BASE_CACHE).glob("*.safetensors")):
        print(f"\n[1/6] Downloading base model...")
        download_model(BASE_MODEL, BASE_CACHE)
    else:
        print(f"\n[1/6] Base model already cached at {BASE_CACHE}")

    if not os.path.exists(FT_CACHE) or not any(Path(FT_CACHE).glob("*.safetensors")):
        print(f"\n[2/6] Downloading finetuned model...")
        download_model(FINETUNED_MODEL, FT_CACHE)
    else:
        print(f"\n[2/6] Finetuned model already cached at {FT_CACHE}")

    print(f"\n[3/6] Loading base model tensors...")
    base_tensors = load_sharded_model(BASE_CACHE)
    print(f"  Loaded {len(base_tensors)} tensors")

    print(f"\n[4/6] Loading finetuned model tensors...")
    ft_tensors = load_sharded_model(FT_CACHE)
    print(f"  Loaded {len(ft_tensors)} tensors")

    common  = set(base_tensors.keys()) & set(ft_tensors.keys())
    only_ft = set(ft_tensors.keys()) - set(base_tensors.keys())
    print(f"\n  Common keys:    {len(common)}")
    print(f"  FT-only keys:   {len(only_ft)}  (kept as-is)")

    print(f"\n[5/6] SLERP merge  (t={FT_RATIO} → {int((1-FT_RATIO)*100)}% base)...")
    slerp_result = merge_tensors_slerp(base_tensors, ft_tensors)
    copy_config_files(FT_CACHE, OUTPUT_SLERP)
    save_sharded(slerp_result, OUTPUT_SLERP)
    print(f"   SLERP → {OUTPUT_SLERP}")
    del slerp_result

    print(f"\n[6/6] DARE-TIES merge  (weight={FT_RATIO}, density={DARE_DENSITY})...")
    dare_result = merge_tensors_dare_ties(base_tensors, ft_tensors)
    copy_config_files(FT_CACHE, OUTPUT_DARE_TIES)
    save_sharded(dare_result, OUTPUT_DARE_TIES)
    print(f"   DARE-TIES → {OUTPUT_DARE_TIES}")

    print("\n" + "=" * 60)
    print("  ALL DONE")
    print(f"  SLERP:     {OUTPUT_SLERP}")
    print(f"  DARE-TIES: {OUTPUT_DARE_TIES}")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/bin/bash
set -e

ENV_PATH="./quant_env"

echo "======================================================"
echo " Qwen3.5 W8A16 / AWQ Quantization — Environment Setup"
echo "======================================================"

echo "[1/7] Creating venv at $ENV_PATH..."
python3 -m venv $ENV_PATH
source $ENV_PATH/bin/activate

echo "[2/7] Installing pip + uv..."
pip install --upgrade pip uv -q

echo "[3/7] Installing torch 2.11.0 + torchvision (CUDA 12.8)..."
uv pip install torch==2.11.0 torchvision \
    --index-url https://download.pytorch.org/whl/cu128

echo "[4/7] Installing transformers from source (Qwen3.5 requires it)..."
uv pip install git+https://github.com/huggingface/transformers.git

echo "[5/7] Installing vLLM nightly..."
uv pip install vllm \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/nightly

echo "[6/7] Installing llm-compressor from source + extras..."
uv pip install git+https://github.com/vllm-project/llm-compressor.git
uv pip install datasets accelerate flash-linear-attention causal-conv1d

echo "[7/7] Re-pinning transformers + huggingface_hub (llm-compressor downgrades them)..."
uv pip install git+https://github.com/huggingface/transformers.git --no-deps
uv pip install huggingface_hub --upgrade --no-deps

echo ""
echo "======================================================"
echo " Verifying installation..."
echo "======================================================"
python -c "
import torch, transformers, llmcompressor, vllm
print('torch          :', torch.__version__)
print('transformers   :', transformers.__version__)
print('llmcompressor  :', llmcompressor.__version__)
print('vllm           :', vllm.__version__)
print('CUDA available :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU            :', torch.cuda.get_device_name(0))
    print('VRAM           :', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
print()
print('ALL OK')
print('Next: source $ENV_PATH/bin/activate')
print('      huggingface-cli login --token YOUR_TOKEN')
print('      python quantize_w8a16.py   # fast, data-free')
print('      python quantize_awq.py     # slower, calibration-based')
"

#!/usr/bin/env bash
# Olly-2 QLoRA training stack setup — Qwen3-14B, RTX 5090 (Blackwell/sm_120)
# Omarchy (Arch Linux) — pacman/yay, NOT apt
set -euo pipefail

VENV_DIR="$HOME/qlora-venv"
LOG() { echo -e "\n=== $1 ===\n"; }

LOG "Step 0: NVIDIA driver check"
if ! command -v nvidia-smi &>/dev/null; then
  echo "ERROR: nvidia-smi not found. Omarchy does not ship NVIDIA drivers by default."
  echo "Install first: sudo pacman -S --needed nvidia-open-dkms nvidia-utils lib32-nvidia-utils"
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "Detected GPU: $GPU_NAME"
if [[ "$GPU_NAME" != *"5090"* ]]; then
  echo "WARNING: expected RTX 5090, found: $GPU_NAME — continuing anyway."
fi

LOG "Step 1: base-devel + python (pacman)"
sudo pacman -S --needed --noconfirm base-devel python python-pip

LOG "Step 2: create venv"
python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip

LOG "Step 3: PyTorch CUDA 12.8 (Blackwell/sm_120 support)"
pip install torch --index-url https://download.pytorch.org/whl/cu128

LOG "Step 4: verify CUDA + GPU visible to PyTorch"
python - <<'PY'
import torch
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Compute capability:", torch.cuda.get_device_capability(0))
PY

LOG "Step 5: QLoRA stack — transformers, peft, trl, bitsandbytes, accelerate, datasets"
pip install --upgrade \
  "transformers>=4.56" \
  "peft>=0.18.0" \
  "trl>=1.1.0" \
  "bitsandbytes>=0.45.5" \
  accelerate datasets sentencepiece protobuf hf_transfer

LOG "Step 6: verify bitsandbytes sees Blackwell GPU"
python - <<'PY'
import torch, bitsandbytes as bnb
print("bitsandbytes:", bnb.__version__)
print("Torch sees GPU:", torch.cuda.get_device_name(0))
PY

LOG "Step 7: smoke test — load Qwen3-14B in 4-bit, attach LoRA adapter (no training yet)"
python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch

MODEL = "Qwen/Qwen3-14B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

print(f"Loading {MODEL} in 4-bit (this downloads ~8-9GB on first run)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# one forward pass to confirm it actually runs on GPU
inputs = tokenizer("The quick brown fox", return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model(**inputs)
print("Forward pass OK. Logits shape:", out.logits.shape)
print("SMOKE TEST PASSED — model loads, LoRA attaches, GPU forward pass works.")
PY

LOG "Setup complete."
echo "Venv: $VENV_DIR"
echo "Activate with: source $VENV_DIR/bin/activate"
echo "No training data wired in yet — smarty.db extraction pipeline still needed before real training."

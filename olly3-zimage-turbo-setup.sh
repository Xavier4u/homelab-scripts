#!/usr/bin/env bash
# Olly-3 Z-Image-Turbo setup — Omarchy (Arch), pacman/yay not apt
# Verified against official docs 2026-09-04:
#   - Tongyi-MAI/Z-Image-Turbo model card (huggingface.co)
#   - vllm-omni quickstart (docs.vllm.ai/projects/vllm-omni)
#   - uv now in official Arch 'extra' repo, no longer AUR-only
set -euo pipefail

LOG() { echo -e "\n=== $1 ===\n"; }

LOG "Step 0: NVIDIA driver check"
if ! command -v nvidia-smi &>/dev/null; then
  echo "NVIDIA driver not found. Installing (Omarchy does not ship it by default)..."
  sudo pacman -S --needed --noconfirm nvidia-dkms nvidia-utils opencl-nvidia cuda
  echo "Driver installed — REBOOT required before continuing, then re-run this script."
  echo "Also ensure ~/.config/hypr/hyprland.conf has:"
  echo "  env = NVD_BACKEND,direct"
  echo "  env = LIBVA_DRIVER_NAME,nvidia"
  echo "  env = __GLX_VENDOR_LIBRARY_NAME,nvidia"
  exit 0
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

LOG "Step 1: install uv (now in official Arch 'extra' repo, not AUR-only)"
sudo pacman -S --needed --noconfirm uv

LOG "Step 2: venv + vLLM + vLLM-Omni"
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm==0.28.0 --torch-backend=auto
if [ ! -d vllm-omni ]; then
  git clone https://github.com/vllm-project/vllm-omni.git
fi
cd vllm-omni
uv pip install -e .
cd ..

LOG "Step 3: smoke test — generate one image (downloads ~6GB model on first run)"
python3 -c "
from vllm_omni.entrypoints.omni import Omni
omni = Omni(model='Tongyi-MAI/Z-Image-Turbo')
outputs = omni.generate('a labeled network diagram showing three servers connected to a switch')
outputs[0].images[0].save('test.png')
print('SMOKE TEST PASSED — image generated, saved to test.png')
"

LOG "Step 4: systemd user unit (Omarchy convention — user unit, not system-wide)"
mkdir -p ~/.config/systemd/user
tee ~/.config/systemd/user/zimage.service << 'EOF'
[Unit]
Description=Z-Image-Turbo image generation server - Olly-3
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=%h/vllm-omni
Environment=PATH=%h/.venv/bin:/usr/bin:/bin
ExecStart=%h/.venv/bin/vllm serve Tongyi-MAI/Z-Image-Turbo --omni --port 3030 --host 0.0.0.0
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now zimage.service
sleep 3
systemctl --user status zimage.service --no-pager

LOG "Step 5: enable linger (survive reboot without active login)"
sudo loginctl enable-linger "$USER"

LOG "Setup complete. Test with:"
echo 'curl -s http://localhost:3030/v1/images/generations -H "Content-Type: application/json" -d '"'"'{"prompt":"a cup of coffee on the table","size":"1024x1024","response_format":"b64_json","seed":42}'"'"' | jq -r ".data[0].b64_json" | base64 -d > test-api.png'

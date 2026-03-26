#!/bin/bash
# =============================================================================
# FedRGBD — Desktop Training Server Setup Script
# =============================================================================
# Target: Linux desktop/laptop with NVIDIA RTX 3060+ (12GB+ VRAM)
# Role: Pre-training, hyperparameter search (NOT used during FL experiments)
# =============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "============================================================"
echo "  FedRGBD — Desktop Training Server Setup"
echo "============================================================"
echo ""

# Check NVIDIA GPU
if ! command -v nvidia-smi &>/dev/null; then
    log_error "nvidia-smi not found. Please install NVIDIA drivers first."
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
log_ok "GPU detected: ${GPU_NAME} (${GPU_VRAM})"

# Create virtual environment
VENV_DIR="$HOME/fedrgbd_venv"
log_info "Creating Python virtual environment at ${VENV_DIR}..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel

# Install PyTorch (standard CUDA build for desktop)
log_info "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
log_info "Installing project dependencies..."
pip install \
    flwr[simulation]==1.13.1 \
    scikit-learn \
    scipy \
    pingouin \
    pandas \
    matplotlib \
    seaborn \
    pyyaml \
    tqdm \
    Pillow \
    opencv-python \
    jupyter \
    notebook

# Verify
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

echo ""
log_ok "Desktop setup complete!"
echo "  Activate: source $VENV_DIR/bin/activate"
echo "  This machine is for pre-training ONLY — FL runs on Jetsons."

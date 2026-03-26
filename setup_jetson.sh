#!/bin/bash
# =============================================================================
# FedRGBD — Jetson Orin Nano 8GB Setup Script
# =============================================================================
# Target: NVIDIA Jetson Orin Nano 8GB with JetPack 6.1 or 6.2
# Cameras: Intel RealSense D435i / D455
# Framework: PyTorch + Flower (Federated Learning)
#
# Usage:
#   chmod +x setup_jetson.sh
#   ./setup_jetson.sh
#
# This script will:
#   1. Detect JetPack version and verify CUDA
#   2. Install system dependencies
#   3. Build librealsense from source (RSUSB backend)
#   4. Install PyTorch (NVIDIA official wheel)
#   5. Install Flower and Python dependencies
#   6. Install jtop for hardware monitoring
#   7. Run verification tests
#
# Estimated time: 30-60 minutes (depends on internet speed)
# Requires: Internet connection, ~5GB free disk space
# =============================================================================

set -e  # Exit on error

# --- Colors for output ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# --- Configuration ---
LIBREALSENSE_VERSION="v2.55.1"
PYTHON_VENV_DIR="$HOME/fedrgbd_venv"
LIBREALSENSE_BUILD_DIR="$HOME/librealsense_build"

# PyTorch wheel URLs (NVIDIA official for JetPack 6)
# JetPack 6.1: PyTorch 2.5.0 official NVIDIA wheel
PYTORCH_JP61_URL="https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"
# JetPack 6.2: Use Jetson AI Lab community mirror
PYTORCH_JP62_INDEX="https://pypi.jetson-ai-lab.io/jp6/cu126"

echo "============================================================"
echo "  FedRGBD — Jetson Orin Nano Setup"
echo "  $(date)"
echo "============================================================"
echo ""

# =============================================================================
# STEP 0: Pre-flight checks
# =============================================================================
log_info "Step 0: Pre-flight checks..."

# Check we're on a Jetson
if [ ! -f /etc/nv_tegra_release ] && [ ! -f /etc/nv_boot_control.conf ]; then
    log_error "This doesn't appear to be a NVIDIA Jetson device."
    log_error "This script is designed for Jetson Orin Nano with JetPack 6.x"
    exit 1
fi

# Detect JetPack version
if command -v dpkg &>/dev/null; then
    JETPACK_VER=$(dpkg -l 2>/dev/null | grep -i "nvidia-jetpack" | awk '{print $3}' | head -1)
fi

if [ -z "$JETPACK_VER" ]; then
    # Fallback: check L4T version
    if [ -f /etc/nv_tegra_release ]; then
        L4T_VER=$(head -1 /etc/nv_tegra_release | sed 's/.*R\([0-9]*\).*/\1/')
        log_warn "Could not detect JetPack version from dpkg. L4T release: R${L4T_VER}"
    else
        log_warn "Could not detect JetPack version. Proceeding with auto-detection."
    fi
else
    log_ok "Detected JetPack version: ${JETPACK_VER}"
fi

# Determine JetPack major.minor for PyTorch selection
# JetPack 6.1 = L4T 36.4.0, JetPack 6.2 = L4T 36.4.3
JETPACK_MAJOR="6"
JETPACK_MINOR="1"  # Default to 6.1

if echo "$JETPACK_VER" | grep -q "6\.2"; then
    JETPACK_MINOR="2"
elif echo "$JETPACK_VER" | grep -q "6\.1"; then
    JETPACK_MINOR="1"
else
    log_warn "Could not determine JetPack minor version. Defaulting to 6.1 path."
    log_warn "If you are on JetPack 6.2, re-run with: JETPACK_MINOR=2 ./setup_jetson.sh"
fi

# Allow override via environment variable
if [ -n "$FORCE_JETPACK_MINOR" ]; then
    JETPACK_MINOR="$FORCE_JETPACK_MINOR"
    log_info "Using forced JetPack minor version: 6.${JETPACK_MINOR}"
fi

log_info "Will use JetPack 6.${JETPACK_MINOR} installation path."

# Check CUDA
if command -v nvcc &>/dev/null; then
    CUDA_VER=$(nvcc --version | grep "release" | awk '{print $6}' | sed 's/,//')
    log_ok "CUDA detected: ${CUDA_VER}"
else
    # CUDA might be installed but not in PATH
    if [ -d /usr/local/cuda ]; then
        export PATH=/usr/local/cuda/bin:$PATH
        export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
        CUDA_VER=$(nvcc --version | grep "release" | awk '{print $6}' | sed 's/,//')
        log_ok "CUDA detected (added to PATH): ${CUDA_VER}"
    else
        log_error "CUDA not found. Please ensure JetPack was installed with CUDA toolkit."
        log_error "Run: sudo apt install nvidia-jetpack"
        exit 1
    fi
fi

# Check available disk space (need ~5GB)
AVAIL_SPACE=$(df -BG /home | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAIL_SPACE" -lt 5 ]; then
    log_error "Insufficient disk space: ${AVAIL_SPACE}GB available, need at least 5GB"
    exit 1
fi
log_ok "Available disk space: ${AVAIL_SPACE}GB"

# Check RAM
TOTAL_RAM=$(free -m | awk '/^Mem:/{print $2}')
log_info "Total RAM: ${TOTAL_RAM}MB"
if [ "$TOTAL_RAM" -lt 7000 ]; then
    log_warn "Less than 7GB RAM detected. This is expected for Orin Nano 8GB (shared with GPU)."
fi

echo ""

# =============================================================================
# STEP 1: System dependencies
# =============================================================================
log_info "Step 1: Installing system dependencies..."

sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    libssl-dev \
    libusb-1.0-0-dev \
    pkg-config \
    libgtk-3-dev \
    libglfw3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    python3-dev \
    python3-pip \
    python3-venv \
    libopenblas-dev \
    libjpeg-dev \
    zlib1g-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    curl \
    wget \
    htop \
    net-tools \
    iproute2 \
    ethtool \
    v4l-utils \
    usbutils

log_ok "System dependencies installed."
echo ""

# =============================================================================
# STEP 2: Set up CUDA environment (persistent)
# =============================================================================
log_info "Step 2: Setting up CUDA environment..."

CUDA_ENV_LINES='
# CUDA environment (added by FedRGBD setup)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda
'

if ! grep -q "FedRGBD setup" "$HOME/.bashrc" 2>/dev/null; then
    echo "$CUDA_ENV_LINES" >> "$HOME/.bashrc"
    log_ok "CUDA paths added to ~/.bashrc"
else
    log_ok "CUDA paths already in ~/.bashrc"
fi

# Apply for current session
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda

echo ""

# =============================================================================
# STEP 3: Build librealsense from source (RSUSB backend)
# =============================================================================
log_info "Step 3: Building librealsense from source (RSUSB backend)..."
log_info "This is the recommended method for JetPack 6.x with IMU-equipped cameras."
log_info "The RSUSB backend avoids kernel patching and HID compatibility issues."

mkdir -p "$LIBREALSENSE_BUILD_DIR"
cd "$LIBREALSENSE_BUILD_DIR"

if [ -d "librealsense" ]; then
    log_warn "librealsense directory already exists. Removing and re-cloning..."
    rm -rf librealsense
fi

git clone --depth 1 --branch "$LIBREALSENSE_VERSION" https://github.com/IntelRealSense/librealsense.git
cd librealsense

# Install udev rules for RealSense cameras
sudo ./scripts/setup_udev_rules.sh

# Build with RSUSB backend and CUDA support
mkdir -p build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DFORCE_RSUSB_BACKEND=true \
    -DBUILD_WITH_CUDA=true \
    -DBUILD_EXAMPLES=true \
    -DBUILD_GRAPHICAL_EXAMPLES=true \
    -DBUILD_PYTHON_BINDINGS:bool=true \
    -DPYTHON_EXECUTABLE=$(which python3) \
    -DCMAKE_INSTALL_PREFIX=/usr/local

# Build using all available cores (but limit to avoid OOM on 8GB)
NUM_CORES=$(nproc)
BUILD_JOBS=$((NUM_CORES > 4 ? 4 : NUM_CORES))
log_info "Building with ${BUILD_JOBS} parallel jobs (of ${NUM_CORES} available cores)..."

make -j${BUILD_JOBS}
sudo make install

# Update shared library cache
sudo ldconfig

log_ok "librealsense ${LIBREALSENSE_VERSION} installed successfully."

# Verify installation
if command -v rs-enumerate-devices &>/dev/null; then
    log_ok "rs-enumerate-devices is available."
else
    log_warn "rs-enumerate-devices not in PATH. You may need to add /usr/local/bin to PATH."
fi

cd "$HOME"
echo ""

# =============================================================================
# STEP 4: Python virtual environment
# =============================================================================
log_info "Step 4: Creating Python virtual environment..."

if [ -d "$PYTHON_VENV_DIR" ]; then
    log_warn "Virtual environment already exists at $PYTHON_VENV_DIR"
    log_warn "To recreate, delete it first: rm -rf $PYTHON_VENV_DIR"
else
    python3 -m venv "$PYTHON_VENV_DIR" --system-site-packages
    log_ok "Virtual environment created at $PYTHON_VENV_DIR"
fi

# Activate venv
source "$PYTHON_VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip setuptools wheel

echo ""

# =============================================================================
# STEP 5: Install PyTorch
# =============================================================================
log_info "Step 5: Installing PyTorch for JetPack 6.${JETPACK_MINOR}..."

# Install cuSPARSELt (required for PyTorch on Jetson)
log_info "Installing cuSPARSELt..."
CUSPARSELT_VER="0.7.1.0"
CUSPARSELT_FILE="libcusparse_lt-linux-aarch64-${CUSPARSELT_VER}-archive.tar.xz"
CUSPARSELT_URL="https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-aarch64/${CUSPARSELT_FILE}"

if [ ! -f "/usr/local/lib/libcusparseLt.so" ]; then
    cd /tmp
    wget -q --show-progress "$CUSPARSELT_URL" -O "$CUSPARSELT_FILE" || {
        log_warn "Could not download cuSPARSELt. PyTorch may still work without it."
    }
    if [ -f "$CUSPARSELT_FILE" ]; then
        tar xf "$CUSPARSELT_FILE"
        sudo cp -a libcusparse_lt-*/include/* /usr/local/include/
        sudo cp -a libcusparse_lt-*/lib/* /usr/local/lib/
        sudo ldconfig
        rm -rf "$CUSPARSELT_FILE" libcusparse_lt-*
        log_ok "cuSPARSELt ${CUSPARSELT_VER} installed."
    fi
    cd "$HOME"
else
    log_ok "cuSPARSELt already installed."
fi

# Install numpy first (specific version for compatibility)
pip install "numpy==1.26.1"

if [ "$JETPACK_MINOR" = "1" ]; then
    log_info "Installing PyTorch 2.5.0 from NVIDIA official wheel (JetPack 6.1)..."
    pip install --no-cache "$PYTORCH_JP61_URL"
elif [ "$JETPACK_MINOR" = "2" ]; then
    log_info "Installing PyTorch from Jetson AI Lab mirror (JetPack 6.2)..."
    pip install torch torchvision --index-url "$PYTORCH_JP62_INDEX"
fi

# Verify PyTorch CUDA
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
" || {
    log_error "PyTorch installation verification failed!"
    log_error "Please check the installation manually."
}

echo ""

# =============================================================================
# STEP 6: Install torchvision (build from source for JetPack 6.1)
# =============================================================================
if [ "$JETPACK_MINOR" = "1" ]; then
    log_info "Step 6: Building torchvision from source (JetPack 6.1)..."
    
    # torchvision 0.20.0 is compatible with PyTorch 2.5.0
    TORCHVISION_VER="v0.20.0"
    
    cd /tmp
    if [ -d "vision" ]; then
        rm -rf vision
    fi
    
    git clone --depth 1 --branch "$TORCHVISION_VER" https://github.com/pytorch/vision.git
    cd vision
    
    # Build with limited parallelism to avoid OOM
    MAX_JOBS=${BUILD_JOBS} pip install -e . --no-build-isolation
    
    cd "$HOME"
    rm -rf /tmp/vision
    
    log_ok "torchvision ${TORCHVISION_VER} installed."
else
    log_info "Step 6: torchvision already installed via Jetson AI Lab mirror."
fi

echo ""

# =============================================================================
# STEP 7: Install Flower and Python dependencies
# =============================================================================
log_info "Step 7: Installing Flower FL framework and Python dependencies..."

pip install \
    flwr[simulation]==1.13.1 \
    pyrealsense2==2.55.1.6643 2>/dev/null || log_warn "pyrealsense2 pip install failed — using system build." 

pip install \
    scikit-learn \
    scipy \
    pingouin \
    pandas \
    matplotlib \
    seaborn \
    pyyaml \
    tqdm \
    Pillow \
    opencv-python-headless

log_ok "Python dependencies installed."
echo ""

# =============================================================================
# STEP 8: Install jtop (Jetson monitoring)
# =============================================================================
log_info "Step 8: Installing jtop for hardware monitoring..."

sudo pip3 install -U jetson-stats 2>/dev/null || {
    sudo pip install -U jetson-stats --break-system-packages 2>/dev/null || {
        log_warn "Could not install jetson-stats via pip. Trying apt..."
        sudo apt-get install -y python3-jetson-stats 2>/dev/null || log_warn "jetson-stats installation failed."
    }
}

# jtop requires a reboot or service restart to work
sudo systemctl restart jtop.service 2>/dev/null || true

log_ok "jtop installed (may require reboot for full functionality)."
echo ""

# =============================================================================
# STEP 9: Configure network for FL
# =============================================================================
log_info "Step 9: Network configuration check..."

# Check Ethernet interface
ETH_IF=$(ip -o link show | awk -F': ' '{print $2}' | grep -E '^eth|^enp|^eno' | head -1)
if [ -n "$ETH_IF" ]; then
    ETH_IP=$(ip -4 addr show "$ETH_IF" 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
    if [ -n "$ETH_IP" ]; then
        log_ok "Ethernet interface: ${ETH_IF}, IP: ${ETH_IP}"
    else
        log_warn "Ethernet interface ${ETH_IF} detected but no IP assigned."
        log_warn "Assign a static IP for FL communication:"
        log_warn "  sudo nmcli connection modify 'Wired connection 1' ipv4.addresses 192.168.1.10/24 ipv4.method manual"
    fi
else
    log_warn "No Ethernet interface found. FL requires wired Ethernet between nodes."
fi

echo ""

# =============================================================================
# STEP 10: Add venv activation to bashrc
# =============================================================================
log_info "Step 10: Finalizing setup..."

VENV_LINE="# FedRGBD venv activation
alias fedrgbd='source $PYTHON_VENV_DIR/bin/activate && cd ~/FedRGBD'"

if ! grep -q "fedrgbd" "$HOME/.bashrc" 2>/dev/null; then
    echo "$VENV_LINE" >> "$HOME/.bashrc"
    log_ok "Added 'fedrgbd' alias to ~/.bashrc (type 'fedrgbd' to activate environment)"
fi

echo ""

# =============================================================================
# VERIFICATION SUMMARY
# =============================================================================
echo "============================================================"
echo "  FedRGBD Setup — Verification Summary"
echo "============================================================"
echo ""

# Python & venv
PYTHON_VER=$(python3 --version 2>&1)
log_info "Python: ${PYTHON_VER}"

# PyTorch
TORCH_VER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "NOT INSTALLED")
TORCH_CUDA=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "N/A")
log_info "PyTorch: ${TORCH_VER} (CUDA: ${TORCH_CUDA})"

# torchvision
TV_VER=$(python3 -c "import torchvision; print(torchvision.__version__)" 2>/dev/null || echo "NOT INSTALLED")
log_info "torchvision: ${TV_VER}"

# Flower
FLWR_VER=$(python3 -c "import flwr; print(flwr.__version__)" 2>/dev/null || echo "NOT INSTALLED")
log_info "Flower: ${FLWR_VER}"

# librealsense
RS_VER=$(rs-enumerate-devices 2>/dev/null | head -1 || echo "NOT AVAILABLE — connect camera and test")
log_info "librealsense: installed (run 'rs-enumerate-devices' with camera connected)"

# pyrealsense2
PYRS_VER=$(python3 -c "import pyrealsense2; print(pyrealsense2.__version__)" 2>/dev/null || echo "using system build")
log_info "pyrealsense2: ${PYRS_VER}"

# jtop
JTOP_VER=$(jtop --version 2>/dev/null || echo "NOT INSTALLED or requires reboot")
log_info "jtop: ${JTOP_VER}"

echo ""
echo "============================================================"
echo "  NEXT STEPS"
echo "============================================================"
echo ""
echo "  1. Reboot the Jetson:  sudo reboot"
echo "  2. After reboot, activate environment:  fedrgbd"
echo "  3. Connect RealSense camera via USB3 port"
echo "  4. Test camera:  rs-enumerate-devices"
echo "  5. Test capture:  python3 src/data/realsense_capture.py --test"
echo ""
echo "  For FL network setup between two Jetsons:"
echo "    Node A (server): 192.168.1.10"
echo "    Node B (client): 192.168.1.11"
echo "    Configure with:"
echo "      sudo nmcli connection modify 'Wired connection 1' \\"
echo "        ipv4.addresses 192.168.1.10/24 ipv4.method manual"
echo ""
echo "  IMPORTANT: If pyrealsense2 import fails, use the system-built version:"
echo "    export PYTHONPATH=\$PYTHONPATH:/usr/local/lib/python3.10/site-packages"
echo ""
echo "============================================================"
echo "  Setup complete! $(date)"
echo "============================================================"

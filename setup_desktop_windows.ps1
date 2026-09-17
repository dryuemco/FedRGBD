# =============================================================================
# FedRGBD - Desktop GPU Setup Script (Windows / CUDA 12.8)
# =============================================================================
# Target: Windows 10/11 desktop with an NVIDIA RTX GPU (tested: RTX 5090,
#         Blackwell sm_120, driver 595.79) and system Python 3.12.
# Role:   Centralized / local-only ACCURACY baselines of the NCAA revision
#         (configs/experiment_matrix.yaml -> revision.baselines_extension).
#         NOT used for FL rounds or for any timing/energy numbers - those
#         stay on the Jetson Orin Nano testbed (see setup_jetson.sh).
#
# Why a separate venv/version set (see docs/DESKTOP_GPU_BASELINES.md):
#   requirements.txt pins the Jetson stack (torch 2.5, numpy 1.26.4). A
#   Blackwell GPU needs torch >= 2.7 built against CUDA 12.8 (sm_120), and
#   those wheels require numpy >= 2. The Jetson pins are therefore relaxed
#   here ONLY for torch/torchvision/numpy; everything else follows
#   requirements.txt.
#
# Usage (PowerShell, from the repo root):
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\setup_desktop_windows.ps1                    # creates C:\Users\<you>\venvs\fedrgbd-gpu
#   .\setup_desktop_windows.ps1 -VenvDir D:\venvs\fedrgbd-gpu
#
# The script refuses to overwrite an existing venv unless -Force is given.
# After setup, run training with PYTHONUTF8=1 set (see docs/DESKTOP_GPU_BASELINES.md).
# =============================================================================

[CmdletBinding()]
param(
    [string]$VenvDir = (Join-Path $env:USERPROFILE "venvs\fedrgbd-gpu"),
    [string]$PythonExe = "",
    [string]$TorchIndex = "https://download.pytorch.org/whl/cu128",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

function Log-Info  { param($m) Write-Host "[INFO] $m"  -ForegroundColor Cyan }
function Log-Ok    { param($m) Write-Host "[OK]   $m"  -ForegroundColor Green }
function Log-Error { param($m) Write-Host "[ERROR] $m" -ForegroundColor Red }

Write-Host "============================================================"
Write-Host "  FedRGBD - Desktop GPU Setup (Windows, CUDA 12.8)"
Write-Host "============================================================"
Write-Host ""

# --- Check NVIDIA GPU ---------------------------------------------------------
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Log-Error "nvidia-smi not found. Install the NVIDIA driver (>= 570 for CUDA 12.8) first."
    exit 1
}
$gpuInfo = (& nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader) | Select-Object -First 1
Log-Ok "GPU detected: $gpuInfo"

# --- Locate Python 3.12 -------------------------------------------------------
if ($PythonExe -eq "") {
    $candidate = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
    if (Test-Path $candidate) {
        $PythonExe = $candidate
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        $PythonExe = (& py -3.12 -c "import sys; print(sys.executable)")
    } else {
        Log-Error "Python 3.12 not found. Install it from python.org or pass -PythonExe."
        exit 1
    }
}
$pyVersion = & $PythonExe --version
Log-Ok "Using $PythonExe ($pyVersion)"

# --- Create the virtual environment ------------------------------------------
if (Test-Path $VenvDir) {
    if (-not $Force) {
        Log-Error "$VenvDir already exists. Re-run with -Force to recreate it, or pass -VenvDir."
        exit 1
    }
    Log-Info "Removing existing venv at $VenvDir (-Force)"
    Remove-Item -Recurse -Force $VenvDir -Confirm:$false
}
Log-Info "Creating virtual environment at $VenvDir ..."
& $PythonExe -m venv $VenvDir
$venvPython = Join-Path $VenvDir "Scripts\python.exe"
& $venvPython -m pip install --upgrade pip setuptools wheel

# The generated run scripts (scripts/print_revision_commands.py --format bash)
# call `python3`; Windows venvs ship only python.exe, so add an alias.
Copy-Item $venvPython (Join-Path $VenvDir "Scripts\python3.exe")

# --- PyTorch with CUDA 12.8 (Blackwell needs torch >= 2.7) -------------------
Log-Info "Installing PyTorch + torchvision from $TorchIndex (multi-GB download) ..."
& $venvPython -m pip install torch torchvision --index-url $TorchIndex
if ($LASTEXITCODE -ne 0) { Log-Error "PyTorch install failed."; exit 1 }

# --- Remaining project dependencies (requirements.txt minus torch/numpy pins) --
# numpy: the cu128 torch/torchvision wheels require numpy >= 2, so the Jetson
#        pin numpy==1.26.4 is NOT applied here (recorded in docs/DESKTOP_GPU_BASELINES.md).
# flwr:  plain flwr[simulation] as pinned; it does not pull a different torch.
Log-Info "Installing project dependencies ..."
& $venvPython -m pip install `
    "flwr[simulation]==1.13.1" `
    "scikit-learn>=1.3.0" `
    "scipy>=1.11.0" `
    "pingouin>=0.5.4" `
    "pandas>=2.0.0" `
    "Pillow>=10.0.0" `
    "opencv-python-headless>=4.8.0" `
    "pyyaml>=6.0" `
    "matplotlib>=3.7.0" `
    "seaborn>=0.12.0" `
    "tqdm>=4.65.0" `
    "pytest>=7.0"
if ($LASTEXITCODE -ne 0) { Log-Error "Dependency install failed."; exit 1 }

# --- Verify -------------------------------------------------------------------
Log-Info "Verifying CUDA ..."
& $venvPython -c @"
import torch, torchvision, numpy, flwr
print(f'PyTorch:      {torch.__version__}')
print(f'torchvision:  {torchvision.__version__}')
print(f'CUDA (build): {torch.version.cuda}')
print(f'numpy:        {numpy.__version__}')
print(f'flwr:         {flwr.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:   {torch.cuda.get_device_name(0)}')
    print(f'VRAM:  {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print(f'archs: {torch.cuda.get_arch_list()}')
    x = torch.randn(1024, 1024, device='cuda'); y = (x @ x).sum().item()
    print(f'CUDA matmul OK ({y:.3e})')
else:
    raise SystemExit('CUDA not available - check driver / wheel index')
"@
if ($LASTEXITCODE -ne 0) { Log-Error "Verification failed."; exit 1 }

Write-Host ""
Log-Ok "Desktop GPU setup complete."
Write-Host "  Activate (PowerShell): $VenvDir\Scripts\Activate.ps1"
Write-Host "  Activate (Git Bash):   source $($VenvDir -replace '\','/')/Scripts/activate"
Write-Host "  Run the CPU tests:     python -m pytest tests -q -k 'not end_to_end' -p no:cacheprovider"
Write-Host "  IMPORTANT: set PYTHONUTF8=1 before running the training scripts on Windows;"
Write-Host "             they print Unicode arrows and crash on a cp1252 console/pipe otherwise."
Write-Host "  Baseline block:        see docs/DESKTOP_GPU_BASELINES.md"
Write-Host "  This machine is for ACCURACY baselines only - FL + timing run on the Jetsons."

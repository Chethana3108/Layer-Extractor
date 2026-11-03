# One-time setup on Windows: finds Python, creates .venv, installs the right PyTorch (GPU or CPU), installs deps.
# Run:  double-click setup.cmd   (or: powershell -ExecutionPolicy Bypass -File setup.ps1)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Fail($msg) { Write-Host "`nERROR: $msg" -ForegroundColor Red; exit 1 }

# ---- 1. Python 3.11 / 3.12 / 3.13 (64-bit) via the py launcher ----
$pyArgs = $null
foreach ($v in "3.12", "3.11", "3.13", "3.10") {
    try {
        $out = & py "-$v" -c "import struct,sys;print(sys.version.split()[0], struct.calcsize('P')*8)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $out -match " 64$") { $pyArgs = "-$v"; Write-Host "Using Python $out-bit"; break }
    } catch { }
}
if (-not $pyArgs) {
    Fail "64-bit Python 3.11/3.12 not found. Install it with:`n  winget install -e --id Python.Python.3.12`nthen open a NEW terminal and run setup again."
}

# ---- 2. venv ----
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating virtual environment .venv ..."
    & py $pyArgs -m venv .venv
    if ($LASTEXITCODE -ne 0) { Fail "venv creation failed" }
}
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $py -m pip install --upgrade pip --quiet

# ---- 3. PyTorch (pinned): pick CUDA build from the NVIDIA driver, else CPU ----
$index = "https://download.pytorch.org/whl/cpu"; $torchPkgs = @("torch==2.5.1", "torchvision==0.20.1")
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
    $smiOut = (& nvidia-smi) -join "`n"
    if ($smiOut -match "CUDA Version:\s*(\d+)\.(\d+)") {
        $cuda = [int]$Matches[1] * 10 + [int]$Matches[2]
        Write-Host "NVIDIA driver supports CUDA $($Matches[1]).$($Matches[2])"
        if     ($cuda -ge 128) { $index = "https://download.pytorch.org/whl/cu128"; $torchPkgs = @("torch==2.7.1", "torchvision==0.22.1") }
        elseif ($cuda -ge 124) { $index = "https://download.pytorch.org/whl/cu124" }
        elseif ($cuda -ge 121) { $index = "https://download.pytorch.org/whl/cu121" }
    }
}
if ($index -like "*cpu") { Write-Host "No usable NVIDIA GPU found -> installing CPU PyTorch (works, just slower)" -ForegroundColor Yellow }
Write-Host "Installing $($torchPkgs -join ', ') from $index (large download, a few minutes) ..."
& $py -m pip install @torchPkgs --index-url $index
if ($LASTEXITCODE -ne 0) { Fail "PyTorch install failed" }

# ---- 4. Remaining dependencies ----
Write-Host "Installing requirements ..."
& $py -m pip install -r requirements.txt --upgrade-strategy only-if-needed
if ($LASTEXITCODE -ne 0) { Fail "requirements install failed" }

# ---- 5. Sanity check ----
& $py -c "import torch, transformers, cv2, PIL; print('torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (CPU)')"
if ($LASTEXITCODE -ne 0) { Fail "import check failed" }
& $py tests\test_imaging.py
& $py tests\test_boxes.py
& $py -m pip check
Write-Host "`nSetup complete. Drag an image onto run-test.cmd to try it." -ForegroundColor Green

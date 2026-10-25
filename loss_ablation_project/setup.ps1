<#
  setup.ps1 - ONE-TIME setup for the loss ablation project (Windows + NVIDIA GPU)

  Run from the VS Code terminal, inside this project folder:
      powershell -ExecutionPolicy Bypass -File .\setup.ps1

  It will:
    1. check the NVIDIA GPU/driver
    2. find Python 3.11 (installs it with winget if missing)
    3. create a virtual environment in .\.venv
    4. install PyTorch 2.7.1 + CUDA 11.8 (same version as the Kaggle run)
    5. install the libraries in requirements.txt
    6. confirm PyTorch can use the GPU
#>
$ErrorActionPreference = "Stop"
$ProgressPreference    = "SilentlyContinue"
$Root  = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv  = Join-Path $Root ".venv"
$VenvPy = Join-Path $Venv "Scripts\python.exe"

function Step($m) { Write-Host "`n=== $m ===" -ForegroundColor Cyan }
function Fail($m) { Write-Host "`nERROR: $m" -ForegroundColor Red; exit 1 }

# 1. GPU ----------------------------------------------------------------------
Step "1/6 Checking NVIDIA GPU"
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Fail "nvidia-smi not found. Install the latest NVIDIA driver (nvidia.com), reboot, run again."
}
& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# 2. Python -------------------------------------------------------------------
Step "2/6 Finding Python (3.11 preferred)"
function Test-PyOk($exe) {
    # returns $true if $exe is a working Python 3.10-3.13
    if (-not $exe -or -not (Test-Path $exe)) { return $false }
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    $out = & $exe -c "import sys; print(1 if (3,10) <= sys.version_info[:2] <= (3,13) else 0)" 2>&1
    $ErrorActionPreference = $old
    return ("$out".Trim() -eq "1")
}
function Find-Python {
    $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try {
        # 1) Python launcher (python.org installs)
        if (Get-Command py -ErrorAction SilentlyContinue) {
            foreach ($v in @("3.11", "3.12", "3.10", "3.13")) {
                $exe = & py "-$v" -c "import sys; print(sys.executable)" 2>$null
                if ($LASTEXITCODE -eq 0 -and $exe -and (Test-PyOk "$exe".Trim())) { return "$exe".Trim() }
            }
        }
        # 2) standard python.org install folders
        foreach ($c in @("$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
                         "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
                         "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe")) {
            if (Test-PyOk $c) { return $c }
        }
        # 3) a project-local conda Python created earlier by this script
        $local = Join-Path $Root ".pybase\python.exe"
        if (Test-PyOk $local) { return $local }
        return $null
    } finally { $ErrorActionPreference = $old }
}
function Find-Conda {
    if ($env:CONDA_EXE -and (Test-Path $env:CONDA_EXE)) { return $env:CONDA_EXE }
    $cmd = Get-Command conda.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($b in @("$env:USERPROFILE\miniconda3", "$env:USERPROFILE\anaconda3",
                     "$env:LOCALAPPDATA\miniconda3", "$env:LOCALAPPDATA\anaconda3",
                     "C:\ProgramData\miniconda3", "C:\ProgramData\anaconda3")) {
        $c = Join-Path $b "Scripts\conda.exe"
        if (Test-Path $c) { return $c }
    }
    return $null
}

$BasePy = Find-Python
if (-not $BasePy) {
    $Conda = Find-Conda
    if ($Conda) {
        # You have conda: make a clean Python 3.11 inside this project folder (does not touch your base env)
        Write-Host "No standalone Python 3.11 found - creating one with conda in .\.pybase (2-5 min)..."
        $old = $ErrorActionPreference; $ErrorActionPreference = "Continue"
        & $Conda create -p (Join-Path $Root ".pybase") python=3.11 -y --override-channels -c conda-forge
        $ErrorActionPreference = $old
        $BasePy = Find-Python
    }
}
if (-not $BasePy) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "Python not found - installing Python 3.11 with winget..."
        & winget install -e --id Python.Python.3.11 --scope user --accept-package-agreements --accept-source-agreements
        $BasePy = Find-Python
    }
    if (-not $BasePy) {
        Fail "Python 3.11 not found. Install it from https://www.python.org/downloads/release/python-3119/ (tick 'Add python.exe to PATH'), restart VS Code, run setup again."
    }
}
Write-Host "Using Python: $BasePy"
& $BasePy --version

# 3. venv ---------------------------------------------------------------------
Step "3/6 Creating virtual environment (.venv)"
if (Test-Path $VenvPy) {
    Write-Host ".venv already exists - reusing it."
} else {
    & $BasePy -m venv $Venv
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $VenvPy)) { Fail "Could not create .venv" }
}
& $VenvPy -m pip install --upgrade pip

# 4. PyTorch ------------------------------------------------------------------
Step "4/6 Installing PyTorch + CUDA 11.8 (about 3 GB, 5-15 min)"
& $VenvPy -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
if ($LASTEXITCODE -ne 0) {
    Write-Host "Pinned version not available for this Python - installing latest CUDA 11.8 build instead." -ForegroundColor Yellow
    & $VenvPy -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    if ($LASTEXITCODE -ne 0) { Fail "PyTorch install failed (check internet and run again)." }
}

# 5. Other libraries ----------------------------------------------------------
Step "5/6 Installing other libraries (requirements.txt)"
& $VenvPy -m pip install -r (Join-Path $Root "requirements.txt")
if ($LASTEXITCODE -ne 0) { Fail "Library install failed (check internet and run again)." }

# 6. Verify -------------------------------------------------------------------
Step "6/6 Checking PyTorch can use the GPU"
$check = & $VenvPy -c "import torch; ok=torch.cuda.is_available(); print('torch', torch.__version__, '| CUDA available:', ok, '|', torch.cuda.get_device_name(0) if ok else 'NO GPU')"
Write-Host $check
if ($check -notmatch "CUDA available: True") {
    Fail "PyTorch cannot see the GPU. Update the NVIDIA driver, delete the .venv folder, and run setup again."
}

Write-Host "`nSETUP COMPLETE." -ForegroundColor Green
Write-Host "In VS Code: Ctrl+Shift+P -> 'Python: Select Interpreter' -> choose .venv"
Write-Host "Next: powershell -ExecutionPolicy Bypass -File .\run.ps1 -DataRoot `"D:\datasets\hierarchical-mimosa`""
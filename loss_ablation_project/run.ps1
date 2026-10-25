<#
  run.ps1 - runs the loss-function ablation (after setup.ps1 has been run once)

  From the VS Code terminal, inside this project folder:
      powershell -ExecutionPolicy Bypass -File .\run.ps1 -DataRoot "D:\datasets"

  -DataRoot   folder that CONTAINS hierarchical_aggression.csv and Img\Img\
              (if omitted or wrong, you will be asked to paste it)
  -NoWorkers  use 0 DataLoader workers (only if you get DataLoader / worker / spawn errors)
  -Script     which ablation to run (default: hrf_loss_ablation.py = proposed HRF model)
#>
param(
    [string]$DataRoot = "D:\datasets",
    [switch]$NoWorkers,
    [string]$Script = "hrf_loss_ablation.py"
)
$ErrorActionPreference = "Stop"
$Root     = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPy   = Join-Path $Root ".venv\Scripts\python.exe"
$PyScript = Join-Path $Root $Script

function Fail($m) { Write-Host "`nERROR: $m" -ForegroundColor Red; exit 1 }

if (-not (Test-Path $VenvPy))   { Fail ".venv not found. Run setup first:  powershell -ExecutionPolicy Bypass -File .\setup.ps1" }
if (-not (Test-Path $PyScript)) { Fail "$Script not found in $Root" }

# ---- dataset check -----------------------------------------------------------
while (-not (Test-Path (Join-Path $DataRoot "hierarchical_aggression.csv"))) {
    Write-Host "hierarchical_aggression.csv was not found in: $DataRoot" -ForegroundColor Yellow
    $DataRoot = (Read-Host "Paste the full path of the folder that CONTAINS hierarchical_aggression.csv").Trim('"', ' ')
}
$DataRoot = $DataRoot.TrimEnd('\')
$ImgDir = Join-Path $DataRoot "Img\Img"
if (-not (Test-Path $ImgDir)) { $ImgDir = Join-Path $DataRoot "Img" }
if (-not (Test-Path $ImgDir)) { Fail "Image folder not found. Expected $DataRoot\Img\Img\" }
$nImg = (Get-ChildItem -Path $ImgDir -File).Count
Write-Host "Dataset : $DataRoot"
Write-Host "Images  : $ImgDir  ($nImg files)"
if ($nImg -lt 4000) { Write-Host "WARNING: expected about 4848 images - check the Drive download." -ForegroundColor Yellow }

# ---- environment for the Python script --------------------------------------
$env:ABLATION_DATA_ROOT = $DataRoot
if ($NoWorkers) { $env:ABLATION_NUM_WORKERS = "0" } else { Remove-Item Env:ABLATION_NUM_WORKERS -ErrorAction SilentlyContinue }
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

# ---- keep the PC awake while training ---------------------------------------
if (-not ("Win32.Power" -as [type])) {
    Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
}
[Win32.Power]::SetThreadExecutionState([uint32]2147483649) | Out-Null   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

$Log = Join-Path $Root ("{0}_log_{1}.txt" -f [IO.Path]::GetFileNameWithoutExtension($Script), (Get-Date -Format "yyyyMMdd_HHmm"))
Write-Host "`nStarting $Script (5 losses x 3 seeds = 15 runs, about 4-6 hours). Keep VS Code open." -ForegroundColor Green
Write-Host "If it stops, run the same command again - finished runs are skipped."
Write-Host "Log: $Log`n"

$ErrorActionPreference = "Continue"   # warnings printed on stderr must not stop the run
Push-Location $Root
try {
    & $VenvPy -u $PyScript 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $Log
    $code = $LASTEXITCODE
} finally {
    Pop-Location
    [Win32.Power]::SetThreadExecutionState([uint32]2147483648) | Out-Null  # normal sleep settings again
}

$Out = if ($Script -like "hrf*") { Join-Path $Root "hrf_loss_ablation_outputs" } else { Join-Path $Root "loss_ablation_outputs" }
if ($code -eq 0) {
    Write-Host "`nDONE. Results folder: $Out" -ForegroundColor Green
    if (Test-Path $Out) { Start-Process explorer.exe $Out }
} else {
    Write-Host "`nRun stopped with an error (exit code $code). Send the last 30 lines of $Log" -ForegroundColor Red
    Write-Host "If it mentions DataLoader / worker / spawn, run again with -NoWorkers"
}
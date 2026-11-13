<#
  run_xai.ps1 - runs xai_analysis.py (explanations for the proposed HRFDCA)

  Examples (VS Code terminal, inside this folder):
    powershell -ExecutionPolicy Bypass -File .\run_xai.ps1 -Smoke                 # quick check first
    powershell -ExecutionPolicy Bypass -File .\run_xai.ps1                        # CE, seed 42
    powershell -ExecutionPolicy Bypass -File .\run_xai.ps1 -Loss focal -Seed 123

  The first run of a (loss, seed) pair trains the model (~13 min) and saves a checkpoint; later runs reuse it.
  Options: -DataRoot "D:\datasets"  -NoWorkers
#>
param(
    [string]$Loss = "ce",
    [int]$Seed = 42,
    [string]$DataRoot = "D:\datasets",
    [switch]$Smoke,
    [switch]$NoWorkers
)
$ErrorActionPreference = "Stop"
$Root   = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
$Py     = Join-Path $Root "xai_analysis.py"
function Fail($m) { Write-Host "`nERROR: $m" -ForegroundColor Red; exit 1 }

if (-not (Test-Path $VenvPy)) { Fail ".venv not found. Run:  powershell -ExecutionPolicy Bypass -File .\setup.ps1" }
if (-not (Test-Path $Py))     { Fail "xai_analysis.py not found in $Root" }
if (-not (Test-Path (Join-Path $DataRoot "hierarchical_aggression.csv"))) {
    Fail "hierarchical_aggression.csv not found in $DataRoot (use -DataRoot)"
}
if ($NoWorkers) { $env:ABLATION_NUM_WORKERS = "0" } else { Remove-Item Env:ABLATION_NUM_WORKERS -ErrorAction SilentlyContinue }
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"; $env:HF_HUB_DISABLE_PROGRESS_BARS = "1"

$args2 = @("-u", $Py, "--loss", $Loss, "--seed", $Seed, "--data-root", $DataRoot.TrimEnd('\'))
if ($Smoke) { $args2 += "--smoke" }
$Log = Join-Path $Root ("xai_log_{0}.txt" -f (Get-Date -Format "yyyyMMdd_HHmm"))
$ErrorActionPreference = "Continue"
Push-Location $Root
try {
    & $VenvPy @args2 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $Log
} finally { Pop-Location }
if ($LASTEXITCODE -ne 0) { Write-Host "`nXAI run failed (see $Log)" -ForegroundColor Red; exit 1 }
Write-Host "`nDONE. Open ablation_suite_outputs\xai\...\xai_report.html in a browser." -ForegroundColor Green

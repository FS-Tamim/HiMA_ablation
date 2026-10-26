<#
  run_suite.ps1 - runs ablation_suite.py (HRFDCA / text-only / image-only loss ablation)

  Examples (VS Code terminal, inside this folder):
    powershell -ExecutionPolicy Bypass -File .\run_suite.ps1 -Part smoke     # quick check first (~15-25 min)
    powershell -ExecutionPolicy Bypass -File .\run_suite.ps1 -Part hrfdca    # ~5 h
    powershell -ExecutionPolicy Bypass -File .\run_suite.ps1 -Part text      # ~2-3 h
    powershell -ExecutionPolicy Bypass -File .\run_suite.ps1 -Part image     # ~2 h
    powershell -ExecutionPolicy Bypass -File .\run_suite.ps1 -Part all       # hrfdca + text + image
    powershell -ExecutionPolicy Bypass -File .\run_suite.ps1 -Part summary   # tables only

  Options: -DataRoot "D:\datasets"  -NoWorkers  -Models "muril,xlmr"  -Losses "ce,focal"  -Seeds "42,123"
#>
param(
    [ValidateSet("smoke", "hrfdca", "text", "image", "all", "summary")]
    [string]$Part = "smoke",
    [string]$DataRoot = "D:\datasets",
    [switch]$NoWorkers,
    [string]$Models = "",
    [string]$Losses = "",
    [string]$Seeds = ""
)
$ErrorActionPreference = "Stop"
$Root   = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
$Py     = Join-Path $Root "ablation_suite.py"
function Fail($m) { Write-Host "`nERROR: $m" -ForegroundColor Red; exit 1 }

if (-not (Test-Path $VenvPy)) { Fail ".venv not found. Run:  powershell -ExecutionPolicy Bypass -File .\setup.ps1" }
if (-not (Test-Path $Py))     { Fail "ablation_suite.py not found in $Root" }

while (-not (Test-Path (Join-Path $DataRoot "hierarchical_aggression.csv"))) {
    Write-Host "hierarchical_aggression.csv not found in: $DataRoot" -ForegroundColor Yellow
    $DataRoot = (Read-Host "Paste the folder that CONTAINS hierarchical_aggression.csv").Trim('"', ' ')
}
$env:ABLATION_DATA_ROOT = $DataRoot.TrimEnd('\')
if ($NoWorkers) { $env:ABLATION_NUM_WORKERS = "0" } else { Remove-Item Env:ABLATION_NUM_WORKERS -ErrorAction SilentlyContinue }
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"; $env:HF_HUB_DISABLE_PROGRESS_BARS = "1"

if (-not ("Win32.Power" -as [type])) {
    Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
}
[Win32.Power]::SetThreadExecutionState([uint32]2147483649) | Out-Null   # keep PC awake

$parts = if ($Part -eq "all") { @("hrfdca", "text", "image") } else { @($Part) }
$allOk = $true
$ErrorActionPreference = "Continue"
Push-Location $Root
try {
    foreach ($p in $parts) {
        $args2 = @("-u", $Py, "--part", $p)
        if ($Models) { $args2 += @("--models", $Models) }
        if ($Losses) { $args2 += @("--losses", $Losses) }
        if ($Seeds)  { $args2 += @("--seeds", $Seeds) }
        $Log = Join-Path $Root ("suite_{0}_log_{1}.txt" -f $p, (Get-Date -Format "yyyyMMdd_HHmm"))
        Write-Host "`n=== Running part: $p  (log: $Log) ===" -ForegroundColor Green
        Write-Host "If it stops, run the same command again - finished runs are skipped.`n"
        & $VenvPy @args2 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $Log
        if ($LASTEXITCODE -ne 0) { $allOk = $false; Write-Host "Part $p ended with errors (see $Log)" -ForegroundColor Red }
    }
} finally {
    Pop-Location
    [Win32.Power]::SetThreadExecutionState([uint32]2147483648) | Out-Null
}

$Out = Join-Path $Root "ablation_suite_outputs"
if ($allOk) {
    Write-Host "`nDONE. Results: $Out" -ForegroundColor Green
    if ($Part -ne "smoke" -and (Test-Path $Out)) { Start-Process explorer.exe $Out }
} else {
    Write-Host "`nSome runs failed. Send the last 40 lines of the log." -ForegroundColor Red
    Write-Host "If it mentions DataLoader / worker / spawn, add -NoWorkers"
}
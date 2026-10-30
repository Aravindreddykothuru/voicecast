<#
.SYNOPSIS
  Run Step 2 Indic Parler benchmark on hi, te, kn as a long-lived detached process.

.DESCRIPTION
  Renders 20 lines x 3 seeds x 2 draw sets per language at RTF 44-74 on CPU
  (~3.3 hours per language, ~10 hours total). Checkpointed per render to
  docs/bench_parler_checkpoint.json. Safe to interrupt and resume anytime.

.EXAMPLE
  .\scripts\run_bench_parler_detached.ps1
  .\scripts\run_bench_parler_detached.ps1 -Langs "hi"
#>
param(
    [string]$Langs = "hi,te,kn"
)

$backend = Split-Path -Parent $PSScriptRoot
Set-Location $backend

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:STORAGE_BACKEND = "local"
$env:ALLOW_REMOTE_STORAGE = "false"

$logFile = "$backend\docs\bench_parler.log"
$errFile = "$backend\docs\bench_parler.err.log"

Write-Host "Starting Indic Parler benchmark for [$Langs]..." -ForegroundColor Cyan
Write-Host "Log output: $logFile" -ForegroundColor DarkGray
Write-Host "Checkpoint: docs/bench_parler_checkpoint.json" -ForegroundColor DarkGray

$python = "$backend\.venv\Scripts\python.exe"
$script = "$backend\scripts\bench_parler_vs_syspin.py"
$argsList = @($script, "--out", "docs", "--langs", $Langs)

& $python @argsList 2>&1 | Tee-Object -FilePath $logFile

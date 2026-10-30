<#
.SYNOPSIS
  Run Step 5 CER comparison (IndicConformer vs Vakyansh vs faster-whisper large-v3)
  as a detached background process immune to session/command timeouts.
#>

$backend = Split-Path -Parent $PSScriptRoot
Set-Location $backend

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:STORAGE_BACKEND = "local"
$env:ALLOW_REMOTE_STORAGE = "false"

$logFile = "$backend\docs\asr_comparison.log"
$python = "$backend\.venv\Scripts\python.exe"
$script = "$backend\scripts\run_step5_cer_comparison.py"

Write-Host "Launching Step 5 CER comparison as detached process..." -ForegroundColor Cyan
Write-Host "Log output: $logFile" -ForegroundColor DarkGray
Write-Host "Checkpoint: docs/asr_comparison_checkpoint.json" -ForegroundColor DarkGray

$process = Start-Process -FilePath $python -ArgumentList @($script) -RedirectStandardOutput $logFile -RedirectStandardError "$backend\docs\asr_comparison.err.log" -PassThru -WindowStyle Hidden

Write-Host "Spawned Process PID: $($process.Id)" -ForegroundColor Green
$process.Id | Out-File -FilePath "$backend\docs\step5.pid" -Encoding utf8

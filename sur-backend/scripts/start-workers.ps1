<#
.SYNOPSIS
  Run both Celery workers as long-lived processes you own, with restart on
  crash and a memory ceiling.

.DESCRIPTION
  The UI needs two consumers or a dub sits in the queue forever: the main
  worker takes every stage except synthesis, and the TTS worker takes
  q.synthesize from its own virtualenv (.venv-tts pins torch/transformers
  versions the other providers cannot use).

  Run this yourself, in its own terminal. It is deliberately NOT started by
  any agent session: workers launched inside one die with it, which is how
  a finished dub was followed by "nothing is consuming q.*".

  Each worker runs under a supervisor loop that restarts it if it exits,
  with a short backoff so a crash loop does not spin the CPU. Memory is
  bounded two ways: --max-memory-per-child makes Celery recycle a worker
  process once it exceeds the ceiling (the real protection on a 15 GB box
  where SYSPIN holds ~2.6 GB and Indic Parler ~3.9 GB), and the supervisor
  refuses to start a worker when free RAM is already below the floor rather
  than letting Windows page the whole machine to a standstill.

  Storage is local disk unless you pass -RemoteStorage: .env carries real
  bucket credentials and three e2e runs wrote to the live bucket because
  nothing overrode STORAGE_BACKEND.

.PARAMETER Role
  both (default), main, or tts.

.PARAMETER MaxMemoryMb
  Per-child ceiling. Celery recycles the child after a task takes it past
  this. Default 4096, which clears Indic Parler's measured 3,925 MB peak
  (docs/tts-warmup-run.json) with a little headroom.

.PARAMETER MinFreeMb
  Refuse to start, or to restart, while free RAM is under this. Default 800.

.PARAMETER Offline
  HF_HUB_OFFLINE=1: load every model from the local cache, touch no network.

.EXAMPLE
  .\scripts\start-workers.ps1
  .\scripts\start-workers.ps1 -Role tts -MaxMemoryMb 5120
  .\scripts\start-workers.ps1 -Offline
#>
[CmdletBinding()]
param(
    [ValidateSet("both", "main", "tts")][string]$Role = "both",
    [int]$MaxMemoryMb = 4096,
    [int]$MinFreeMb = 800,
    [switch]$Offline,
    [switch]$RemoteStorage
)

$ErrorActionPreference = "Continue"
$backend = Split-Path -Parent $PSScriptRoot
Set-Location $backend

$env:PYTHONUTF8 = "1"
if ($Offline) { $env:HF_HUB_OFFLINE = "1" }

if ($RemoteStorage) {
    $env:ALLOW_REMOTE_STORAGE = "true"
    Write-Host "storage: REMOTE -- this writes to the real bucket" -ForegroundColor Yellow
} else {
    $env:STORAGE_BACKEND = "local"
    $env:ALLOW_REMOTE_STORAGE = "false"
    Write-Host "storage: local disk (data/storage)" -ForegroundColor DarkGray
}

$MAIN_QUEUES = "q.extract_audio,q.chunk_and_diarize,q.transcribe,q.detect_emotion,q.translate,q.mux_export"
$TTS_QUEUES = "q.synthesize"

function Get-FreeMb {
    try { [int](Get-Counter '\Memory\Available MBytes' -ErrorAction Stop).CounterSamples[0].CookedValue }
    catch { 99999 }   # cannot measure: do not block on it
}

function Test-Prereqs {
    $ok = $true
    foreach ($p in @(".\.venv\Scripts\python.exe", ".\.venv-tts\Scripts\python.exe")) {
        if (-not (Test-Path $p)) { Write-Host "missing interpreter: $p" -ForegroundColor Red; $ok = $false }
    }
    # Redis is the broker; without it the workers start and silently idle.
    $redis = $env:CELERY_BROKER_URL
    if (-not $redis) { $redis = "redis://localhost:6381/1" }
    if ($redis -match ":(\d+)") {
        $port = [int]$Matches[1]
        $probe = Test-NetConnection -ComputerName 127.0.0.1 -Port $port -WarningAction SilentlyContinue
        if (-not $probe.TcpTestSucceeded) {
            Write-Host "broker not reachable on port $port -- start it with:" -ForegroundColor Red
            Write-Host "    docker compose -f docker-compose.dev.yml up -d" -ForegroundColor Yellow
            $ok = $false
        }
    }
    return $ok
}

function Start-Supervised {
    param([string]$Name, [string]$Python, [string]$Queues, [string]$NodeName)

    $flags = @(
        "-m", "celery", "-A", "app.celery_app", "worker",
        "-Q", $Queues, "-n", $NodeName,
        "--without-gossip", "--without-mingle", "--without-heartbeat",
        "--loglevel=INFO",
        # Celery recycles a child once a task pushes it past this, which is
        # what actually bounds a run away on this box.
        "--max-memory-per-child=$($MaxMemoryMb * 1024)"
    )

    $script = {
        param($Name, $Python, $Flags, $MinFreeMb, $Backend)
        Set-Location $Backend
        $attempt = 0
        while ($true) {
            $free = try { [int](Get-Counter '\Memory\Available MBytes' -ErrorAction Stop).CounterSamples[0].CookedValue } catch { 99999 }
            if ($free -lt $MinFreeMb) {
                Write-Host "[$Name] only ${free} MB free (floor $MinFreeMb) -- waiting 30s" -ForegroundColor Yellow
                Start-Sleep -Seconds 30
                continue
            }
            $attempt++
            Write-Host "[$Name] starting (attempt $attempt, ${free} MB free)" -ForegroundColor Cyan
            & $Python @Flags
            $code = $LASTEXITCODE
            # 0 and 15 are a clean stop (Ctrl+C / SIGTERM): do not resurrect.
            if ($code -eq 0 -or $code -eq 15) { Write-Host "[$Name] stopped cleanly" -ForegroundColor DarkGray; break }
            $backoff = [Math]::Min(60, 5 * $attempt)
            Write-Host "[$Name] exited $code -- restarting in ${backoff}s" -ForegroundColor Red
            Start-Sleep -Seconds $backoff
        }
    }

    Start-Job -Name $Name -ScriptBlock $script -ArgumentList $Name, $Python, $flags, $MinFreeMb, $backend | Out-Null
    Write-Host "supervising $Name ($Queues)" -ForegroundColor Green
}

if (-not (Test-Prereqs)) { Write-Host "refusing to start" -ForegroundColor Red; exit 1 }

$free = Get-FreeMb
Write-Host "free RAM: ${free} MB | per-child ceiling: ${MaxMemoryMb} MB | floor: ${MinFreeMb} MB"

if ($Role -eq "both" -or $Role -eq "main") {
    Start-Supervised -Name "worker-main" -Python ".\.venv\Scripts\python.exe" -Queues $MAIN_QUEUES -NodeName "main@%h"
}
if ($Role -eq "both" -or $Role -eq "tts") {
    Start-Supervised -Name "worker-tts" -Python ".\.venv-tts\Scripts\python.exe" -Queues $TTS_QUEUES -NodeName "tts@%h"
}

Write-Host ""
Write-Host "Workers are supervised in this window. Ctrl+C to stop them all." -ForegroundColor Green
Write-Host "  Receive-Job -Name worker-main -Keep   # see its output"
Write-Host "  Get-Job                               # state of both"
Write-Host ""

try {
    while ($true) {
        Get-Job | ForEach-Object { Receive-Job -Job $_ }
        Start-Sleep -Seconds 2
    }
} finally {
    Write-Host "stopping workers..." -ForegroundColor Yellow
    Get-Job | Stop-Job -ErrorAction SilentlyContinue
    Get-Job | Remove-Job -Force -ErrorAction SilentlyContinue
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like '*celery*app.celery_app*' } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

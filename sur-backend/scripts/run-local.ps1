<#
.SYNOPSIS
  Run one piece of the Sur backend natively on Windows (no Docker for Python).

.DESCRIPTION
  Three roles, each its own long-running process:

    api         FastAPI on :8000                      (.venv)
    worker      every stage except synthesis          (.venv)
    tts-worker  q.synthesize only                     (.venv-tts)

  The split mirrors docker-compose.yml: the TTS stack (CosyVoice2 voice
  conversion) pins torch/transformers versions the other providers can't use,
  so it lives in a second virtualenv. See requirements-tts.txt.

  Infra first:  docker compose -f docker-compose.dev.yml up -d
  Schema:       .\.venv\Scripts\python.exe -m alembic upgrade head

  -Offline sets HF_HUB_OFFLINE=1: workers load every Hugging Face model from
  the local cache and never touch the network. That is also how the gated
  models (pyannote, IndicTrans2) load without HF_TOKEN once cached.

  Storage defaults to local disk here whatever .env says, because .env holds
  real bucket credentials and this script is the local-development entry
  point: three e2e runs on 2026-10-01 wrote to the production bucket because
  nothing overrode STORAGE_BACKEND=s3. Pass -RemoteStorage to opt in, which
  also sets ALLOW_REMOTE_STORAGE so get_storage() permits it.

  Celery runs with the solo pool on Windows (set in app/celery_app.py);
  prefork is unsupported there.

.EXAMPLE
  .\scripts\run-local.ps1 -Role worker -Offline
#>
param(
    [Parameter(Mandatory)][ValidateSet("api", "worker", "tts-worker")][string]$Role,
    [switch]$Offline,
    [switch]$RemoteStorage,
    [int]$Port = 8000
)

$backend = Split-Path -Parent $PSScriptRoot
Set-Location $backend

if ($Offline) { $env:HF_HUB_OFFLINE = "1" }
$env:PYTHONUTF8 = "1"

# Local by default, remote only when asked for out loud. .env is configured
# for the real bucket, so without this a local run silently writes there.
if ($RemoteStorage) {
    $env:ALLOW_REMOTE_STORAGE = "true"
    Write-Host "storage: REMOTE (ALLOW_REMOTE_STORAGE=true) -- this writes to the real bucket" -ForegroundColor Yellow
} else {
    $env:STORAGE_BACKEND = "local"
    $env:ALLOW_REMOTE_STORAGE = "false"
    Write-Host "storage: local disk (data/storage). Pass -RemoteStorage to use the bucket." -ForegroundColor DarkGray
}
# Deliberately NOT $ErrorActionPreference = "Stop": in Windows PowerShell 5.1
# that turns the first line a native process writes to stderr -- any Python
# warning -- into a terminating error once output is redirected, killing a
# healthy server. Python's own exit code is what's propagated below.
$ErrorActionPreference = "Continue"

# The solo pool can't service gossip/heartbeat events while a task runs, and a
# CPU synthesize task runs for minutes: every peer's queued heartbeats then
# arrive "late" and Celery logs "Substantial drift ... clocks are out of sync"
# for clocks that are fine. These workers don't use gossip, mingle or
# broker heartbeats (Redis broker; one worker per queue set), so they're off.
$workerFlags = @("--without-gossip", "--without-mingle", "--without-heartbeat", "--loglevel=INFO")

switch ($Role) {
    "api" {
        & "$backend\.venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port $Port
    }
    "worker" {
        $queues = "q.extract_audio,q.chunk_and_diarize,q.transcribe,q.detect_emotion,q.translate,q.mux_export"
        & "$backend\.venv\Scripts\python.exe" -m celery -A app.celery_app worker -Q $queues -n "main@%h" @workerFlags
    }
    "tts-worker" {
        & "$backend\.venv-tts\Scripts\python.exe" -m celery -A app.celery_app worker -Q q.synthesize -n "tts@%h" @workerFlags
    }
}
exit $LASTEXITCODE

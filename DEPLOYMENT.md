# VoiceCast (Sur) — Production Deployment Guide

This guide covers deploying the full VoiceCast dubbing stack (FastAPI Backend, Celery ML Workers, PostgreSQL, Redis, and React/Vite Frontend) to a Cloud Virtual Machine (AWS EC2, RunPod, Hetzner, DigitalOcean, or GCP) using Docker.

---

## 1. System Requirements & Recommended Sizing

The VoiceCast pipeline runs deep-learning models for ASR (Whisper), Diarization (Pyannote), Translation (IndicTrans2 1B), Emotion Recognition (wav2vec2), and Voice Synthesis (SYSPIN VITS + OpenVoice V2).

| Profile | Specs | Ideal Cloud Instances | Throughput |
|---|---|---|---|
| **Production GPU (Recommended)** | 1x NVIDIA GPU (16GB VRAM: T4 / A10G / L4), 4–8 vCPUs, 16–32 GB RAM, 80+ GB SSD | AWS `g4dn.xlarge` or `g5.xlarge`, RunPod `1x RTX 3090/4090/A4000`, Lambda Labs | **~0.15–0.3x RTF** (1 min video dubs in ~15–20s) |
| **Budget CPU** | 4–8 vCPUs, 16 GB RAM, 60+ GB SSD | AWS `c6i.2xlarge` or `t3.2xlarge`, Hetzner `CPX41` (8 vCPU / 16GB RAM) | **~1.5–3.0x RTF** (1 min video dubs in ~2–3 min) |

> [!IMPORTANT]
> A minimum of **16 GB RAM** and **60 GB disk space** is required to hold the model checkpoints and prevent out-of-memory errors during model initialization.

---

## 2. Prerequisites on the Host VM

1. **Operating System**: Ubuntu 22.04 LTS or 24.04 LTS (recommended).
2. **Docker & Docker Compose**:
   ```bash
   # Install Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker $USER
   newgrp docker
   ```
3. *(For GPU instances only)* **NVIDIA Container Toolkit**:
   ```bash
   sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

---

## 3. Deployment Steps

### Step 1: Clone the Repository
```bash
git clone <YOUR_GIT_REPO_URL> voicecast
cd voicecast
```

### Step 2: Configure Environment (`.env`)
Create `.env` in the repository root:
```bash
cat << 'EOF' > .env
# --- Security & Secrets ---
HF_TOKEN=hf_your_huggingface_token_here
POSTGRES_USER=sur
POSTGRES_PASSWORD=generate_a_strong_password_here
POSTGRES_DB=sur

DATABASE_URL=postgresql+psycopg2://sur:<YOUR_DB_PASSWORD>@postgres:5432/sur
ASYNC_DATABASE_URL=postgresql+asyncpg://sur:<YOUR_DB_PASSWORD>@postgres:5432/sur
REDIS_URL=redis://redis:6379/0
CELERY_BROKER_URL=redis://redis:6379/1
CELERY_RESULT_BACKEND=redis://redis:6379/2

# --- Storage Configuration ---
# Option A: Local disk storage on the host (recommended default)
STORAGE_BACKEND=local
ALLOW_REMOTE_STORAGE=false
LOCAL_STORAGE_ROOT=/srv/data/storage

# Option B: AWS S3 or Cloudflare R2 (Uncomment and set if using cloud bucket)
# STORAGE_BACKEND=s3
# ALLOW_REMOTE_STORAGE=true
# STORAGE_ACCESS_KEY=your_access_key
# STORAGE_SECRET_KEY=your_secret_key
# STORAGE_BUCKET=your_bucket_name
# STORAGE_REGION=ap-south-2
# STORAGE_USE_SSL=true

# --- Pipeline & Providers ---
ASR_PROVIDER=real
DIARIZATION_PROVIDER=real
TRANSLATION_PROVIDER=real
EMOTION_PROVIDER=real
TTS_PROVIDER=real
TTS_ENGINE=syspin
TTS_REQUIRE_COMMERCIAL_LICENSE=true
TTS_VOICE_CLONE=true
TTS_VOICE_CLONE_ENGINE=openvoice

# --- CORS ---
API_CORS_ORIGINS=*
EOF
```

### Step 3: Build & Launch All Containers
```bash
# Build images and start all services in the background
docker compose -f docker-compose.prod.yml up -d --build
```

### Step 4: Verify the Services
Check container status:
```bash
docker compose -f docker-compose.prod.yml ps
```

Verify backend and worker health:
```bash
# Check API health
curl http://localhost/healthz
# Response: {"status":"ok"}

# Check Celery Worker Health
curl http://localhost/api/health/workers
# Response: {"workers_online": 2, "queues": {"q.extract_audio": 0, "q.synthesize": 0, ...}}
```

Check live logs:
```bash
# Watch worker logs during model self-checks
docker compose -f docker-compose.prod.yml logs -f worker-main worker-tts
```

---

## 4. Architecture & Port Mapping

| Service | Container Port | Host Port | Purpose |
|---|---|---|---|
| **frontend** | `80` | `80` (HTTP) | Nginx reverse proxy + React SPA |
| **api** | `8000` | `8000` | FastAPI Web API |
| **worker-main** | Internal | Internal | Celery worker (ASR, Diarize, Translate, Emotion, Mux) |
| **worker-tts** | Internal | Internal | Celery worker (VITS TTS + Voice Cloning) |
| **postgres** | `5432` | `5432` | Relational database (metadata, segments, projects) |
| **redis** | `6379` | `6379` | Celery broker & result backend |

> **Reverse Proxy Note**: The `frontend` container automatically reverse proxies all `/api/*` and `/ws/*` requests to the `api` container. Opening `http://<VM_PUBLIC_IP>/` in any browser connects to the full application.

---

## 5. Domain & HTTPS / SSL Setup (Optional)

To secure the deployment with SSL (HTTPS) on a custom domain:

1. Point your domain DNS **A record** (`dub.yourdomain.com`) to your VM's public IP.
2. Install Certbot on the host:
   ```bash
   sudo apt-get install -y certbot python3-certbot-nginx
   ```
3. Run Certbot to generate and configure SSL certificates:
   ```bash
   sudo certbot --nginx -d dub.yourdomain.com
   ```
   *Or use Cloudflare in Proxy mode (Orange Cloud) with Universal SSL for instant zero-configuration HTTPS.*

---

## 6. Maintenance Commands

- **Restart the Stack**:
  ```bash
  docker compose -f docker-compose.prod.yml restart
  ```
- **Update to Latest Code**:
  ```bash
  git pull origin main
  docker compose -f docker-compose.prod.yml up -d --build
  ```
- **Stop the Stack**:
  ```bash
  docker compose -f docker-compose.prod.yml down
  ```
- **Backup Database**:
  ```bash
  docker compose -f docker-compose.prod.yml exec postgres pg_dump -U sur sur > backup_$(date +%F).sql
  ```

---

## 7. Continuous Deployment (Automatic Updates on Git Push)

A GitHub Actions workflow ([.github/workflows/deploy.yml](.github/workflows/deploy.yml)) is included so that any change pushed to the `main` branch automatically updates the live project without manual SSH commands.

### How to Enable:
1. In your GitHub repository (`Aravindreddykothuru/voicecast`), navigate to **Settings** → **Secrets and variables** → **Actions**.
2. Click **New repository secret** and add:
   - `DEPLOY_HOST`: The public IP or hostname of your live server.
   - `DEPLOY_USER`: The SSH user (e.g. `ubuntu` or `root`).
   - `DEPLOY_KEY`: The private SSH key (e.g. `id_rsa` or `id_ed25519`) with access to the server.
   - `DEPLOY_PATH`: The directory path on the server (e.g. `/home/ubuntu/voicecast`).
   - `DEPLOY_PORT`: (Optional) SSH port if different from `22`.

Once set, whenever you `git push origin main`:
- GitHub Actions connects to the live server.
- Pulls the latest commits.
- Runs `docker compose -f docker-compose.prod.yml up -d --build`.
- Gracefully restarts updated containers with zero manual intervention.


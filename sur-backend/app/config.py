"""Centralized, .env-driven configuration.

Every knob that differs between local dev, CI, and production lives here --
nothing else in the codebase should call os.environ directly. This is what
makes the "run without a GPU" and "swap a provider" requirements possible
without touching pipeline code.
"""
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine.url import make_url

ProviderMode = Literal["mock", "real"]

# sur-backend/. Relative paths in settings (model dirs, tool binaries, local
# storage) resolve against this, NOT the process's working directory: the API
# and each worker are separate processes, often started from different
# directories, and they must all agree on where a file lives. A relative
# "./data/storage" meant an upload written by the API could be invisible to a
# worker started one directory up.
BACKEND_ROOT = Path(__file__).resolve().parent.parent


def resolve_backend_path(value: str) -> str:
    """Absolute paths pass through; relative ones anchor at BACKEND_ROOT."""
    p = Path(value)
    return str(p if p.is_absolute() else (BACKEND_ROOT / p).resolve())


# The signing key a dev checkout gets. Named once so the field default and
# the production guard below cannot drift apart, and so it is greppable:
# this repository is public, which means this exact string is public. Any
# deployment still using it can have session tokens forged for any user.
DEV_JWT_SECRET = "dev-only-insecure-secret-override-in-production"  # pragma: allowlist secret
MIN_JWT_SECRET_LENGTH = 32


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- App ---
    app_name: str = "sur-backend"
    environment: str = "development"
    log_level: str = "INFO"
    api_cors_origins: str = "http://localhost:3000"

    # --- Database ---
    database_url: str = "postgresql+psycopg2://sur:sur@localhost:5432/sur"
    async_database_url: str = "postgresql+asyncpg://sur:sur@localhost:5432/sur"

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # --- Object storage ---
    storage_backend: str = "local"  # "local" or "s3"
    # A second, deliberate switch for anything that is not local disk. `.env`
    # carries real bucket credentials, so STORAGE_BACKEND=s3 alone is one
    # forgotten override away from a local run writing to production -- which
    # has happened. get_storage() refuses a remote backend unless this is on.
    allow_remote_storage: bool = False
    # None means "use the real AWS S3 endpoint for storage_region" -- boto3's
    # own client() treats endpoint_url=None as "no override" natively. Only
    # set an explicit http://... value when pointing at MinIO, R2, or another
    # S3-compatible store that isn't AWS itself.
    storage_endpoint_url: str | None = None
    storage_public_endpoint_url: str | None = None
    storage_access_key: str = "sur_minio"
    storage_secret_key: str = "sur_minio_secret"
    storage_bucket: str = "sur-media"
    storage_region: str = "us-east-1"
    storage_use_ssl: bool = False
    # STORAGE_BACKEND=local only. Relative values anchor at BACKEND_ROOT so
    # the API (which serves /api/storage/*) and every worker share one tree.
    local_storage_root: str = "data/storage"

    # --- Media tooling ---
    # Bare names are looked up on PATH (the Docker images apt-install
    # ffmpeg). A native Windows checkout usually has no ffmpeg on PATH, so
    # point these at the binaries instead, e.g.
    # .tools/ffmpeg-9.0.1-essentials_build/bin/ffmpeg.exe (relative paths
    # anchor at BACKEND_ROOT). A missing binary is a hard, non-retried error
    # naming the variable to set -- see app/pipeline/ffmpeg_utils.py.
    ffmpeg_binary: str = "ffmpeg"
    ffprobe_binary: str = "ffprobe"

    # --- Providers ---
    asr_provider: ProviderMode = "mock"
    diarization_provider: ProviderMode = "mock"
    translation_provider: ProviderMode = "mock"
    emotion_provider: ProviderMode = "mock"
    tts_provider: ProviderMode = "mock"

    # NOTE: hf_token is deliberately NOT a Settings field. A long-lived
    # credential must not live in a file next to the code -- see hf_token()
    # below, which reads it from the process environment at call time.

    asr_model_name: str = "large-v3"
    asr_device: str = "cpu"
    # Source language for ASR. "auto" lets Whisper detect it per chunk, which
    # is what a dubbing tool wants -- pinning it to "en" (as this used to)
    # forces every non-English source to be decoded as English. Set an
    # explicit code (e.g. "en", "hi") when the source language is known, since
    # that's both faster and more accurate than detection on short chunks.
    asr_language: str = "auto"
    # One model per IndicTrans2 direction, loaded lazily by whichever
    # (source, target) pair actually shows up -- a run that never dubs a
    # non-English source never pays to download indic-en/indic-indic.
    # translation_src_lang was removed: it hardcoded every source as English
    # regardless of what ASR detected, which is the exact bug that let a
    # 98%-confidence Chinese detection get silently translated as English.
    # See CONTRACTS.md #3 and #5.
    translation_model_name: str = "ai4bharat/indictrans2-en-indic-1B"
    translation_indic_en_model_name: str = "ai4bharat/indictrans2-indic-en-1B"
    translation_indic_indic_model_name: str = "ai4bharat/indictrans2-indic-indic-1B"
    # --- TTS ---
    # TTS_PROVIDER=real renders speech with MMS-TTS (facebook/mms-tts-<lang>,
    # one small VITS checkpoint per language; the per-language suffix lives
    # in app/capabilities.py). CosyVoice2 used to be the renderer, but it was
    # never trained on any Indic language: on CPU it spent 30+ minutes on one
    # 3-second Telugu sentence and returned ~21s of unintelligible audio.
    tts_mms_model_prefix: str = "facebook/mms-tts-"
    # Which TTS engine renders speech (app/providers/tts/). Voices per
    # language and their licenses live in app/capabilities.py.
    #   syspin -- IISc SYSPIN VITS, CC-BY-4.0: commercial use with attribution
    #   mms    -- facebook/mms-tts, CC-BY-NC-4.0: NOT for commercial use
    tts_engine: Literal["syspin", "mms"] = "syspin"
    # Feature flag, off by default: when true and tts_engine=syspin, the raw
    # SYSPIN render happens in an isolated subprocess (app/tts_runtime) rather
    # than in-process, so a crash, hang or runaway TorchScript call in the
    # render can no longer take the whole Celery worker down with it. Every
    # other step (prosody, time-fit, voice cloning, loudness normalize) is
    # unchanged, and the two paths are proven byte-identical on a fixed sample
    # set: tests/test_runtime_syspin_provider.py. One-line rollback: set this
    # back to false (or unset TTS_USE_RUNTIME).
    tts_use_runtime: bool = False
    # When true, only voices whose license permits commercial use are
    # offered (languages without one report tts_available=false) and a
    # non-commercial engine refuses to start. Production must set this.
    tts_require_commercial_license: bool = True
    # Voice cloning (a project's clone_voice flag): CosyVoice2 *voice
    # conversion* re-voices the MMS speech as the original speaker. VC uses
    # no text model, so the Indic-language limitation above doesn't apply.
    # Off unless enabled -- it loads CosyVoice2 (~4GB) into the TTS worker.
    tts_voice_clone: bool = False
    # openvoice (OpenVoice V2, MIT) or cosyvoice (CosyVoice2 VC). See
    # app/providers/tts/voice_clone.py for the budget each must pass.
    tts_voice_clone_engine: Literal["openvoice", "cosyvoice"] = "openvoice"
    # Conversion must run at or below this multiple of real time on the TTS
    # worker, checked at startup (best of a few runs -- see check_budget).
    # Measured on an 8-core CPU box, 3s probe: OpenVoice V2 1.9-3.0x,
    # CosyVoice2 VC ~25x. 4.0 clears the faster converter's spread with room
    # for a loaded machine and still refuses one an order of magnitude slower.
    # This is a "don't ship something unusable" bound, not a latency target:
    # cloning is opt-in and runs offline, in a worker, on whole clips.
    tts_voice_clone_max_rtf: float = 4.0
    # OpenVoice source checkout, put on sys.path by the converter.
    openvoice_src_dir: str | None = None
    # CosyVoice2 checkpoint directory, used only for voice conversion.
    # Relative paths anchor at BACKEND_ROOT.
    tts_model_name: str = "cosyvoice2"
    # CosyVoice ships as a source checkout, not a package. When set, this
    # directory (and its third_party/Matcha-TTS) is put on sys.path by the
    # provider itself, so a worker no longer depends on someone remembering
    # to export PYTHONPATH -- the local TTS venv could not import cosyvoice
    # at all without it. Docker sets PYTHONPATH instead and leaves this unset.
    cosyvoice_src_dir: str | None = None
    tts_device: str = "cpu"
    # Must be a checkpoint whose classification head actually loads under the
    # installed transformers. Two earlier picks failed that bar:
    # audeering/...-24-ft-... doesn't exist on the Hub, and
    # ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition stores its head
    # as classifier.dense/classifier.output, which today's
    # Wav2Vec2ForSequenceClassification ignores -- so it silently loaded with a
    # RANDOM head and emitted noise at ~chance confidence. SUPERB's ER models
    # use the standard head and load cleanly (verified by loading twice and
    # comparing classifier.weight). Trade-off: IEMOCAP's 4 labels only
    # (neutral/happy/angry/sad), so fear and surprise are never predicted.
    # superb/hubert-large-superb-er is a drop-in, more accurate, slower swap.
    emotion_model_name: str = "superb/wav2vec2-base-superb-er"

    # Emotion predictions below this confidence are shown as "uncertain"
    # rather than as a label. One number, served via /api/capabilities, so the
    # UI never hardcodes its own threshold.
    emotion_confidence_floor: float = 0.4

    # --- Pipeline ---
    default_target_language: str = "te"
    sync_tolerance_pct: float = 10.0
    # Liveness (app/pipeline/liveness.py). Workers stamp a heartbeat this
    # often while a task runs; a queued/processing run silent for longer than
    # stall_after_seconds is reported as stalled and may be restarted.
    # Served via /api/capabilities so the UI never hardcodes its own timeout.
    heartbeat_interval_seconds: int = 20
    stall_after_seconds: int = 300

    # --- Upload limits ---
    # Enforced server-side (routes_projects.py), not just advisory copy in the
    # UI: an upload whose content_type isn't listed here, or whose confirmed
    # size exceeds this, is rejected rather than silently accepted and failing
    # deep in the pipeline later. Served via /api/capabilities so the UI's
    # file picker can't offer a choice the backend will then refuse.
    max_upload_mb: int = 2048
    accepted_video_formats: str = "video/mp4,video/quicktime,video/x-matroska,video/webm"

    @property
    def accepted_video_format_list(self) -> list[str]:
        return [f.strip() for f in self.accepted_video_formats.split(",") if f.strip()]

    @property
    def compute_device(self) -> str:
        """Single honest device summary for /api/capabilities' Runtime panel.

        Derived from the configured per-stage device settings, not a live
        torch.cuda.is_available() probe: the API process doesn't necessarily
        have torch installed (it isn't in requirements.txt, only
        requirements-ml.txt/-tts.txt, which run in the worker containers) and
        even if it did, the API's own hardware isn't necessarily the worker's.
        Reports "cuda" only if every real-provider device knob actually asks
        for it -- a mixed deployment (e.g. ASR on GPU, TTS still on CPU)
        reports "cpu" so ETA estimates stay conservative rather than
        overpromising. See CONTRACTS.md #5 (no silent defaults / no
        optimistic guessing).
        """
        return "cuda" if self.asr_device == "cuda" and self.tts_device == "cuda" else "cpu"

    @property
    def asr_autodetect(self) -> bool:
        return self.asr_language == "auto"

    @property
    def voice_clone_available(self) -> bool:
        """Whether a clone_voice=True run can be served. Published via
        /api/capabilities and enforced by /process, so the UI cannot offer a
        toggle the TTS worker would then fail on."""
        return self.tts_provider == "mock" or self.tts_voice_clone

    # Hosts that hold real user data. A process that is not explicitly
    # ENVIRONMENT=production must never open a connection to one of these.
    # This is a crash, not a warning, because the convention "don't point
    # your local .env at prod" already failed once: migrations were applied
    # and rows were deleted against the database holding real users during a
    # debugging session, because dev and prod were the same instance.
    production_db_hosts: str = "pooler.supabase.com,db.bwwdpjkdxmgfdlyffgzr.supabase.co"

    @property
    def production_db_host_list(self) -> list[str]:
        return [h.strip() for h in self.production_db_hosts.split(",") if h.strip()]

    @model_validator(mode="after")
    def _refuse_production_database_outside_production(self) -> "Settings":
        if self.environment.strip().lower() == "production":
            return self
        host = (make_url(self.database_url).host or "").lower()
        for prod_host in self.production_db_host_list:
            if prod_host.lower() in host:
                raise RuntimeError(
                    f"REFUSING TO START: DATABASE_URL points at the production host "
                    f"{host!r} but ENVIRONMENT={self.environment!r}, not 'production'.\n"
                    "This process would read and write real user data. Point "
                    "DATABASE_URL at your dev database (see docker-compose.test.yml "
                    "or a second Supabase project), or set ENVIRONMENT=production if "
                    "this really is the production deployment.\n"
                    "Production credentials belong in deploy config, never in a local .env."
                )
        return self

    def require_secure_production_runtime(self) -> None:
        """Refuse the values a dev checkout ships with, for processes that
        actually serve authenticated requests.

        Called from app/main.py at import, NOT from a model validator.
        Everything loads Settings -- Alembic, Celery, one-off scripts -- and
        none of those sign or verify a token or answer a CORS preflight.
        Enforcing it globally meant `alembic upgrade head` against
        production died with a JWT error, which is both confusing and a
        reason to reach for a workaround during an incident.

        The database guard stays a model validator, because THAT one does
        apply to every process: Alembic and Celery must not touch the
        production database from a dev environment either.

        This repository is public, so DEV_JWT_SECRET is not merely weak, it
        is published. HS256 tokens signed with a known key can be forged for
        any user id: no password, no failed login, nothing unusual in a log.
        """
        if self.environment.strip().lower() != "production":
            return

        secret = self.jwt_secret_key.strip()
        if secret == DEV_JWT_SECRET:
            raise RuntimeError(
                "REFUSING TO START: JWT_SECRET_KEY is still the development "
                "default, and this repository is public, so that value is "
                "known to everyone. Session tokens could be forged for any "
                "user. Set JWT_SECRET_KEY in your deploy config to a random "
                "value, e.g. `python -c \"import secrets; print(secrets.token_urlsafe(48))\"`."
            )
        if len(secret) < MIN_JWT_SECRET_LENGTH:
            raise RuntimeError(
                f"REFUSING TO START: JWT_SECRET_KEY is {len(secret)} characters; "
                f"at least {MIN_JWT_SECRET_LENGTH} are required. A short HS256 key "
                "is brute-forceable offline from a single captured token."
            )

        if "*" in self.cors_origin_list:
            raise RuntimeError(
                "REFUSING TO START: API_CORS_ORIGINS is '*' while credentials are "
                "allowed (app/main.py sets allow_credentials=True). That combination "
                "lets any site on the internet make authenticated requests as a "
                "logged-in user. List the real origins instead."
            )

    # --- Auth ---
    # Falls back to the X-User-Email dev stub (app/core/security.py) when no
    # request carries a real bearer token -- keeps every existing route and
    # test working unchanged while /api/auth/{signup,login} issue real,
    # password-verified tokens for anyone who goes through them.
    dev_default_user_email: str = "dev@sur.local"

    @property
    def dev_email_auth_enabled(self) -> bool:
        """Whether the X-User-Email identity stub is accepted.

        Never in production, and deliberately with no override. The stub
        authenticates on a header alone: `curl -H "X-User-Email: someone@..."`
        with no token and no password returns that user's data, and creates
        the account if it does not exist. That is fine for a local dev tool
        and is unauthenticated account takeover anywhere else.

        Derived from ENVIRONMENT rather than configurable, because a
        security control with an "enable it anyway" flag is one env var away
        from being off in the place it matters.
        """
        return self.environment.strip().lower() != "production"
    # A real deployment MUST override this (env var, not this default) --
    # anyone who knows it can forge a valid token for any user id. Kept as a
    # plain default (not a hard-fail) only because this is still a
    # single-user local dev tool; see CONTRACTS.md #4 for the same rule
    # already enforced for HF_TOKEN.
    jwt_secret_key: str = DEV_JWT_SECRET
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 60 * 24 * 7  # 7 days

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.api_cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------
# Read from the process environment at call time, never persisted to .env and
# never cached in Settings, so rotating the token is a restart rather than an
# edit-and-redeploy, and a stale value can't linger in a cached object.
# See CONTRACTS.md invariant #4 and README "Rotating the Hugging Face token".
HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
# huggingface_hub's own switches, read by the library itself from the process
# environment at import time -- so they are read from the same place here,
# never from .env (a .env value would claim offline while the library, which
# never sees it, still went to the network).
HF_OFFLINE_ENV_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")


def hf_token() -> str | None:
    """The Hugging Face token, or None if unset. Callers that require it should
    use require_hf_token() so the failure names the variable to set."""
    for var in HF_TOKEN_ENV_VARS:
        value = os.environ.get(var)
        if value and value.strip():
            return value.strip()
    return None


def hf_hub_offline() -> bool:
    """True when huggingface_hub is forbidden from touching the network.

    That is the deployment mode where weights are provisioned ahead of time
    (a models volume, a warmed cache) and workers only ever read them. It is
    also the ONLY way to load a gated model that is already cached without a
    token: online, hf_hub_download re-raises GatedRepoError on its metadata
    HEAD request even when every file is on disk, and pyannote resolves its
    sub-models through that same call.
    """
    return any(
        os.environ.get(var, "").strip().lower() in ("1", "true", "yes", "on")
        for var in HF_OFFLINE_ENV_VARS
    )


def require_hf_token(reason: str) -> str | None:
    """Fail with an actionable message rather than letting a gated download
    return an opaque 401. CONTRACTS.md #3: never substitute a silent default.

    Returns None under HF_HUB_OFFLINE: nothing will be downloaded, so there is
    nothing to authenticate. Offline does not mean "unchecked" -- a gated
    model missing from the cache still fails its load, loudly, and workers
    load every configured model at startup (startup_checks.verify_models).
    """
    token = hf_token()
    if token:
        return token
    if hf_hub_offline():
        return None
    raise RuntimeError(
        f"{reason} requires a Hugging Face token, but none of "
        f"{', '.join(HF_TOKEN_ENV_VARS)} is set in the environment. "
        "Create one at https://huggingface.co/settings/tokens (read scope), "
        "accept the gated model's licence with that account, then export it "
        "before starting the worker. It is intentionally not read from .env. "
        "If the weights are already cached locally, set HF_HUB_OFFLINE=1 "
        "instead to load them without a token."
    )

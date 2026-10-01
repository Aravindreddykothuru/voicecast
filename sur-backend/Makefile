.PHONY: up down logs migrate revision test lint fmt shell setup-tts tts-warmup

up:
	docker compose up --build

down:
	docker compose down -v

logs:
	docker compose logs -f api worker-cpu worker-gpu worker-tts

migrate:
	docker compose run --rm api alembic upgrade head

revision:
	docker compose run --rm api alembic revision --autogenerate -m "$(m)"

test:
	pip install -r requirements-dev.txt --break-system-packages -q
	pytest -v --cov=app

lint:
	ruff check app tests

fmt:
	ruff check --fix app tests

shell:
	docker compose run --rm api python

# Self-healing TTS runtime (docs/tts-runtime.md). Idempotent: safe to re-run
# after any interruption -- venvs are created only if missing, pip skips what
# is installed, downloads resume from their partial files and verified files
# are not fetched again. Gated models need HF_TOKEN in the environment (never
# in .env); without it `setup` names them and exits 1, because a chain
# that names an unprovisioned model is a half-set-up deployment. To run
# anyway, say so in config: TTS_CONFIG=tts_chains.syspin-only.yaml.
PY ?= python
VENV_PY = $(if $(wildcard .venv/Scripts/python.exe),.venv/Scripts/python.exe,.venv/bin/python)
PARLER_PY = $(if $(filter Windows_NT,$(OS)),.venv-parler/Scripts/python.exe,.venv-parler/bin/python)

# TTS_CONFIG points setup at a different chain file. On a box with no
# HF_TOKEN the gated models cannot be provisioned and `setup` exits non-zero
# naming them; run the SYSPIN-only profile instead, which is a stated choice
# rather than a silent skip:
#     make setup-tts TTS_CONFIG=tts_chains.syspin-only.yaml
TTS_CONFIG ?=
TTS_CONFIG_ARG = $(if $(TTS_CONFIG),--config $(TTS_CONFIG),)

setup-tts:
	test -x $(PARLER_PY) || $(PY) -m venv .venv-parler
	$(PARLER_PY) -m pip install -q -r requirements-parler.txt
	$(VENV_PY) -m app.tts_runtime $(TTS_CONFIG_ARG) setup
	$(VENV_PY) -m app.tts_runtime $(TTS_CONFIG_ARG) warmup

tts-warmup:
	$(VENV_PY) -m app.tts_runtime warmup

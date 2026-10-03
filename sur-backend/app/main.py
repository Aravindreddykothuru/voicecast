from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import (
    routes_auth,
    routes_capabilities,
    routes_health,
    routes_projects,
    routes_segments,
    routes_storage,
    ws,
)
from app.config import get_settings
from app.logging_conf import configure_logging

settings = get_settings()

# This is the process that signs and verifies session tokens and answers
# CORS preflights, so this is where the production secret rules are
# enforced -- at import, before the app can serve a single request. It is
# deliberately not a Settings validator: Alembic and Celery load Settings
# too, and making `alembic upgrade head` fail on a JWT error during a
# production migration is both confusing and an invitation to work around
# the check.
settings.require_secure_production_runtime()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sur API",
    description=(
        "Backend for Sur, an emotion-aware AI dubbing pipeline. "
        "REST under /api for project/segment CRUD and control; a WebSocket "
        "at /ws/projects/{id} for live pipeline progress. OpenAPI schema "
        "here is the contract the frontend generates its TypeScript types "
        "from (see frontend build prompt, PRD section 08)."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_health.router)
app.include_router(routes_auth.router)
app.include_router(routes_capabilities.router)
app.include_router(routes_projects.router)
app.include_router(routes_segments.router)
app.include_router(routes_storage.router)
app.include_router(ws.router)


from app.db import Base, engine
from app import models as _models  # noqa: F401


@app.on_event("startup")
async def on_startup() -> None:
    logger.info("sur-backend starting up", extra={"environment": settings.environment})
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables verified/created")
    except Exception as e:
        logger.warning("Could not auto-create database tables: %s", e)

    # Cheap, model-free invariants. The API serves /api/capabilities, so an
    # incomplete language table here would be published to every client.
    # Model loading is checked in the workers, which actually own the models.
    from app.startup_checks import verify_capabilities
    from app.pipeline.maintenance import sweep_stuck_projects

    verify_capabilities()
    sweep_stuck_projects()


# Serve built frontend so frontend and backend work together on http://localhost:8000
from pathlib import Path
from fastapi.responses import FileResponse
from fastapi import HTTPException

_FRONTEND_DIST_CANDIDATES = [
    Path(__file__).resolve().parent.parent.parent / "sur-frontend" / "dist",
    Path("/app/frontend/dist"),
    Path(__file__).resolve().parent / "static",
]

_FRONTEND_DIST = next((p for p in _FRONTEND_DIST_CANDIDATES if p.is_dir()), None)

if _FRONTEND_DIST:
    logger.info("Mounted frontend distribution directory from %s", _FRONTEND_DIST)

    @app.api_route("/assets/{asset_path:path}", methods=["GET", "HEAD"], include_in_schema=False)
    async def serve_frontend_asset(asset_path: str):
        target = _FRONTEND_DIST / "assets" / asset_path
        if target.is_file():
            return FileResponse(str(target))
        raise HTTPException(status_code=404, detail="Asset not found")

    @app.api_route("/{full_path:path}", methods=["GET", "HEAD"], include_in_schema=False)
    async def serve_spa_frontend(full_path: str):
        # Do not intercept API, WS, docs, health endpoints
        if full_path.startswith(("api/", "api", "ws/", "ws", "docs", "redoc", "openapi.json", "healthz", "health/")):
            raise HTTPException(status_code=404, detail="Not Found")
        target = _FRONTEND_DIST / full_path
        if full_path and target.is_file():
            return FileResponse(str(target))
        index_file = _FRONTEND_DIST / "index.html"
        if index_file.is_file():
            return FileResponse(str(index_file))
        raise HTTPException(status_code=404, detail="Frontend build index.html not found")



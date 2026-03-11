"""FastAPI application factory."""
from __future__ import annotations

import logging

from fastapi import FastAPI

from predictagent.config import Settings
from predictagent.api.routers.forecast import router, init_router

logger = logging.getLogger(__name__)


def create_app(settings: Settings) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Validated application settings.

    Returns:
        Configured FastAPI instance.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    app = FastAPI(
        title="predictagent",
        description="Cell load forecast API",
        version="0.1.0",
    )
    init_router(settings)
    app.include_router(router)
    logger.info("App created; registry at %s", settings.registry.model_dir)
    return app

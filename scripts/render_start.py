"""Startup script for Render deployment."""

import logging
import os
import sys

sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_database():
    """Create all tables if they don't exist (idempotent)."""
    from edgefinder.db.engine import Base, get_engine
    from edgefinder.db import models  # noqa: F401 — registers ORM models

    engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("Database tables verified/created")
    engine.dispose()


def main():
    init_database()

    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "dashboard.app:app",
        host="0.0.0.0",
        port=port,
        workers=1,
    )


if __name__ == "__main__":
    main()

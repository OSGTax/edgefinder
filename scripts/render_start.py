"""Startup script for Render deployment."""

import os
import uvicorn


def main():
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "dashboard.app:app",
        host="0.0.0.0",
        port=port,
        workers=1,
    )


if __name__ == "__main__":
    main()

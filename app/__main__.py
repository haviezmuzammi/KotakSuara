"""Entry point: `python -m app` (this is what the NSSM service runs)."""
from __future__ import annotations

import uvicorn

from app.config import load_settings
from app.service import build_service, create_app, setup_logging


def main() -> None:
    settings = load_settings()
    setup_logging(settings)
    app = create_app(build_service(settings))
    uvicorn.run(app, host=settings.host, port=settings.port, log_config=None, access_log=True)


if __name__ == "__main__":
    main()

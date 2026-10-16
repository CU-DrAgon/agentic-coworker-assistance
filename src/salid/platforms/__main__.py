from __future__ import annotations

import asyncio
import logging
import sys

from ..config import Settings
from .config import PlatformSettings
from .runtime import run


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    for noisy in ("httpx", "httpcore", "slack_bolt", "slack_sdk", "aiohttp", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)  # their debug/info lines can echo urls
    s, ps = Settings.from_env(), PlatformSettings()
    if not ps.enabled_platforms():
        sys.exit(
            "No platform configured: set SALID_SLACK_BOT_TOKEN + SALID_SLACK_APP_TOKEN and/or "
            "SALID_LINE_CHANNEL_SECRET + SALID_LINE_CHANNEL_ACCESS_TOKEN and/or "
            "SALID_TELEGRAM_BOT_TOKEN (see docs/platforms/)"
        )
    asyncio.run(run(s, ps))


if __name__ == "__main__":
    main()

"""Which chat platforms would start, and do their tokens work? Read-only: Slack auth.test and
LINE get-bot-info. Reads SALID_* from the environment / .env; prints no secrets.

    uv run python scripts/platforms_smoke.py
"""

import asyncio
import sys

from salid.platforms.smoke import main

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

from __future__ import annotations

import logging
import sys

from .bot import SecretaryBot
from .config import Settings
from .llm import make_client
from .store import make_store


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    s = Settings.from_env()
    if not s.discord_token:
        sys.exit("SALID_DISCORD_TOKEN is not set (see .env.example)")
    SecretaryBot(s, make_store(s), make_client(s)).run(s.discord_token, log_handler=None)


if __name__ == "__main__":
    main()

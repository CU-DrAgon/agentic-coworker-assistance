"""What `scripts/platforms_smoke.py` prints: which platforms would start, and whether their tokens
work, using one read-only API call each (Slack auth.test, LINE get bot info, Telegram getMe). Never prints a
secret, only whether it is set and what it is called."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from .config import PlatformSettings
from .core import safe_exc

Checker = Callable[[PlatformSettings], Awaitable[dict]]


def plan(ps: PlatformSettings) -> list[str]:
    """Which platforms would start, and what is missing for the others."""

    def need(**vars_: bool) -> str:
        missing = [f"SALID_{k.upper()}" for k, ok in vars_.items() if not ok]
        return "will start" if not missing else "off, missing " + ", ".join(missing)

    return [
        "slack: "
        + need(slack_bot_token=bool(ps.slack_bot_token), slack_app_token=bool(ps.slack_app_token)),
        "line:  "
        + need(
            line_channel_secret=bool(ps.line_channel_secret),
            line_channel_access_token=bool(ps.line_channel_access_token),
        )
        + (f" (webhook port {ps.line_port})" if ps.line_enabled else ""),
        "telegram: " + need(telegram_bot_token=bool(ps.telegram_bot_token)),
        f"slack executives: {len(ps.slack_exec_user_ids)} id(s) plus workspace admins/owners",
        f"line executives:  {len(ps.line_exec_user_ids)} id(s)",
        f"telegram executives: {len(ps.telegram_exec_user_ids)} id(s) (group admins do not count)",
    ]


async def check(
    ps: PlatformSettings,
    slack_check: Checker | None = None,
    line_check: Checker | None = None,
    telegram_check: Checker | None = None,
) -> list[str]:
    """Validate the tokens of the platforms that would start."""
    out: list[str] = []
    if ps.slack_enabled:
        if slack_check is None:
            from .slack import check as slack_check
        out.append(await _one("slack", slack_check, ps))
    if ps.line_enabled:
        if line_check is None:
            from .line import check as line_check
        out.append(await _one("line", line_check, ps))
    if ps.telegram_enabled:
        if telegram_check is None:
            from .telegram import check as telegram_check
        out.append(await _one("telegram", telegram_check, ps))
    return out


async def _one(name: str, checker: Checker, ps: PlatformSettings) -> str:
    try:
        info = await asyncio.wait_for(checker(ps), 15)
    except Exception as e:  # the exception text may contain request details: type only
        return f"{name}: token check FAILED ({safe_exc(e)})"
    shown = ", ".join(f"{k}={v}" for k, v in info.items() if v)
    return f"{name}: token ok ({shown})"


async def main() -> int:
    ps = PlatformSettings()
    print("\n".join(plan(ps)))
    results = await check(ps)
    if results:
        print("\n".join(results))
    return 1 if any("FAILED" in r for r in results) else 0

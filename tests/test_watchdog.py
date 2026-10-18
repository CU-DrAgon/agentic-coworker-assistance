import asyncio
import logging
import time

from meeting_scribe import watchdog


def test_a_blocked_event_loop_is_logged_with_its_stack(caplog):
    async def main():
        watchdog.start(asyncio.get_running_loop(), stall_s=0.3, every_s=0.05)
        await asyncio.sleep(0.1)
        time.sleep(0.8)  # a synchronous wait on the loop  # noqa: ASYNC251
        await asyncio.sleep(0.1)

    with caplog.at_level(logging.ERROR, logger="meeting_scribe.watchdog"):
        asyncio.run(main())
    msgs = [r.getMessage() for r in caplog.records]
    assert len(msgs) == 1 and "event loop stalled" in msgs[0] and "time.sleep" in msgs[0]


def test_a_healthy_loop_logs_nothing(caplog):
    async def main():
        watchdog.start(asyncio.get_running_loop(), stall_s=0.3, every_s=0.05)
        await asyncio.sleep(0.6)

    with caplog.at_level(logging.ERROR, logger="meeting_scribe.watchdog"):
        asyncio.run(main())
    assert not caplog.records

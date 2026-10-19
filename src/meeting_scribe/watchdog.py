"""Event-loop watchdog: on 18 Oct 2026 the one web instance stopped answering for two minutes
(CPU idle, memory fine) right after a Live voice call opened, so some synchronous wait was
running on the event loop. A daemon thread pings the loop; when a ping is not answered within
`stall_s` it logs the stack of the loop's thread once per stall, so the next one names the line."""

import asyncio
import logging
import sys
import threading
import time
import traceback

logger = logging.getLogger(__name__)


def loop_stack(thread_id: int) -> str:
    frame = sys._current_frames().get(thread_id)
    return "".join(traceback.format_stack(frame)) if frame else "(thread gone)"


def start(
    loop: asyncio.AbstractEventLoop, stall_s: float = 10.0, every_s: float = 2.0
) -> threading.Thread:
    loop_thread = threading.get_ident()  # called from inside the loop
    state = {"seen": time.monotonic()}

    def pong() -> None:
        state["seen"] = time.monotonic()

    def watch() -> None:
        reported = False
        while not loop.is_closed():
            try:
                loop.call_soon_threadsafe(pong)
            except RuntimeError:
                return  # loop closed
            time.sleep(every_s)
            stalled = time.monotonic() - state["seen"]
            if stalled > stall_s and not reported:
                reported = True
                logger.error(
                    "event loop stalled %.0f s; loop thread stack:\n%s",
                    stalled,
                    loop_stack(loop_thread),
                )
            elif stalled <= stall_s:
                reported = False

    t = threading.Thread(target=watch, name="loop-watchdog", daemon=True)
    t.start()
    return t

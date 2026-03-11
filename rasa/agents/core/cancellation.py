"""Cancellation token for interrupting long-running agent operations."""

import asyncio


class CancellationToken:
    """A lightweight, lock-free signaling primitive for cooperative cancellation.

    Wraps an ``asyncio.Event`` so that any external source (session timer,
    channel disconnect, explicit user action) can interrupt a long-running
    operation — such as A2A polling — without acquiring the conversation lock.

    Typical usage inside a polling loop::

        while True:
            ...
            if cancellation_token:
                cancelled = await cancellation_token.wait(timeout=delay)
                if cancelled:
                    break
            else:
                await asyncio.sleep(delay)
    """

    def __init__(self) -> None:
        self._event = asyncio.Event()

    def cancel(self) -> None:
        """Signal cancellation. Lock-free, safe to call from anywhere."""
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        """Non-blocking check."""
        return self._event.is_set()

    async def wait_until_cancelled(self) -> None:
        """Block indefinitely until cancellation is signalled.

        Unlike :meth:`wait`, this method has no timeout — it returns only
        when :meth:`cancel` has been called.  Use it to create an
        ``asyncio.Task`` that can be raced against another coroutine via
        ``asyncio.wait``::

            consume_task = asyncio.create_task(do_work())
            cancel_task = asyncio.create_task(token.wait_until_cancelled())
            done, pending = await asyncio.wait(
                {consume_task, cancel_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
        """
        await self._event.wait()

    async def wait(self, timeout: float) -> bool:
        """Sleep for up to *timeout* seconds, waking immediately on cancellation.

        Use as a cancellation-aware replacement for ``asyncio.sleep`` inside
        polling loops::

            cancelled = await token.wait(timeout=delay)
            if cancelled:
                break

        Returns:
            ``True`` if cancelled before the timeout, ``False`` if the timeout
            expired normally.
        """
        if self._event.is_set():
            return True
        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

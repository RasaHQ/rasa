"""Unit tests for the CancellationToken."""

import asyncio
import time

from rasa.agents.core.cancellation import CancellationToken


async def test_cancellation_token_starts_not_cancelled():
    token = CancellationToken()
    assert token.is_cancelled is False


async def test_cancellation_token_cancel_sets_flag():
    token = CancellationToken()
    token.cancel()
    assert token.is_cancelled is True


async def test_cancellation_token_cancel_is_idempotent():
    token = CancellationToken()
    token.cancel()
    token.cancel()
    assert token.is_cancelled is True


async def test_cancellation_token_wait_returns_true_when_cancelled():
    token = CancellationToken()
    token.cancel()
    result = await token.wait(timeout=5.0)
    assert result is True


async def test_cancellation_token_wait_returns_false_on_timeout():
    token = CancellationToken()
    result = await token.wait(timeout=0.05)
    assert result is False
    assert token.is_cancelled is False


async def test_cancellation_token_wait_wakes_immediately_on_cancel():
    token = CancellationToken()

    async def _cancel_after(delay: float) -> None:
        await asyncio.sleep(delay)
        token.cancel()

    cancel_delay = 0.1
    background_task = asyncio.create_task(_cancel_after(cancel_delay))

    start = time.monotonic()
    result = await token.wait(timeout=5.0)
    elapsed = time.monotonic() - start

    assert result is True
    assert elapsed < 1.0, f"wait() should have returned quickly, took {elapsed:.2f}s"
    assert background_task.done()


async def test_cancellation_token_wait_zero_timeout_not_cancelled():
    token = CancellationToken()
    result = await token.wait(timeout=0)
    assert result is False


async def test_cancellation_token_wait_zero_timeout_already_cancelled():
    token = CancellationToken()
    token.cancel()
    result = await token.wait(timeout=0)
    assert result is True


async def test_wait_until_cancelled_completes_on_cancel():
    token = CancellationToken()

    async def _cancel_after(delay: float) -> None:
        await asyncio.sleep(delay)
        token.cancel()

    background_task = asyncio.create_task(_cancel_after(0.05))

    start = time.monotonic()
    await token.wait_until_cancelled()
    elapsed = time.monotonic() - start

    assert token.is_cancelled is True
    assert elapsed < 1.0
    assert background_task.done()


async def test_wait_until_cancelled_returns_immediately_if_already_cancelled():
    token = CancellationToken()
    token.cancel()

    start = time.monotonic()
    await token.wait_until_cancelled()
    elapsed = time.monotonic() - start

    assert elapsed < 0.1

import time

import pytest

from rasa.builder import config
from rasa.builder.guardrails.constants import BLOCK_SCOPE_PROJECT, BLOCK_SCOPE_USER
from rasa.builder.guardrails.store import GuardrailsInMemoryStore

USER_ID = "test-user-id"
ANOTHER_USER_ID = "another_test-user-id"


@pytest.fixture()
def store() -> GuardrailsInMemoryStore:
    """Return a fresh store per test."""
    return GuardrailsInMemoryStore()


@pytest.fixture(autouse=True)
def set_default_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default thresholds and duration for tests; override per-test as needed."""
    monkeypatch.setattr(config, "GUARDRAILS_USER_MAX_STRIKES", 3, raising=False)
    monkeypatch.setattr(config, "GUARDRAILS_PROJECT_MAX_STRIKES", 3, raising=False)
    monkeypatch.setattr(config, "GUARDRAILS_BLOCK_DURATION_SECONDS", 10, raising=False)


@pytest.mark.asyncio
async def test_initial_state_not_blocked(store: GuardrailsInMemoryStore):
    assert await store.check_block_scope(USER_ID) is None


@pytest.mark.asyncio
async def test_violation_below_threshold_no_blocks(
    store: GuardrailsInMemoryStore, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(config, "GUARDRAILS_USER_MAX_STRIKES", 5, raising=False)
    monkeypatch.setattr(config, "GUARDRAILS_PROJECT_MAX_STRIKES", 5, raising=False)

    result = await store.record_violation(USER_ID)
    assert not result.user_blocked_now
    assert not result.project_blocked_now
    assert await store.check_block_scope(USER_ID) is None


@pytest.mark.asyncio
async def test_user_block_on_threshold(
    store: GuardrailsInMemoryStore, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(config, "GUARDRAILS_USER_MAX_STRIKES", 2, raising=False)
    monkeypatch.setattr(config, "GUARDRAILS_PROJECT_MAX_STRIKES", 99, raising=False)

    result1 = await store.record_violation(USER_ID)
    result2 = await store.record_violation(USER_ID)

    assert not result1.user_blocked_now
    assert result2.user_blocked_now
    assert not result2.project_blocked_now

    assert await store.check_block_scope(USER_ID) == BLOCK_SCOPE_USER
    # Another user is unaffected
    assert await store.check_block_scope(ANOTHER_USER_ID) is None


@pytest.mark.asyncio
async def test_project_block_on_threshold(
    store: GuardrailsInMemoryStore, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(config, "GUARDRAILS_USER_MAX_STRIKES", 99, raising=False)
    monkeypatch.setattr(config, "GUARDRAILS_PROJECT_MAX_STRIKES", 2, raising=False)

    # Different users both count toward project-wide violations
    await store.record_violation(USER_ID)
    result = await store.record_violation(ANOTHER_USER_ID)

    assert result.project_blocked_now
    assert not result.user_blocked_now

    # Check project block using a user that is not user-blocked
    assert await store.check_block_scope("user-id") == BLOCK_SCOPE_PROJECT


@pytest.mark.asyncio
async def test_both_blocks_priority_user(
    store: GuardrailsInMemoryStore, monkeypatch: pytest.MonkeyPatch
):
    # Both thresholds at 1 so first violation triggers both
    monkeypatch.setattr(config, "GUARDRAILS_USER_MAX_STRIKES", 1, raising=False)
    monkeypatch.setattr(config, "GUARDRAILS_PROJECT_MAX_STRIKES", 1, raising=False)

    result = await store.record_violation(USER_ID)
    assert result.user_blocked_now
    assert result.project_blocked_now

    # User-level block should take priority for the same user
    assert await store.check_block_scope(USER_ID) == BLOCK_SCOPE_USER
    # Another user sees the project block
    assert await store.check_block_scope(ANOTHER_USER_ID) == BLOCK_SCOPE_PROJECT


@pytest.mark.asyncio
async def test_user_block_expires(
    store: GuardrailsInMemoryStore, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(config, "GUARDRAILS_USER_MAX_STRIKES", 1, raising=False)
    monkeypatch.setattr(config, "GUARDRAILS_BLOCK_DURATION_SECONDS", 2, raising=False)

    base_time = time.time()
    monkeypatch.setattr("time.time", lambda: base_time, raising=False)

    res = await store.record_violation(USER_ID)
    assert res.user_blocked_now
    assert await store.check_block_scope(USER_ID) == BLOCK_SCOPE_USER

    # Advance beyond duration; a check should clear the expired user block
    monkeypatch.setattr("time.time", lambda: base_time + 3, raising=False)
    assert await store.check_block_scope(USER_ID) is None


@pytest.mark.asyncio
async def test_project_block_expires(
    store: GuardrailsInMemoryStore, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(config, "GUARDRAILS_PROJECT_MAX_STRIKES", 1, raising=False)
    monkeypatch.setattr(config, "GUARDRAILS_BLOCK_DURATION_SECONDS", 2, raising=False)

    base_time = time.time()
    monkeypatch.setattr("time.time", lambda: base_time, raising=False)

    await store.record_violation(USER_ID)
    # Check project block via a user who is not user-blocked
    assert await store.check_block_scope(ANOTHER_USER_ID) == BLOCK_SCOPE_PROJECT

    # Advance beyond duration; a check should clear the expired project block
    monkeypatch.setattr("time.time", lambda: base_time + 3, raising=False)
    assert await store.check_block_scope(ANOTHER_USER_ID) is None


@pytest.mark.asyncio
async def test_indefinite_blocks_do_not_expire(
    store: GuardrailsInMemoryStore, monkeypatch: pytest.MonkeyPatch
):
    # Duration 0 is indefinite block
    monkeypatch.setattr(config, "GUARDRAILS_BLOCK_DURATION_SECONDS", 0, raising=False)
    monkeypatch.setattr(config, "GUARDRAILS_USER_MAX_STRIKES", 1, raising=False)
    monkeypatch.setattr(config, "GUARDRAILS_PROJECT_MAX_STRIKES", 99, raising=False)

    base = time.time()
    monkeypatch.setattr("time.time", lambda: base, raising=False)

    # Trigger a user block
    await store.record_violation(USER_ID)
    assert await store.check_block_scope(USER_ID) == BLOCK_SCOPE_USER

    # Far in the future, still blocked (since duration is indefinite)
    monkeypatch.setattr("time.time", lambda: base + 10_000, raising=False)
    assert await store.check_block_scope(USER_ID) == BLOCK_SCOPE_USER


@pytest.mark.asyncio
async def test_user_blocked_now_only_once(
    store: GuardrailsInMemoryStore, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(config, "GUARDRAILS_USER_MAX_STRIKES", 1, raising=False)
    monkeypatch.setattr(config, "GUARDRAILS_PROJECT_MAX_STRIKES", 99, raising=False)

    result1 = await store.record_violation(USER_ID)
    result2 = await store.record_violation(USER_ID)

    assert result1.user_blocked_now
    assert not result2.user_blocked_now  # already blocked
    assert await store.check_block_scope(USER_ID) == BLOCK_SCOPE_USER


@pytest.mark.asyncio
async def test_project_blocked_now_only_once(
    store: GuardrailsInMemoryStore, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(config, "GUARDRAILS_USER_MAX_STRIKES", 99, raising=False)
    monkeypatch.setattr(config, "GUARDRAILS_PROJECT_MAX_STRIKES", 1, raising=False)

    result1 = await store.record_violation(USER_ID)
    result2 = await store.record_violation(USER_ID)

    assert result1.project_blocked_now
    assert not result2.project_blocked_now  # already blocked
    # Check project block via a user who is not user-blocked
    assert await store.check_block_scope(ANOTHER_USER_ID) == BLOCK_SCOPE_PROJECT


@pytest.mark.asyncio
async def test_per_user_isolation_for_user_block(
    store: GuardrailsInMemoryStore, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(config, "GUARDRAILS_USER_MAX_STRIKES", 1, raising=False)
    monkeypatch.setattr(config, "GUARDRAILS_PROJECT_MAX_STRIKES", 99, raising=False)

    # Block first user
    await store.record_violation(USER_ID)
    assert await store.check_block_scope(USER_ID) == BLOCK_SCOPE_USER

    # Second user should not be user-blocked
    assert await store.check_block_scope(ANOTHER_USER_ID) is None


@pytest.mark.asyncio
async def test_project_block_affects_all_users(
    store: GuardrailsInMemoryStore, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(config, "GUARDRAILS_PROJECT_MAX_STRIKES", 1, raising=False)
    monkeypatch.setattr(config, "GUARDRAILS_USER_MAX_STRIKES", 99, raising=False)

    # Trigger project block with first user
    await store.record_violation(USER_ID)
    # Second user sees project-level block
    assert await store.check_block_scope(ANOTHER_USER_ID) == BLOCK_SCOPE_PROJECT


@pytest.mark.asyncio
async def test_check_block_scope_clears_expired_and_returns_none(
    store: GuardrailsInMemoryStore, monkeypatch: pytest.MonkeyPatch
):
    # Thresholds at 1 so first violation blocks both
    monkeypatch.setattr(config, "GUARDRAILS_USER_MAX_STRIKES", 1, raising=False)
    monkeypatch.setattr(config, "GUARDRAILS_PROJECT_MAX_STRIKES", 1, raising=False)
    monkeypatch.setattr(config, "GUARDRAILS_BLOCK_DURATION_SECONDS", 1, raising=False)

    base = time.time()
    monkeypatch.setattr("time.time", lambda: base, raising=False)

    await store.record_violation(USER_ID)
    assert await store.check_block_scope(USER_ID) == BLOCK_SCOPE_USER
    # Another user sees the project block
    assert await store.check_block_scope(ANOTHER_USER_ID) == BLOCK_SCOPE_PROJECT

    # After expiration, a call to check should clear both blocks
    monkeypatch.setattr("time.time", lambda: base + 2, raising=False)
    assert await store.check_block_scope(USER_ID) is None
    assert await store.check_block_scope(ANOTHER_USER_ID) is None


@pytest.mark.asyncio
async def test_unblock_user_clears_block_but_preserves_context(
    store: GuardrailsInMemoryStore, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(config, "GUARDRAILS_USER_MAX_STRIKES", 2, raising=False)

    await store.record_violation(USER_ID)
    await store.record_violation(USER_ID)
    assert await store.check_block_scope(USER_ID) == BLOCK_SCOPE_USER

    # Unblock user (strikes preserved)
    await store.unblock_user(USER_ID)
    assert await store.check_block_scope(USER_ID) is None

    # Next violation re-triggers the block because strikes are still >= threshold
    res = await store.record_violation(USER_ID)
    assert res.user_blocked_now
    assert await store.check_block_scope(USER_ID) == BLOCK_SCOPE_USER


@pytest.mark.asyncio
async def test_reset_user_clears_block_and_violations(
    store: GuardrailsInMemoryStore, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(config, "GUARDRAILS_USER_MAX_STRIKES", 2, raising=False)
    monkeypatch.setattr(config, "GUARDRAILS_PROJECT_MAX_STRIKES", 5, raising=False)

    await store.record_violation(USER_ID)
    await store.record_violation(USER_ID)
    assert await store.check_block_scope(USER_ID) == BLOCK_SCOPE_USER

    # Reset user - clear strikes and block
    await store.reset_user(USER_ID)
    assert await store.check_block_scope(USER_ID) is None

    # Violation shouldn't trigger block
    result = await store.record_violation(USER_ID)
    assert not result.user_blocked_now
    assert await store.check_block_scope(USER_ID) is None

    # Second violation should trigger block
    result = await store.record_violation(USER_ID)
    assert result.user_blocked_now
    assert await store.check_block_scope(USER_ID) == BLOCK_SCOPE_USER


@pytest.mark.asyncio
async def test_unblock_project_clears_block_but_preserves_context(
    store: GuardrailsInMemoryStore, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(config, "GUARDRAILS_PROJECT_MAX_STRIKES", 2, raising=False)

    await store.record_violation("alice")
    await store.record_violation("bob")
    assert await store.check_block_scope("charlie") == BLOCK_SCOPE_PROJECT

    # Unblock project - strikes preserved
    await store.unblock_project()
    assert await store.check_block_scope("charlie") is None

    # Next violation re-triggers the project block
    res = await store.record_violation("dave")
    assert res.project_blocked_now
    assert await store.check_block_scope("erin") == BLOCK_SCOPE_PROJECT


@pytest.mark.asyncio
async def test_reset_project_clears_block_and_violations(
    store: GuardrailsInMemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "GUARDRAILS_PROJECT_MAX_STRIKES", 2, raising=False)

    await store.record_violation("alice")
    await store.record_violation("bob")
    assert await store.check_block_scope("carol") == BLOCK_SCOPE_PROJECT

    # Reset project (clear strikes and block)
    await store.reset_project()
    assert await store.check_block_scope("carol") is None

    result = await store.record_violation("dave")
    assert not result.project_blocked_now
    assert await store.check_block_scope("erin") is None

    result = await store.record_violation("erin")
    assert result.project_blocked_now
    assert await store.check_block_scope("grace") == BLOCK_SCOPE_PROJECT


@pytest.mark.asyncio
async def test_reset_all_clears_everything(
    store: GuardrailsInMemoryStore, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(config, "GUARDRAILS_USER_MAX_STRIKES", 1, raising=False)
    monkeypatch.setattr(config, "GUARDRAILS_PROJECT_MAX_STRIKES", 1, raising=False)

    await store.record_violation("alice")
    assert await store.check_block_scope("alice") == BLOCK_SCOPE_USER
    assert await store.check_block_scope("bob") == BLOCK_SCOPE_PROJECT

    # Reset all
    await store.reset_all()
    assert await store.check_block_scope("alice") is None
    assert await store.check_block_scope("bob") is None

    # Raise thresholds again and verify clean slate behavior
    monkeypatch.setattr(config, "GUARDRAILS_USER_MAX_STRIKES", 2, raising=False)
    monkeypatch.setattr(config, "GUARDRAILS_PROJECT_MAX_STRIKES", 2, raising=False)

    result = await store.record_violation("alice")
    assert not result.user_blocked_now
    assert not result.project_blocked_now
    assert await store.check_block_scope("alice") is None
    assert await store.check_block_scope("bob") is None

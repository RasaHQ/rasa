import asyncio
import time
from typing import Optional

import structlog

from rasa.builder import config
from rasa.builder.guardrails.constants import (
    BLOCK_SCOPE_PROJECT,
    BLOCK_SCOPE_USER,
    BlockScope,
)
from rasa.builder.guardrails.models import BlockResult, ProjectState, ScopeState

structlogger = structlog.get_logger()


class GuardrailsInMemoryStore:
    """In-memory strike and block tracking for a single project instance."""

    def __init__(self) -> None:
        self._state: ProjectState = ProjectState()
        self._lock = asyncio.Lock()

    @staticmethod
    def _normalize_user_id(user_id: Optional[str]) -> str:
        """Normalize and validate user_id.

        Args:
            user_id: User identifier.

        Returns:
            Validated, non-empty user id.
        """
        uid = (user_id or "").strip()
        if not uid:
            raise ValueError("user_id is required for guardrails tracking.")
        return uid

    def _get_or_create_user(self, user_id: str) -> ScopeState:
        """Return (and create if needed) the per-user scope state.

        Args:
            user_id: User identifier.

        Returns:
            Mutable ScopeState for the user.
        """
        uid = self._normalize_user_id(user_id)
        return self._state.users.setdefault(uid, ScopeState())

    def _clear_expired_project_block(self, now: float) -> None:
        """Clear project-level block if expired.

        Args:
            now: Current time as a UNIX timestamp.
        """
        project = self._state.project
        if project.blocked_until and now >= project.blocked_until:
            project.blocked_until = None

    def _clear_expired_user_block(self, user_id: str, now: float) -> None:
        """Clear user-level block if expired.

        Args:
            user_id: User identifier.
            now: Current time as a UNIX timestamp.
        """
        try:
            uid = self._normalize_user_id(user_id)
        except ValueError:
            return

        user_state = self._state.users.get(uid)
        if user_state and user_state.blocked_until and now >= user_state.blocked_until:
            user_state.blocked_until = None

    @staticmethod
    def _apply_user_block_if_needed(user_id: str, user_state: ScopeState) -> bool:
        """Apply a user-level block if the threshold is reached.

        Args:
            user_id: User identifier.
            user_state: Current user's scope state.

        Returns:
            True if a block was applied during this call; otherwise False.
        """
        now = time.time()
        if user_state.is_blocked(now):
            return False

        if len(user_state.violations) < config.GUARDRAILS_USER_MAX_STRIKES:
            return False

        duration_seconds = config.GUARDRAILS_BLOCK_DURATION_SECONDS
        block_until = float("inf") if duration_seconds <= 0 else now + duration_seconds
        user_state.blocked_until = block_until
        structlogger.info(
            "guardrails.store.user_blocked",
            project_id=config.HELLO_RASA_PROJECT_ID,
            user_id=user_id,
            strikes=len(user_state.violations),
            duration_sec=duration_seconds,
        )
        return True

    def _apply_project_block_if_needed(self) -> bool:
        """Apply a project-level block if the threshold is reached.

        Returns:
            True if a block was applied during this call; otherwise False.
        """
        now = time.time()
        project_state = self._state.project
        if project_state.is_blocked(now):
            return False

        if len(project_state.violations) < config.GUARDRAILS_PROJECT_MAX_STRIKES:
            return False

        duration_seconds = config.GUARDRAILS_BLOCK_DURATION_SECONDS
        block_until = float("inf") if duration_seconds <= 0 else now + duration_seconds
        project_state.blocked_until = block_until
        structlogger.info(
            "guardrails.store.project_blocked",
            project_id=config.HELLO_RASA_PROJECT_ID,
            strikes=len(project_state.violations),
            duration_sec=duration_seconds,
        )
        return True

    async def check_block_scope(self, user_id: str) -> Optional[BlockScope]:
        """Return current block scope if blocked.

        Args:
            user_id: User identifier.

        Returns:
            'user' if user blocked, 'project' if project blocked, otherwise None.
        """
        uid = self._normalize_user_id(user_id)

        async with self._lock:
            now = time.time()

            # Clear expired blocks
            self._clear_expired_project_block(now)
            self._clear_expired_user_block(uid, now)

            user_state = self._state.users.get(uid)

            if user_state and user_state.is_blocked(now):
                return BLOCK_SCOPE_USER
            if self._state.project.is_blocked(now):
                return BLOCK_SCOPE_PROJECT
            return None

    async def record_violation(self, user_id: str) -> BlockResult:
        """Record a violation and apply thresholds/blocks.

        Args:
            user_id: User identifier.

        Returns:
            BlockResult indicating whether a new user or project block was applied.
        """
        uid = self._normalize_user_id(user_id)
        result = BlockResult()

        async with self._lock:
            now = time.time()
            user_state = self._get_or_create_user(uid)

            # Accumulate violations (no sliding window)
            self._state.project.violations.append(now)
            user_state.violations.append(now)

            # Apply user and project blocks
            result.user_blocked_now = self._apply_user_block_if_needed(
                user_id=uid,
                user_state=user_state,
            )
            result.project_blocked_now = self._apply_project_block_if_needed()

        return result

    async def unblock_user(self, user_id: str) -> None:
        """Unblock a user without altering strike history.

        Args:
            user_id: User identifier.
        """
        uid = self._normalize_user_id(user_id)

        async with self._lock:
            if state := self._state.users.get(uid):
                state.blocked_until = None

    async def reset_user(self, user_id: str) -> None:
        """Unblock a user and clear their strikes.

        Args:
            user_id: User identifier.
        """
        uid = self._normalize_user_id(user_id)

        async with self._lock:
            if state := self._state.users.get(uid):
                state.blocked_until = None
                state.violations.clear()

    async def unblock_project(self) -> None:
        """Unblock the project without altering strike history."""
        async with self._lock:
            self._state.project.blocked_until = None

    async def reset_project(self) -> None:
        """Unblock the project and clear project-wide strikes."""
        async with self._lock:
            self._state.project.blocked_until = None
            self._state.project.violations.clear()

    async def reset_all(self) -> None:
        """Reset all guardrail state (project and users)."""
        async with self._lock:
            # Clear project scope
            self._state.project.blocked_until = None
            self._state.project.violations.clear()

            # Clear all users
            for state in self._state.users.values():
                state.blocked_until = None
                state.violations.clear()


# Singleton instance for application-wide use
guardrails_store = GuardrailsInMemoryStore()

import asyncio
import json
import logging
import typing
from typing import Text, Optional

import time

if typing.TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class LockCounter(asyncio.Lock):
    """Decorated asyncio lock that counts how many coroutines are waiting.
    The counter can be used to discard the lock when there is no coroutine
    waiting for it. For this to work, there should not be any execution yield
    between retrieving the lock and acquiring it, otherwise there might be
    race conditions."""

    def __init__(self) -> None:
        super().__init__()
        self.wait_counter = 0

    async def acquire(self) -> bool:
        """Acquire the lock, makes sure only one coroutine can retrieve it."""

        self.wait_counter += 1
        try:
            return await super(LockCounter, self).acquire()  # type: ignore
        finally:
            self.wait_counter -= 1

    def is_someone_waiting(self) -> bool:
        """Check if a coroutine is waiting for this lock to be freed."""

        return self.wait_counter != 0


class RedisTicketLock:
    """Redis ticket lock."""

    def __init__(
        self,
        conversation_id: Text,
        record_exp: float,
        now_serving: Optional[int] = None,
        last_issued: Optional[int] = None,
        red=None,
    ):
        self.conversation_id = conversation_id
        self.record_exp = record_exp
        self.now_serving = self._now_serving(now_serving)
        self.last_issued = self._last_issued(last_issued)
        self.red = red

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._release()

    @staticmethod
    def _now_serving(now_serving: Optional[int] = None) -> int:
        if now_serving is not None:
            return now_serving

        return 0

    @staticmethod
    def _last_issued(last_issued: Optional[int] = None) -> int:
        if last_issued is not None:
            return last_issued

        return 0

    def is_locked(self, ticket: int) -> bool:
        """Return whether `ticket` is locked.

        Returns False if lock has expired.
        Otherwise returns True if `now_serving` is not equal to `ticket`.
        """

        if self.has_lock_expired():
            return False

        return self.now_serving != ticket

    def has_lock_expired(self):
        """Returns True if Redis record for `conversation_id` no longer exists."""
        return self.red.get(self.conversation_id) is None

    def is_someone_waiting(self) -> bool:
        """Return whether someone is waiting on the lock to become available.

        Returns True if `now_serving` is greater than `last_issued`.
        """

        return self.now_serving > self.last_issued

    def acquire(self):
        """Issues a RedisTicketLock.

        Updates the existing record.
        """

        return self

    def issue_new_ticket(self) -> None:
        """Issues a new ticket and return its value.

        Updates the lock's `last_issued` count by 1 and persists the lock.
        """

        self.last_issued += 1
        self._persist()

    def _release(self) -> None:
        """Increment its `now_serving` count by one and persist the lock."""
        self.now_serving += 1
        self._persist()

    def _persist(self) -> None:
        self.red.set(self.conversation_id, self.dumps(), ex=self.record_exp)

    def dumps(self) -> Text:
        """Returns json dump of `RedisTicketLock`."""

        return json.dumps(
            dict(
                conversation_id=self.conversation_id,
                now_serving=self.now_serving,
                last_issued=self.last_issued,
                expires=self.record_exp,
            )
        )

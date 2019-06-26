import asyncio
import json
import logging
import time
import typing
from typing import Text, Optional

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
        expires: float,
        now_serving: Optional[int] = None,
        last_served: Optional[int] = None,
        red=None,
    ):
        self.conversation_id = conversation_id
        self.expires = expires
        self.now_serving = self._now_serving(now_serving)
        self.last_served = last_served
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

    def _increment_now_serving(self):
        self.now_serving += 1

    def _increment_last_served(self):
        self.last_served += 1

    def is_locked(self):
        """Return whether lock is locked.

        Always returns False if lock has expired. Otherwise returns False if
        `now_serving` is equal to `last_served`.
        """

        if time.time() > self.expires:
            return False

        is_locked = self.now_serving != self.last_served + 1

        return is_locked

    def is_someone_waiting(self) -> bool:
        """Return whether someone is waiting on the lock to become available.

        Returns True if `now_serving` is greater than `last_served`.
        """

        return self.now_serving == self.last_served

    def acquire(self, update_expires: float):
        """Issues a RedisTicketLock.

        Updates a lock's expiration time, increments its `last_served` count
        and persists the lock.
        """

        if update_expires:
            self._update_expiration(update_expires)
        self._increment_now_serving()
        self._persist()

        return self

    def _release(self):
        self._increment_last_served()
        self._persist()

    def _update_expiration(self, expires):
        self.expires = expires

    def _persist(self):
        self.red.set(self.conversation_id, self.dumps())

    def dumps(self):
        return json.dumps(
            dict(
                conversation_id=self.conversation_id,
                now_serving=self.now_serving,
                last_served=self.last_served,
                expires=self.expires,
            )
        )

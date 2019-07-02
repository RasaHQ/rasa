import asyncio
import json
import logging
import time
import typing
from typing import Text, Optional, Union

from redis import Redis

if typing.TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class TicketLock(object):
    def __init__(
        self,
        conversation_id: Text,
        now_serving: Optional[int] = None,
        last_issued: Optional[int] = None,
        lifetime: Optional[Union[int, float]] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        self.conversation_id = conversation_id
        self.now_serving = self._arg_or_number(now_serving, default=0)
        self.last_issued = self._arg_or_number(last_issued, default=0)
        self.lifetime = self._arg_or_number(lifetime, default=60)
        self.timestamp = self._arg_or_number(timestamp, default=time.time())

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._release()

    @staticmethod
    def _arg_or_number(
        arg: Optional[int] = None, default: Union[int, float] = 0
    ) -> int:
        return arg if arg is not None else default

    def is_locked(self, ticket: int) -> bool:
        """Return whether `ticket` is locked.

        Returns False if lock has expired.
        Otherwise returns True if `now_serving` is not equal to `ticket`.
        """

        if self.has_lock_expired():
            return False

        return self.now_serving != ticket

    async def acquire(self) -> "TicketLock":
        """Acquire the lock, makes sure only one coroutine can retrieve it."""
        return self

    def is_someone_waiting(self) -> bool:
        """Return whether someone is waiting on the lock to become available.

        Returns True if `now_serving` is greater than `last_issued`.
        """

        return self.now_serving > self.last_issued

    def has_lock_expired(self) -> bool:
        return time.time() > self.timestamp + self.lifetime

    def dumps(self) -> Text:
        """Returns json dump of `RedisTicketLock`."""

        return json.dumps(
            dict(
                conversation_id=self.conversation_id,
                now_serving=self.now_serving,
                last_issued=self.last_issued,
                lifetime=self.lifetime,
                timestamp=self.timestamp,
            )
        )

    @classmethod
    def create_lock(cls, *args, **kwargs) -> "TicketLock":
        raise NotImplementedError

    def issue_new_ticket(self) -> None:
        """Issues a new ticket.

        Updates the lock's `last_issued` count by 1 and persists the lock.
        """

        self.last_issued += 1

    def _release(self) -> None:
        """Increment its `now_serving` count by one and persist the lock."""

        self.now_serving += 1


class InMemoryTicketLock(TicketLock):
    """Decorated asyncio lock that counts how many coroutines are waiting.
    The counter can be used to discard the lock when there is no coroutine
    waiting for it. For this to work, there should not be any execution yield
    between retrieving the lock and acquiring it, otherwise there might be
    race conditions."""

    @classmethod
    def create_lock(
        cls,
        conversation_id: Text,
        now_serving: Optional[int] = None,
        last_issued: Optional[int] = None,
        lifetime: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> "InMemoryTicketLock":
        return cls(conversation_id, now_serving, last_issued, lifetime, timestamp)


class RedisTicketLock(TicketLock):
    """Redis ticket lock."""

    def __init__(
        self,
        conversation_id: Text,
        now_serving: Optional[int] = None,
        last_issued: Optional[int] = None,
        lifetime: Optional[Union[int, float]] = None,
        timestamp: Optional[float] = None,
        red: Optional[Redis] = None,
    ):
        self.red = red
        super().__init__(conversation_id, now_serving, last_issued, lifetime, timestamp)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._release()

    @classmethod
    def create_lock(
        cls,
        conversation_id: Text,
        now_serving: Optional[int] = None,
        last_issued: Optional[int] = None,
        lifetime: Optional[int] = None,
        timestamp: Optional[float] = None,
        red: Optional = None,
    ) -> "RedisTicketLock":
        return cls(conversation_id, now_serving, last_issued, lifetime, timestamp, red)

    def has_lock_expired(self) -> bool:
        """Returns True if Redis record for `conversation_id` no longer exists."""

        return self.red.get(self.conversation_id) is None

    def acquire(self):
        """Issues a RedisTicketLock and updates the existing record."""

        self.persist()

        return self

    def issue_new_ticket(self) -> None:
        """Issues a new ticket and return its value.

        Updates the lock's `last_issued` count by 1 and persists the lock.
        """

        self.last_issued += 1
        self.persist()

    def _release(self) -> None:
        """Increment its `now_serving` count by one and persist the lock."""

        self.now_serving += 1
        self.persist()

    def persist(self) -> None:
        self.red.set(self.conversation_id, self.dumps(), ex=self.lifetime)

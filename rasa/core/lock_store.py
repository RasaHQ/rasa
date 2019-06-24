import asyncio
import logging
import typing
from typing import Text, Optional

from aioredlock import Aioredlock

if typing.TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class LockStore(object):
    async def acquire(self, conversation_id: Text):
        """Acquire lock for `conversation_id`."""

        raise NotImplementedError

    def cleanup(self, conversation_id: Text):
        """Acquire lock for `conversation_id`."""

        pass


class CounterLockStore(LockStore):
    """Decorated asyncio lock that counts how many coroutines are waiting.

    The counter can be used to discard the lock when there is no coroutine
    waiting for it. For this to work, there should not be any execution yield
    between retrieving the lock and acquiring it, otherwise there might be
    race conditions.

    This makes sure that there can always only be one coroutine
    handling a conversation at any point in time.
    Note: this doesn't support multi-processing, it just works
    for coroutines. If there are multiple processes handling
    messages, an external system needs to make sure messages
    for the same conversation are always processed by the same
    process.
    """

    def __init__(self) -> None:
        self.conversations_in_processing = {}

    # TODO: make this a namedtuple
    def lock_for_conversation_id(self, conversation_id: Text) -> Optional[asyncio.Lock]:
        return self.conversations_in_processing.get(conversation_id, {}).get("lock")

    def increment_wait_counter(self, conversation_id: Text) -> None:
        lock_dict = self.conversations_in_processing.get(conversation_id)
        if lock_dict:
            lock_dict["wait_counter"] += 1

    def decrement_wait_counter(self, conversation_id: Text) -> None:
        lock_dict = self.conversations_in_processing.get(conversation_id)
        if lock_dict:
            lock_dict["wait_counter"] -= 1

    def _create_lock(self, conversation_id: Text) -> asyncio.Lock:
        lock = asyncio.Lock()
        self.conversations_in_processing[conversation_id] = dict(
            lock=lock, wait_counter=0
        )
        logger.debug("Created a new lock for conversation '{}'".format(conversation_id))

        return lock

    async def acquire(self, conversation_id: Text) -> bool:
        """Acquire the lock, makes sure only one coroutine can retrieve it."""

        lock = self.lock_for_conversation_id(conversation_id)

        if not lock:
            lock = self._create_lock(conversation_id)

        self.increment_wait_counter(conversation_id)

        try:
            return await lock.acquire()
        finally:
            self.decrement_wait_counter(conversation_id)

    def is_someone_waiting(self, conversation_id: Text) -> bool:
        """Check if a coroutine is waiting for this lock to be freed."""

        return self.lock_for_conversation_id(conversation_id).get("wait_counter") != 0

    def cleanup(self, conversation_id: Text) -> None:
        if not self.is_someone_waiting(conversation_id):
            # dispose of the lock if no one needs it to avoid
            # accumulating locks
            del self.conversations_in_processing[conversation_id]
            logger.debug(
                "Deleted lock for conversation '{}' (unused)".format(conversation_id)
            )


class RedisLockStore(LockStore, Aioredlock):
    def __init__(
        self,
        host: Optional[Text] = None,
        port: Optional[int] = None,
        db: Text = "rasa",
        password: Text = None,
        lock_timeout: float = 1,
        retry_count: int = 3,
    ) -> None:
        redis_instances = [
            {"host": host, "port": port, "db": db, "password": password}
        ]
        self.lock_manager = Aioredlock(
            redis_connections=[redis_instances],
            lock_timeout=lock_timeout,
            retry_count=retry_count,
        )

    async def acquire(self, conversation_id: Text) -> bool:
        """Acquire the lock, makes sure only one coroutine can retrieve it."""

        return await self.lock_manager.lock(conversation_id)

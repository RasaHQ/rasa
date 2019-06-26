import asyncio
import logging
import typing
from typing import Text, Optional

if typing.TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class LockStore(object):
    def lock(self, conversation_id: Text):
        raise NotImplementedError

    def cleanup(self, conversation_id: Text):
        """Dispose of the lock if no one needs it to avoid accumulating locks."""

        pass


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


class CounterLockStore(LockStore):
    """Store for LockCounter locks."""

    def __init__(self) -> None:
        self.conversation_locks = {}

    def lock(self, conversation_id: Text) -> LockCounter:
        lock = self._get_lock(conversation_id)
        if not lock:
            lock = self._create_lock(conversation_id)

        return lock

    def _get_lock(self, conversation_id: Text) -> Optional[LockCounter]:
        return self.conversation_locks.get(conversation_id)

    def _create_lock(self, conversation_id: Text) -> LockCounter:
        lock = LockCounter()
        self.conversation_locks[conversation_id] = lock
        return lock

    def _is_someone_waiting(self, conversation_id: Text) -> bool:
        lock = self._get_lock(conversation_id)
        if lock:
            return lock.is_someone_waiting()

        return False

    def cleanup(self, conversation_id: Text) -> None:
        if not self._is_someone_waiting(conversation_id):
            del self.conversation_locks[conversation_id]
            logger.debug(
                "Deleted lock for conversation '{}' (unused)".format(conversation_id)
            )


class RedisLockStore(LockStore):
    def __init__(
        self,
        host: Optional[Text] = None,
        port: Optional[int] = None,
        db: Text = "rasa",
        password: Text = None,
        lock_timeout: float = 0.5,
        retry_count: int = 20,
    ) -> None:
        from aioredlock import Aioredlock

        redis_instances = [{"host": host, "port": port, "db": db, "password": password}]
        self.lock_manager = Aioredlock(
            redis_connections=[redis_instances],
            lock_timeout=lock_timeout,
            retry_count=retry_count,
        )

    def lock(self, conversation_id: Text) -> "Aioredlock":
        return self.lock_manager.lock(conversation_id)

import asyncio
import json
import logging
import typing
from typing import Text, Optional, Tuple

from rasa.core.lock import InMemoryTicketLock, RedisTicketLock

if typing.TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

ACCEPTED_LOCKS = {"lock_counter": InMemoryTicketLock, "redis": RedisTicketLock}


class LockStore(object):
    def __init__(self, lock_class=None, lifetime: int = 60):
        self.lock_class = lock_class
        self.lifetime = lifetime
        # raise ValueError(
        #     "Cannot create lock of type '{}'. One of the following lock "
        #     "types need to be specified: {}."
        #     "".format("lock_type", ", ".join(ACCEPTED_LOCKS.keys()))
        # )

    def create_lock(self, conversation_id: Text) -> RedisTicketLock:
        raise NotImplementedError

    def get_or_create_lock(self, conversation_id: Text):
        raise NotImplementedError

    def get_lock(self, conversation_id: Text):
        raise NotImplementedError

    async def lock(self, conversation_id: Text):
        _lock = self.get_or_create_lock(conversation_id)
        ticket = _lock.last_issued

        while True:

            # fetch lock again because things might have changed
            _lock = self.get_lock(conversation_id)

            if not _lock.is_locked(ticket):
                return _lock.acquire()

            await asyncio.sleep(1)


class RedisLockStore(LockStore):
    def __init__(
        self,
        host: Text = "localhost",
        port: int = 6379,
        db: int = 1,
        password: Optional[Text] = None,
        lifetime: int = 60,
    ):
        import redis

        self.red = redis.StrictRedis(host=host, port=port, db=db, password=password)
        super().__init__(RedisTicketLock, lifetime)

    def get_lock(self, conversation_id: Text) -> Optional[RedisTicketLock]:
        existing_lock = self.red.get(conversation_id)
        if existing_lock:
            return self.lock_object_from_dump(existing_lock)

        return None

    def create_lock(self, conversation_id: Text):
        _lock = self.lock_class.create_lock(conversation_id)
        _lock.persist()
        return _lock

    def get_or_create_lock(self, conversation_id: Text) -> RedisTicketLock:
        existing_lock = self.get_lock(conversation_id)

        # issue new ticket if lock exists
        if existing_lock:
            existing_lock.issue_new_ticket()
            return existing_lock

        return self.create_lock(conversation_id)

    def lock_object_from_dump(self, existing_lock: Text) -> RedisTicketLock:
        lock_json = json.loads(existing_lock)
        return self.lock_class.create_lock(
            lock_json.get("conversation_id"),
            lock_json.get("now_serving"),
            lock_json.get("last_issued"),
            lock_json.get("timestamp"),
            self.lifetime,
            self.red,
        )

    def cleanup(self, conversation_id: Text) -> None:
        """Delete lock if no one is waiting on it."""

        _lock = self.get_lock(conversation_id)
        if _lock and not _lock.is_someone_waiting():
            self.red.delete(conversation_id)


class CounterLockStore(LockStore):
    """Store for InMemoryTicketLock locks."""

    def __init__(self):
        self.conversation_locks = {}
        super().__init__()

    async def lock(self, conversation_id: Text) -> InMemoryTicketLock:
        lock = self.get_lock(conversation_id)
        if not lock:
            lock = self.create_lock(conversation_id)

        return lock

    def get_lock(self, conversation_id: Text) -> Optional[InMemoryTicketLock]:
        return self.conversation_locks.get(conversation_id)

    def create_lock(self, conversation_id: Text) -> InMemoryTicketLock:
        lock = InMemoryTicketLock()
        self.conversation_locks[conversation_id] = lock
        return lock

    def get_or_create_lock(self, conversation_id: Text) -> InMemoryTicketLock:
        existing_lock = self.get_lock(conversation_id)

        # issue new ticket if lock exists
        if existing_lock:
            existing_lock.issue_new_ticket()
            return existing_lock

        return self.create_lock(conversation_id)

    def is_someone_waiting(self, conversation_id: Text) -> bool:
        lock = self.get_lock(conversation_id)
        if lock:
            return lock.is_someone_waiting()

        return False

    def cleanup(self, conversation_id: Text) -> None:
        if not self.is_someone_waiting(conversation_id):
            del self.conversation_locks[conversation_id]
            logger.debug(
                "Deleted lock for conversation '{}' (unused)".format(conversation_id)
            )

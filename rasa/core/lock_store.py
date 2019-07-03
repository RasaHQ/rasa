import asyncio
import logging
import typing
from typing import Text, Optional, Dict

from rasa.core.lock import TicketLock, RedisTicketLock

if typing.TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

ACCEPTED_LOCK_STORES = ["in_memory", "redis"]


class LockError(Exception):
    pass


class LockStore(object):
    def __init__(self, lifetime: int = 60):
        self.lifetime = lifetime

    @staticmethod
    def find_lock_store(store=None):
        if store is None or store.type is None or store.type == "in_memory":
            lock_store = LockStore()
        elif store.type == "redis":
            lock_store = RedisLockStore(host=store.url, **store.kwargs)
        else:
            raise ValueError(
                "Cannot create lock of type '{}'. One of the following lock "
                "types need to be specified: {}."
                "".format("lock_type", ", ".join(ACCEPTED_LOCK_STORES))
            )

        logger.debug(
            "Connected to lock store {}.".format(lock_store.__class__.__name__)
        )

        return lock_store

    def create_lock(self, conversation_id: Text) -> TicketLock:
        raise NotImplementedError

    def get_lock(self, conversation_id: Text) -> Optional[TicketLock]:
        raise NotImplementedError

    def delete_lock(self, conversation_id: Text):
        raise NotImplementedError

    async def lock(self, conversation_id: Text) -> "LockStore":
        lock = self.get_or_create_lock(conversation_id)
        ticket = lock.issue_ticket()

        attempts = 60

        while attempts > 0:

            # acquire lock if it isn't locked
            if not lock.is_locked(ticket):
                return self

            # sleep and update locks
            await asyncio.sleep(1)
            self.update_lock(conversation_id)

            # fetch lock again because things might have changed
            lock = self.get_lock(conversation_id)

            attempts -= 1

        raise LockError(
            "Could not acquire lock for conversation_id '{}'."
            "".format(conversation_id)
        )

    def update_lock(self, conversation_id: Text) -> None:
        """Fetch lock from memory, remove expired tickets and save lock."""

        lock = self.get_lock(conversation_id)
        if lock:
            lock.remove_expired_tickets()

    def get_or_create_lock(self, conversation_id: Text) -> TicketLock:
        existing_lock = self.get_lock(conversation_id)

        if existing_lock:
            return existing_lock

        lock = self.create_lock(conversation_id)

        return lock

    def is_someone_waiting(self, conversation_id: Text) -> bool:
        lock = self.get_lock(conversation_id)
        if lock:
            return lock.is_someone_waiting()

        return False

    def cleanup(self, conversation_id: Text) -> None:
        """Remove lock for `conversation_id` if no one is waiting."""
        # update memory

        if not self.is_someone_waiting(conversation_id):
            self.delete_lock(conversation_id)
            logger.debug(
                "Deleted lock for conversation '{}' (unused)".format(conversation_id)
            )


class RedisLockStore(LockStore):
    """Redis store for ticket locks."""

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
        super().__init__(lifetime)

    def create_lock(self, conversation_id: Text) -> TicketLock:
        lock = RedisTicketLock(conversation_id, self.lifetime, self.red)
        lock.persist()
        return lock

    def get_lock(self, conversation_id: Text) -> Optional[TicketLock]:
        return self.red.get(conversation_id)

    def delete_lock(self, conversation_id: Text):
        self.red.delete(conversation_id)


class InMemoryLockStore(LockStore):
    """In-memory store for ticket locks."""

    def __init__(self, lifetime: int = 60):
        self.conversation_locks = {}  # type: Dict[Text, TicketLock]
        super().__init__(lifetime)

    def create_lock(self, conversation_id: Text) -> TicketLock:
        return TicketLock(conversation_id, self.lifetime)

    def get_lock(self, conversation_id: Text) -> Optional[TicketLock]:
        return self.conversation_locks.get(conversation_id)

    def delete_lock(self, conversation_id: Text):
        self.conversation_locks.pop(conversation_id)

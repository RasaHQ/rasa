import asyncio
import json
import logging
import pickle
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
            lock_store = InMemoryLockStore()
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

    async def lock(self, conversation_id: Text) -> TicketLock:
        lock = self.get_or_create_lock(conversation_id)
        ticket = lock.issue_ticket(self.lifetime)

        attempts = 60

        while attempts > 0:
            # exit loop if lock does not exist anymore (expired)
            if not lock:
                break

            # acquire lock if it isn't locked
            if not lock.is_locked(ticket):
                return lock

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

        self.update_lock(conversation_id)

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

        self.host = host
        self.port = port
        self.db = db
        self.password = password

        self.red = redis.StrictRedis(
            host=self.host, port=self.port, db=self.db, password=self.password
        )
        super().__init__(lifetime)

    def create_lock(self, conversation_id: Text) -> RedisTicketLock:
        lock = RedisTicketLock(
            conversation_id, self.host, self.port, self.db, self.password
        )
        lock.persist()
        return lock

    def get_lock(self, conversation_id: Text) -> Optional[RedisTicketLock]:
        serialised_lock = self.red.get(conversation_id)
        if serialised_lock:
            lock = json.loads(serialised_lock)
            return RedisTicketLock(
                lock["conversation_id"],
                lock["host"],
                lock["port"],
                lock["db"],
                lock["password"],
            )

    def delete_lock(self, conversation_id: Text):
        self.red.delete(conversation_id)


class InMemoryLockStore(LockStore):
    """In-memory store for ticket locks."""

    def __init__(self, lifetime: int = 60):
        self.conversation_locks = {}  # type: Dict[Text, TicketLock]
        super().__init__(lifetime)

    def create_lock(self, conversation_id: Text) -> TicketLock:
        return TicketLock(conversation_id)

    def get_lock(self, conversation_id: Text) -> Optional[TicketLock]:
        return self.conversation_locks.get(conversation_id)

    def delete_lock(self, conversation_id: Text):
        if conversation_id in self.conversation_locks:
            del self.conversation_locks[conversation_id]

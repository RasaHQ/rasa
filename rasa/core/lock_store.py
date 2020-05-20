import asyncio
import json
import logging
import os

from async_generator import asynccontextmanager
from typing import Text, Union, Optional, AsyncGenerator

from rasa.core.constants import DEFAULT_LOCK_LIFETIME
from rasa.utils import common
from rasa.core.lock import TicketLock
from rasa.utils.endpoints import EndpointConfig

logger = logging.getLogger(__name__)

LOCK_LIFETIME = int(os.environ.get("TICKET_LOCK_LIFETIME", 0)) or DEFAULT_LOCK_LIFETIME


# noinspection PyUnresolvedReferences
class LockError(Exception):
    """Exception that is raised when a lock cannot be acquired.

     Attributes:
          message (str): explanation of which `conversation_id` raised the error
    """

    pass


class LockStore:
    @staticmethod
    def create(obj: Union["LockStore", EndpointConfig, None]) -> "LockStore":
        """Factory to create a lock store."""

        if isinstance(obj, LockStore):
            return obj
        else:
            return _create_from_endpoint_config(obj)

    @staticmethod
    def create_lock(conversation_id: Text) -> TicketLock:
        """Create a new `TicketLock` for `conversation_id`."""

        return TicketLock(conversation_id)

    def get_lock(self, conversation_id: Text) -> Optional[TicketLock]:
        """Fetch lock for `conversation_id` from storage."""

        raise NotImplementedError

    def delete_lock(self, conversation_id: Text) -> None:
        """Delete lock for `conversation_id` from storage."""

        raise NotImplementedError

    def save_lock(self, lock: TicketLock) -> None:
        """Commit `lock` to storage."""

        raise NotImplementedError

    def issue_ticket(
        self, conversation_id: Text, lock_lifetime: float = LOCK_LIFETIME
    ) -> int:
        """Issue new ticket with `lock_lifetime` for lock associated with
        `conversation_id`.

        Creates a new lock if none is found.
        """

        lock = self.get_or_create_lock(conversation_id)
        ticket = lock.issue_ticket(lock_lifetime)
        self.save_lock(lock)

        return ticket

    @asynccontextmanager
    async def lock(
        self,
        conversation_id: Text,
        lock_lifetime: float = LOCK_LIFETIME,
        wait_time_in_seconds: float = 1,
    ) -> AsyncGenerator[TicketLock, None]:
        """Acquire lock with lifetime `lock_lifetime`for `conversation_id`.

        Try acquiring lock with a wait time of `wait_time_in_seconds` seconds
        between attempts. Raise a `LockError` if lock has expired.
        """

        ticket = self.issue_ticket(conversation_id, lock_lifetime)

        try:
            yield await self._acquire_lock(
                conversation_id, ticket, wait_time_in_seconds
            )

        finally:
            self.cleanup(conversation_id, ticket)

    async def _acquire_lock(
        self, conversation_id: Text, ticket: int, wait_time_in_seconds: float
    ) -> TicketLock:

        while True:
            # fetch lock in every iteration because lock might no longer exist
            lock = self.get_lock(conversation_id)

            # exit loop if lock does not exist anymore (expired)
            if not lock:
                break

            # acquire lock if it isn't locked
            if not lock.is_locked(ticket):
                return lock

            logger.debug(
                f"Failed to acquire lock for conversation ID '{conversation_id}'. "
                f"Retrying..."
            )

            # sleep and update lock
            await asyncio.sleep(wait_time_in_seconds)
            self.update_lock(conversation_id)

        raise LockError(
            f"Could not acquire lock for conversation_id '{conversation_id}'."
        )

    def update_lock(self, conversation_id: Text) -> None:
        """Fetch lock for `conversation_id`, remove expired tickets and save lock."""

        lock = self.get_lock(conversation_id)
        if lock:
            lock.remove_expired_tickets()
            self.save_lock(lock)

    def get_or_create_lock(self, conversation_id: Text) -> TicketLock:
        """Fetch existing lock for `conversation_id` or create a new one if
        it doesn't exist."""

        existing_lock = self.get_lock(conversation_id)

        if existing_lock:
            return existing_lock

        return self.create_lock(conversation_id)

    def is_someone_waiting(self, conversation_id: Text) -> bool:
        """Return whether someone is waiting for lock associated with
        `conversation_id`."""

        lock = self.get_lock(conversation_id)
        if lock:
            return lock.is_someone_waiting()

        return False

    def finish_serving(self, conversation_id: Text, ticket_number: int) -> None:
        """Finish serving ticket with `ticket_number` for `conversation_id`.

        Removes ticket from lock and saves lock.
        """

        lock = self.get_lock(conversation_id)
        if lock:
            lock.remove_ticket_for(ticket_number)
            self.save_lock(lock)

    def cleanup(self, conversation_id: Text, ticket_number: int) -> None:
        """Remove lock for `conversation_id` if no one is waiting."""

        self.finish_serving(conversation_id, ticket_number)
        if not self.is_someone_waiting(conversation_id):
            self.delete_lock(conversation_id)

    @staticmethod
    def _log_deletion(conversation_id: Text, deletion_successful: bool) -> None:
        if deletion_successful:
            logger.debug(f"Deleted lock for conversation '{conversation_id}'.")
        else:
            logger.debug(f"Could not delete lock for conversation '{conversation_id}'.")


class RedisLockStore(LockStore):
    """Redis store for ticket locks."""

    def __init__(
        self,
        host: Text = "localhost",
        port: int = 6379,
        db: int = 1,
        password: Optional[Text] = None,
        use_ssl: bool = False,
    ):
        import redis

        self.red = redis.StrictRedis(
            host=host, port=int(port), db=int(db), password=password, ssl=use_ssl
        )
        super().__init__()

    def get_lock(self, conversation_id: Text) -> Optional[TicketLock]:
        serialised_lock = self.red.get(conversation_id)
        if serialised_lock:
            return TicketLock.from_dict(json.loads(serialised_lock))

    def delete_lock(self, conversation_id: Text) -> None:
        deletion_successful = self.red.delete(conversation_id)
        self._log_deletion(conversation_id, deletion_successful)

    def save_lock(self, lock: TicketLock) -> None:
        self.red.set(lock.conversation_id, lock.dumps())


class InMemoryLockStore(LockStore):
    """In-memory store for ticket locks."""

    def __init__(self) -> None:
        self.conversation_locks = {}
        super().__init__()

    def get_lock(self, conversation_id: Text) -> Optional[TicketLock]:
        return self.conversation_locks.get(conversation_id)

    def delete_lock(self, conversation_id: Text) -> None:
        deleted_lock = self.conversation_locks.pop(conversation_id, None)
        self._log_deletion(
            conversation_id, deletion_successful=deleted_lock is not None
        )

    def save_lock(self, lock: TicketLock) -> None:
        self.conversation_locks[lock.conversation_id] = lock


def _create_from_endpoint_config(
    endpoint_config: Optional[EndpointConfig] = None,
) -> "LockStore":
    """Given an endpoint configuration, create a proper `LockStore` object."""

    if (
        endpoint_config is None
        or endpoint_config.type is None
        or endpoint_config.type == "in_memory"
    ):
        # this is the default type if no lock store type is set

        lock_store = InMemoryLockStore()
    elif endpoint_config.type == "redis":
        lock_store = RedisLockStore(host=endpoint_config.url, **endpoint_config.kwargs)
    else:
        lock_store = _load_from_module_string(endpoint_config.type)

    logger.debug(f"Connected to lock store '{lock_store.__class__.__name__}'.")

    return lock_store


def _load_from_module_string(endpoint_config: EndpointConfig) -> "LockStore":
    """Given the name of a `LockStore` module tries to retrieve it."""

    try:
        lock_store_class = common.class_from_module_path(endpoint_config.type)
        return lock_store_class(endpoint_config=endpoint_config)
    except (AttributeError, ImportError) as e:
        raise Exception(
            f"Could not find a class based on the module path "
            f"'{endpoint_config.type}'. Failed to create a `LockStore` "
            f"instance. Error: {e}"
        )

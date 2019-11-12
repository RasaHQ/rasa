import asyncio
import json
import logging
import os
from typing import Text, Optional, Union, AsyncGenerator

from async_generator import asynccontextmanager

from rasa.core.constants import DEFAULT_LOCK_LIFETIME
from rasa.core.lock import TicketLock, NO_TICKET_ISSUED
from rasa.utils.endpoints import EndpointConfig

logger = logging.getLogger(__name__)

ACCEPTED_LOCK_STORES = ["in_memory", "redis"]

LOCK_LIFETIME = int(os.environ.get("TICKET_LOCK_LIFETIME", 0)) or DEFAULT_LOCK_LIFETIME


# noinspection PyUnresolvedReferences
class LockError(Exception):
    """Exception that is raised when a lock cannot be acquired.

     Attributes:
          message (str): explanation of which `conversation_id` raised the error
    """

    pass


# noinspection PyUnresolvedReferences
class TicketExistsError(Exception):
    """Exception that is raised when an already-existing ticket for a conversation
    has been issued.

     Attributes:
          message (str): explanation of which `conversation_id` raised the error
    """

    pass


class LockStore:
    @staticmethod
    def find_lock_store(store: EndpointConfig = None) -> "LockStore":
        if store is None or store.type is None or store.type == "in_memory":
            lock_store = InMemoryLockStore()
        elif store.type == "redis":
            lock_store = RedisLockStore(host=store.url, **store.kwargs)
        else:
            logger.debug(
                "Could not load built-in `LockStore`, which needs to be of "
                "type: {}. Trying to load `LockStore` from module path '{}' "
                "instead."
                "".format(store.type, ", ".join(ACCEPTED_LOCK_STORES), store.type)
            )
            lock_store = LockStore.load_lock_store_from_module_path(store.type)

        logger.debug(f"Connected to lock store '{lock_store.__class__.__name__}'.")

        return lock_store

    @staticmethod
    def load_lock_store_from_module_path(module_path: Text) -> "LockStore":
        """Given the name of a `LockStore` module tries to retrieve it."""

        from rasa.utils.common import class_from_module_path

        try:
            return class_from_module_path(module_path)
        except ImportError:
            raise ImportError(f"Cannot retrieve `LockStore` from path '{module_path}'.")

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
        self, conversation_id: Text, lock_lifetime: Union[float, int] = LOCK_LIFETIME
    ) -> int:
        """Issue new ticket with `lock_lifetime` for lock associated with
        `conversation_id`.

        Creates a new lock if none is found.
        """

        lock = self.get_or_create_lock(conversation_id)
        ticket = lock.issue_ticket(lock_lifetime)

        while True:
            try:
                self.ensure_ticket_available(lock)
                break
            except TicketExistsError:
                # issue a new ticket if current ticket number has been issued twice
                logger.exception(
                    "Ticket could not be issued. Issuing new ticket and retrying..."
                )
                ticket = lock.issue_ticket(lock_lifetime)

        self.save_lock(lock)

        return ticket

    @asynccontextmanager
    async def lock(
        self,
        conversation_id: Text,
        lock_lifetime: int = LOCK_LIFETIME,
        wait_time_in_seconds: Union[int, float] = 1,
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
        self,
        conversation_id: Text,
        ticket: int,
        wait_time_in_seconds: Union[int, float],
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
                "Failed to acquire lock for conversation ID '{}'. Retrying..."
                "".format(conversation_id)
            )

            # sleep and update lock
            await asyncio.sleep(wait_time_in_seconds)
            self.update_lock(conversation_id)

        raise LockError(
            "Could not acquire lock for conversation_id '{}'."
            "".format(conversation_id)
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

    def ensure_ticket_available(self, lock: TicketLock) -> None:
        """Check for duplicate tickets issued for `lock`.

        This function should be called before saving `lock`. Raises `TicketExistsError`
        if the last issued ticket for `lock` does not match the last ticket issued
        for a lock fetched from storage for `lock.conversation_id`. This indicates
        that some other process has issued a ticket for `lock` in the meantime.
        """

        existing_lock = self.get_lock(lock.conversation_id)
        if not existing_lock or existing_lock.last_issued == NO_TICKET_ISSUED:
            # lock does not yet exist for conversation or no ticket has been issued
            return

        # raise if the last issued ticket number of `existing_lock` is not the same as
        # that of the one being acquired
        if existing_lock.last_issued != lock.last_issued:
            raise TicketExistsError(
                "Ticket '{}' already exists for conversation ID '{}'."
                "".format(existing_lock.last_issued, lock.conversation_id)
            )


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

    def __init__(self):
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

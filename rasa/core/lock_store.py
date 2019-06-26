import asyncio
import json
import logging
import typing
from typing import Text, Optional, Tuple

from rasa.core.lock import LockCounter, RedisTicketLock

if typing.TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class LockStore(object):
    async def lock(self, conversation_id: Text):
        raise NotImplementedError

    def cleanup(self, conversation_id: Text):
        """Dispose of the lock if no one needs it to avoid accumulating locks."""

        pass


class CounterLockStore(LockStore):
    """Store for LockCounter locks."""

    def __init__(self) -> None:
        self.conversation_locks = {}

    async def lock(self, conversation_id: Text) -> LockCounter:
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
    """Lock store for the `RedisTicketLock`"""

    def __init__(self, host="localhost", port=6379, db=1, password=None, record_exp=10):
        import redis

        self.red = redis.StrictRedis(host=host, port=port, db=db, password=password)
        self.record_exp = record_exp

    async def lock(self, conversation_id: Text):
        _lock = self._get_or_create_lock(conversation_id)
        ticket = _lock.last_issued

        while True:

            # todo: update the fetched lock record as things might have changed in
            # the meantime - although acquire() should set now_serving to ticket
            if not _lock.is_locked(ticket):
                return _lock.acquire()

            await asyncio.sleep(1)

    def _create_lock_object(
        self,
        conversation_id: Text,
        now_serving: Optional[int] = None,
        last_issued: Optional[int] = None,
    ) -> RedisTicketLock:
        return RedisTicketLock(
            conversation_id, self.record_exp, now_serving, last_issued, self.red
        )

    def _create_new_lock(self, conversation_id: Text) -> RedisTicketLock:
        _lock = self._create_lock_object(conversation_id)
        _lock._persist()
        return _lock

    def _get_lock(self, conversation_id: Text) -> Optional[RedisTicketLock]:
        existing_lock = self.red.get(conversation_id)
        if existing_lock:
            return self._lock_object_from_dump(existing_lock)

        return None

    def _get_or_create_lock(self, conversation_id: Text) -> RedisTicketLock:
        existing_lock = self._get_lock(conversation_id)

        # issue new ticket if lock exists
        if existing_lock:
            existing_lock.issue_new_ticket()
            return existing_lock

        return self._create_new_lock(conversation_id)

    def _lock_object_from_dump(self, existing_lock: Text) -> RedisTicketLock:
        lock_json = json.loads(existing_lock)
        return self._create_lock_object(
            lock_json.get("conversation_id"),
            lock_json.get("now_serving"),
            lock_json.get("last_issued"),
        )

    def cleanup(self, conversation_id: Text) -> None:
        """Delete lock if no one is waiting on it."""

        _lock = self._get_lock(conversation_id)
        if _lock and not _lock.is_someone_waiting():
            self.red.delete(conversation_id)

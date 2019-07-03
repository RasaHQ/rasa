import json
import logging
import typing
from collections import deque
from typing import Text, Optional, Union

import time

if typing.TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class Ticket:
    def __init__(self, number: int, expires: float):
        self.number = number
        self.expires = expires

    def has_expired(self):
        return time.time() > self.expires


class TicketLock(object):
    def __init__(self, conversation_id: Text) -> None:
        self.conversation_id = conversation_id
        self.tickets = deque()  # type: deque[Ticket]

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Remove the ticket most recently served ticket."""

        self.tickets.popleft()
        self.persist()

    def is_locked(self, ticket_number: int) -> bool:
        """Return whether `ticket_number` is locked.

        Returns False if lock has expired.
        Otherwise returns True if `now_serving` is not equal to `ticket`.
        """

        return self.now_serving != ticket_number

    def persist(self) -> None:
        pass

    def issue_ticket(self, lifetime: Union[float, int]) -> int:
        """Issue a new ticket and return its number."""

        self.remove_expired_tickets()

        number = self.last_issued + 1
        ticket = Ticket(number, time.time() + lifetime)
        self.tickets.append(ticket)

        self.persist()

        return number

    def remove_expired_tickets(self) -> None:
        """Remove expired tickets."""

        # iterate over copy of self.tickets so we can remove items
        for ticket in list(self.tickets):
            if ticket.has_expired():
                self.tickets.remove(ticket)

        self.persist()

    @property
    def last_issued(self) -> int:
        """Return number of the ticket that was last added.

        Return -1 if no tickets exist.
        """

        return self._ticket_number_for_index(-1) or -1

    @property
    def now_serving(self) -> Optional[int]:
        """Return number of the ticket to be served next.

        Return 0 if no tickets exists.
        """

        return self._ticket_number_for_index(0) or 0

    def _ticket_number_for_index(self, idx: int) -> Optional[int]:
        """Return ticket number for `idx`.

        Return None if there are no tickets, or if `idx` is out of bounds of
        `self.tickets`.
        """

        self.remove_expired_tickets()

        if not self.tickets or len(self.tickets) < abs(idx):
            return None

        return self.tickets[idx].number

    def _ticket_for_ticket_number(self, ticket_number: int) -> Optional[Ticket]:
        """Return expiration time for `ticket_number`."""

        self.remove_expired_tickets()

        return next((t for t in self.tickets if t.number == ticket_number), None)

    def is_someone_waiting(self) -> bool:
        """Return whether someone is waiting for the lock to become available.

        Returns True if the ticket queue has length greater than 0.
        """

        return len(self.tickets) > 0

    def has_lock_expired(self, ticket_number: int) -> Optional[bool]:
        """Return whether ticket for `ticket_number` has expired.

        Return True if ticket was not found.
        """

        ticket = self._ticket_for_ticket_number(ticket_number)
        if ticket:
            return ticket.has_expired()

        return True


class RedisTicketLock(TicketLock):
    """`TicketLock` with `Redis` connection for persistence."""

    def __init__(
        self,
        conversation_id: Text,
        host: Text = "localhost",
        port: int = 6379,
        db: int = 1,
        password: Optional[Text] = None,
    ) -> None:
        import redis

        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.red = redis.StrictRedis(
            host=self.host, port=self.port, db=self.db, password=self.password
        )
        super().__init__(conversation_id)

    def dumps(self):
        return json.dumps(
            dict(
                conversation_id=self.conversation_id,
                tickets=self.tickets,
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
            )
        )

    def persist(self) -> None:
        self.red.set(self.conversation_id, self.dumps())

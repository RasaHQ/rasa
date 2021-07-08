import json
import logging
from collections import deque

import time
from typing import Text, Optional, Union, Deque, Dict, Any

logger = logging.getLogger(__name__)

NO_TICKET_ISSUED = -1  # index of latest issued ticket if no tickets exist


class Ticket:
    def __init__(self, number: int, expires: float) -> None:
        self.number = number
        self.expires = expires

    def has_expired(self) -> bool:
        return time.time() > self.expires

    def as_dict(self) -> Dict[Text, Any]:
        return dict(number=self.number, expires=self.expires)

    def dumps(self) -> Text:
        """Return json dump of `Ticket` as dictionary."""

        return json.dumps(self.as_dict())

    @classmethod
    def from_dict(cls, data: Dict[Text, Union[int, float]]) -> "Ticket":
        """Creates `Ticket` from dictionary."""

        return cls(number=data["number"], expires=data["expires"])

    def __repr__(self) -> Text:
        return f"Ticket(number: {self.number}, expires: {self.expires})"


class TicketLock:
    """Locking mechanism that issues tickets managing access to conversation IDs.

    Tickets are issued in the order in which they are requested. A detailed
    explanation of the ticket lock algorithm can be found at
    http://pages.cs.wisc.edu/~remzi/OSTEP/threads-locks.pdf#page=13
    """

    def __init__(
        self, conversation_id: Text, tickets: Optional[Deque[Ticket]] = None
    ) -> None:
        self.conversation_id = conversation_id
        self.tickets = tickets or deque()

    @classmethod
    def from_dict(cls, data: Dict[Text, Any]) -> "TicketLock":
        """Create `TicketLock` from dictionary."""

        tickets = [Ticket.from_dict(json.loads(d)) for d in data.get("tickets")]
        return cls(data.get("conversation_id"), deque(tickets))

    def dumps(self) -> Text:
        """Return json dump of `TicketLock`."""

        tickets = [ticket.dumps() for ticket in self.tickets]
        return json.dumps(dict(conversation_id=self.conversation_id, tickets=tickets))

    def is_locked(self, ticket_number: int) -> bool:
        """Return whether `ticket_number` is locked.

        Returns:
             True if `now_serving` is not equal to `ticket`.
        """

        return self.now_serving != ticket_number

    def issue_ticket(self, lifetime: float) -> int:
        """Issue a new ticket and return its number."""

        self.remove_expired_tickets()
        number = self.last_issued + 1
        ticket = Ticket(number, time.time() + lifetime)
        self.tickets.append(ticket)

        return number

    def remove_expired_tickets(self) -> None:
        """Remove expired tickets."""

        # iterate over copy of self.tickets so we can remove items
        for ticket in list(self.tickets):
            if ticket.has_expired():
                self.tickets.remove(ticket)

    @property
    def last_issued(self) -> int:
        """Return number of the ticket that was last added.

        Returns:
             Number of `Ticket` that was last added. `NO_TICKET_ISSUED` if no
             tickets exist.
        """

        ticket_number = self._ticket_number_for(-1)

        return ticket_number if ticket_number is not None else NO_TICKET_ISSUED

    @property
    def now_serving(self) -> Optional[int]:
        """Get number of the ticket to be served next.

        Returns:
             Number of `Ticket` that is served next. 0 if no `Ticket` exists.
        """

        return self._ticket_number_for(0) or 0

    def _ticket_number_for(self, ticket_index: int) -> Optional[int]:
        """Get ticket number for `ticket_index`.

        Returns:
             Ticket number for `Ticket` with index `ticket_index`. None if there are no
             tickets, or if `ticket_index` is out of bounds of `self.tickets`.
        """

        self.remove_expired_tickets()

        try:
            return self.tickets[ticket_index].number
        except IndexError:
            return None

    def _ticket_for_ticket_number(self, ticket_number: int) -> Optional[Ticket]:
        """Return ticket for `ticket_number`."""

        self.remove_expired_tickets()

        return next((t for t in self.tickets if t.number == ticket_number), None)

    def is_someone_waiting(self) -> bool:
        """Return whether someone is waiting for the lock to become available.

        Returns:
             True if the `self.tickets` queue has length greater than 0.
        """

        return len(self.tickets) > 0

    def remove_ticket_for(self, ticket_number: int) -> None:
        """Remove `Ticket` for `ticket_number."""

        ticket = self._ticket_for_ticket_number(ticket_number)
        if ticket:
            self.tickets.remove(ticket)

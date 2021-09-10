---
sidebar_label: rasa.core.lock
title: rasa.core.lock
---
## Ticket Objects

```python
class Ticket()
```

#### dumps

```python
def dumps() -> Text
```

Return json dump of `Ticket` as dictionary.

#### from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[Text, Union[int, float]]) -> "Ticket"
```

Creates `Ticket` from dictionary.

## TicketLock Objects

```python
class TicketLock()
```

Locking mechanism that issues tickets managing access to conversation IDs.

Tickets are issued in the order in which they are requested. A detailed
explanation of the ticket lock algorithm can be found at
http://pages.cs.wisc.edu/~remzi/OSTEP/threads-locks.pdf#page=13

#### from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[Text, Any]) -> "TicketLock"
```

Create `TicketLock` from dictionary.

#### dumps

```python
def dumps() -> Text
```

Return json dump of `TicketLock`.

#### is\_locked

```python
def is_locked(ticket_number: int) -> bool
```

Return whether `ticket_number` is locked.

**Returns**:

  True if `now_serving` is not equal to `ticket`.

#### issue\_ticket

```python
def issue_ticket(lifetime: float) -> int
```

Issue a new ticket and return its number.

#### remove\_expired\_tickets

```python
def remove_expired_tickets() -> None
```

Remove expired tickets.

#### last\_issued

```python
@property
def last_issued() -> int
```

Return number of the ticket that was last added.

**Returns**:

  Number of `Ticket` that was last added. `NO_TICKET_ISSUED` if no
  tickets exist.

#### now\_serving

```python
@property
def now_serving() -> Optional[int]
```

Get number of the ticket to be served next.

**Returns**:

  Number of `Ticket` that is served next. 0 if no `Ticket` exists.

#### is\_someone\_waiting

```python
def is_someone_waiting() -> bool
```

Return whether someone is waiting for the lock to become available.

**Returns**:

  True if the `self.tickets` queue has length greater than 0.

#### remove\_ticket\_for

```python
def remove_ticket_for(ticket_number: int) -> None
```

Remove `Ticket` for `ticket_number.


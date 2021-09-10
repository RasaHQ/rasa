---
sidebar_label: rasa.core.lock_store
title: rasa.core.lock_store
---
## LockError Objects

```python
class LockError(RasaException)
```

Exception that is raised when a lock cannot be acquired.

**Attributes**:

- `message` _str_ - explanation of which `conversation_id` raised the error

## LockStore Objects

```python
class LockStore()
```

#### create

```python
@staticmethod
def create(obj: Union["LockStore", EndpointConfig, None]) -> "LockStore"
```

Factory to create a lock store.

#### create\_lock

```python
@staticmethod
def create_lock(conversation_id: Text) -> TicketLock
```

Create a new `TicketLock` for `conversation_id`.

#### get\_lock

```python
def get_lock(conversation_id: Text) -> Optional[TicketLock]
```

Fetch lock for `conversation_id` from storage.

#### delete\_lock

```python
def delete_lock(conversation_id: Text) -> None
```

Delete lock for `conversation_id` from storage.

#### save\_lock

```python
def save_lock(lock: TicketLock) -> None
```

Commit `lock` to storage.

#### issue\_ticket

```python
def issue_ticket(conversation_id: Text, lock_lifetime: float = LOCK_LIFETIME) -> int
```

Issue new ticket with `lock_lifetime` for lock associated with
`conversation_id`.

Creates a new lock if none is found.

#### lock

```python
@asynccontextmanager
async def lock(conversation_id: Text, lock_lifetime: float = LOCK_LIFETIME, wait_time_in_seconds: float = 1) -> AsyncGenerator[TicketLock, None]
```

Acquire lock with lifetime `lock_lifetime`for `conversation_id`.

Try acquiring lock with a wait time of `wait_time_in_seconds` seconds
between attempts. Raise a `LockError` if lock has expired.

#### update\_lock

```python
def update_lock(conversation_id: Text) -> None
```

Fetch lock for `conversation_id`, remove expired tickets and save lock.

#### get\_or\_create\_lock

```python
def get_or_create_lock(conversation_id: Text) -> TicketLock
```

Fetch existing lock for `conversation_id` or create a new one if
it doesn&#x27;t exist.

#### is\_someone\_waiting

```python
def is_someone_waiting(conversation_id: Text) -> bool
```

Return whether someone is waiting for lock associated with
`conversation_id`.

#### finish\_serving

```python
def finish_serving(conversation_id: Text, ticket_number: int) -> None
```

Finish serving ticket with `ticket_number` for `conversation_id`.

Removes ticket from lock and saves lock.

#### cleanup

```python
def cleanup(conversation_id: Text, ticket_number: int) -> None
```

Remove lock for `conversation_id` if no one is waiting.

## RedisLockStore Objects

```python
class RedisLockStore(LockStore)
```

Redis store for ticket locks.

#### \_\_init\_\_

```python
def __init__(host: Text = "localhost", port: int = 6379, db: int = 1, password: Optional[Text] = None, use_ssl: bool = False, key_prefix: Optional[Text] = None, socket_timeout: float = DEFAULT_SOCKET_TIMEOUT_IN_SECONDS) -> None
```

Create a lock store which uses Redis for persistence.

**Arguments**:

- `host` - The host of the redis server.
- `port` - The port of the redis server.
- `db` - The name of the database within Redis which should be used by Rasa
  Open Source.
- `password` - The password which should be used for authentication with the
  Redis database.
- `use_ssl` - `True` if SSL should be used for the connection to Redis.
- `key_prefix` - prefix to prepend to all keys used by the lock store. Must be
  alphanumeric.
- `socket_timeout` - Timeout in seconds after which an exception will be raised
  in case Redis doesn&#x27;t respond within `socket_timeout` seconds.

#### get\_lock

```python
def get_lock(conversation_id: Text) -> Optional[TicketLock]
```

Retrieves lock (see parent docstring for more information).

#### delete\_lock

```python
def delete_lock(conversation_id: Text) -> None
```

Deletes lock for conversation ID.

## InMemoryLockStore Objects

```python
class InMemoryLockStore(LockStore)
```

In-memory store for ticket locks.


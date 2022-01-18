---
sidebar_label: rasa.core.tracker_store
title: rasa.core.tracker_store
---
## TrackerDeserialisationException Objects

```python
class TrackerDeserialisationException(RasaException)
```

Raised when an error is encountered while deserialising a tracker.

## TrackerStore Objects

```python
class TrackerStore()
```

Represents common behavior and interface for all `TrackerStore`s.

#### \_\_init\_\_

```python
 | __init__(domain: Optional[Domain], event_broker: Optional[EventBroker] = None, **kwargs: Dict[Text, Any], ,) -> None
```

Create a TrackerStore.

**Arguments**:

- `domain` - The `Domain` to initialize the `DialogueStateTracker`.
- `event_broker` - An event broker to publish any new events to another
  destination.
- `kwargs` - Additional kwargs.

#### create

```python
 | @staticmethod
 | create(obj: Union["TrackerStore", EndpointConfig, None], domain: Optional[Domain] = None, event_broker: Optional[EventBroker] = None) -> "TrackerStore"
```

Factory to create a tracker store.

#### get\_or\_create\_tracker

```python
 | get_or_create_tracker(sender_id: Text, max_event_history: Optional[int] = None, append_action_listen: bool = True) -> "DialogueStateTracker"
```

Returns tracker or creates one if the retrieval returns None.

**Arguments**:

- `sender_id` - Conversation ID associated with the requested tracker.
- `max_event_history` - Value to update the tracker store&#x27;s max event history to.
- `append_action_listen` - Whether or not to append an initial `action_listen`.

#### init\_tracker

```python
 | init_tracker(sender_id: Text) -> "DialogueStateTracker"
```

Returns a Dialogue State Tracker

#### create\_tracker

```python
 | create_tracker(sender_id: Text, append_action_listen: bool = True) -> DialogueStateTracker
```

Creates a new tracker for `sender_id`.

The tracker begins with a `SessionStarted` event and is initially listening.

**Arguments**:

- `sender_id` - Conversation ID associated with the tracker.
- `append_action_listen` - Whether or not to append an initial `action_listen`.
  

**Returns**:

  The newly created tracker for `sender_id`.

#### save

```python
 | save(tracker: DialogueStateTracker) -> None
```

Save method that will be overridden by specific tracker.

#### exists

```python
 | exists(conversation_id: Text) -> bool
```

Checks if tracker exists for the specified ID.

This method may be overridden by the specific tracker store for
faster implementations.

**Arguments**:

- `conversation_id` - Conversation ID to check if the tracker exists.
  

**Returns**:

  `True` if the tracker exists, `False` otherwise.

#### retrieve

```python
 | retrieve(sender_id: Text) -> Optional[DialogueStateTracker]
```

Retrieves tracker for the latest conversation session.

This method will be overridden by the specific tracker store.

**Arguments**:

- `sender_id` - Conversation ID to fetch the tracker for.
  

**Returns**:

  Tracker containing events from the latest conversation sessions.

#### retrieve\_full\_tracker

```python
 | retrieve_full_tracker(conversation_id: Text) -> Optional[DialogueStateTracker]
```

Retrieve method for fetching all tracker events across conversation sessions
that may be overridden by specific tracker.

The default implementation uses `self.retrieve()`.

**Arguments**:

- `conversation_id` - The conversation ID to retrieve the tracker for.
  

**Returns**:

  The fetch tracker containing all events across session starts.

#### stream\_events

```python
 | stream_events(tracker: DialogueStateTracker) -> None
```

Streams events to a message broker

#### number\_of\_existing\_events

```python
 | number_of_existing_events(sender_id: Text) -> int
```

Return number of stored events for a given sender id.

#### keys

```python
 | keys() -> Iterable[Text]
```

Returns the set of values for the tracker store&#x27;s primary key

#### serialise\_tracker

```python
 | @staticmethod
 | serialise_tracker(tracker: DialogueStateTracker) -> Text
```

Serializes the tracker, returns representation of the tracker.

#### deserialise\_tracker

```python
 | deserialise_tracker(sender_id: Text, serialised_tracker: Union[Text, bytes]) -> Optional[DialogueStateTracker]
```

Deserializes the tracker and returns it.

## InMemoryTrackerStore Objects

```python
class InMemoryTrackerStore(TrackerStore)
```

Stores conversation history in memory

#### save

```python
 | save(tracker: DialogueStateTracker) -> None
```

Updates and saves the current conversation state

#### keys

```python
 | keys() -> Iterable[Text]
```

Returns sender_ids of the Tracker Store in memory

## RedisTrackerStore Objects

```python
class RedisTrackerStore(TrackerStore)
```

Stores conversation history in Redis

#### save

```python
 | save(tracker: DialogueStateTracker, timeout: Optional[float] = None) -> None
```

Saves the current conversation state.

#### retrieve

```python
 | retrieve(sender_id: Text) -> Optional[DialogueStateTracker]
```

Retrieves tracker for the latest conversation session.

The Redis key is formed by appending a prefix to sender_id.

**Arguments**:

- `sender_id` - Conversation ID to fetch the tracker for.
  

**Returns**:

  Tracker containing events from the latest conversation sessions.

#### keys

```python
 | keys() -> Iterable[Text]
```

Returns keys of the Redis Tracker Store.

## DynamoTrackerStore Objects

```python
class DynamoTrackerStore(TrackerStore)
```

Stores conversation history in DynamoDB

#### \_\_init\_\_

```python
 | __init__(domain: Domain, table_name: Text = "states", region: Text = "us-east-1", event_broker: Optional[EndpointConfig] = None, **kwargs: Dict[Text, Any], ,) -> None
```

Initialize `DynamoTrackerStore`.

**Arguments**:

- `domain` - Domain associated with this tracker store.
- `table_name` - The name of the DynamoDB table, does not need to be present a
  priori.
- `region` - The name of the region associated with the client.
  A client is associated with a single region.
- `event_broker` - An event broker used to publish events.
- `kwargs` - Additional kwargs.

#### get\_or\_create\_table

```python
 | get_or_create_table(table_name: Text) -> "boto3.resources.factory.dynamodb.Table"
```

Returns table or creates one if the table name is not in the table list.

#### save

```python
 | save(tracker: DialogueStateTracker) -> None
```

Saves the current conversation state.

#### serialise\_tracker

```python
 | serialise_tracker(tracker: "DialogueStateTracker") -> Dict
```

Serializes the tracker, returns object with decimal types.

#### retrieve

```python
 | retrieve(sender_id: Text) -> Optional[DialogueStateTracker]
```

Retrieve dialogues for a sender_id in reverse-chronological order.

Based on the session_date sort key.

#### keys

```python
 | keys() -> Iterable[Text]
```

Returns sender_ids of the `DynamoTrackerStore`.

## MongoTrackerStore Objects

```python
class MongoTrackerStore(TrackerStore)
```

Stores conversation history in Mongo.

Property methods:
    conversations: returns the current conversation

#### conversations

```python
 | @property
 | conversations() -> Collection
```

Returns the current conversation.

#### save

```python
 | save(tracker: DialogueStateTracker) -> None
```

Saves the current conversation state.

#### retrieve

```python
 | retrieve(sender_id: Text) -> Optional[DialogueStateTracker]
```

Retrieves tracker for the latest conversation session.

#### retrieve\_full\_tracker

```python
 | retrieve_full_tracker(conversation_id: Text) -> Optional[DialogueStateTracker]
```

Fetching all tracker events across conversation sessions.

#### keys

```python
 | keys() -> Iterable[Text]
```

Returns sender_ids of the Mongo Tracker Store.

#### is\_postgresql\_url

```python
is_postgresql_url(url: Union[Text, "URL"]) -> bool
```

Determine whether `url` configures a PostgreSQL connection.

**Arguments**:

- `url` - SQL connection URL.
  

**Returns**:

  `True` if `url` is a PostgreSQL connection URL.

#### create\_engine\_kwargs

```python
create_engine_kwargs(url: Union[Text, "URL"]) -> Dict[Text, Any]
```

Get `sqlalchemy.create_engine()` kwargs.

**Arguments**:

- `url` - SQL connection URL.
  

**Returns**:

  kwargs to be passed into `sqlalchemy.create_engine()`.

#### ensure\_schema\_exists

```python
ensure_schema_exists(session: "Session") -> None
```

Ensure that the requested PostgreSQL schema exists in the database.

**Arguments**:

- `session` - Session used to inspect the database.
  

**Raises**:

  `ValueError` if the requested schema does not exist.

## SQLTrackerStore Objects

```python
class SQLTrackerStore(TrackerStore)
```

Store which can save and retrieve trackers from an SQL database.

## SQLEvent Objects

```python
class SQLEvent(Base)
```

Represents an event in the SQL Tracker Store.

#### get\_db\_url

```python
 | @staticmethod
 | get_db_url(dialect: Text = "sqlite", host: Optional[Text] = None, port: Optional[int] = None, db: Text = "rasa.db", username: Text = None, password: Text = None, login_db: Optional[Text] = None, query: Optional[Dict] = None) -> Union[Text, "URL"]
```

Build an SQLAlchemy `URL` object representing the parameters needed
to connect to an SQL database.

**Arguments**:

- `dialect` - SQL database type.
- `host` - Database network host.
- `port` - Database network port.
- `db` - Database name.
- `username` - User name to use when connecting to the database.
- `password` - Password for database user.
- `login_db` - Alternative database name to which initially connect, and create
  the database specified by `db` (PostgreSQL only).
- `query` - Dictionary of options to be passed to the dialect and/or the
  DBAPI upon connect.
  

**Returns**:

  URL ready to be used with an SQLAlchemy `Engine` object.

#### session\_scope

```python
 | @contextlib.contextmanager
 | session_scope() -> Generator["Session", None, None]
```

Provide a transactional scope around a series of operations.

#### keys

```python
 | keys() -> Iterable[Text]
```

Returns sender_ids of the SQLTrackerStore.

#### retrieve

```python
 | retrieve(sender_id: Text) -> Optional[DialogueStateTracker]
```

Retrieves tracker for the latest conversation session.

#### retrieve\_full\_tracker

```python
 | retrieve_full_tracker(conversation_id: Text) -> Optional[DialogueStateTracker]
```

Fetching all tracker events across conversation sessions.

#### save

```python
 | save(tracker: DialogueStateTracker) -> None
```

Update database with events from the current conversation.

## FailSafeTrackerStore Objects

```python
class FailSafeTrackerStore(TrackerStore)
```

Wraps a tracker store so that we can fallback to a different tracker store in
case of errors.

#### \_\_init\_\_

```python
 | __init__(tracker_store: TrackerStore, on_tracker_store_error: Optional[Callable[[Exception], None]] = None, fallback_tracker_store: Optional[TrackerStore] = None) -> None
```

Create a `FailSafeTrackerStore`.

**Arguments**:

- `tracker_store` - Primary tracker store.
- `on_tracker_store_error` - Callback which is called when there is an error
  in the primary tracker store.


---
sidebar_label: rasa.core.brokers.pika
title: rasa.core.brokers.pika
---
## PikaEventBroker Objects

```python
class PikaEventBroker(EventBroker)
```

Pika-based event broker for publishing messages to RabbitMQ.

#### \_\_init\_\_

```python
 | __init__(host: Text, username: Text, password: Text, port: Union[int, Text] = 5672, queues: Union[List[Text], Tuple[Text], Text, None] = None, should_keep_unpublished_messages: bool = True, raise_on_failure: bool = False, event_loop: Optional[AbstractEventLoop] = None, connection_attempts: int = 20, retry_delay_in_seconds: float = 5, exchange_name: Text = RABBITMQ_EXCHANGE, **kwargs: Any, ,)
```

Initialise RabbitMQ event broker.

**Arguments**:

- `host` - Pika host.
- `username` - Username for authentication with Pika host.
- `password` - Password for authentication with Pika host.
- `port` - port of the Pika host.
- `queues` - Pika queues to declare and publish to.
- `should_keep_unpublished_messages` - Whether or not the event broker should
  maintain a queue of unpublished messages to be published later in
  case of errors.
- `raise_on_failure` - Whether to raise an exception if publishing fails. If
  `False`, keep retrying.
- `event_loop` - The event loop which will be used to run `async` functions. If
  `None` `asyncio.get_event_loop()` is used to get a loop.
- `connection_attempts` - Number of attempts for connecting to RabbitMQ before
  an exception is thrown.
- `retry_delay_in_seconds` - Time in seconds between connection attempts.
- `exchange_name` - Exchange name to which the queues binds to.
  If nothing is mentioned then the default exchange name would be used.

#### from\_endpoint\_config

```python
 | @classmethod
 | async from_endpoint_config(cls, broker_config: Optional["EndpointConfig"], event_loop: Optional[AbstractEventLoop] = None) -> Optional["PikaEventBroker"]
```

Creates broker. See the parent class for more information.

#### connect

```python
 | async connect() -> None
```

Connects to RabbitMQ.

#### close

```python
 | async close() -> None
```

Closes connection to RabbitMQ.

#### is\_ready

```python
 | is_ready() -> bool
```

Return `True` if a connection was established.

#### publish

```python
 | publish(event: Dict[Text, Any], headers: Optional[Dict[Text, Text]] = None) -> None
```

Publishes `event` to Pika queues.

**Arguments**:

- `event` - Serialised event to be published.
- `headers` - Message headers to append to the published message. The headers
  can be retrieved in the consumer from the `headers` attribute of the
  message&#x27;s `BasicProperties`.

#### rasa\_environment

```python
 | @rasa.shared.utils.common.lazy_property
 | rasa_environment() -> Optional[Text]
```

Get value of the `RASA_ENVIRONMENT` environment variable.


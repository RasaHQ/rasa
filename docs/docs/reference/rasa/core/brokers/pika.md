---
sidebar_label: rasa.core.brokers.pika
title: rasa.core.brokers.pika
---

#### initialise\_pika\_connection

```python
initialise_pika_connection(host: Text, username: Text, password: Text, port: Union[Text, int] = 5672, connection_attempts: int = 20, retry_delay_in_seconds: float = 5) -> "BlockingConnection"
```

Create a Pika `BlockingConnection`.

**Arguments**:

- `host` - Pika host
- `username` - username for authentication with Pika host
- `password` - password for authentication with Pika host
- `port` - port of the Pika host
- `connection_attempts` - number of channel attempts before giving up
- `retry_delay_in_seconds` - delay in seconds between channel attempts
  

**Returns**:

  `pika.BlockingConnection` with provided parameters

#### create\_rabbitmq\_ssl\_options

```python
create_rabbitmq_ssl_options(rabbitmq_host: Optional[Text] = None) -> Optional["pika.SSLOptions"]
```

Create RabbitMQ SSL options.

Requires the following environment variables to be set:

RABBITMQ_SSL_CLIENT_CERTIFICATE - path to the SSL client certificate (required)
RABBITMQ_SSL_CLIENT_KEY - path to the SSL client key (required)
RABBITMQ_SSL_CA_FILE - path to the SSL CA file for verification (optional)
RABBITMQ_SSL_KEY_PASSWORD - SSL private key password (optional)

Details on how to enable RabbitMQ TLS support can be found here:
https://www.rabbitmq.com/ssl.html#enabling-tls

**Arguments**:

- `rabbitmq_host` - RabbitMQ hostname
  

**Returns**:

  Pika SSL context of type `pika.SSLOptions` if
  the RABBITMQ_SSL_CLIENT_CERTIFICATE and RABBITMQ_SSL_CLIENT_KEY
  environment variables are valid paths, else `None`.

#### initialise\_pika\_select\_connection

```python
initialise_pika_select_connection(parameters: "Parameters", on_open_callback: Callable[["SelectConnection"], None], on_open_error_callback: Callable[["SelectConnection", Text], None]) -> "SelectConnection"
```

Create a non-blocking Pika `SelectConnection`.

**Arguments**:

- `parameters` - Parameters which should be used to connect.
- `on_open_callback` - Callback which is called when the connection was established.
- `on_open_error_callback` - Callback which is called when connecting to the broker
  failed.
  

**Returns**:

  A callback-based connection to the RabbitMQ event broker.

#### initialise\_pika\_channel

```python
initialise_pika_channel(host: Text, queue: Text, username: Text, password: Text, port: Union[Text, int] = 5672, connection_attempts: int = 20, retry_delay_in_seconds: float = 5) -> "BlockingChannel"
```

Initialise a Pika channel with a durable queue.

**Arguments**:

- `host` - Pika host.
- `queue` - Pika queue to declare.
- `username` - Username for authentication with Pika host.
- `password` - Password for authentication with Pika host.
- `port` - port of the Pika host.
- `connection_attempts` - Number of channel attempts before giving up.
- `retry_delay_in_seconds` - Delay in seconds between channel attempts.
  

**Returns**:

  Pika `BlockingChannel` with declared queue.

#### close\_pika\_channel

```python
close_pika_channel(channel: "Channel", attempts: int = 1000, time_between_attempts_in_seconds: float = 0.001) -> None
```

Attempt to close Pika channel and wait until it is closed.

**Arguments**:

- `channel` - Pika `Channel` to close.
- `attempts` - How many times to try to confirm that the channel has indeed been
  closed.
- `time_between_attempts_in_seconds` - Wait time between attempts to confirm closed
  state.

#### close\_pika\_connection

```python
close_pika_connection(connection: "Connection") -> None
```

Attempt to close Pika connection.

## PikaMessageProcessor Objects

```python
class PikaMessageProcessor()
```

A class that holds all the Pika connection details and processes Pika messages.

#### \_\_init\_\_

```python
 | __init__(parameters: "Parameters", get_message: Callable[[], Message], queues: Union[List[Text], Tuple[Text], Text, None]) -> None
```

Initialise Pika connector.

**Arguments**:

- `parameters` - Pika connection parameters
- `queues` - Pika queues to declare and publish to

#### close

```python
 | close() -> None
```

Close the Pika connection.

#### is\_connected

```python
 | @property
 | is_connected() -> bool
```

Indicates if Pika is connected and the channel is initialized.

**Returns**:

  A boolean value indicating if the connection is established.

#### is\_ready

```python
 | is_ready(attempts: int = 1000, wait_time_between_attempts_in_seconds: float = 0.01) -> bool
```

Spin until the connector is ready to process messages.

It typically takes 50 ms or so for the pika channel to open. We&#x27;ll wait up
to 10 seconds just in case.

**Arguments**:

- `attempts` - Number of retries.
- `wait_time_between_attempts_in_seconds` - Wait time between retries.
  

**Returns**:

  `True` if the channel is available, `False` otherwise.

#### process\_messages

```python
 | process_messages() -> None
```

Start to process messages.

#### run

```python
 | run()
```

Run the message processor by connecting to RabbitMQ and then
starting the IOLoop to block and allow the SelectConnection to operate.

This function is blocking and indefinite thus it
should be started in a separate process.

## PikaEventBroker Objects

```python
class PikaEventBroker(EventBroker)
```

Pika-based event broker for publishing messages to RabbitMQ.

#### \_\_init\_\_

```python
 | __init__(host: Text, username: Text, password: Text, port: Union[int, Text] = 5672, queues: Union[List[Text], Tuple[Text], Text, None] = None, should_keep_unpublished_messages: bool = True, raise_on_failure: bool = False, log_level: Union[Text, int] = os.environ.get(
 |             ENV_LOG_LEVEL_LIBRARIES, DEFAULT_LOG_LEVEL_LIBRARIES
 |         ), **kwargs: Any, ,) -> None
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
- `log_level` - Logging level.

#### close

```python
 | close() -> None
```

Close the Pika connector.

#### from\_endpoint\_config

```python
 | @classmethod
 | from_endpoint_config(cls, broker_config: Optional["EndpointConfig"]) -> Optional["PikaEventBroker"]
```

Initialise `PikaEventBroker` from `EndpointConfig`.

**Arguments**:

- `broker_config` - `EndpointConfig` to read.
  

**Returns**:

  `PikaEventBroker` if `broker_config` was supplied, else `None`.

#### is\_ready

```python
 | is_ready(attempts: int = 1000, wait_time_between_attempts_in_seconds: float = 0.01) -> bool
```

Spin until Pika is ready to process messages.

It typically takes 50 ms or so for the pika channel to open. We&#x27;ll wait up
to 10 seconds just in case.

**Arguments**:

- `attempts` - Number of retries.
- `wait_time_between_attempts_in_seconds` - Wait time between retries.
  

**Returns**:

  `True` if the channel is available, `False` otherwise.

#### publish

```python
 | publish(event: Dict[Text, Any], retries: int = 60, retry_delay_in_seconds: int = 5, headers: Optional[Dict[Text, Text]] = None) -> None
```

Publish `event` into Pika queue.

**Arguments**:

- `event` - Serialised event to be published.
- `retries` - Number of retries if publishing fails
- `retry_delay_in_seconds` - Delay in seconds between retries.
- `headers` - Message headers to append to the published message (key-value
  dictionary). The headers can be retrieved in the consumer from the
  `headers` attribute of the message&#x27;s `BasicProperties`.


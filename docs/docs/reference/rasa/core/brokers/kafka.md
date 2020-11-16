---
sidebar_label: kafka
title: rasa.core.brokers.kafka
---

## KafkaEventBroker Objects

```python
class KafkaEventBroker(EventBroker)
```

#### \_\_init\_\_

```python
 | __init__(url: Union[Text, List[Text], None], topic: Text = "rasa_core_events", client_id: Optional[Text] = None, sasl_username: Optional[Text] = None, sasl_password: Optional[Text] = None, ssl_cafile: Optional[Text] = None, ssl_certfile: Optional[Text] = None, ssl_keyfile: Optional[Text] = None, ssl_check_hostname: bool = False, security_protocol: Text = "SASL_PLAINTEXT", loglevel: Union[int, Text] = logging.ERROR, **kwargs: Any, ,) -> None
```

Kafka event broker.

**Arguments**:

- `url` - &#x27;url[:port]&#x27; string (or list of &#x27;url[:port]&#x27;
  strings) that the producer should contact to bootstrap initial
  cluster metadata. This does not have to be the full node list.
  It just needs to have at least one broker that will respond to a
  Metadata API Request.
- `topic` - Topics to subscribe to.
- `client_id` - A name for this client. This string is passed in each request
  to servers and can be used to identify specific server-side log entries
  that correspond to this client. Also submitted to `GroupCoordinator` for
  logging with respect to producer group administration.
- `group_id` - The name of the producer group to join for dynamic partition
  assignment (if enabled), and to use for fetching and committing offsets.
  If None, auto-partition assignment (via group coordinator) and offset
  commits are disabled.
- `sasl_username` - Username for plain authentication.
- `sasl_password` - Password for plain authentication.
- `ssl_cafile` - Optional filename of ca file to use in certificate
  verification.
- `ssl_certfile` - Optional filename of file in pem format containing
  the client certificate, as well as any ca certificates needed to
  establish the certificate&#x27;s authenticity.
- `ssl_keyfile` - Optional filename containing the client private key.
- `ssl_check_hostname` - Flag to configure whether ssl handshake
  should verify that the certificate matches the brokers hostname.
- `security_protocol` - Protocol used to communicate with brokers.
  Valid values are: PLAINTEXT, SSL, SASL_PLAINTEXT, SASL_SSL.
- `loglevel` - Logging level of the kafka logger.

#### from\_endpoint\_config

```python
 | @classmethod
 | async from_endpoint_config(cls, broker_config: EndpointConfig, event_loop: Optional[AbstractEventLoop] = None) -> Optional["KafkaEventBroker"]
```

Creates broker. See the parent class for more information.


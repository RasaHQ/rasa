---
sidebar_label: rasa.core.run
title: rasa.core.run
---
#### create\_http\_input\_channels

```python
create_http_input_channels(channel: Optional[Text], credentials_file: Optional[Text]) -> List["InputChannel"]
```

Instantiate the chosen input channel.

#### configure\_app

```python
configure_app(input_channels: Optional[List["InputChannel"]] = None, cors: Optional[Union[Text, List[Text], None]] = None, auth_token: Optional[Text] = None, enable_api: bool = True, response_timeout: int = constants.DEFAULT_RESPONSE_TIMEOUT, jwt_secret: Optional[Text] = None, jwt_method: Optional[Text] = None, route: Optional[Text] = "/webhooks/", port: int = constants.DEFAULT_SERVER_PORT, endpoints: Optional[AvailableEndpoints] = None, log_file: Optional[Text] = None, conversation_id: Optional[Text] = uuid.uuid4().hex, use_syslog: bool = False, syslog_address: Optional[Text] = None, syslog_port: Optional[int] = None, syslog_protocol: Optional[Text] = None, request_timeout: Optional[int] = None) -> Sanic
```

Run the agent.

#### serve\_application

```python
serve_application(model_path: Optional[Text] = None, channel: Optional[Text] = None, interface: Optional[Text] = constants.DEFAULT_SERVER_INTERFACE, port: int = constants.DEFAULT_SERVER_PORT, credentials: Optional[Text] = None, cors: Optional[Union[Text, List[Text]]] = None, auth_token: Optional[Text] = None, enable_api: bool = True, response_timeout: int = constants.DEFAULT_RESPONSE_TIMEOUT, jwt_secret: Optional[Text] = None, jwt_method: Optional[Text] = None, endpoints: Optional[AvailableEndpoints] = None, remote_storage: Optional[Text] = None, log_file: Optional[Text] = None, ssl_certificate: Optional[Text] = None, ssl_keyfile: Optional[Text] = None, ssl_ca_file: Optional[Text] = None, ssl_password: Optional[Text] = None, conversation_id: Optional[Text] = uuid.uuid4().hex, use_syslog: Optional[bool] = False, syslog_address: Optional[Text] = None, syslog_port: Optional[int] = None, syslog_protocol: Optional[Text] = None, request_timeout: Optional[int] = None) -> None
```

Run the API entrypoint.

#### load\_agent\_on\_start

```python
async load_agent_on_start(model_path: Text, endpoints: AvailableEndpoints, remote_storage: Optional[Text], app: Sanic, loop: AbstractEventLoop) -> Agent
```

Load an agent.

Used to be scheduled on server start
(hence the `app` and `loop` arguments).

#### close\_resources

```python
async close_resources(app: Sanic, _: AbstractEventLoop) -> None
```

Gracefully closes resources when shutting down server.

**Arguments**:

- `app` - The Sanic application.
- `_` - The current Sanic worker event loop.


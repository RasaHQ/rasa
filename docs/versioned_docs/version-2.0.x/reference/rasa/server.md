---
sidebar_label: rasa.server
title: rasa.server
---

#### ensure\_loaded\_agent

```python
ensure_loaded_agent(app: Sanic, require_core_is_ready=False)
```

Wraps a request handler ensuring there is a loaded and usable agent.

Require the agent to have a loaded Core model if `require_core_is_ready` is
`True`.

#### requires\_auth

```python
requires_auth(app: Sanic, token: Optional[Text] = None) -> Callable[[Any], Any]
```

Wraps a request handler with token authentication.

#### event\_verbosity\_parameter

```python
event_verbosity_parameter(request: Request, default_verbosity: EventVerbosity) -> EventVerbosity
```

Create `EventVerbosity` object using request params if present.

#### get\_tracker

```python
async get_tracker(processor: "MessageProcessor", conversation_id: Text) -> DialogueStateTracker
```

Get tracker object from `MessageProcessor`.

#### validate\_request\_body

```python
validate_request_body(request: Request, error_message: Text)
```

Check if `request` has a body.

#### authenticate

```python
async authenticate(request: Request)
```

Callback for authentication failed.

#### create\_ssl\_context

```python
create_ssl_context(ssl_certificate: Optional[Text], ssl_keyfile: Optional[Text], ssl_ca_file: Optional[Text] = None, ssl_password: Optional[Text] = None) -> Optional["SSLContext"]
```

Create an SSL context if a proper certificate is passed.

**Arguments**:

- `ssl_certificate` - path to the SSL client certificate
- `ssl_keyfile` - path to the SSL key file
- `ssl_ca_file` - path to the SSL CA file for verification (optional)
- `ssl_password` - SSL private key password (optional)
  

**Returns**:

  SSL context if a valid certificate chain can be loaded, `None` otherwise.

#### configure\_cors

```python
configure_cors(app: Sanic, cors_origins: Union[Text, List[Text], None] = "") -> None
```

Configure CORS origins for the given app.

#### add\_root\_route

```python
add_root_route(app: Sanic)
```

Add &#x27;/&#x27; route to return hello.

#### create\_app

```python
create_app(agent: Optional["Agent"] = None, cors_origins: Union[Text, List[Text], None] = "*", auth_token: Optional[Text] = None, response_timeout: int = DEFAULT_RESPONSE_TIMEOUT, jwt_secret: Optional[Text] = None, jwt_method: Text = "HS256", endpoints: Optional[AvailableEndpoints] = None)
```

Class representing a Rasa HTTP server.


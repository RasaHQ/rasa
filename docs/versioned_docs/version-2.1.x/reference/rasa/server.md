---
sidebar_label: server
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

#### get\_test\_stories

```python
get_test_stories(processor: "MessageProcessor", conversation_id: Text, until_time: Optional[float], fetch_all_sessions: bool = False) -> Text
```

Retrieves test stories from `processor` for all conversation sessions for
`conversation_id`.

**Arguments**:

- `processor` - An instance of `MessageProcessor`.
- `conversation_id` - Conversation ID to fetch stories for.
- `until_time` - Timestamp up to which to include events.
- `fetch_all_sessions` - Whether to fetch stories for all conversation sessions.
  If `False`, only the last conversation session is retrieved.
  

**Returns**:

  The stories for `conversation_id` in test format.

#### update\_conversation\_with\_events

```python
async update_conversation_with_events(conversation_id: Text, processor: "MessageProcessor", domain: Domain, events: List[Event]) -> DialogueStateTracker
```

Fetches or creates a tracker for `conversation_id` and appends `events` to it.

**Arguments**:

- `conversation_id` - The ID of the conversation to update the tracker for.
- `processor` - An instance of `MessageProcessor`.
- `domain` - The domain associated with the current `Agent`.
- `events` - The events to append to the tracker.
  

**Returns**:

  The tracker for `conversation_id` with the updated events.

#### validate\_request\_body

```python
validate_request_body(request: Request, error_message: Text) -> None
```

Check if `request` has a body.

#### authenticate

```python
async authenticate(_: Request) -> NoReturn
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


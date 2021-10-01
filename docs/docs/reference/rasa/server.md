---
sidebar_label: rasa.server
title: rasa.server
---
## ErrorResponse Objects

```python
class ErrorResponse(Exception)
```

Common exception to handle failing API requests.

#### \_\_init\_\_

```python
def __init__(status: Union[int, HTTPStatus], reason: Text, message: Text, details: Any = None, help_url: Optional[Text] = None) -> None
```

Creates error.

**Arguments**:

- `status` - The HTTP status code to return.
- `reason` - Short summary of the error.
- `message` - Detailed explanation of the error.
- `details` - Additional details which describe the error. Must be serializable.
- `help_url` - URL where users can get further help (e.g. docs).

#### ensure\_loaded\_agent

```python
def ensure_loaded_agent(app: Sanic, require_core_is_ready: bool = False) -> Callable[[Callable], Callable[..., Any]]
```

Wraps a request handler ensuring there is a loaded and usable agent.

Require the agent to have a loaded Core model if `require_core_is_ready` is
`True`.

#### ensure\_conversation\_exists

```python
def ensure_conversation_exists() -> Callable[["SanicView"], "SanicView"]
```

Wraps a request handler ensuring the conversation exists.

#### requires\_auth

```python
def requires_auth(app: Sanic, token: Optional[Text] = None) -> Callable[["SanicView"], "SanicView"]
```

Wraps a request handler with token authentication.

#### event\_verbosity\_parameter

```python
def event_verbosity_parameter(request: Request, default_verbosity: EventVerbosity) -> EventVerbosity
```

Create `EventVerbosity` object using request params if present.

#### get\_test\_stories

```python
def get_test_stories(processor: "MessageProcessor", conversation_id: Text, until_time: Optional[float], fetch_all_sessions: bool = False) -> Text
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
async def update_conversation_with_events(conversation_id: Text, processor: "MessageProcessor", domain: Domain, events: List[Event]) -> DialogueStateTracker
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
def validate_request_body(request: Request, error_message: Text) -> None
```

Check if `request` has a body.

#### validate\_events\_in\_request\_body

```python
def validate_events_in_request_body(request: Request) -> None
```

Validates events format in request body.

#### authenticate

```python
async def authenticate(_: Request) -> NoReturn
```

Callback for authentication failed.

#### create\_ssl\_context

```python
def create_ssl_context(ssl_certificate: Optional[Text], ssl_keyfile: Optional[Text], ssl_ca_file: Optional[Text] = None, ssl_password: Optional[Text] = None) -> Optional["SSLContext"]
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
def configure_cors(app: Sanic, cors_origins: Union[Text, List[Text], None] = "") -> None
```

Configure CORS origins for the given app.

#### add\_root\_route

```python
def add_root_route(app: Sanic) -> None
```

Add &#x27;/&#x27; route to return hello.

#### async\_if\_callback\_url

```python
def async_if_callback_url(f: Callable[..., Coroutine]) -> Callable
```

Decorator to enable async request handling.

If the incoming HTTP request specified a `callback_url` query parameter, the request
will return immediately with a 204 while the actual request response will
be sent to the `callback_url`. If an error happens, the error payload will also
be sent to the `callback_url`.

**Arguments**:

- `f` - The request handler function which should be decorated.
  

**Returns**:

  The decorated function.

#### run\_in\_thread

```python
def run_in_thread(f: Callable[..., Coroutine]) -> Callable
```

Decorator which runs request on a separate thread.

Some requests (e.g. training or cross-validation) are computional intense requests.
This means that they will block the event loop and hence the processing of other
requests. This decorator can be used to process these requests on a separate thread
to avoid blocking the processing of incoming requests.

**Arguments**:

- `f` - The request handler function which should be decorated.
  

**Returns**:

  The decorated function.

#### inject\_temp\_dir

```python
def inject_temp_dir(f: Callable[..., Coroutine]) -> Callable
```

Decorator to inject a temporary directory before a request and clean up after.

**Arguments**:

- `f` - The request handler function which should be decorated.
  

**Returns**:

  The decorated function.

#### create\_app

```python
def create_app(agent: Optional["Agent"] = None, cors_origins: Union[Text, List[Text], None] = "*", auth_token: Optional[Text] = None, response_timeout: int = DEFAULT_RESPONSE_TIMEOUT, jwt_secret: Optional[Text] = None, jwt_method: Text = "HS256", endpoints: Optional[AvailableEndpoints] = None) -> Sanic
```

Class representing a Rasa HTTP server.


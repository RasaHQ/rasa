---
sidebar_label: rasa.utils.endpoints
title: rasa.utils.endpoints
---
#### read\_endpoint\_config

```python
def read_endpoint_config(filename: Text, endpoint_type: Text) -> Optional["EndpointConfig"]
```

Read an endpoint configuration file from disk and extract one

config.

#### concat\_url

```python
def concat_url(base: Text, subpath: Optional[Text]) -> Text
```

Append a subpath to a base url.

Strips leading slashes from the subpath if necessary. This behaves
differently than `urlparse.urljoin` and will not treat the subpath
as a base url if it starts with `/` but will always append it to the
`base`.

**Arguments**:

- `base` - Base URL.
- `subpath` - Optional path to append to the base URL.
  

**Returns**:

  Concatenated URL with base and subpath.

## EndpointConfig Objects

```python
class EndpointConfig()
```

Configuration for an external HTTP endpoint.

#### \_\_init\_\_

```python
def __init__(url: Optional[Text] = None, params: Optional[Dict[Text, Any]] = None, headers: Optional[Dict[Text, Any]] = None, basic_auth: Optional[Dict[Text, Text]] = None, token: Optional[Text] = None, token_name: Text = "token", cafile: Optional[Text] = None, **kwargs: Any, ,) -> None
```

Creates an `EndpointConfig` instance.

#### session

```python
def session() -> aiohttp.ClientSession
```

Creates and returns a configured aiohttp client session.

#### request

```python
async def request(method: Text = "post", subpath: Optional[Text] = None, content_type: Optional[Text] = "application/json", **kwargs: Any, ,) -> Optional[Any]
```

Send a HTTP request to the endpoint. Return json response, if available.

All additional arguments will get passed through
to aiohttp&#x27;s `session.request`.

#### bool\_arg

```python
def bool_arg(request: Request, name: Text, default: bool = True) -> bool
```

Returns a passed boolean argument of the request or a default.

Checks the `name` parameter of the request if it contains a valid
boolean value. If not, `default` is returned.

**Arguments**:

- `request` - Sanic request.
- `name` - Name of argument.
- `default` - Default value for `name` argument.
  

**Returns**:

  A bool value if `name` is a valid boolean, `default` otherwise.

#### float\_arg

```python
def float_arg(request: Request, key: Text, default: Optional[float] = None) -> Optional[float]
```

Returns a passed argument cast as a float or None.

Checks the `key` parameter of the request if it contains a valid
float value. If not, `default` is returned.

**Arguments**:

- `request` - Sanic request.
- `key` - Name of argument.
- `default` - Default value for `key` argument.
  

**Returns**:

  A float value if `key` is a valid float, `default` otherwise.

#### int\_arg

```python
def int_arg(request: Request, key: Text, default: Optional[int] = None) -> Optional[int]
```

Returns a passed argument cast as an int or None.

Checks the `key` parameter of the request if it contains a valid
int value. If not, `default` is returned.

**Arguments**:

- `request` - Sanic request.
- `key` - Name of argument.
- `default` - Default value for `key` argument.
  

**Returns**:

  An int value if `key` is a valid integer, `default` otherwise.


---
sidebar_label: endpoints
title: rasa.utils.endpoints
---

#### read\_endpoint\_config

```python
read_endpoint_config(filename: Text, endpoint_type: Text) -> Optional["EndpointConfig"]
```

Read an endpoint configuration file from disk and extract one

config.

#### concat\_url

```python
concat_url(base: Text, subpath: Optional[Text]) -> Text
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

#### request

```python
 | async request(method: Text = "post", subpath: Optional[Text] = None, content_type: Optional[Text] = "application/json", **kwargs: Any, ,) -> Optional[Any]
```

Send a HTTP request to the endpoint. Return json response, if available.

All additional arguments will get passed through
to aiohttp&#x27;s `session.request`.

#### bool\_arg

```python
bool_arg(request: Request, name: Text, default: bool = True) -> bool
```

Return a passed boolean argument of the request or a default.

Checks the `name` parameter of the request if it contains a valid
boolean value. If not, `default` is returned.

#### float\_arg

```python
float_arg(request: Request, key: Text, default: Optional[float] = None) -> Optional[float]
```

Return a passed argument cast as a float or None.

Checks the `name` parameter of the request if it contains a valid
float value. If not, `None` is returned.


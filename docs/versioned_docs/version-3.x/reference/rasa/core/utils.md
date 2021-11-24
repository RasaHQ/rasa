---
sidebar_label: rasa.core.utils
title: rasa.core.utils
---
#### configure\_file\_logging

```python
configure_file_logging(logger_obj: logging.Logger, log_file: Optional[Text], use_syslog: Optional[bool], syslog_address: Optional[Text] = None, syslog_port: Optional[int] = None, syslog_protocol: Optional[Text] = None) -> None
```

Configure logging to a file.

**Arguments**:

- `logger_obj` - Logger object to configure.
- `log_file` - Path of log file to write to.
- `use_syslog` - Add syslog as a logger.
- `syslog_address` - Adress of the syslog server.
- `syslog_port` - Port of the syslog server.
- `syslog_protocol` - Protocol with the syslog server

#### one\_hot

```python
one_hot(hot_idx: int, length: int, dtype: Optional[Text] = None) -> np.ndarray
```

Create a one-hot array.

**Arguments**:

- `hot_idx` - Index of the hot element.
- `length` - Length of the array.
- `dtype` - ``numpy.dtype`` of the array.
  

**Returns**:

  One-hot array.

#### dump\_obj\_as\_yaml\_to\_file

```python
dump_obj_as_yaml_to_file(filename: Union[Text, Path], obj: Any, should_preserve_key_order: bool = False) -> None
```

Writes `obj` to the filename in YAML repr.

**Arguments**:

- `filename` - Target filename.
- `obj` - Object to dump.
- `should_preserve_key_order` - Whether to preserve key order in `obj`.

#### list\_routes

```python
list_routes(app: Sanic) -> Dict[Text, Text]
```

List all the routes of a sanic application. Mainly used for debugging.

#### extract\_args

```python
extract_args(kwargs: Dict[Text, Any], keys_to_extract: Set[Text]) -> Tuple[Dict[Text, Any], Dict[Text, Any]]
```

Go through the kwargs and filter out the specified keys.

Return both, the filtered kwargs as well as the remaining kwargs.

#### is\_limit\_reached

```python
is_limit_reached(num_messages: int, limit: Optional[int]) -> bool
```

Determine whether the number of messages has reached a limit.

**Arguments**:

- `num_messages` - The number of messages to check.
- `limit` - Limit on the number of messages.
  

**Returns**:

  `True` if the limit has been reached, otherwise `False`.

#### file\_as\_bytes

```python
file_as_bytes(path: Text) -> bytes
```

Read in a file as a byte array.

## AvailableEndpoints Objects

```python
class AvailableEndpoints()
```

Collection of configured endpoints.

#### read\_endpoints\_from\_path

```python
read_endpoints_from_path(endpoints_path: Union[Path, Text, None] = None) -> AvailableEndpoints
```

Get `AvailableEndpoints` object from specified path.

**Arguments**:

- `endpoints_path` - Path of the endpoints file to be read. If `None` the
  default path for that file is used (`endpoints.yml`).
  

**Returns**:

  `AvailableEndpoints` object read from endpoints file.

#### replace\_floats\_with\_decimals

```python
replace_floats_with_decimals(obj: Any, round_digits: int = 9) -> Any
```

Convert all instances in `obj` of `float` to `Decimal`.

**Arguments**:

- `obj` - Input object.
- `round_digits` - Rounding precision of `Decimal` values.
  

**Returns**:

  Input `obj` with all `float` types replaced by `Decimal`s rounded to
  `round_digits` decimal places.

## DecimalEncoder Objects

```python
class DecimalEncoder(json.JSONEncoder)
```

`json.JSONEncoder` that dumps `Decimal`s as `float`s.

#### default

```python
 | default(obj: Any) -> Any
```

Get serializable object for `o`.

**Arguments**:

- `obj` - Object to serialize.
  

**Returns**:

  `obj` converted to `float` if `o` is a `Decimals`, else the base class
  `default()` method.

#### replace\_decimals\_with\_floats

```python
replace_decimals_with_floats(obj: Any) -> Any
```

Convert all instances in `obj` of `Decimal` to `float`.

**Arguments**:

- `obj` - A `List` or `Dict` object.
  

**Returns**:

  Input `obj` with all `Decimal` types replaced by `float`s.

#### number\_of\_sanic\_workers

```python
number_of_sanic_workers(lock_store: Union[EndpointConfig, LockStore, None]) -> int
```

Get the number of Sanic workers to use in `app.run()`.

If the environment variable constants.ENV_SANIC_WORKERS is set and is not equal to
1, that value will only be permitted if the used lock store is not the
`InMemoryLockStore`.


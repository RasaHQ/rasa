---
sidebar_label: rasa.core.utils
title: rasa.core.utils
---

#### configure\_file\_logging

```python
configure_file_logging(logger_obj: logging.Logger, log_file: Optional[Text]) -> None
```

Configure logging to a file.

**Arguments**:

- `logger_obj` - Logger object to configure.
- `log_file` - Path of log file to write to.

#### is\_int

```python
is_int(value: Any) -> bool
```

Checks if a value is an integer.

The type of the value is not important, it might be an int or a float.

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

## HashableNDArray Objects

```python
class HashableNDArray()
```

Hashable wrapper for ndarray objects.

Instances of ndarray are not hashable, meaning they cannot be added to
sets, nor used as keys in dictionaries. This is by design - ndarray
objects are mutable, and therefore cannot reliably implement the
__hash__() method.

The hashable class allows a way around this limitation. It implements
the required methods for hashable objects in terms of an encapsulated
ndarray object. This can be either a copied instance (which is safer)
or the original object (which requires the user to be careful enough
not to modify it).

#### \_\_init\_\_

```python
 | __init__(wrapped, tight=False) -> None
```

Creates a new hashable object encapsulating an ndarray.

wrapped
    The wrapped ndarray.

tight
    Optional. If True, a copy of the input ndaray is created.
    Defaults to False.

#### unwrap

```python
 | unwrap() -> np.ndarray
```

Returns the encapsulated ndarray.

If the wrapper is &quot;tight&quot;, a copy of the encapsulated ndarray is
returned. Otherwise, the encapsulated ndarray itself is returned.

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
list_routes(app: Sanic)
```

List all the routes of a sanic application.

Mainly used for debugging.

#### extract\_args

```python
extract_args(kwargs: Dict[Text, Any], keys_to_extract: Set[Text]) -> Tuple[Dict[Text, Any], Dict[Text, Any]]
```

Go through the kwargs and filter out the specified keys.

Return both, the filtered kwargs as well as the remaining kwargs.

#### is\_limit\_reached

```python
is_limit_reached(num_messages: int, limit: int) -> bool
```

Determine whether the number of messages has reached a limit.

**Arguments**:

- `num_messages` - The number of messages to check.
- `limit` - Limit on the number of messages.
  

**Returns**:

  `True` if the limit has been reached, otherwise `False`.

#### read\_lines

```python
read_lines(filename, max_line_limit=None, line_pattern=".*") -> Generator[Text, Any, None]
```

Read messages from the command line and print bot responses.

#### file\_as\_bytes

```python
file_as_bytes(path: Text) -> bytes
```

Read in a file as a byte array.

#### convert\_bytes\_to\_string

```python
convert_bytes_to_string(data: Union[bytes, bytearray, Text]) -> Text
```

Convert `data` to string if it is a bytes-like object.

#### get\_file\_hash

```python
get_file_hash(path: Text) -> Text
```

Calculate the md5 hash of a file.

#### download\_file\_from\_url

```python
async download_file_from_url(url: Text) -> Text
```

Download a story file from a url and persists it into a temp file.

**Arguments**:

- `url` - url to download from
  

**Returns**:

  The file path of the temp file that contains the
  downloaded content.

#### pad\_lists\_to\_size

```python
pad_lists_to_size(list_x: List, list_y: List, padding_value: Optional[Any] = None) -> Tuple[List, List]
```

Compares list sizes and pads them to equal length.

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

#### set\_default\_subparser

```python
set_default_subparser(parser, default_subparser) -> None
```

default subparser selection. Call after setup, just before parse_args()

parser: the name of the parser you&#x27;re making changes to
default_subparser: the name of the subparser to call by default

#### create\_task\_error\_logger

```python
create_task_error_logger(error_message: Text = "") -> Callable[[Future], None]
```

Error logger to be attached to a task.

This will ensure exceptions are properly logged and won&#x27;t get lost.

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


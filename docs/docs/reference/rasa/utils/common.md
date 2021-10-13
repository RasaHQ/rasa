---
sidebar_label: rasa.utils.common
title: rasa.utils.common
---
## TempDirectoryPath Objects

```python
class TempDirectoryPath(str,  ContextManager)
```

Represents a path to an temporary directory.

When used as a context manager, it erases the contents of the directory on exit.

#### read\_global\_config

```python
def read_global_config(path: Text) -> Dict[Text, Any]
```

Read global Rasa configuration.

**Arguments**:

- `path` - Path to the configuration

**Returns**:

  The global configuration

#### set\_log\_level

```python
def set_log_level(log_level: Optional[int] = None) -> None
```

Set log level of Rasa and Tensorflow either to the provided log level or
to the log level specified in the environment variable &#x27;LOG_LEVEL&#x27;. If none is set
a default log level will be used.

#### update\_tensorflow\_log\_level

```python
def update_tensorflow_log_level() -> None
```

Sets Tensorflow log level based on env variable &#x27;LOG_LEVEL_LIBRARIES&#x27;.

#### update\_sanic\_log\_level

```python
def update_sanic_log_level(log_file: Optional[Text] = None) -> None
```

Set the log level of sanic loggers to the log level specified in the environment
variable &#x27;LOG_LEVEL_LIBRARIES&#x27;.

#### update\_asyncio\_log\_level

```python
def update_asyncio_log_level() -> None
```

Set the log level of asyncio to the log level specified in the environment
variable &#x27;LOG_LEVEL_LIBRARIES&#x27;.

#### update\_matplotlib\_log\_level

```python
def update_matplotlib_log_level() -> None
```

Set the log level of matplotlib to the log level specified in the environment
variable &#x27;LOG_LEVEL_LIBRARIES&#x27;.

#### set\_log\_and\_warnings\_filters

```python
def set_log_and_warnings_filters() -> None
```

Set log filters on the root logger, and duplicate filters for warnings.

Filters only propagate on handlers, not loggers.

#### sort\_list\_of\_dicts\_by\_first\_key

```python
def sort_list_of_dicts_by_first_key(dicts: List[Dict]) -> List[Dict]
```

Sorts a list of dictionaries by their first key.

#### write\_global\_config\_value

```python
def write_global_config_value(name: Text, value: Any) -> bool
```

Read global Rasa configuration.

**Arguments**:

- `name` - Name of the configuration key
- `value` - Value the configuration key should be set to
  

**Returns**:

  `True` if the operation was successful.

#### read\_global\_config\_value

```python
def read_global_config_value(name: Text, unavailable_ok: bool = True) -> Any
```

Read a value from the global Rasa configuration.

#### update\_existing\_keys

```python
def update_existing_keys(original: Dict[Any, Any], updates: Dict[Any, Any]) -> Dict[Any, Any]
```

Iterate through all the updates and update a value in the original dictionary.

If the updates contain a key that is not present in the original dict, it will
be ignored.

## RepeatedLogFilter Objects

```python
class RepeatedLogFilter(logging.Filter)
```

Filter repeated log records.

#### filter

```python
def filter(record: logging.LogRecord) -> bool
```

Determines whether current log is different to last log.

#### call\_potential\_coroutine

```python
async def call_potential_coroutine(coroutine_or_return_value: Union[Any, Coroutine]) -> Any
```

Awaits coroutine or returns value directly if it&#x27;s not a coroutine.

**Arguments**:

- `coroutine_or_return_value` - Either the return value of a synchronous function
  call or a coroutine which needs to be await first.
  

**Returns**:

  The return value of the function.

#### directory\_size\_in\_mb

```python
def directory_size_in_mb(path: Path, filenames_to_exclude: Optional[List[Text]] = None) -> float
```

Calculates the size of a directory.

**Arguments**:

- `path` - The path to the directory.
- `filenames_to_exclude` - Allows excluding certain files from the calculation.
  

**Returns**:

  Directory size in MiB.

#### copy\_directory

```python
def copy_directory(source: Path, destination: Path) -> None
```

Copies the content of one directory into another.

Unlike `shutil.copytree` this doesn&#x27;t raise if `destination` already exists.

# TODO: Drop this in favor of `shutil.copytree(..., dirs_exist_ok=True)` when
# dropping Python 3.7.

**Arguments**:

- `source` - The directory whose contents should be copied to `destination`.
- `destination` - The directory which should contain the content `source` in the end.
  

**Raises**:

- `ValueError` - If destination is not empty.

#### find\_unavailable\_packages

```python
def find_unavailable_packages(package_names: List[Text]) -> Set[Text]
```

Tries to import all package names and returns the packages where it failed.

**Arguments**:

- `package_names` - The package names to import.
  

**Returns**:

  Package names that could not be imported.

#### module\_path\_from\_class

```python
def module_path_from_class(clazz: Type) -> Text
```

Return the module path of an instance&#x27;s class.


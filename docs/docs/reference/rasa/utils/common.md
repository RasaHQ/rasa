---
sidebar_label: rasa.utils.common
title: rasa.utils.common
---
## TempDirectoryPath Objects

```python
class TempDirectoryPath(str)
```

Represents a path to an temporary directory. When used as a context
manager, it erases the contents of the directory on exit.

#### read\_global\_config

```python
read_global_config(path: Text) -> Dict[Text, Any]
```

Read global Rasa configuration.

**Arguments**:

- `path` - Path to the configuration

**Returns**:

  The global configuration

#### set\_log\_level

```python
set_log_level(log_level: Optional[int] = None) -> None
```

Set log level of Rasa and Tensorflow either to the provided log level or
to the log level specified in the environment variable &#x27;LOG_LEVEL&#x27;. If none is set
a default log level will be used.

#### update\_tensorflow\_log\_level

```python
update_tensorflow_log_level() -> None
```

Sets Tensorflow log level based on env variable &#x27;LOG_LEVEL_LIBRARIES&#x27;.

#### update\_sanic\_log\_level

```python
update_sanic_log_level(log_file: Optional[Text] = None) -> None
```

Set the log level of sanic loggers to the log level specified in the environment
variable &#x27;LOG_LEVEL_LIBRARIES&#x27;.

#### update\_asyncio\_log\_level

```python
update_asyncio_log_level() -> None
```

Set the log level of asyncio to the log level specified in the environment
variable &#x27;LOG_LEVEL_LIBRARIES&#x27;.

#### update\_matplotlib\_log\_level

```python
update_matplotlib_log_level() -> None
```

Set the log level of matplotlib to the log level specified in the environment
variable &#x27;LOG_LEVEL_LIBRARIES&#x27;.

#### set\_log\_and\_warnings\_filters

```python
set_log_and_warnings_filters() -> None
```

Set log filters on the root logger, and duplicate filters for warnings.

Filters only propagate on handlers, not loggers.

#### sort\_list\_of\_dicts\_by\_first\_key

```python
sort_list_of_dicts_by_first_key(dicts: List[Dict]) -> List[Dict]
```

Sorts a list of dictionaries by their first key.

#### write\_global\_config\_value

```python
write_global_config_value(name: Text, value: Any) -> bool
```

Read global Rasa configuration.

**Arguments**:

- `name` - Name of the configuration key
- `value` - Value the configuration key should be set to
  

**Returns**:

  `True` if the operation was successful.

#### read\_global\_config\_value

```python
read_global_config_value(name: Text, unavailable_ok: bool = True) -> Any
```

Read a value from the global Rasa configuration.

#### update\_existing\_keys

```python
update_existing_keys(original: Dict[Any, Any], updates: Dict[Any, Any]) -> Dict[Any, Any]
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
 | filter(record: logging.LogRecord) -> bool
```

Determines whether current log is different to last log.

#### run\_in\_loop

```python
run_in_loop(f: Coroutine[Any, Any, T], loop: Optional[asyncio.AbstractEventLoop] = None) -> T
```

Execute the awaitable in the passed loop.

If no loop is passed, the currently existing one is used or a new one is created
if no loop has been started in the current context.

After the awaitable is finished, all remaining tasks on the loop will be
awaited as well (background tasks).

WARNING: don&#x27;t use this if there are never ending background tasks scheduled.
in this case, this function will never return.

**Arguments**:

- `f` - function to execute
- `loop` - loop to use for the execution
  

**Returns**:

  return value from the function

#### call\_potential\_coroutine

```python
async call_potential_coroutine(coroutine_or_return_value: Union[Any, Coroutine]) -> Any
```

Awaits coroutine or returns value directly if it&#x27;s not a coroutine.

**Arguments**:

- `coroutine_or_return_value` - Either the return value of a synchronous function
  call or a coroutine which needs to be await first.
  

**Returns**:

  The return value of the function.


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

#### arguments\_of

```python
arguments_of(func: Callable) -> List[Text]
```

Return the parameters of the function `func` as a list of names.

#### read\_global\_config

```python
read_global_config() -> Dict[Text, Any]
```

Read global Rasa configuration.

#### set\_log\_level

```python
set_log_level(log_level: Optional[int] = None)
```

Set log level of Rasa and Tensorflow either to the provided log level or
to the log level specified in the environment variable &#x27;LOG_LEVEL&#x27;. If none is set
a default log level will be used.

#### update\_tensorflow\_log\_level

```python
update_tensorflow_log_level() -> None
```

Set the log level of Tensorflow to the log level specified in the environment
variable &#x27;LOG_LEVEL_LIBRARIES&#x27;.

#### update\_sanic\_log\_level

```python
update_sanic_log_level(log_file: Optional[Text] = None)
```

Set the log level of sanic loggers to the log level specified in the environment
variable &#x27;LOG_LEVEL_LIBRARIES&#x27;.

#### update\_asyncio\_log\_level

```python
update_asyncio_log_level() -> None
```

Set the log level of asyncio to the log level specified in the environment
variable &#x27;LOG_LEVEL_LIBRARIES&#x27;.

#### set\_log\_and\_warnings\_filters

```python
set_log_and_warnings_filters() -> None
```

Set log filters on the root logger, and duplicate filters for warnings.

Filters only propagate on handlers, not loggers.

#### obtain\_verbosity

```python
obtain_verbosity() -> int
```

Returns a verbosity level according to the set log level.

#### sort\_list\_of\_dicts\_by\_first\_key

```python
sort_list_of_dicts_by_first_key(dicts: List[Dict]) -> List[Dict]
```

Sorts a list of dictionaries by their first key.

#### transform\_collection\_to\_sentence

```python
transform_collection_to_sentence(collection: Collection[Text]) -> Text
```

Transforms e.g. a list like [&#x27;A&#x27;, &#x27;B&#x27;, &#x27;C&#x27;] into a sentence &#x27;A, B and C&#x27;.

#### minimal\_kwargs

```python
minimal_kwargs(kwargs: Dict[Text, Any], func: Callable, excluded_keys: Optional[List] = None) -> Dict[Text, Any]
```

Returns only the kwargs which are required by a function. Keys, contained in
the exception list, are not included.

**Arguments**:

- `kwargs` - All available kwargs.
- `func` - The function which should be called.
- `excluded_keys` - Keys to exclude from the result.
  

**Returns**:

  Subset of kwargs which are accepted by `func`.

#### write\_global\_config\_value

```python
write_global_config_value(name: Text, value: Any) -> None
```

Read global Rasa configuration.

#### read\_global\_config\_value

```python
read_global_config_value(name: Text, unavailable_ok: bool = True) -> Any
```

Read a value from the global Rasa configuration.

#### mark\_as\_experimental\_feature

```python
mark_as_experimental_feature(feature_name: Text) -> None
```

Warns users that they are using an experimental feature.

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


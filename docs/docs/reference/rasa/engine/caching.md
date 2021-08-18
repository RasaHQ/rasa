---
sidebar_label: rasa.engine.caching
title: rasa.engine.caching
---
## TrainingCache Objects

```python
class TrainingCache(abc.ABC)
```

Stores training results in a persistent cache.

Used to minimize re-retraining when the data / config didn&#x27;t change in between
training runs.

#### cache\_output

```python
 | @abc.abstractmethod
 | cache_output(fingerprint_key: Text, output: Any, output_fingerprint: Text, model_storage: ModelStorage) -> None
```

Adds the output to the cache.

If the output is of type `Cacheable` the output is persisted to disk in addition
to its fingerprint.

**Arguments**:

- `fingerprint_key` - The fingerprint key serves as key for the cache. Graph
  components can use their fingerprint key to lookup fingerprints of
  previous training runs.
- `output` - The output. The output is only cached to disk if it&#x27;s of type
  `Cacheable`.
- `output_fingerprint` - The fingerprint of their output. This can be used
  to lookup potentially persisted outputs on disk.
- `model_storage` - Required for caching `Resource` instances. E.g. `Resource`s
  use that to copy data from the model storage to the cache.

#### get\_cached\_output\_fingerprint

```python
 | @abc.abstractmethod
 | get_cached_output_fingerprint(fingerprint_key: Text) -> Optional[Text]
```

Retrieves fingerprint of output based on fingerprint key.

**Arguments**:

- `fingerprint_key` - The fingerprint serves as key for the lookup of output
  fingerprints.
  

**Returns**:

  The fingerprint of a matching output or `None` in case no cache entry was
  found for the given fingerprint key.

#### get\_cached\_result

```python
 | @abc.abstractmethod
 | get_cached_result(output_fingerprint_key: Text, node_name: Text, model_storage: ModelStorage) -> Optional[Cacheable]
```

Returns a potentially cached output result.

**Arguments**:

- `output_fingerprint_key` - The fingerprint key of the output serves as lookup
  key for a potentially cached version of this output.
- `node_name` - The name of the graph node which wants to use this cached result.
- `model_storage` - The current model storage (e.g. used when restoring
  `Resource` objects so that they can fill the model storage with data).
  

**Returns**:

  `None` if no matching result was found or restored `Cacheable`.

## Cacheable Objects

```python
@runtime_checkable
class Cacheable(Protocol)
```

Protocol for cacheable graph component outputs.

We only cache graph component outputs which are `Cacheable`. We only store the
output fingerprint for everything else.

#### to\_cache

```python
 | to_cache(directory: Path, model_storage: ModelStorage) -> None
```

Persists `Cacheable` to disk.

**Arguments**:

- `directory` - The directory where the `Cacheable` can persist itself to.
- `model_storage` - The current model storage (e.g. used when caching `Resource`
  objects.

#### from\_cache

```python
 | @classmethod
 | from_cache(cls, node_name: Text, directory: Path, model_storage: ModelStorage) -> Cacheable
```

Loads `Cacheable` from cache.

**Arguments**:

- `node_name` - The name of the graph node which wants to use this cached result.
- `directory` - Directory containing the persisted `Cacheable`.
- `model_storage` - The current model storage (e.g. used when restoring
  `Resource` objects so that they can fill the model storage with data).
  

**Returns**:

  Instantiated `Cacheable`.

## LocalTrainingCache Objects

```python
class LocalTrainingCache(TrainingCache)
```

Caches training results on local disk (see parent class for full docstring).

## CacheEntry Objects

```python
class CacheEntry(Base)
```

Stores metadata about a single cache entry.

#### \_\_init\_\_

```python
 | __init__() -> None
```

Creates cache.

The `Cache` setting can be configured via environment variables.

#### cache\_output

```python
 | cache_output(fingerprint_key: Text, output: Any, output_fingerprint: Text, model_storage: ModelStorage) -> None
```

Adds the output to the cache (see parent class for full docstring).

#### get\_cached\_output\_fingerprint

```python
 | get_cached_output_fingerprint(fingerprint_key: Text) -> Optional[Text]
```

Returns cached output fingerprint (see parent class for full docstring).

#### get\_cached\_result

```python
 | get_cached_result(output_fingerprint_key: Text, node_name: Text, model_storage: ModelStorage) -> Optional[Cacheable]
```

Returns a potentially cached output (see parent class for full docstring).


---
sidebar_label: rasa.engine.storage.resource
title: rasa.engine.storage.resource
---
## Resource Objects

```python
@dataclass
class Resource()
```

Represents a persisted graph component in the graph.

#### from\_cache

```python
@classmethod
def from_cache(cls, node_name: Text, directory: Path, model_storage: ModelStorage) -> Resource
```

Loads a `Resource` from the cache.

This automatically loads the persisted resource into the given `ModelStorage`.

**Arguments**:

- `node_name` - The node name of the `Resource`.
- `directory` - The directory with the cached `Resource`.
- `model_storage` - The `ModelStorage` which the cached `Resource` will be added
  to so that the `Resource` is accessible for other graph nodes.
  

**Returns**:

  The ready-to-use and accessible `Resource`.

#### to\_cache

```python
def to_cache(directory: Path, model_storage: ModelStorage) -> None
```

Persists the `Resource` to the cache.

**Arguments**:

- `directory` - The directory which receives the persisted `Resource`.
- `model_storage` - The model storage which currently contains the persisted
  `Resource`.

#### fingerprint

```python
def fingerprint() -> Text
```

Provides fingerprint for `Resource`.

The fingerprint can be just the name as the persisted resource only changes
if the used training data (which is loaded in previous nodes) or the config
(which is fingerprinted separately) changes.

**Returns**:

  Fingerprint for `Resource`.


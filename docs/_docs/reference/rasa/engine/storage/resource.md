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

**Attributes**:

- `name` - The unique identifier for the `Resource`. Used to locate the associated
  data from a `ModelStorage`. Normally matches the name of the node which
  created it.
- `output_fingerprint` - An unique identifier for a specific instantiation of a
  `Resource`. Used to distinguish a specific persistence for the same
  `Resource` when saving to the cache.

#### from\_cache

```python
 | @classmethod
 | from_cache(cls, node_name: Text, directory: Path, model_storage: ModelStorage, output_fingerprint: Text) -> Resource
```

Loads a `Resource` from the cache.

This automatically loads the persisted resource into the given `ModelStorage`.

**Arguments**:

- `node_name` - The node name of the `Resource`.
- `directory` - The directory with the cached `Resource`.
- `model_storage` - The `ModelStorage` which the cached `Resource` will be added
  to so that the `Resource` is accessible for other graph nodes.
- `output_fingerprint` - The fingerprint of the cached `Resource`.
  

**Returns**:

  The ready-to-use and accessible `Resource`.

#### to\_cache

```python
 | to_cache(directory: Path, model_storage: ModelStorage) -> None
```

Persists the `Resource` to the cache.

**Arguments**:

- `directory` - The directory which receives the persisted `Resource`.
- `model_storage` - The model storage which currently contains the persisted
  `Resource`.

#### fingerprint

```python
 | fingerprint() -> Text
```

Provides fingerprint for `Resource`.

A unique fingerprint is created on initialization of a `Resource` however we
also allow a value to be provided for when we retrieve a `Resource` from the
cache (see `Resource.from_cache`).

**Returns**:

  Fingerprint for `Resource`.


---
sidebar_label: rasa.engine.storage.storage
title: rasa.engine.storage.storage
---
## ModelStorage Objects

```python
class ModelStorage(abc.ABC)
```

Serves as storage backend for `GraphComponents` which need persistence.

#### create

```python
@classmethod
@abc.abstractmethod
def create(cls, storage_path: Path) -> ModelStorage
```

Creates the storage.

**Arguments**:

- `storage_path` - Directory which will contain the persisted graph components.

#### from\_model\_archive

```python
@classmethod
@abc.abstractmethod
def from_model_archive(cls, storage_path: Path, model_archive_path: Union[Text, Path]) -> Tuple[ModelStorage, ModelMetadata]
```

Unpacks a model archive and initializes a `ModelStorage`.

**Arguments**:

- `storage_path` - Directory which will contain the persisted graph components.
- `model_archive_path` - The path to the model archive.
  

**Returns**:

  Initialized model storage, and metadata about the model.

#### write\_to

```python
@contextmanager
@abc.abstractmethod
def write_to(resource: Resource) -> ContextManager[Path]
```

Persists data for a given resource.

This `Resource` can then be accessed in dependent graph nodes via
`model_storage.read_from`.

**Arguments**:

- `resource` - The resource which should be persisted.
  

**Returns**:

  A directory which can be used to persist data for the given `Resource`.

#### read\_from

```python
@contextmanager
@abc.abstractmethod
def read_from(resource: Resource) -> ContextManager[Path]
```

Provides the data of a persisted `Resource`.

**Arguments**:

- `resource` - The `Resource` whose persisted should be accessed.
  

**Returns**:

  A directory containing the data of the persisted `Resource`.
  

**Raises**:

- `ValueError` - In case no persisted data for the given `Resource` exists.

#### create\_model\_package

```python
def create_model_package(model_archive_path: Union[Text, Path], train_schema: GraphSchema, predict_schema: GraphSchema, domain: Domain) -> ModelMetadata
```

Creates a model archive containing all data to load and run the model.

**Arguments**:

- `model_archive_path` - The path to the archive which should be created.
- `train_schema` - The schema which was used to train the graph model.
- `predict_schema` - The schema for running predictions with the trained model.
- `domain` - The `Domain` which was used to train the model.
  

**Returns**:

  The model metadata.

## ModelMetadata Objects

```python
@dataclass()
class ModelMetadata()
```

Describes a trained model.

#### as\_dict

```python
def as_dict() -> Dict[Text, Any]
```

Returns serializable version of the `ModelMetadata`.

#### from\_dict

```python
@classmethod
def from_dict(cls, serialized: Dict[Text, Any]) -> ModelMetadata
```

Loads `ModelMetadata` which has been serialized using `metadata.as_dict()`.

**Arguments**:

- `serialized` - Serialized `ModelMetadata` (e.g. read from disk).
  

**Returns**:

  Instantiated `ModelMetadata`.


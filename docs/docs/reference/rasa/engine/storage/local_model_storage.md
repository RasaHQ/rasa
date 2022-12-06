---
sidebar_label: rasa.engine.storage.local_model_storage
title: rasa.engine.storage.local_model_storage
---
#### windows\_safe\_temporary\_directory

```python
@contextmanager
def windows_safe_temporary_directory(
        suffix: Optional[Text] = None,
        prefix: Optional[Text] = None,
        dir: Optional[Text] = None) -> Generator[Text, None, None]
```

Like `tempfile.TemporaryDirectory`, but works with Windows and long file names.

On Windows by default there is a restriction on long path names.
Using the prefix below allows to bypass this restriction in environments
where it&#x27;s not possible to override this behavior, mostly for internal
policy reasons.

Reference: https://stackoverflow.com/a/49102229

## LocalModelStorage Objects

```python
class LocalModelStorage(ModelStorage)
```

Stores and provides output of `GraphComponents` on local disk.

#### \_\_init\_\_

```python
def __init__(storage_path: Path) -> None
```

Creates storage (see parent class for full docstring).

#### create

```python
@classmethod
def create(cls, storage_path: Path) -> ModelStorage
```

Creates a new instance (see parent class for full docstring).

#### from\_model\_archive

```python
@classmethod
def from_model_archive(
    cls, storage_path: Path, model_archive_path: Union[Text, Path]
) -> Tuple[LocalModelStorage, ModelMetadata]
```

Initializes storage from archive (see parent class for full docstring).

#### metadata\_from\_archive

```python
@classmethod
def metadata_from_archive(
        cls, model_archive_path: Union[Text, Path]) -> ModelMetadata
```

Retrieves metadata from archive (see parent class for full docstring).

#### write\_to

```python
@contextmanager
def write_to(resource: Resource) -> Generator[Path, None, None]
```

Persists data for a resource (see parent class for full docstring).

#### read\_from

```python
@contextmanager
def read_from(resource: Resource) -> Generator[Path, None, None]
```

Provides the data of a `Resource` (see parent class for full docstring).

#### create\_model\_package

```python
def create_model_package(model_archive_path: Union[Text, Path],
                         model_configuration: GraphModelConfiguration,
                         domain: Domain) -> ModelMetadata
```

Creates model package (see parent class for full docstring).


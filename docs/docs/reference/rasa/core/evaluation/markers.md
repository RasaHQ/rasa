---
sidebar_label: rasa.core.evaluation.markers
title: rasa.core.evaluation.markers
---
## InvalidMarkersConfig Objects

```python
class InvalidMarkersConfig(RasaException)
```

Exception that can be raised when markers config is not valid.

## MarkerConfig Objects

```python
class MarkerConfig()
```

A class that represents the markers config.

A markers config contains the markers and the conditions for when they apply.
The class reads the config, validates the schema, and validates the conditions.

#### empty\_config

```python
@classmethod
def empty_config(cls) -> Dict
```

Returns an empty config file.

#### load\_config\_from\_path

```python
@classmethod
def load_config_from_path(cls, path: Union[Text, Path]) -> Dict
```

Loads the config from a file or directory.

#### from\_file

```python
@classmethod
def from_file(cls, path: Text) -> Dict
```

Loads the config from a YAML file.

#### from\_yaml

```python
@classmethod
def from_yaml(cls, yaml: Text, filename: Text = "") -> Dict
```

Loads the config from YAML text after validating it.

#### from\_directory

```python
@classmethod
def from_directory(cls, path: Text) -> Dict
```

Loads and appends multiple configs from a directory tree.

#### validate\_config

```python
@classmethod
def validate_config(cls, config: Dict, filename: Text = "") -> bool
```

Validates the markers config according to the schema.

#### config\_format\_spec

```python
@staticmethod
def config_format_spec() -> Dict
```

Returns expected schema for a markers config.


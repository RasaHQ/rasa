---
sidebar_label: rasa.core.training.converters.responses_prefix_converter
title: rasa.core.training.converters.responses_prefix_converter
---
#### normalize\_utter\_action

```python
normalize_utter_action(action_name: Text) -> Text
```

Ensure that response names start with `utter_`.

**Arguments**:

- `action_name` - The name of the response.
  

**Returns**:

  The name of the response, starting with `utter_`.

## StoryResponsePrefixConverter Objects

```python
class StoryResponsePrefixConverter(TrainingDataConverter)
```

Converter responsible for ensuring that retrieval intent actions in stories
start with `utter_` instead of `respond_`.

#### filter

```python
 | @classmethod
 | filter(cls, source_path: Path) -> bool
```

Only accept YAML story files.

**Arguments**:

- `source_path` - Path to a training data file.
  

**Returns**:

  `True` if the given file is a YAML stories file, `False` otherwise.

#### convert\_and\_write

```python
 | @classmethod
 | async convert_and_write(cls, source_path: Path, output_path: Path) -> None
```

Migrate retrieval intent responses to the new 2.0 format in stories.

Before 2.0, retrieval intent responses needed to start
with `respond_`. Now, they need to start with `utter_`.

**Arguments**:

- `source_path` - the source YAML stories file.
- `output_path` - Path to the output directory.

## DomainResponsePrefixConverter Objects

```python
class DomainResponsePrefixConverter(TrainingDataConverter)
```

Converter responsible for ensuring that retrieval intent actions in domain
start with `utter_` instead of `respond_`.

#### filter

```python
 | @classmethod
 | filter(cls, source_path: Path) -> bool
```

Only accept domain files.

**Arguments**:

- `source_path` - Path to a domain file.
  

**Returns**:

  `True` if the given file can is a valid domain file, `False` otherwise.

#### convert\_and\_write

```python
 | @classmethod
 | async convert_and_write(cls, source_path: Path, output_path: Path) -> None
```

Migrate retrieval intent responses to the new 2.0 format in domains.

Before 2.0, retrieval intent responses needed to start
with `respond_`. Now, they need to start with `utter_`.

**Arguments**:

- `source_path` - The source domain file.
- `output_path` - Path to the output directory.


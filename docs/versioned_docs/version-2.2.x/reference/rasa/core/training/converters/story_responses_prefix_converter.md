---
sidebar_label: story_responses_prefix_converter
title: rasa.core.training.converters.story_responses_prefix_converter
---

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

Only consider YAML story files.

**Arguments**:

- `source_path` - Path to the training data file.
  

**Returns**:

  `True` if the given file can is a YAML stories file, `False` otherwise

#### convert\_and\_write

```python
 | @classmethod
 | async convert_and_write(cls, source_path: Path, output_path: Path) -> None
```

Migrate retrieval intent responses to the new 2.0 format in stories.

Before 2.0, retrieval intent responses needed to start
with `respond_`. Now, they need to start with `utter_`.

**Arguments**:

- `source_path` - the source YAML stories file
- `output_path` - Path to the output directory.


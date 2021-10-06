---
sidebar_label: rasa.nlu.training_data.converters.nlu_markdown_to_yaml_converter
title: rasa.nlu.training_data.converters.nlu_markdown_to_yaml_converter
---
## NLUMarkdownToYamlConverter Objects

```python
class NLUMarkdownToYamlConverter(TrainingDataConverter)
```

#### filter

```python
 | @classmethod
 | filter(cls, source_path: Path) -> bool
```

Checks if the given training data file contains NLU data in `Markdown` format
and can be converted to `YAML`.

**Arguments**:

- `source_path` - Path to the training data file.
  

**Returns**:

  `True` if the given file can be converted, `False` otherwise

#### convert\_and\_write

```python
 | @classmethod
 | async convert_and_write(cls, source_path: Path, output_path: Path) -> None
```

Converts the given training data file and saves it to the output directory.

**Arguments**:

- `source_path` - Path to the training data file.
- `output_path` - Path to the output directory.


---
sidebar_label: rasa.utils.converter
title: rasa.utils.converter
---
## TrainingDataConverter Objects

```python
class TrainingDataConverter()
```

Interface for any training data format conversion.

#### filter

```python
@classmethod
def filter(cls, source_path: Path) -> bool
```

Checks if the concrete implementation of TrainingDataConverter can convert
training data file.

**Arguments**:

- `source_path` - Path to the training data file.
  

**Returns**:

  `True` if the given file can be converted, `False` otherwise

#### convert\_and\_write

```python
@classmethod
async def convert_and_write(cls, source_path: Path, output_path: Path) -> None
```

Converts the given training data file and saves it to the output directory.

**Arguments**:

- `source_path` - Path to the training data file.
- `output_path` - Path to the output directory.

#### generate\_path\_for\_converted\_training\_data\_file

```python
@classmethod
def generate_path_for_converted_training_data_file(cls, source_file_path: Path, output_directory: Path) -> Path
```

Generates path for a training data file converted to YAML format.

**Arguments**:

- `source_file_path` - Path to the original file.
- `output_directory` - Path to the target directory.
  

**Returns**:

  Path to the target converted training data file.

#### converted\_file\_suffix

```python
@classmethod
def converted_file_suffix(cls) -> Text
```

Returns suffix that should be appended to the converted
training data file.


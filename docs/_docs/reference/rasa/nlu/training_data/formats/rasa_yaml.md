---
sidebar_label: rasa.nlu.training_data.formats.rasa_yaml
title: rasa.nlu.training_data.formats.rasa_yaml
---
## RasaYAMLReader Objects

```python
class RasaYAMLReader(TrainingDataReader)
```

Reads YAML training data and creates a TrainingData object.

#### validate

```python
 | @staticmethod
 | validate(string: Text) -> None
```

Check if the string adheres to the NLU yaml data schema.

If the string is not in the right format, an exception will be raised.

#### reads

```python
 | reads(string: Text, **kwargs: Any) -> "TrainingData"
```

Reads TrainingData in YAML format from a string.

**Arguments**:

- `string` - String with YAML training data.
- `**kwargs` - Keyword arguments.
  

**Returns**:

  New `TrainingData` object with parsed training data.

#### is\_yaml\_nlu\_file

```python
 | @staticmethod
 | is_yaml_nlu_file(filename: Text) -> bool
```

Checks if the specified file possibly contains NLU training data in YAML.

**Arguments**:

- `filename` - name of the file to check.
  

**Returns**:

  `True` if the `filename` is possibly a valid YAML NLU file,
  `False` otherwise.

## RasaYAMLWriter Objects

```python
class RasaYAMLWriter(TrainingDataWriter)
```

Writes training data into a file in a YAML format.

#### dumps

```python
 | dumps(training_data: "TrainingData") -> Text
```

Turns TrainingData into a string.

#### dump

```python
 | dump(target: Union[Text, Path, StringIO], training_data: "TrainingData") -> None
```

Writes training data into a file in a YAML format.

**Arguments**:

- `target` - Name of the target object to write the YAML to.
- `training_data` - TrainingData object.


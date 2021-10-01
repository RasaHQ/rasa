---
sidebar_label: rasa.shared.nlu.training_data.formats.rasa_yaml
title: rasa.shared.nlu.training_data.formats.rasa_yaml
---
## RasaYAMLReader Objects

```python
class RasaYAMLReader(TrainingDataReader)
```

Reads YAML training data and creates a TrainingData object.

#### validate

```python
def validate(string: Text) -> None
```

Check if the string adheres to the NLU yaml data schema.

If the string is not in the right format, an exception will be raised.

#### reads

```python
def reads(string: Text, **kwargs: Any) -> "TrainingData"
```

Reads TrainingData in YAML format from a string.

**Arguments**:

- `string` - String with YAML training data.
- `**kwargs` - Keyword arguments.
  

**Returns**:

  New `TrainingData` object with parsed training data.

#### is\_yaml\_nlu\_file

```python
@staticmethod
def is_yaml_nlu_file(filename: Union[Text, Path]) -> bool
```

Checks if the specified file possibly contains NLU training data in YAML.

**Arguments**:

- `filename` - name of the file to check.
  

**Returns**:

  `True` if the `filename` is possibly a valid YAML NLU file,
  `False` otherwise.
  

**Raises**:

- `YamlException` - if the file seems to be a YAML file (extension) but
  can not be read / parsed.

## RasaYAMLWriter Objects

```python
class RasaYAMLWriter(TrainingDataWriter)
```

Writes training data into a file in a YAML format.

#### dumps

```python
def dumps(training_data: "TrainingData") -> Text
```

Turns TrainingData into a string.

#### dump

```python
def dump(target: Union[Text, Path, StringIO], training_data: "TrainingData") -> None
```

Writes training data into a file in a YAML format.

**Arguments**:

- `target` - Name of the target object to write the YAML to.
- `training_data` - TrainingData object.

#### training\_data\_to\_dict

```python
@classmethod
def training_data_to_dict(cls, training_data: "TrainingData") -> Optional[OrderedDict]
```

Represents NLU training data to a dict/list structure ready to be
serialized as YAML.

**Arguments**:

- `training_data` - `TrainingData` to convert.
  

**Returns**:

  `OrderedDict` containing all training data.

#### process\_lookup\_tables

```python
@classmethod
def process_lookup_tables(cls, training_data: "TrainingData") -> List[OrderedDict]
```

Serializes the look up tables.

**Arguments**:

- `training_data` - The training data object with potential look up tables.
  

**Returns**:

  The serialized lookup tables.

#### process\_training\_examples\_by\_key

```python
@staticmethod
def process_training_examples_by_key(training_examples: Dict[Text, List[Union[Dict, Text]]], key_name: Text, key_examples: Text, example_extraction_predicate: Callable[[Dict[Text, Any]], Text]) -> List[OrderedDict]
```

Prepares training examples  to be written to YAML.

This can be any NLU training data (intent examples, lookup tables, etc.)

**Arguments**:

- `training_examples` - Multiple training examples. Mappings in case additional
  values were specified for an example (e.g. metadata) or just the plain
  value.
- `key_name` - The top level key which the examples belong to (e.g. `intents`)
- `key_examples` - The sub key which the examples should be added to
  (e.g. `examples`).
- `example_extraction_predicate` - Function to extract example value (e.g. the
  the text for an intent example)
  

**Returns**:

  NLU training data examples prepared for writing to YAML.


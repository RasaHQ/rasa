---
sidebar_label: rasa.shared.utils.validation
title: rasa.shared.utils.validation
---
## YamlValidationException Objects

```python
class YamlValidationException(YamlException,  ValueError)
```

Raised if a yaml file does not correspond to the expected schema.

#### \_\_init\_\_

```python
 | __init__(message: Text, validation_errors: Optional[List[SchemaError.SchemaErrorEntry]] = None, filename: Optional[Text] = None, content: Any = None) -> None
```

Create The Error.

**Arguments**:

- `message` - error message
- `validation_errors` - validation errors
- `filename` - name of the file which was validated
- `content` - yaml content loaded from the file (used for line information)

#### validate\_yaml\_schema

```python
validate_yaml_schema(yaml_file_content: Text, schema_path: Text) -> None
```

Validate yaml content.

**Arguments**:

- `yaml_file_content` - the content of the yaml file to be validated
- `schema_path` - the schema of the yaml file

#### validate\_training\_data

```python
validate_training_data(json_data: Dict[Text, Any], schema: Dict[Text, Any]) -> None
```

Validate rasa training data format to ensure proper training.

**Arguments**:

- `json_data` - the data to validate
- `schema` - the schema
  

**Raises**:

  SchemaValidationError if validation fails.

#### validate\_training\_data\_format\_version

```python
validate_training_data_format_version(yaml_file_content: Dict[Text, Any], filename: Optional[Text]) -> bool
```

Validates version on the training data content using `version` field
and warns users if the file is not compatible with the current version of
Rasa Open Source.

**Arguments**:

- `yaml_file_content` - Raw content of training data file as a dictionary.
- `filename` - Name of the validated file.
  

**Returns**:

  `True` if the file can be processed by current version of Rasa Open Source,
  `False` otherwise.


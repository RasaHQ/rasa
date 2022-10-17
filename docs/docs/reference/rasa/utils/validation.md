---
sidebar_label: rasa.utils.validation
title: rasa.utils.validation
---
## InvalidYamlFileError Objects

```python
class InvalidYamlFileError(ValueError)
```

Raised if an invalid yaml file was provided.

#### validate\_yaml\_schema

```python
validate_yaml_schema(yaml_file_content: Text, schema_path: Text, show_validation_errors: bool = True) -> None
```

Validate yaml content.

**Arguments**:

- `yaml_file_content` - the content of the yaml file to be validated
- `schema_path` - the schema of the yaml file
- `show_validation_errors` - if true, validation errors are shown

#### validate\_training\_data

```python
validate_training_data(json_data: Dict[Text, Any], schema: Dict[Text, Any]) -> None
```

Validate rasa training data format to ensure proper training.

**Arguments**:

- `json_data` - the data to validate
- `schema` - the schema
  

**Raises**:

  ValidationError if validation fails.


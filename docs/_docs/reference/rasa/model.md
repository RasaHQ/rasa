---
sidebar_label: rasa.model
title: rasa.model
---
#### get\_local\_model

```python
get_local_model(model_path: Text = DEFAULT_MODELS_PATH) -> Text
```

Returns verified path to local model archive.

**Arguments**:

- `model_path` - Path to the zipped model. If it&#x27;s a directory, the latest
  trained model is returned.
  

**Returns**:

  Path to the zipped model. If it&#x27;s a directory, the latest
  trained model is returned.
  

**Raises**:

  ModelNotFound Exception: When no model could be found at the provided path.

#### get\_latest\_model

```python
get_latest_model(model_path: Text = DEFAULT_MODELS_PATH) -> Optional[Text]
```

Get the latest model from a path.

**Arguments**:

- `model_path` - Path to a directory containing zipped models.
  

**Returns**:

  Path to latest model in the given directory.

#### get\_model\_for\_finetuning

```python
get_model_for_finetuning(previous_model_file: Union[Path, Text]) -> Optional[Path]
```

Gets validated path for model to finetune.

**Arguments**:

- `previous_model_file` - Path to model file which should be used for finetuning or
  a directory in case the latest trained model should be used.
  

**Returns**:

  Path to model archive. `None` if there is no model.

#### project\_fingerprint

```python
project_fingerprint() -> Optional[Text]
```

Create a hash for the project in the current working directory.

**Returns**:

  project hash


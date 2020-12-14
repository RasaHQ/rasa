---
sidebar_label: model
title: rasa.model
---

## Section Objects

```python
class Section(NamedTuple)
```

Defines relevant fingerprint sections which are used to decide whether a model
should be retrained.

## FingerprintComparisonResult Objects

```python
class FingerprintComparisonResult()
```

#### \_\_init\_\_

```python
 | __init__(nlu: bool = True, core: bool = True, nlg: bool = True, force_training: bool = False)
```

Creates a `FingerprintComparisonResult` instance.

**Arguments**:

- `nlu` - `True` if the NLU model should be retrained.
- `core` - `True` if the Core model should be retrained.
- `nlg` - `True` if the responses in the domain should be updated.
- `force_training` - `True` if a training of all parts is forced.

#### is\_training\_required

```python
 | is_training_required() -> bool
```

Check if anything has to be retrained.

#### should\_retrain\_core

```python
 | should_retrain_core() -> bool
```

Check if the Core model has to be updated.

#### should\_retrain\_nlg

```python
 | should_retrain_nlg() -> bool
```

Check if the responses have to be updated.

#### should\_retrain\_nlu

```python
 | should_retrain_nlu() -> bool
```

Check if the NLU model has to be updated.

#### get\_model

```python
get_model(model_path: Text = DEFAULT_MODELS_PATH) -> TempDirectoryPath
```

Get a model and unpack it. Raises a `ModelNotFound` exception if
no model could be found at the provided path.

**Arguments**:

- `model_path` - Path to the zipped model. If it&#x27;s a directory, the latest
  trained model is returned.
  

**Returns**:

  Path to the unpacked model.

#### get\_latest\_model

```python
get_latest_model(model_path: Text = DEFAULT_MODELS_PATH) -> Optional[Text]
```

Get the latest model from a path.

**Arguments**:

- `model_path` - Path to a directory containing zipped models.
  

**Returns**:

  Path to latest model in the given directory.

#### unpack\_model

```python
unpack_model(model_file: Text, working_directory: Optional[Union[Path, Text]] = None) -> TempDirectoryPath
```

Unpack a zipped Rasa model.

**Arguments**:

- `model_file` - Path to zipped model.
- `working_directory` - Location where the model should be unpacked to.
  If `None` a temporary directory will be created.
  

**Returns**:

  Path to unpacked Rasa model.

#### get\_model\_subdirectories

```python
get_model_subdirectories(unpacked_model_path: Text) -> Tuple[Optional[Text], Optional[Text]]
```

Return paths for Core and NLU model directories, if they exist.
If neither directories exist, a `ModelNotFound` exception is raised.

**Arguments**:

- `unpacked_model_path` - Path to unpacked Rasa model.
  

**Returns**:

  Tuple (path to Core subdirectory if it exists or `None` otherwise,
  path to NLU subdirectory if it exists or `None` otherwise).

#### create\_package\_rasa

```python
create_package_rasa(training_directory: Text, output_filename: Text, fingerprint: Optional[Fingerprint] = None) -> Text
```

Create a zipped Rasa model from trained model files.

**Arguments**:

- `training_directory` - Path to the directory which contains the trained
  model files.
- `output_filename` - Name of the zipped model file to be created.
- `fingerprint` - A unique fingerprint to identify the model version.
  

**Returns**:

  Path to zipped model.

#### project\_fingerprint

```python
project_fingerprint() -> Optional[Text]
```

Create a hash for the project in the current working directory.

**Returns**:

  project hash

#### model\_fingerprint

```python
async model_fingerprint(file_importer: "TrainingDataImporter") -> Fingerprint
```

Create a model fingerprint from its used configuration and training data.

**Arguments**:

- `file_importer` - File importer which provides the training data and model config.
  

**Returns**:

  The fingerprint.

#### fingerprint\_from\_path

```python
fingerprint_from_path(model_path: Text) -> Fingerprint
```

Load a persisted fingerprint.

**Arguments**:

- `model_path` - Path to directory containing the fingerprint.
  

**Returns**:

  The fingerprint or an empty dict if no fingerprint was found.

#### persist\_fingerprint

```python
persist_fingerprint(output_path: Text, fingerprint: Fingerprint)
```

Persist a model fingerprint.

**Arguments**:

- `output_path` - Directory in which the fingerprint should be saved.
- `fingerprint` - The fingerprint to be persisted.

#### did\_section\_fingerprint\_change

```python
did_section_fingerprint_change(fingerprint1: Fingerprint, fingerprint2: Fingerprint, section: Section) -> bool
```

Check whether the fingerprint of a section has changed.

#### move\_model

```python
move_model(source: Text, target: Text) -> bool
```

Move two model directories.

**Arguments**:

- `source` - The original folder which should be merged in another.
- `target` - The destination folder where it should be moved to.
  

**Returns**:

  `True` if the merge was successful, else `False`.

#### should\_retrain

```python
should_retrain(new_fingerprint: Fingerprint, old_model: Text, train_path: Union[Text, Path], force_training: bool = False) -> FingerprintComparisonResult
```

Check which components of a model should be retrained.

**Arguments**:

- `new_fingerprint` - The fingerprint of the new model to be trained.
- `old_model` - Path to the old zipped model file.
- `train_path` - Path to the directory in which the new model will be trained.
- `force_training` - Indicates if the model needs to be retrained even if the data has not changed.
  

**Returns**:

  A FingerprintComparisonResult object indicating whether Rasa Core and/or Rasa NLU needs
  to be retrained or not.

#### package\_model

```python
package_model(fingerprint: Fingerprint, output_directory: Text, train_path: Text, fixed_model_name: Optional[Text] = None, model_prefix: Text = "") -> Text
```

Compress a trained model.

**Arguments**:

- `fingerprint` - fingerprint of the model
- `output_directory` - path to the directory in which the model should be stored
- `train_path` - path to uncompressed model
- `fixed_model_name` - name of the compressed model file
- `model_prefix` - prefix of the compressed model file
  
- `Returns` - path to &#x27;tar.gz&#x27; model file

#### update\_model\_with\_new\_domain

```python
async update_model_with_new_domain(importer: "TrainingDataImporter", unpacked_model_path: Union[Path, Text]) -> None
```

Overwrites the domain of an unpacked model with a new domain.

**Arguments**:

- `importer` - Importer which provides the new domain.
- `unpacked_model_path` - Path to the unpacked model.


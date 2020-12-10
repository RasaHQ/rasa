---
sidebar_label: train
title: rasa.train
---

#### train\_async

```python
async train_async(domain: Union[Domain, Text], config: Text, training_files: Optional[Union[Text, List[Text]]], output: Text = DEFAULT_MODELS_PATH, force_training: bool = False, fixed_model_name: Optional[Text] = None, persist_nlu_training_data: bool = False, core_additional_arguments: Optional[Dict] = None, nlu_additional_arguments: Optional[Dict] = None) -> Optional[Text]
```

Trains a Rasa model (Core and NLU).

**Arguments**:

- `domain` - Path to the domain file.
- `config` - Path to the config for Core and NLU.
- `training_files` - Paths to the training data for Core and NLU.
- `output_path` - Output path.
- `force_training` - If `True` retrain model even if data has not changed.
- `fixed_model_name` - Name of model to be stored.
- `persist_nlu_training_data` - `True` if the NLU training data should be persisted
  with the model.
- `core_additional_arguments` - Additional training parameters for core training.
- `nlu_additional_arguments` - Additional training parameters forwarded to training
  method of each NLU component.
  

**Returns**:

  Path of the trained model archive.

#### train\_core\_async

```python
async train_core_async(domain: Union[Domain, Text], config: Text, stories: Text, output: Text, train_path: Optional[Text] = None, fixed_model_name: Optional[Text] = None, additional_arguments: Optional[Dict] = None) -> Optional[Text]
```

Trains a Core model.

**Arguments**:

- `domain` - Path to the domain file.
- `config` - Path to the config file for Core.
- `stories` - Path to the Core training data.
- `output` - Output path.
- `train_path` - If `None` the model will be trained in a temporary
  directory, otherwise in the provided directory.
- `fixed_model_name` - Name of model to be stored.
- `additional_arguments` - Additional training parameters.
  

**Returns**:

  If `train_path` is given it returns the path to the model archive,
  otherwise the path to the directory with the trained model files.

#### train\_nlu

```python
train_nlu(config: Text, nlu_data: Text, output: Text, train_path: Optional[Text] = None, fixed_model_name: Optional[Text] = None, persist_nlu_training_data: bool = False, additional_arguments: Optional[Dict] = None, domain: Optional[Union[Domain, Text]] = None) -> Optional[Text]
```

Trains an NLU model.

**Arguments**:

- `config` - Path to the config file for NLU.
- `nlu_data` - Path to the NLU training data.
- `output` - Output path.
- `train_path` - If `None` the model will be trained in a temporary
  directory, otherwise in the provided directory.
- `fixed_model_name` - Name of the model to be stored.
- `persist_nlu_training_data` - `True` if the NLU training data should be persisted
  with the model.
- `additional_arguments` - Additional training parameters which will be passed to
  the `train` method of each component.
- `domain` - Path to the optional domain file/Domain object.
  
  

**Returns**:

  If `train_path` is given it returns the path to the model archive,
  otherwise the path to the directory with the trained model files.


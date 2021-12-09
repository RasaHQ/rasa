---
sidebar_label: rasa.model_training
title: rasa.model_training
---
## TrainingResult Objects

```python
class TrainingResult(NamedTuple)
```

Holds information about the results of training.

#### train

```python
train(domain: Text, config: Text, training_files: Optional[Union[Text, List[Text]]], output: Text = rasa.shared.constants.DEFAULT_MODELS_PATH, dry_run: bool = False, force_training: bool = False, fixed_model_name: Optional[Text] = None, persist_nlu_training_data: bool = False, core_additional_arguments: Optional[Dict] = None, nlu_additional_arguments: Optional[Dict] = None, model_to_finetune: Optional[Text] = None, finetuning_epoch_fraction: float = 1.0) -> TrainingResult
```

Trains a Rasa model (Core and NLU).

**Arguments**:

- `domain` - Path to the domain file.
- `config` - Path to the config file.
- `training_files` - List of paths to training data files.
- `output` - Output directory for the trained model.
- `dry_run` - If `True` then no training will be done, and the information about
  whether the training needs to be done will be printed.
- `force_training` - If `True` retrain model even if data has not changed.
- `fixed_model_name` - Name of model to be stored.
- `persist_nlu_training_data` - `True` if the NLU training data should be persisted
  with the model.
- `core_additional_arguments` - Additional training parameters for core training.
- `nlu_additional_arguments` - Additional training parameters forwarded to training
  method of each NLU component.
- `model_to_finetune` - Optional path to a model which should be finetuned or
  a directory in case the latest trained model should be used.
- `finetuning_epoch_fraction` - The fraction currently specified training epochs
  in the model configuration which should be used for finetuning.
  

**Returns**:

  An instance of `TrainingResult`.

#### train\_core

```python
train_core(domain: Union[Domain, Text], config: Text, stories: Text, output: Text, fixed_model_name: Optional[Text] = None, additional_arguments: Optional[Dict] = None, model_to_finetune: Optional[Text] = None, finetuning_epoch_fraction: float = 1.0) -> Optional[Text]
```

Trains a Core model.

**Arguments**:

- `domain` - Path to the domain file.
- `config` - Path to the config file for Core.
- `stories` - Path to the Core training data.
- `output` - Output path.
- `fixed_model_name` - Name of model to be stored.
- `additional_arguments` - Additional training parameters.
- `model_to_finetune` - Optional path to a model which should be finetuned or
  a directory in case the latest trained model should be used.
- `finetuning_epoch_fraction` - The fraction currently specified training epochs
  in the model configuration which should be used for finetuning.
  

**Returns**:

  Path to the model archive.

#### train\_nlu

```python
train_nlu(config: Text, nlu_data: Optional[Text], output: Text, fixed_model_name: Optional[Text] = None, persist_nlu_training_data: bool = False, additional_arguments: Optional[Dict] = None, domain: Optional[Union[Domain, Text]] = None, model_to_finetune: Optional[Text] = None, finetuning_epoch_fraction: float = 1.0) -> Optional[Text]
```

Trains an NLU model.

**Arguments**:

- `config` - Path to the config file for NLU.
- `nlu_data` - Path to the NLU training data.
- `output` - Output path.
- `fixed_model_name` - Name of the model to be stored.
- `persist_nlu_training_data` - `True` if the NLU training data should be persisted
  with the model.
- `additional_arguments` - Additional training parameters which will be passed to
  the `train` method of each component.
- `domain` - Path to the optional domain file/Domain object.
- `model_to_finetune` - Optional path to a model which should be finetuned or
  a directory in case the latest trained model should be used.
- `finetuning_epoch_fraction` - The fraction currently specified training epochs
  in the model configuration which should be used for finetuning.
  

**Returns**:

  Path to the model archive.


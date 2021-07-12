---
sidebar_label: rasa.model_training
title: rasa.model_training
---
## TrainingResult Objects

```python
class TrainingResult(NamedTuple)
```

Holds information about the results of training.

#### train\_async

```python
async train_async(domain: Union[Domain, Text], config: Text, training_files: Optional[Union[Text, List[Text]]], output: Text = DEFAULT_MODELS_PATH, dry_run: bool = False, force_training: bool = False, fixed_model_name: Optional[Text] = None, persist_nlu_training_data: bool = False, core_additional_arguments: Optional[Dict] = None, nlu_additional_arguments: Optional[Dict] = None, model_to_finetune: Optional[Text] = None, finetuning_epoch_fraction: float = 1.0) -> TrainingResult
```

Trains a Rasa model (Core and NLU).

**Arguments**:

- `domain` - Path to the domain file.
- `config` - Path to the config for Core and NLU.
- `training_files` - Paths to the training data for Core and NLU.
- `output_path` - Output path.
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

#### handle\_domain\_if\_not\_exists

```python
async handle_domain_if_not_exists(file_importer: TrainingDataImporter, output_path: Text, fixed_model_name: Optional[Text]) -> Text
```

Trains only the nlu model and prints a warning about missing domain.

#### dry\_run\_result

```python
dry_run_result(fingerprint_comparison: FingerprintComparisonResult) -> Tuple[int, List[Text]]
```

Returns a dry run result.

**Arguments**:

- `fingerprint_comparison` - A result of fingerprint comparison operation.
  

**Returns**:

  A tuple where the first element is the result code and the second
  is the list of human-readable texts that need to be printed to the end user.

#### train\_core\_async

```python
async train_core_async(domain: Union[Domain, Text], config: Text, stories: Text, output: Text, train_path: Optional[Text] = None, fixed_model_name: Optional[Text] = None, additional_arguments: Optional[Dict] = None, model_to_finetune: Optional[Text] = None, finetuning_epoch_fraction: float = 1.0) -> Optional[Text]
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
- `model_to_finetune` - Optional path to a model which should be finetuned or
  a directory in case the latest trained model should be used.
- `finetuning_epoch_fraction` - The fraction currently specified training epochs
  in the model configuration which should be used for finetuning.
  

**Returns**:

  If `train_path` is given it returns the path to the model archive,
  otherwise the path to the directory with the trained model files.

#### train\_nlu

```python
train_nlu(config: Text, nlu_data: Text, output: Text, train_path: Optional[Text] = None, fixed_model_name: Optional[Text] = None, persist_nlu_training_data: bool = False, additional_arguments: Optional[Dict] = None, domain: Optional[Union[Domain, Text]] = None, model_to_finetune: Optional[Text] = None, finetuning_epoch_fraction: float = 1.0) -> Optional[Text]
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
- `model_to_finetune` - Optional path to a model which should be finetuned or
  a directory in case the latest trained model should be used.
- `finetuning_epoch_fraction` - The fraction currently specified training epochs
  in the model configuration which should be used for finetuning.
  

**Returns**:

  If `train_path` is given it returns the path to the model archive,
  otherwise the path to the directory with the trained model files.

#### train\_nlu\_async

```python
async train_nlu_async(config: Text, nlu_data: Text, output: Text, train_path: Optional[Text] = None, fixed_model_name: Optional[Text] = None, persist_nlu_training_data: bool = False, additional_arguments: Optional[Dict] = None, domain: Optional[Union[Domain, Text]] = None, model_to_finetune: Optional[Text] = None, finetuning_epoch_fraction: float = 1.0) -> Optional[Text]
```

Trains an NLU model asynchronously.


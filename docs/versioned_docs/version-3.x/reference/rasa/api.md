---
sidebar_label: rasa.api
title: rasa.api
---
#### run

```python
run(model: "Text", endpoints: "Text", connector: "Text" = None, credentials: "Text" = None, **kwargs: "Dict[Text, Any]", ,) -> "NoReturn"
```

Runs a Rasa model.

**Arguments**:

- `model` - Path to model archive.
- `endpoints` - Path to endpoints file.
- `connector` - Connector which should be use (overwrites `credentials`
  field).
- `credentials` - Path to channel credentials file.
- `**kwargs` - Additional arguments which are passed to
  `rasa.core.run.serve_application`.

#### train

```python
train(domain: "Text", config: "Text", training_files: "Union[Text, List[Text]]", output: "Text" = rasa.shared.constants.DEFAULT_MODELS_PATH, dry_run: bool = False, force_training: bool = False, fixed_model_name: "Optional[Text]" = None, persist_nlu_training_data: bool = False, core_additional_arguments: "Optional[Dict]" = None, nlu_additional_arguments: "Optional[Dict]" = None, model_to_finetune: "Optional[Text]" = None, finetuning_epoch_fraction: float = 1.0) -> "TrainingResult"
```

Runs Rasa Core and NLU training in `async` loop.

**Arguments**:

- `domain` - Path to the domain file.
- `config` - Path to the config for Core and NLU.
- `training_files` - Paths to the training data for Core and NLU.
- `output` - Output path.
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

#### test

```python
test(model: "Text", stories: "Text", nlu_data: "Text", output: "Text" = rasa.shared.constants.DEFAULT_RESULTS_PATH, additional_arguments: "Optional[Dict]" = None) -> None
```

Test a Rasa model against a set of test data.

**Arguments**:

- `model` - model to test
- `stories` - path to the dialogue test data
- `nlu_data` - path to the NLU test data
- `output` - path to folder where all output will be stored
- `additional_arguments` - additional arguments for the test call


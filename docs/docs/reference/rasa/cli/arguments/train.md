---
sidebar_label: rasa.cli.arguments.train
title: rasa.cli.arguments.train
---
#### set\_train\_arguments

```python
def set_train_arguments(parser: argparse.ArgumentParser) -> None
```

Specifies CLI arguments for `rasa train`.

#### set\_train\_core\_arguments

```python
def set_train_core_arguments(parser: argparse.ArgumentParser) -> None
```

Specifies CLI arguments for `rasa train core`.

#### set\_train\_nlu\_arguments

```python
def set_train_nlu_arguments(parser: argparse.ArgumentParser) -> None
```

Specifies CLI arguments for `rasa train nlu`.

#### add\_force\_param

```python
def add_force_param(parser: Union[argparse.ArgumentParser, argparse._ActionsContainer]) -> None
```

Specifies if the model should be trained from scratch.

#### add\_data\_param

```python
def add_data_param(parser: Union[argparse.ArgumentParser, argparse._ActionsContainer]) -> None
```

Specifies path to training data.

#### add\_dry\_run\_param

```python
def add_dry_run_param(parser: Union[argparse.ArgumentParser, argparse._ActionsContainer]) -> None
```

Adds `--dry-run` argument to a specified `parser`.

**Arguments**:

- `parser` - An instance of `ArgumentParser` or `_ActionsContainer`.

#### add\_augmentation\_param

```python
def add_augmentation_param(parser: Union[argparse.ArgumentParser, argparse._ActionsContainer]) -> None
```

Sets the augmentation factor for the Core training.

**Arguments**:

- `parser` - An instance of `ArgumentParser` or `_ActionsContainer`.

#### add\_debug\_plots\_param

```python
def add_debug_plots_param(parser: Union[argparse.ArgumentParser, argparse._ActionsContainer]) -> None
```

Specifies if conversation flow should be visualized.

#### add\_persist\_nlu\_data\_param

```python
def add_persist_nlu_data_param(parser: Union[argparse.ArgumentParser, argparse._ActionsContainer]) -> None
```

Adds parameters for persisting the NLU training data with the model.

#### add\_finetune\_params

```python
def add_finetune_params(parser: Union[argparse.ArgumentParser, argparse._ActionsContainer]) -> None
```

Adds parameters for model finetuning.


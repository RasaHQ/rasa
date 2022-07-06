---
sidebar_label: rasa.cli.train
title: rasa.cli.train
---
#### add\_subparser

```python
def add_subparser(subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]) -> None
```

Add all training parsers.

**Arguments**:

- `subparsers` - subparser we are going to attach to
- `parents` - Parent parsers, needed to ensure tree structure in argparse

#### run\_training

```python
def run_training(args: argparse.Namespace, can_exit: bool = False) -> Optional[Text]
```

Trains a model.

**Arguments**:

- `args` - Namespace arguments.
- `can_exit` - If `True`, the operation can send `sys.exit` in the case
  training was not successful.
  

**Returns**:

  Path to a trained model or `None` if training was not successful.

#### run\_core\_training

```python
def run_core_training(args: argparse.Namespace) -> Optional[Text]
```

Trains a Rasa Core model only.

**Arguments**:

- `args` - Command-line arguments to configure training.
  

**Returns**:

  Path to a trained model or `None` if training was not successful.

#### run\_nlu\_training

```python
def run_nlu_training(args: argparse.Namespace) -> Optional[Text]
```

Trains an NLU model.

**Arguments**:

- `args` - Namespace arguments.
  

**Returns**:

  Path to a trained model or `None` if training was not successful.


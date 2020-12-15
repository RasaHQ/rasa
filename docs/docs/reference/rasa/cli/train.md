---
sidebar_label: train
title: rasa.cli.train
---

#### add\_subparser

```python
add_subparser(subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]) -> None
```

Add all training parsers.

**Arguments**:

- `subparsers` - subparser we are going to attach to
- `parents` - Parent parsers, needed to ensure tree structure in argparse

#### train

```python
train(args: argparse.Namespace, can_exit: bool = False) -> Optional[Text]
```

Trains a model.

**Arguments**:

- `args` - Namespace arguments.
- `can_exit` - If `True`, the operation can send `sys.exit` in the case
  training was not successful.
  

**Returns**:

  Path to a trained model or `None` if training was not successful.

#### train\_core

```python
train_core(args: argparse.Namespace, train_path: Optional[Text] = None) -> Optional[Text]
```

Trains a Rasa Core model only.

**Arguments**:

- `args` - Command-line arguments to configure training.
- `train_path` - Path where trained model but not unzipped model should be stored.
  

**Returns**:

  Path to a trained model or `None` if training was not successful.

#### train\_nlu

```python
train_nlu(args: argparse.Namespace, train_path: Optional[Text] = None) -> Optional[Text]
```

Trains an NLU model.

**Arguments**:

- `args` - Namespace arguments.
- `train_path` - Directory where models should be stored.
  

**Returns**:

  Path to a trained model or `None` if training was not successful.


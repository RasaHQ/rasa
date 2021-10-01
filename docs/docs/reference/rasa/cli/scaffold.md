---
sidebar_label: rasa.cli.scaffold
title: rasa.cli.scaffold
---
#### add\_subparser

```python
def add_subparser(subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]) -> None
```

Add all init parsers.

**Arguments**:

- `subparsers` - subparser we are going to attach to
- `parents` - Parent parsers, needed to ensure tree structure in argparse

#### print\_train\_or\_instructions

```python
def print_train_or_instructions(args: argparse.Namespace, path: Text) -> None
```

Train a model if the user wants to.


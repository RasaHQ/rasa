---
sidebar_label: rasa.cli.scaffold
title: rasa.cli.scaffold
---
#### add\_subparser

```python
add_subparser(subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]) -> None
```

Add all init parsers.

**Arguments**:

- `subparsers` - subparser we are going to attach to
- `parents` - Parent parsers, needed to ensure tree structure in argparse

#### print\_train\_or\_instructions

```python
print_train_or_instructions(args: argparse.Namespace) -> None
```

Train a model if the user wants to.

#### init\_project

```python
init_project(args: argparse.Namespace, path: Text) -> None
```

Inits project.

#### create\_initial\_project

```python
create_initial_project(path: Text) -> None
```

Creates directory structure and templates for initial project.


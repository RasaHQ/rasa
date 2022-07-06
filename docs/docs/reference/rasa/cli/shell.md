---
sidebar_label: rasa.cli.shell
title: rasa.cli.shell
---
#### add\_subparser

```python
def add_subparser(subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]) -> None
```

Add all shell parsers.

**Arguments**:

- `subparsers` - subparser we are going to attach to
- `parents` - Parent parsers, needed to ensure tree structure in argparse

#### shell\_nlu

```python
def shell_nlu(args: argparse.Namespace) -> None
```

Talk with an NLU only bot though the command line.

#### shell

```python
def shell(args: argparse.Namespace) -> None
```

Talk with a bot though the command line.


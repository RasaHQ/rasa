---
sidebar_label: rasa.cli.data
title: rasa.cli.data
---
#### add\_subparser

```python
def add_subparser(subparsers: SubParsersAction,
                  parents: List[argparse.ArgumentParser]) -> None
```

Add all data parsers.

**Arguments**:

- `subparsers` - subparser we are going to attach to
- `parents` - Parent parsers, needed to ensure tree structure in argparse

#### split\_nlu\_data

```python
def split_nlu_data(args: argparse.Namespace) -> None
```

Load data from a file path and split the NLU data into test and train examples.

**Arguments**:

- `args` - Commandline arguments

#### split\_stories\_data

```python
def split_stories_data(args: argparse.Namespace) -> None
```

Load data from a file path and split stories into test and train examples.

**Arguments**:

- `args` - Commandline arguments


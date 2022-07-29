---
sidebar_label: rasa.cli.x
title: rasa.cli.x
---
#### add\_subparser

```python
def add_subparser(subparsers: SubParsersAction,
                  parents: List[argparse.ArgumentParser]) -> None
```

Add all rasa x parsers.

**Arguments**:

- `subparsers` - subparser we are going to attach to
- `parents` - Parent parsers, needed to ensure tree structure in argparse

#### rasa\_x

```python
def rasa_x(args: argparse.Namespace) -> None
```

Run Rasa with the `x` subcommand.

#### run\_in\_enterprise\_connection\_mode

```python
def run_in_enterprise_connection_mode(args: argparse.Namespace) -> None
```

Run Rasa in a mode that enables using Rasa X as the config endpoint.


---
sidebar_label: rasa.cli.export
title: rasa.cli.export
---
#### add\_subparser

```python
def add_subparser(subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]) -> None
```

Add subparser for `rasa export`.

**Arguments**:

- `subparsers` - Subparsers action object to which `argparse.ArgumentParser`
  objects can be added.
- `parents` - `argparse.ArgumentParser` objects whose arguments should also be
  included.

#### export\_trackers

```python
def export_trackers(args: argparse.Namespace) -> None
```

Export events for a connected tracker store using an event broker.

**Arguments**:

- `args` - Command-line arguments to process.


---
sidebar_label: x
title: rasa.cli.x
---

#### add\_subparser

```python
add_subparser(subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]) -> None
```

Add all rasa x parsers.

**Arguments**:

- `subparsers` - subparser we are going to attach to
- `parents` - Parent parsers, needed to ensure tree structure in argparse

#### start\_rasa\_for\_local\_rasa\_x

```python
start_rasa_for_local_rasa_x(args: argparse.Namespace, rasa_x_token: Text)
```

Starts the Rasa X API with Rasa as a background process.

#### is\_rasa\_x\_installed

```python
is_rasa_x_installed() -> bool
```

Check if Rasa X is installed.

#### generate\_rasa\_x\_token

```python
generate_rasa_x_token(length: int = 16)
```

Generate a hexadecimal secret token used to access the Rasa X API.

A new token is generated on every `rasa x` command.

#### run\_locally

```python
run_locally(args: argparse.Namespace) -> None
```

Run a Rasa X instance locally.

**Arguments**:

- `args` - commandline arguments


---
sidebar_label: rasa.cli.arguments.run
title: rasa.cli.arguments.run
---
#### set\_run\_arguments

```python
def set_run_arguments(parser: argparse.ArgumentParser) -> None
```

Arguments for running Rasa directly using `rasa run`.

#### set\_run\_action\_arguments

```python
def set_run_action_arguments(parser: argparse.ArgumentParser) -> None
```

Set arguments for running Rasa SDK.

#### add\_interface\_argument

```python
def add_interface_argument(parser: Union[argparse.ArgumentParser, argparse._ArgumentGroup]) -> None
```

Binds the RASA process to a network interface.

#### add\_port\_argument

```python
def add_port_argument(parser: Union[argparse.ArgumentParser, argparse._ArgumentGroup]) -> None
```

Add an argument for port.

#### add\_server\_arguments

```python
def add_server_arguments(parser: argparse.ArgumentParser) -> None
```

Add arguments for running API endpoint.


---
sidebar_label: rasa.shared.utils.cli
title: rasa.shared.utils.cli
---
#### print\_color

```python
def print_color(*args: Any, color: Text) -> None
```

Print the given arguments to STDOUT in the specified color.

**Arguments**:

- `args` - A list of objects to be printed.
- `color` - A textual representation of the color.

#### print\_success

```python
def print_success(*args: Any) -> None
```

Print the given arguments to STDOUT in green, indicating success.

**Arguments**:

- `args` - A list of objects to be printed.

#### print\_info

```python
def print_info(*args: Any) -> None
```

Print the given arguments to STDOUT in blue.

**Arguments**:

- `args` - A list of objects to be printed.

#### print\_warning

```python
def print_warning(*args: Any) -> None
```

Print the given arguments to STDOUT in a color indicating a warning.

**Arguments**:

- `args` - A list of objects to be printed.

#### print\_error

```python
def print_error(*args: Any) -> None
```

Print the given arguments to STDOUT in a color indicating an error.

**Arguments**:

- `args` - A list of objects to be printed.

#### print\_error\_and\_exit

```python
def print_error_and_exit(message: Text, exit_code: int = 1) -> NoReturn
```

Print an error message and exit the application.

**Arguments**:

- `message` - The error message to be printed.
- `exit_code` - The program exit code, defaults to 1.


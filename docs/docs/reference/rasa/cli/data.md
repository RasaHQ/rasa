---
sidebar_label: rasa.cli.data
title: rasa.cli.data
---

#### split\_nlu\_data

```python
split_nlu_data(args: argparse.Namespace) -> None
```

Load data from a file path and split the NLU data into test and train examples.

**Arguments**:

- `args` - Commandline arguments

#### validate\_files

```python
validate_files(args: argparse.Namespace, stories_only: bool = False) -> None
```

Validates either the story structure or the entire project.

**Arguments**:

- `args` - Commandline arguments
- `stories_only` - If `True`, only the story structure is validated.

#### validate\_stories

```python
validate_stories(args: argparse.Namespace) -> None
```

Validates that training data file content conforms to training data spec.

**Arguments**:

- `args` - Commandline arguments


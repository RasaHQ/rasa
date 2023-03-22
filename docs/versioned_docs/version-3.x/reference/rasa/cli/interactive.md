---
sidebar_label: rasa.cli.interactive
title: rasa.cli.interactive
---
#### add\_subparser

```python
def add_subparser(subparsers: SubParsersAction,
                  parents: List[argparse.ArgumentParser]) -> None
```

Add all interactive cli parsers.

**Arguments**:

- `subparsers` - subparser we are going to attach to
- `parents` - Parent parsers, needed to ensure tree structure in argparse

#### perform\_interactive\_learning

```python
def perform_interactive_learning(args: argparse.Namespace,
                                 zipped_model: Union[Text, "Path"],
                                 file_importer: TrainingDataImporter) -> None
```

Performs interactive learning.

**Arguments**:

- `args` - Namespace arguments.
- `zipped_model` - Path to zipped model.
- `file_importer` - File importer which provides the training data and model config.

#### get\_provided\_model

```python
def get_provided_model(arg_model: Text) -> Optional[Union[Text, Path]]
```

Checks model path input and selects model from it.

#### validate\_assistant\_id\_key\_in\_config

```python
def validate_assistant_id_key_in_config(
        file_importer: TrainingDataImporter) -> None
```

Verifies that config contains a unique value for assistant identifier.


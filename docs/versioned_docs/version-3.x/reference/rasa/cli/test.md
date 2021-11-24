---
sidebar_label: rasa.cli.test
title: rasa.cli.test
---
#### add\_subparser

```python
add_subparser(subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]) -> None
```

Add all test parsers.

**Arguments**:

- `subparsers` - subparser we are going to attach to
- `parents` - Parent parsers, needed to ensure tree structure in argparse

#### run\_core\_test\_async

```python
async run_core_test_async(args: argparse.Namespace) -> None
```

Run core tests.

#### run\_nlu\_test\_async

```python
async run_nlu_test_async(config: Optional[Union[Text, List[Text]]], data_path: Text, models_path: Text, output_dir: Text, cross_validation: bool, percentages: List[int], runs: int, no_errors: bool, all_args: Dict[Text, Any]) -> None
```

Runs NLU tests.

**Arguments**:

- `all_args` - all arguments gathered in a Dict so we can pass it as one argument
  to other functions.
- `config` - it refers to the model configuration file. It can be a single file or
  a list of multiple files or a folder with multiple config files inside.
- `data_path` - path for the nlu data.
- `models_path` - path to a trained Rasa model.
- `output_dir` - output path for any files created during the evaluation.
- `cross_validation` - indicates if it should test the model using cross validation
  or not.
- `percentages` - defines the exclusion percentage of the training data.
- `runs` - number of comparison runs to make.
- `no_errors` - indicates if incorrect predictions should be written to a file
  or not.

#### run\_nlu\_test

```python
run_nlu_test(args: argparse.Namespace) -> None
```

Runs NLU tests.

**Arguments**:

- `args` - the parsed CLI arguments for &#x27;rasa test nlu&#x27;.

#### run\_core\_test

```python
run_core_test(args: argparse.Namespace) -> None
```

Runs Core tests.

**Arguments**:

- `args` - the parsed CLI arguments for &#x27;rasa test core&#x27;.

#### test

```python
test(args: argparse.Namespace) -> None
```

Run end-to-end tests.


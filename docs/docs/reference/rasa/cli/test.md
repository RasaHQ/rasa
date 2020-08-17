---
sidebar_label: rasa.cli.test
title: rasa.cli.test
---

#### add\_subparser

```python
add_subparser(subparsers: argparse._SubParsersAction, parents: List[argparse.ArgumentParser])
```

Adds a test subparser.

#### run\_core\_test

```python
run_core_test(args: argparse.Namespace) -> None
```

Run core tests.

#### run\_nlu\_test

```python
run_nlu_test(args: argparse.Namespace) -> None
```

Run NLU tests.

#### test

```python
test(args: argparse.Namespace)
```

Run end-to-end tests.


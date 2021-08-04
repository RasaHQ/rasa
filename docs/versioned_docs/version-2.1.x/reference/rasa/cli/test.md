---
sidebar_label: test
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


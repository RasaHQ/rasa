---
sidebar_label: rasa.core.evaluation.marker_tracker_loader
title: rasa.core.evaluation.marker_tracker_loader
---
#### strategy\_all

```python
def strategy_all(keys: List[Text], count: int) -> Iterable[Text]
```

Selects all keys from the set of keys.

#### strategy\_first\_n

```python
def strategy_first_n(keys: List[Text], count: int) -> Iterable[Text]
```

Takes the first N keys from the set of keys.

#### strategy\_sample

```python
def strategy_sample(keys: List[Text], count: int) -> Iterable[Text]
```

Samples N unique keys from the set of keys.

## MarkerTrackerLoader Objects

```python
class MarkerTrackerLoader()
```

Represents a wrapper over a `TrackerStore` with a configurable access pattern.

#### \_\_init\_\_

```python
def __init__(tracker_store: TrackerStore, strategy: str, count: int = None, seed: Any = None) -> None
```

Creates a MarkerTrackerLoader.

**Arguments**:

- `tracker_store` - The underlying tracker store to access.
- `strategy` - The strategy to use for selecting trackers,
  can be &#x27;all&#x27;, &#x27;sample&#x27;, or &#x27;first_n&#x27;.
- `count` - Number of trackers to return, can only be None if strategy is &#x27;all&#x27;.
- `seed` - Optional seed to set up random number generator,
  only useful if strategy is &#x27;sample&#x27;.

#### load

```python
def load() -> Iterator[Optional[DialogueStateTracker]]
```

Loads trackers according to strategy.


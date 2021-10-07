---
sidebar_label: rasa.core.evaluation.marker_stats
title: rasa.core.evaluation.marker_stats
---
## MarkerStats Objects

```python
class MarkerStats(TypedDict)
```

A TypedDict for statistics computed over extracted markers.

#### load\_extracted\_markers\_json\_file

```python
def load_extracted_markers_json_file(path: Union[Text, Path]) -> List
```

Reads a json marker file.

**Arguments**:

- `path` - path to a json file.

#### compute\_summary\_stats

```python
def compute_summary_stats(data_points: Union[List, np.ndarray]) -> MarkerStats
```

Computes summary statistics for a given array.

Computes size, mean, median, min, and max.
If size is == 0 returns np.nan for mean, median.

**Arguments**:

- `data_points` - can be a numpy array or a list of numbers.

#### compute\_single\_tracker\_stats

```python
def compute_single_tracker_stats(single_tracker_markers: Dict[str, Any]) -> Dict[str, MarkerStats]
```

Computes summary statistics for a single tracker.

#### compute\_multi\_tracker\_stats

```python
def compute_multi_tracker_stats(multi_tracker_markers: list) -> Tuple[Dict, Dict[Any, Dict[str, MarkerStats]]]
```

Computes summary statistics for multiple trackers.

#### write\_stats

```python
def write_stats(path: Union[Text, Path], stats: dict, per_tracker_stats: dict) -> None
```

Outputs statistics to JSON file.

#### np\_encoder

```python
def np_encoder(obj: Any) -> Any
```

Encodes numpy array values to make them JSON serializable.

adapted from: https://bit.ly/3ajjTwp


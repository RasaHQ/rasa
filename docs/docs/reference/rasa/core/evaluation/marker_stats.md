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
load_extracted_markers_json_file(path: Union[Text, Path]) -> List
```

Reads a json marker file.

**Arguments**:

- `path` - path to a json file.

#### compute\_summary\_stats

```python
compute_summary_stats(data_points: Union[List[float], np.ndarray]) -> MarkerStats
```

Computes summary statistics for a given array.

**Arguments**:

- `data_points` - can be a numpy array or a list of numbers.
  

**Returns**:

  A MarkerStats object containing size, mean, median, min, and max.
  If the given array of data points is empty, it returns 0 for size, and
  `np.nan` for every statistic.

#### compute\_single\_tracker\_stats

```python
compute_single_tracker_stats(single_tracker_markers: Dict[str, Any]) -> Dict[str, MarkerStats]
```

Computes summary statistics for a single tracker.

**Arguments**:

- `single_tracker_markers` - a dictionary containing the extracted
  markers for one tracker.
  

**Returns**:

  A dictionary containing statistics computed for each marker.

#### compute\_multi\_tracker\_stats

```python
compute_multi_tracker_stats(multi_tracker_markers: List[Dict[str, Any]]) -> Tuple[Dict[str, Union[int, MarkerStats]], Dict[Any, Dict[str, MarkerStats]]]
```

Computes summary statistics for multiple trackers.

**Arguments**:

- `multi_tracker_markers` - a list of dictionaries each containing the
  extracted markers for one tracker.
  

**Returns**:

  A dictionary containing summary statistics computed per
  marker over all trackers.
  A dictionary containing summary statistics computed
  per tracker.

#### write\_stats

```python
write_stats(path: Union[Text, Path], stats: dict, per_tracker_stats: dict) -> None
```

Outputs statistics to JSON file.

#### np\_encoder

```python
np_encoder(obj: Any) -> Any
```

Encodes numpy array values to make them JSON serializable.

adapted from: https://bit.ly/3ajjTwp


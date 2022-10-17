---
sidebar_label: rasa.core.evaluation.marker_stats
title: rasa.core.evaluation.marker_stats
---
#### compute\_statistics

```python
compute_statistics(values: List[Union[float, int]]) -> Dict[Text, Union[int, np.float]]
```

Computes some statistics over the given numbers.

## MarkerStatistics Objects

```python
class MarkerStatistics()
```

Computes some statistics on marker extraction results.

(1) Number of sessions where markers apply:

For each marker, we compute the total number (as well as the percentage) of
sessions in which a marker applies at least once.
Moreover, we output the total number of sessions that were parsed.

(2) Number of user turns preceding relevant events - per sessions:

For each marker, we consider all relevant events where that marker applies.
Everytime a marker applies, we check how many user turns precede that event.
We collect all these numbers and compute basic statistics (e.g. count and mean)
on them.

This means, per session, we compute how often a marker applies and how many
user turns precede a relevant marker application on average, in that session.

(3) Number of user turns preceding relevant events - over all sessions:

Here, for each marker, we consider all relevant events where a marker applies
*in any of the sessions*. Then, we again calculate basic statistics over the
respective number of user turns that precede each of these events.

This means, we compute how many events the marker applies in total and we
compute an estimate of the expected number of user turns preceding that
precede an (relevant) event where a marker applies.

#### \_\_init\_\_

```python
 | __init__() -> None
```

Creates a new marker statistics object.

#### process

```python
 | process(sender_id: Text, session_idx: int, meta_data_on_relevant_events_per_marker: Dict[Text, List[EventMetaData]]) -> None
```

Processes the meta data that was extracted from a single session.

Internally, this method ..
1. computes some statistics for the given meta data and saves it for later
2. keeps track of the total number of sessions processed and the
collects all metadata to be able to compute meta data over *all*

**Arguments**:

- `sender_id` - an id that, together with the `session_idx` identifies
  the session from which the markers where extracted
- `session_idx` - an index that, together with the `sender_id` identifies
  the session from which the markers where extracted
- `meta_data_on_relevant_events_per_marker` - marker extraction results,
  i.e. a dictionary mapping
  marker names to the meta data describing relevant events
  for those markers

#### overall\_statistic\_to\_csv

```python
 | overall_statistic_to_csv(path: Path, overwrite: bool = False) -> None
```

Exports the overall statistics (over all processes sessions) to a csv file.

**Arguments**:

- `path` - path to where the csv file should be written.
- `overwrite` - set to `True` to enable overwriting an existing file

#### per\_session\_statistics\_to\_csv

```python
 | per_session_statistics_to_csv(path: Path, overwrite: bool = False) -> None
```

Exports the resulting statistics to a csv file.

**Arguments**:

- `path` - path to where the csv file should be written.
- `overwrite` - set to `True` to enable overwriting an existing file


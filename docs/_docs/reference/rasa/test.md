---
sidebar_label: rasa.test
title: rasa.test
---
#### plot\_core\_results

```python
plot_core_results(output_directory: Text, number_of_examples: List[int]) -> None
```

Plot core model comparison graph.

**Arguments**:

- `output_directory` - path to the output directory
- `number_of_examples` - number of examples per run

#### test\_core

```python
test_core(model: Optional[Text] = None, stories: Optional[Text] = None, output: Text = DEFAULT_RESULTS_PATH, additional_arguments: Optional[Dict] = None) -> None
```

Tests a trained Core model against a set of test stories.

#### test\_nlu

```python
async test_nlu(model: Optional[Text], nlu_data: Optional[Text], output_directory: Text = DEFAULT_RESULTS_PATH, additional_arguments: Optional[Dict] = None)
```

Tests the NLU Model.

#### compare\_nlu\_models

```python
async compare_nlu_models(configs: List[Text], nlu: Text, output: Text, runs: int, exclusion_percentages: List[int])
```

Trains multiple models, compares them and saves the results.

#### plot\_nlu\_results

```python
plot_nlu_results(output_directory: Text, number_of_examples: List[int]) -> None
```

Plot NLU model comparison graph.

**Arguments**:

- `output_directory` - path to the output directory
- `number_of_examples` - number of examples per run

#### get\_evaluation\_metrics

```python
get_evaluation_metrics(targets: Iterable[Any], predictions: Iterable[Any], output_dict: bool = False, exclude_label: Optional[Text] = None) -> Tuple[Union[Text, Dict[Text, Dict[Text, float]]], float, float, float]
```

Compute the f1, precision, accuracy and summary report from sklearn.

**Arguments**:

- `targets` - target labels
- `predictions` - predicted labels
- `output_dict` - if True sklearn returns a summary report as dict, if False the
  report is in string format
- `exclude_label` - labels to exclude from evaluation
  

**Returns**:

  Report from sklearn, precision, f1, and accuracy values.

#### clean\_labels

```python
clean_labels(labels: Iterable[Text]) -> List[Text]
```

Remove `None` labels. sklearn metrics do not support them.

**Arguments**:

- `labels` - list of labels
  

**Returns**:

  Cleaned labels.

#### get\_unique\_labels

```python
get_unique_labels(targets: Iterable[Text], exclude_label: Optional[Text]) -> List[Text]
```

Get unique labels. Exclude &#x27;exclude_label&#x27; if specified.

**Arguments**:

- `targets` - labels
- `exclude_label` - label to exclude
  

**Returns**:

  Unique labels.


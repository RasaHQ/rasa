---
sidebar_label: rasa.model_testing
title: rasa.model_testing
---
#### test\_core\_models\_in\_directory

```python
async test_core_models_in_directory(model_directory: Text, stories: Text, output: Text, use_conversation_test_files: bool = False) -> None
```

Evaluates a directory with multiple Core models using test data.

**Arguments**:

- `model_directory` - Directory containing multiple model files.
- `stories` - Path to a conversation test file.
- `output` - Output directory to store results to.
- `use_conversation_test_files` - `True` if conversation test files should be used
  for testing instead of regular Core story files.

#### plot\_core\_results

```python
plot_core_results(output_directory: Text, number_of_examples: List[int]) -> None
```

Plot core model comparison graph.

**Arguments**:

- `output_directory` - path to the output directory
- `number_of_examples` - number of examples per run

#### test\_core\_models

```python
async test_core_models(models: List[Text], stories: Text, output: Text, use_conversation_test_files: bool = False) -> None
```

Compares multiple Core models based on test data.

**Arguments**:

- `models` - A list of models files.
- `stories` - Path to test data.
- `output` - Path to output directory for test results.
- `use_conversation_test_files` - `True` if conversation test files should be used
  for testing instead of regular Core story files.

#### test\_core

```python
async test_core(model: Optional[Text] = None, stories: Optional[Text] = None, output: Text = DEFAULT_RESULTS_PATH, additional_arguments: Optional[Dict] = None, use_conversation_test_files: bool = False) -> None
```

Tests a trained Core model against a set of test stories.

#### test\_nlu

```python
async test_nlu(model: Optional[Text], nlu_data: Optional[Text], output_directory: Text = DEFAULT_RESULTS_PATH, additional_arguments: Optional[Dict] = None) -> None
```

Tests the NLU Model.

#### compare\_nlu\_models

```python
async compare_nlu_models(configs: List[Text], test_data: TrainingData, output: Text, runs: int, exclusion_percentages: List[int]) -> None
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

#### perform\_nlu\_cross\_validation

```python
async perform_nlu_cross_validation(config: Dict[Text, Any], data: TrainingData, output: Text, additional_arguments: Optional[Dict[Text, Any]]) -> None
```

Runs cross-validation on test data.

**Arguments**:

- `config` - The model configuration.
- `data` - The data which is used for the cross-validation.
- `output` - Output directory for the cross-validation results.
- `additional_arguments` - Additional arguments which are passed to the
  cross-validation, like number of `disable_plotting`.

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


---
sidebar_label: rasa.nlu.test
title: rasa.nlu.test
---
## CVEvaluationResult Objects

```python
class CVEvaluationResult(NamedTuple)
```

Stores NLU cross-validation results.

#### log\_evaluation\_table

```python
def log_evaluation_table(report: Text, precision: float, f1: float, accuracy: float) -> None
```

Log the sklearn evaluation metrics.

#### remove\_empty\_intent\_examples

```python
def remove_empty_intent_examples(intent_results: List[IntentEvaluationResult]) -> List[IntentEvaluationResult]
```

Remove those examples without an intent.

**Arguments**:

- `intent_results` - intent evaluation results
  
- `Returns` - intent evaluation results

#### remove\_empty\_response\_examples

```python
def remove_empty_response_examples(response_results: List[ResponseSelectionEvaluationResult]) -> List[ResponseSelectionEvaluationResult]
```

Remove those examples without a response.

**Arguments**:

- `response_results` - response selection evaluation results
  
- `Returns` - response selection evaluation results

#### drop\_intents\_below\_freq

```python
def drop_intents_below_freq(training_data: TrainingData, cutoff: int = 5) -> TrainingData
```

Remove intent groups with less than cutoff instances.

**Arguments**:

- `training_data` - training data
- `cutoff` - threshold
  
- `Returns` - updated training data

#### write\_intent\_successes

```python
def write_intent_successes(intent_results: List[IntentEvaluationResult], successes_filename: Text) -> None
```

Write successful intent predictions to a file.

**Arguments**:

- `intent_results` - intent evaluation result
- `successes_filename` - filename of file to save successful predictions to

#### write\_response\_successes

```python
def write_response_successes(response_results: List[ResponseSelectionEvaluationResult], successes_filename: Text) -> None
```

Write successful response selection predictions to a file.

**Arguments**:

- `response_results` - response selection evaluation result
- `successes_filename` - filename of file to save successful predictions to

#### plot\_attribute\_confidences

```python
def plot_attribute_confidences(results: Union[
        List[IntentEvaluationResult], List[ResponseSelectionEvaluationResult]
    ], hist_filename: Optional[Text], target_key: Text, prediction_key: Text, title: Text) -> None
```

Create histogram of confidence distribution.

**Arguments**:

- `results` - evaluation results
- `hist_filename` - filename to save plot to
- `target_key` - key of target in results
- `prediction_key` - key of predictions in results
- `title` - title of plot

#### plot\_entity\_confidences

```python
def plot_entity_confidences(merged_targets: List[Text], merged_predictions: List[Text], merged_confidences: List[float], hist_filename: Text, title: Text) -> None
```

Creates histogram of confidence distribution.

**Arguments**:

- `merged_targets` - Entity labels.
- `merged_predictions` - Predicted entities.
- `merged_confidences` - Confidence scores of predictions.
- `hist_filename` - filename to save plot to
- `title` - title of plot

#### evaluate\_response\_selections

```python
def evaluate_response_selections(response_selection_results: List[ResponseSelectionEvaluationResult], output_directory: Optional[Text], successes: bool, errors: bool, disable_plotting: bool, report_as_dict: Optional[bool] = None) -> Dict
```

Creates summary statistics for response selection.

Only considers those examples with a set response.
Others are filtered out. Returns a dictionary of containing the
evaluation result.

**Arguments**:

- `response_selection_results` - response selection evaluation results
- `output_directory` - directory to store files to
- `successes` - if True success are written down to disk
- `errors` - if True errors are written down to disk
- `disable_plotting` - if True no plots are created
- `report_as_dict` - `True` if the evaluation report should be returned as `dict`.
  If `False` the report is returned in a human-readable text format. If `None`
  `report_as_dict` is considered as `True` in case an `output_directory` is
  given.
  
- `Returns` - dictionary with evaluation results

#### evaluate\_intents

```python
def evaluate_intents(intent_results: List[IntentEvaluationResult], output_directory: Optional[Text], successes: bool, errors: bool, disable_plotting: bool, report_as_dict: Optional[bool] = None) -> Dict
```

Creates summary statistics for intents.

Only considers those examples with a set intent. Others are filtered out.
Returns a dictionary of containing the evaluation result.

**Arguments**:

- `intent_results` - intent evaluation results
- `output_directory` - directory to store files to
- `successes` - if True correct predictions are written to disk
- `errors` - if True incorrect predictions are written to disk
- `disable_plotting` - if True no plots are created
- `report_as_dict` - `True` if the evaluation report should be returned as `dict`.
  If `False` the report is returned in a human-readable text format. If `None`
  `report_as_dict` is considered as `True` in case an `output_directory` is
  given.
  
- `Returns` - dictionary with evaluation results

#### merge\_labels

```python
def merge_labels(aligned_predictions: List[Dict], extractor: Optional[Text] = None) -> List[Text]
```

Concatenates all labels of the aligned predictions.

Takes the aligned prediction labels which are grouped for each message
and concatenates them.

**Arguments**:

- `aligned_predictions` - aligned predictions
- `extractor` - entity extractor name
  
- `Returns` - concatenated predictions

#### merge\_confidences

```python
def merge_confidences(aligned_predictions: List[Dict], extractor: Optional[Text] = None) -> List[float]
```

Concatenates all confidences of the aligned predictions.

Takes the aligned prediction confidences which are grouped for each message
and concatenates them.

**Arguments**:

- `aligned_predictions` - aligned predictions
- `extractor` - entity extractor name
  
- `Returns` - concatenated confidences

#### substitute\_labels

```python
def substitute_labels(labels: List[Text], old: Text, new: Text) -> List[Text]
```

Replaces label names in a list of labels.

**Arguments**:

- `labels` - list of labels
- `old` - old label name that should be replaced
- `new` - new label name
  
- `Returns` - updated labels

#### collect\_incorrect\_entity\_predictions

```python
def collect_incorrect_entity_predictions(entity_results: List[EntityEvaluationResult], merged_predictions: List[Text], merged_targets: List[Text]) -> List["EntityPrediction"]
```

Get incorrect entity predictions.

**Arguments**:

- `entity_results` - entity evaluation results
- `merged_predictions` - list of predicted entity labels
- `merged_targets` - list of true entity labels
  
- `Returns` - list of incorrect predictions

#### write\_successful\_entity\_predictions

```python
def write_successful_entity_predictions(entity_results: List[EntityEvaluationResult], merged_targets: List[Text], merged_predictions: List[Text], successes_filename: Text) -> None
```

Write correct entity predictions to a file.

**Arguments**:

- `entity_results` - response selection evaluation result
- `merged_predictions` - list of predicted entity labels
- `merged_targets` - list of true entity labels
- `successes_filename` - filename of file to save correct predictions to

#### collect\_successful\_entity\_predictions

```python
def collect_successful_entity_predictions(entity_results: List[EntityEvaluationResult], merged_predictions: List[Text], merged_targets: List[Text]) -> List["EntityPrediction"]
```

Get correct entity predictions.

**Arguments**:

- `entity_results` - entity evaluation results
- `merged_predictions` - list of predicted entity labels
- `merged_targets` - list of true entity labels
  
- `Returns` - list of correct predictions

#### evaluate\_entities

```python
def evaluate_entities(entity_results: List[EntityEvaluationResult], extractors: Set[Text], output_directory: Optional[Text], successes: bool, errors: bool, disable_plotting: bool, report_as_dict: Optional[bool] = None) -> Dict
```

Creates summary statistics for each entity extractor.

Logs precision, recall, and F1 per entity type for each extractor.

**Arguments**:

- `entity_results` - entity evaluation results
- `extractors` - entity extractors to consider
- `output_directory` - directory to store files to
- `successes` - if True correct predictions are written to disk
- `errors` - if True incorrect predictions are written to disk
- `disable_plotting` - if True no plots are created
- `report_as_dict` - `True` if the evaluation report should be returned as `dict`.
  If `False` the report is returned in a human-readable text format. If `None`
  `report_as_dict` is considered as `True` in case an `output_directory` is
  given.
  
- `Returns` - dictionary with evaluation results

#### is\_token\_within\_entity

```python
def is_token_within_entity(token: Token, entity: Dict) -> bool
```

Checks if a token is within the boundaries of an entity.

#### does\_token\_cross\_borders

```python
def does_token_cross_borders(token: Token, entity: Dict) -> bool
```

Checks if a token crosses the boundaries of an entity.

#### determine\_intersection

```python
def determine_intersection(token: Token, entity: Dict) -> int
```

Calculates how many characters a given token and entity share.

#### do\_entities\_overlap

```python
def do_entities_overlap(entities: List[Dict]) -> bool
```

Checks if entities overlap.

I.e. cross each others start and end boundaries.

**Arguments**:

- `entities` - list of entities
  
- `Returns` - true if entities overlap, false otherwise.

#### find\_intersecting\_entities

```python
def find_intersecting_entities(token: Token, entities: List[Dict]) -> List[Dict]
```

Finds the entities that intersect with a token.

**Arguments**:

- `token` - a single token
- `entities` - entities found by a single extractor
  
- `Returns` - list of entities

#### pick\_best\_entity\_fit

```python
def pick_best_entity_fit(token: Token, candidates: List[Dict[Text, Any]]) -> Optional[Dict[Text, Any]]
```

Determines the best fitting entity given intersecting entities.

**Arguments**:

- `token` - a single token
- `candidates` - entities found by a single extractor
- `attribute_key` - the attribute key of interest
  

**Returns**:

  the value of the attribute key of the best fitting entity

#### determine\_token\_labels

```python
def determine_token_labels(token: Token, entities: List[Dict], extractors: Optional[Set[Text]] = None, attribute_key: Text = ENTITY_ATTRIBUTE_TYPE) -> Text
```

Determines the token label for the provided attribute key given entities that do
not overlap.

**Arguments**:

- `token` - a single token
- `entities` - entities found by a single extractor
- `extractors` - list of extractors
- `attribute_key` - the attribute key for which the entity type should be returned

**Returns**:

  entity type

#### determine\_entity\_for\_token

```python
def determine_entity_for_token(token: Token, entities: List[Dict[Text, Any]], extractors: Optional[Set[Text]] = None) -> Optional[Dict[Text, Any]]
```

Determines the best fitting entity for the given token, given entities that do
not overlap.

**Arguments**:

- `token` - a single token
- `entities` - entities found by a single extractor
- `extractors` - list of extractors
  

**Returns**:

  entity type

#### do\_extractors\_support\_overlap

```python
def do_extractors_support_overlap(extractors: Optional[Set[Text]]) -> bool
```

Checks if extractors support overlapping entities

#### align\_entity\_predictions

```python
def align_entity_predictions(result: EntityEvaluationResult, extractors: Set[Text]) -> Dict
```

Aligns entity predictions to the message tokens.

Determines for every token the true label based on the
prediction targets and the label assigned by each
single extractor.

**Arguments**:

- `result` - entity evaluation result
- `extractors` - the entity extractors that should be considered
  
- `Returns` - dictionary containing the true token labels and token labels
  from the extractors

#### align\_all\_entity\_predictions

```python
def align_all_entity_predictions(entity_results: List[EntityEvaluationResult], extractors: Set[Text]) -> List[Dict]
```

Aligns entity predictions to the message tokens for the whole dataset
using align_entity_predictions.

**Arguments**:

- `entity_results` - list of entity prediction results
- `extractors` - the entity extractors that should be considered
  
- `Returns` - list of dictionaries containing the true token labels and token
  labels from the extractors

#### get\_eval\_data

```python
def get_eval_data(interpreter: Interpreter, test_data: TrainingData) -> Tuple[
    List[IntentEvaluationResult],
    List[ResponseSelectionEvaluationResult],
    List[EntityEvaluationResult],
]
```

Runs the model for the test set and extracts targets and predictions.

Returns intent results (intent targets and predictions, the original
messages and the confidences of the predictions), response results (
response targets and predictions) as well as entity results
(entity_targets, entity_predictions, and tokens).

**Arguments**:

- `interpreter` - the interpreter
- `test_data` - test data
  
- `Returns` - intent, response, and entity evaluation results

#### run\_evaluation

```python
def run_evaluation(data_path: Text, model_path: Text, output_directory: Optional[Text] = None, successes: bool = False, errors: bool = False, component_builder: Optional[ComponentBuilder] = None, disable_plotting: bool = False, report_as_dict: Optional[bool] = None) -> Dict
```

Evaluate intent classification, response selection and entity extraction.

**Arguments**:

- `data_path` - path to the test data
- `model_path` - path to the model
- `output_directory` - path to folder where all output will be stored
- `successes` - if true successful predictions are written to a file
- `errors` - if true incorrect predictions are written to a file
- `component_builder` - component builder
- `disable_plotting` - if true confusion matrix and histogram will not be rendered
- `report_as_dict` - `True` if the evaluation report should be returned as `dict`.
  If `False` the report is returned in a human-readable text format. If `None`
  `report_as_dict` is considered as `True` in case an `output_directory` is
  given.
  
- `Returns` - dictionary containing evaluation results

#### generate\_folds

```python
def generate_folds(n: int, training_data: TrainingData) -> Iterator[Tuple[TrainingData, TrainingData]]
```

Generates n cross validation folds for given training data.

#### combine\_result

```python
def combine_result(intent_metrics: IntentMetrics, entity_metrics: EntityMetrics, response_selection_metrics: ResponseSelectionMetrics, interpreter: Interpreter, data: TrainingData, intent_results: Optional[List[IntentEvaluationResult]] = None, entity_results: Optional[List[EntityEvaluationResult]] = None, response_selection_results: Optional[
        List[ResponseSelectionEvaluationResult]
    ] = None) -> Tuple[IntentMetrics, EntityMetrics, ResponseSelectionMetrics]
```

Collects intent, response selection and entity metrics for cross validation
folds.

If `intent_results`, `response_selection_results` or `entity_results` is provided
as a list, prediction results are also collected.

**Arguments**:

- `intent_metrics` - intent metrics
- `entity_metrics` - entity metrics
- `response_selection_metrics` - response selection metrics
- `interpreter` - the interpreter
- `data` - training data
- `intent_results` - intent evaluation results
- `entity_results` - entity evaluation results
- `response_selection_results` - reponse selection evaluation results
  
- `Returns` - intent, entity, and response selection metrics

#### cross\_validate

```python
def cross_validate(data: TrainingData, n_folds: int, nlu_config: Union[RasaNLUModelConfig, Text, Dict], output: Optional[Text] = None, successes: bool = False, errors: bool = False, disable_plotting: bool = False, report_as_dict: Optional[bool] = None) -> Tuple[CVEvaluationResult, CVEvaluationResult, CVEvaluationResult]
```

Stratified cross validation on data.

**Arguments**:

- `data` - Training Data
- `n_folds` - integer, number of cv folds
- `nlu_config` - nlu config file
- `output` - path to folder where reports are stored
- `successes` - if true successful predictions are written to a file
- `errors` - if true incorrect predictions are written to a file
- `disable_plotting` - if true no confusion matrix and historgram plates are created
- `report_as_dict` - `True` if the evaluation report should be returned as `dict`.
  If `False` the report is returned in a human-readable text format. If `None`
  `report_as_dict` is considered as `True` in case an `output_directory` is
  given.
  

**Returns**:

  dictionary with key, list structure, where each entry in list
  corresponds to the relevant result for one fold

#### compute\_metrics

```python
def compute_metrics(interpreter: Interpreter, training_data: TrainingData) -> Tuple[
    IntentMetrics,
    EntityMetrics,
    ResponseSelectionMetrics,
    List[IntentEvaluationResult],
    List[EntityEvaluationResult],
    List[ResponseSelectionEvaluationResult],
]
```

Computes metrics for intent classification, response selection and entity
extraction.

**Arguments**:

- `interpreter` - the interpreter
- `training_data` - training data
  
- `Returns` - intent, response selection and entity metrics, and prediction results.

#### compare\_nlu

```python
async def compare_nlu(configs: List[Text], data: TrainingData, exclusion_percentages: List[int], f_score_results: Dict[Text, Any], model_names: List[Text], output: Text, runs: int) -> List[int]
```

Trains and compares multiple NLU models.
For each run and exclusion percentage a model per config file is trained.
Thereby, the model is trained only on the current percentage of training data.
Afterwards, the model is tested on the complete test data of that run.
All results are stored in the provided output directory.

**Arguments**:

- `configs` - config files needed for training
- `data` - training data
- `exclusion_percentages` - percentages of training data to exclude during comparison
- `f_score_results` - dictionary of model name to f-score results per run
- `model_names` - names of the models to train
- `output` - the output directory
- `runs` - number of comparison runs
  
- `Returns` - training examples per run

#### log\_results

```python
def log_results(results: IntentMetrics, dataset_name: Text) -> None
```

Logs results of cross validation.

**Arguments**:

- `results` - dictionary of results returned from cross validation
- `dataset_name` - string of which dataset the results are from, e.g. test/train

#### log\_entity\_results

```python
def log_entity_results(results: EntityMetrics, dataset_name: Text) -> None
```

Logs entity results of cross validation.

**Arguments**:

- `results` - dictionary of dictionaries of results returned from cross validation
- `dataset_name` - string of which dataset the results are from, e.g. test/train


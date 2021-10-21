---
sidebar_label: rasa.core.test
title: rasa.core.test
---
## WrongPredictionException Objects

```python
class WrongPredictionException(RasaException,  ValueError)
```

Raised if a wrong prediction is encountered.

## WarningPredictedAction Objects

```python
class WarningPredictedAction(ActionExecuted)
```

The model predicted the correct action with warning.

#### \_\_init\_\_

```python
 | __init__(action_name_prediction: Text, action_name: Optional[Text] = None, policy: Optional[Text] = None, confidence: Optional[float] = None, timestamp: Optional[float] = None, metadata: Optional[Dict] = None)
```

Creates event `action_unlikely_intent` predicted as warning.

See the docstring of the parent class for more information.

#### inline\_comment

```python
 | inline_comment() -> Text
```

A comment attached to this event. Used during dumping.

## WronglyPredictedAction Objects

```python
class WronglyPredictedAction(ActionExecuted)
```

The model predicted the wrong action.

Mostly used to mark wrong predictions and be able to
dump them as stories.

#### \_\_init\_\_

```python
 | __init__(action_name_target: Text, action_text_target: Text, action_name_prediction: Text, policy: Optional[Text] = None, confidence: Optional[float] = None, timestamp: Optional[float] = None, metadata: Optional[Dict] = None, predicted_action_unlikely_intent: bool = False) -> None
```

Creates event for a successful event execution.

See the docstring of the parent class `ActionExecuted` for more information.

#### inline\_comment

```python
 | inline_comment() -> Text
```

A comment attached to this event. Used during dumping.

#### as\_story\_string

```python
 | as_story_string() -> Text
```

Returns the story equivalent representation.

#### \_\_repr\_\_

```python
 | __repr__() -> Text
```

Returns event as string for debugging.

## EvaluationStore Objects

```python
class EvaluationStore()
```

Class storing action, intent and entity predictions and targets.

#### \_\_init\_\_

```python
 | __init__(action_predictions: Optional[PredictionList] = None, action_targets: Optional[PredictionList] = None, intent_predictions: Optional[PredictionList] = None, intent_targets: Optional[PredictionList] = None, entity_predictions: Optional[List["EntityPrediction"]] = None, entity_targets: Optional[List["EntityPrediction"]] = None) -> None
```

Initialize store attributes.

#### add\_to\_store

```python
 | add_to_store(action_predictions: Optional[PredictionList] = None, action_targets: Optional[PredictionList] = None, intent_predictions: Optional[PredictionList] = None, intent_targets: Optional[PredictionList] = None, entity_predictions: Optional[List["EntityPrediction"]] = None, entity_targets: Optional[List["EntityPrediction"]] = None) -> None
```

Add items or lists of items to the store.

#### merge\_store

```python
 | merge_store(other: "EvaluationStore") -> None
```

Add the contents of other to self.

#### check\_prediction\_target\_mismatch

```python
 | check_prediction_target_mismatch() -> bool
```

Checks if intent, entity or action predictions don&#x27;t match expected ones.

#### serialise

```python
 | serialise() -> Tuple[PredictionList, PredictionList]
```

Turn targets and predictions to lists of equal size for sklearn.

## EndToEndUserUtterance Objects

```python
class EndToEndUserUtterance(UserUttered)
```

End-to-end user utterance.

Mostly used to print the full end-to-end user message in the
`failed_test_stories.yml` output file.

#### as\_story\_string

```python
 | as_story_string(e2e: bool = True) -> Text
```

Returns the story equivalent representation.

## WronglyClassifiedUserUtterance Objects

```python
class WronglyClassifiedUserUtterance(UserUttered)
```

The NLU model predicted the wrong user utterance.

Mostly used to mark wrong predictions and be able to
dump them as stories.

#### \_\_init\_\_

```python
 | __init__(event: UserUttered, eval_store: EvaluationStore) -> None
```

Set `predicted_intent` and `predicted_entities` attributes.

#### inline\_comment

```python
 | inline_comment() -> Optional[Text]
```

A comment attached to this event. Used during dumping.

#### inline\_comment\_for\_entity

```python
 | @staticmethod
 | inline_comment_for_entity(predicted: Dict[Text, Any], entity: Dict[Text, Any]) -> Optional[Text]
```

Returns the predicted entity which is then printed as a comment.

#### as\_story\_string

```python
 | as_story_string(e2e: bool = True) -> Text
```

Returns text representation of event.

#### emulate\_loop\_rejection

```python
emulate_loop_rejection(partial_tracker: DialogueStateTracker) -> None
```

Add `ActionExecutionRejected` event to the tracker.

During evaluation, we don&#x27;t run action server, therefore in order to correctly
test unhappy paths of the loops, we need to emulate loop rejection.

**Arguments**:

- `partial_tracker` - a :class:`rasa.core.trackers.DialogueStateTracker`

#### test

```python
async test(stories: Text, agent: "Agent", max_stories: Optional[int] = None, out_directory: Optional[Text] = None, fail_on_prediction_errors: bool = False, e2e: bool = False, disable_plotting: bool = False, successes: bool = False, errors: bool = True, warnings: bool = True) -> Dict[Text, Any]
```

Run the evaluation of the stories, optionally plot the results.

**Arguments**:

- `stories` - the stories to evaluate on
- `agent` - the agent
- `max_stories` - maximum number of stories to consider
- `out_directory` - path to directory to results to
- `fail_on_prediction_errors` - boolean indicating whether to fail on prediction
  errors or not
- `e2e` - boolean indicating whether to use end to end evaluation or not
- `disable_plotting` - boolean indicating whether to disable plotting or not
- `successes` - boolean indicating whether to write down successful predictions or
  not
- `errors` - boolean indicating whether to write down incorrect predictions or not
- `warnings` - boolean indicating whether to write down prediction warnings or not
  

**Returns**:

  Evaluation summary.

#### compare\_models\_in\_dir

```python
async compare_models_in_dir(model_dir: Text, stories_file: Text, output: Text, use_conversation_test_files: bool = False) -> None
```

Evaluates multiple trained models in a directory on a test set.

**Arguments**:

- `model_dir` - path to directory that contains the models to evaluate
- `stories_file` - path to the story file
- `output` - output directory to store results to
- `use_conversation_test_files` - `True` if conversation test files should be used
  for testing instead of regular Core story files.

#### compare\_models

```python
async compare_models(models: List[Text], stories_file: Text, output: Text, use_conversation_test_files: bool = False) -> None
```

Evaluates multiple trained models on a test set.

**Arguments**:

- `models` - Paths to model files.
- `stories_file` - path to the story file
- `output` - output directory to store results to
- `use_conversation_test_files` - `True` if conversation test files should be used
  for testing instead of regular Core story files.


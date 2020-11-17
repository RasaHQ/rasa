---
sidebar_label: test
title: rasa.core.test
---

## WrongPredictionException Objects

```python
class WrongPredictionException(RasaException,  ValueError)
```

Raised if a wrong prediction is encountered.

## EvaluationStore Objects

```python
class EvaluationStore()
```

Class storing action, intent and entity predictions and targets.

#### add\_to\_store

```python
 | add_to_store(action_predictions: Optional[List[Text]] = None, action_targets: Optional[List[Text]] = None, intent_predictions: Optional[List[Text]] = None, intent_targets: Optional[List[Text]] = None, entity_predictions: Optional[List[Dict[Text, Any]]] = None, entity_targets: Optional[List[Dict[Text, Any]]] = None) -> None
```

Add items or lists of items to the store

#### merge\_store

```python
 | merge_store(other: "EvaluationStore") -> None
```

Add the contents of other to self

#### serialise

```python
 | serialise() -> Tuple[List[Text], List[Text]]
```

Turn targets and predictions to lists of equal size for sklearn.

## WronglyPredictedAction Objects

```python
class WronglyPredictedAction(ActionExecuted)
```

The model predicted the wrong action.

Mostly used to mark wrong predictions and be able to
dump them as stories.

#### inline\_comment

```python
 | inline_comment() -> Text
```

A comment attached to this event. Used during dumping.

## EndToEndUserUtterance Objects

```python
class EndToEndUserUtterance(UserUttered)
```

End-to-end user utterance.

Mostly used to print the full end-to-end user message in the
`failed_test_stories.yml` output file.

## WronglyClassifiedUserUtterance Objects

```python
class WronglyClassifiedUserUtterance(UserUttered)
```

The NLU model predicted the wrong user utterance.

Mostly used to mark wrong predictions and be able to
dump them as stories.

#### inline\_comment

```python
 | inline_comment() -> Text
```

A comment attached to this event. Used during dumping.

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
async test(stories: Text, agent: "Agent", max_stories: Optional[int] = None, out_directory: Optional[Text] = None, fail_on_prediction_errors: bool = False, e2e: bool = False, disable_plotting: bool = False, successes: bool = False, errors: bool = True) -> Dict[Text, Any]
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
  

**Returns**:

  Evaluation summary.

#### compare\_models\_in\_dir

```python
async compare_models_in_dir(model_dir: Text, stories_file: Text, output: Text) -> None
```

Evaluate multiple trained models in a directory on a test set.

**Arguments**:

- `model_dir` - path to directory that contains the models to evaluate
- `stories_file` - path to the story file
- `output` - output directory to store results to

#### compare\_models

```python
async compare_models(models: List[Text], stories_file: Text, output: Text) -> None
```

Evaluate provided trained models on a test set.

**Arguments**:

- `models` - list of trained model paths
- `stories_file` - path to the story file
- `output` - output directory to store results to


---
sidebar_label: rasa.core.policies.fallback
title: rasa.core.policies.fallback
---
## FallbackPolicy Objects

```python
class FallbackPolicy(Policy)
```

Policy which predicts fallback actions.

A fallback can be triggered by a low confidence score on a
NLU prediction or by a low confidence score on an action
prediction.

#### \_\_init\_\_

```python
 | __init__(priority: int = FALLBACK_POLICY_PRIORITY, nlu_threshold: float = DEFAULT_NLU_FALLBACK_THRESHOLD, ambiguity_threshold: float = DEFAULT_NLU_FALLBACK_AMBIGUITY_THRESHOLD, core_threshold: float = DEFAULT_CORE_FALLBACK_THRESHOLD, fallback_action_name: Text = ACTION_DEFAULT_FALLBACK_NAME, **kwargs: Any, ,) -> None
```

Create a new Fallback policy.

**Arguments**:

- `priority` - Fallback policy priority.
- `core_threshold` - if NLU confidence threshold is met,
  predict fallback action with confidence `core_threshold`.
  If this is the highest confidence in the ensemble,
  the fallback action will be executed.
- `nlu_threshold` - minimum threshold for NLU confidence.
  If intent prediction confidence is lower than this,
  predict fallback action with confidence 1.0.
- `ambiguity_threshold` - threshold for minimum difference
  between confidences of the top two predictions
- `fallback_action_name` - name of the action to execute as a fallback

#### train

```python
 | train(training_trackers: List[TrackerWithCachedStates], domain: Domain, interpreter: NaturalLanguageInterpreter, **kwargs: Any, ,) -> None
```

Does nothing. This policy is deterministic.

#### nlu\_confidence\_below\_threshold

```python
 | nlu_confidence_below_threshold(nlu_data: Dict[Text, Any]) -> Tuple[bool, float]
```

Check if the highest confidence is lower than ``nlu_threshold``.

#### nlu\_prediction\_ambiguous

```python
 | nlu_prediction_ambiguous(nlu_data: Dict[Text, Any]) -> Tuple[bool, Optional[float]]
```

Check if top 2 confidences are closer than ``ambiguity_threshold``.

#### should\_nlu\_fallback

```python
 | should_nlu_fallback(nlu_data: Dict[Text, Any], last_action_name: Text) -> bool
```

Check if fallback action should be predicted.

Checks for:
- predicted NLU confidence is lower than ``nlu_threshold``
- difference in top 2 NLU confidences lower than ``ambiguity_threshold``
- last action is action listen

#### fallback\_scores

```python
 | fallback_scores(domain: Domain, fallback_score: float = 1.0) -> List[float]
```

Prediction scores used if a fallback is necessary.

#### predict\_action\_probabilities

```python
 | predict_action_probabilities(tracker: DialogueStateTracker, domain: Domain, interpreter: NaturalLanguageInterpreter, **kwargs: Any, ,) -> PolicyPrediction
```

Predicts a fallback action.

The fallback action is predicted if the NLU confidence is low
or no other policy has a high-confidence prediction.


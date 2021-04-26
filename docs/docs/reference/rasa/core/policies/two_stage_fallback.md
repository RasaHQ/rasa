---
sidebar_label: rasa.core.policies.two_stage_fallback
title: rasa.core.policies.two_stage_fallback
---
## TwoStageFallbackPolicy Objects

```python
class TwoStageFallbackPolicy(FallbackPolicy)
```

This policy handles low NLU confidence in multiple stages.

If a NLU prediction has a low confidence score,
the user is asked to affirm whether they really had this intent.
If they affirm, the story continues as if the intent was classified
with high confidence from the beginning.
If they deny, the user is asked to rephrase his intent.
If the classification for the rephrased intent was confident, the story
continues as if the user had this intent from the beginning.
If the rephrased intent was not classified with high confidence,
the user is asked to affirm the classified intent.
If the user affirm the intent, the story continues as if the user had
this intent from the beginning.
If the user denies, an ultimate fallback action is triggered
(e.g. a hand-off to a human).

#### \_\_init\_\_

```python
 | __init__(priority: int = FALLBACK_POLICY_PRIORITY, nlu_threshold: float = DEFAULT_NLU_FALLBACK_THRESHOLD, ambiguity_threshold: float = DEFAULT_NLU_FALLBACK_AMBIGUITY_THRESHOLD, core_threshold: float = DEFAULT_CORE_FALLBACK_THRESHOLD, fallback_core_action_name: Text = ACTION_DEFAULT_FALLBACK_NAME, fallback_nlu_action_name: Text = ACTION_DEFAULT_FALLBACK_NAME, deny_suggestion_intent_name: Text = USER_INTENT_OUT_OF_SCOPE, **kwargs: Any, ,) -> None
```

Create a new Two-stage Fallback policy.

**Arguments**:

- `priority` - The fallback policy priority.
- `nlu_threshold` - minimum threshold for NLU confidence.
  If intent prediction confidence is lower than this,
  predict fallback action with confidence 1.0.
- `ambiguity_threshold` - threshold for minimum difference
  between confidences of the top two predictions
- `core_threshold` - if NLU confidence threshold is met,
  predict fallback action with confidence
  `core_threshold`. If this is the highest confidence in
  the ensemble, the fallback action will be executed.
- `fallback_core_action_name` - This action is executed if the Core
  threshold is not met.
- `fallback_nlu_action_name` - This action is executed if the user
  denies the recognised intent for the second time.
- `deny_suggestion_intent_name` - The name of the intent which is used
  to detect that the user denies the suggested intents.

#### predict\_action\_probabilities

```python
 | predict_action_probabilities(tracker: DialogueStateTracker, domain: Domain, interpreter: NaturalLanguageInterpreter, **kwargs: Any, ,) -> PolicyPrediction
```

Predicts the next action if NLU confidence is low.


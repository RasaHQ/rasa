---
sidebar_label: rasa.graph_components.adders.nlu_prediction_to_history_adder
title: rasa.graph_components.adders.nlu_prediction_to_history_adder
---
## NLUPredictionToHistoryAdder Objects

```python
class NLUPredictionToHistoryAdder(GraphComponent)
```

Adds NLU predictions to DialogueStateTracker.

#### create

```python
 | @classmethod
 | create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> NLUPredictionToHistoryAdder
```

Creates component (see parent class for full docstring).

#### add

```python
 | add(predictions: List[Message], tracker: DialogueStateTracker, domain: Domain, original_message: UserMessage) -> DialogueStateTracker
```

Adds NLU predictions to the tracker.

**Arguments**:

- `predictions` - A list of NLU predictions wrapped as Messages
- `tracker` - The tracker the predictions should be attached to
- `domain` - The domain of the model.
- `original_message` - An original message from the user with
  extra metadata to annotate the predictions (e.g. channel)
  

**Returns**:

  The original tracker updated with events created from the predictions


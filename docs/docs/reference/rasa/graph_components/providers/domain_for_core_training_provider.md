---
sidebar_label: rasa.graph_components.providers.domain_for_core_training_provider
title: rasa.graph_components.providers.domain_for_core_training_provider
---
## DomainForCoreTrainingProvider Objects

```python
class DomainForCoreTrainingProvider(GraphComponent)
```

Provides domain without information that is irrelevant for core training.

The information that we retain includes:
- intents and their &quot;used&quot; and &quot;ignored&quot; entities because intents influence the
next action prediction directly and the latter flags determine whether the
listed entities influence the next action prediction
- entities, their roles and groups, and their `influence_conversation` flag because
all of those items are used by policies
- slots names along with their types, since this type information determines the
pre-featurization of slot values
- response keys (i.e. `utter_*) because those keys may appear in stories
- form names because those appear in stories
- how slots are filled (i.e. &#x27;mappings&#x27; key under &#x27;slots&#x27;) because a domain instance
needs to be created by core during training time to parse the training data
properly

This information that we drop (or replace with default values) includes:
- the &#x27;session_config&#x27; which determines details of a session e.g. whether data is
transferred from one session to the next (this is replaced with defaults as it
cannot just be removed)
- the actual text of a &#x27;response&#x27; because those are only used by response selectors
- the actual configuration of &#x27;forms&#x27; because those are not actually executed
by core components

**References**:

  - `rasa.core.featurizer.tracker_featurizer.py` (used by all policies)
  - `rasa.core.featurizer.single_state_featurizer.py` (used by ML policies)
  - `rasa.shared.core.domain.get_active_state` (used by above references)
  - `rasa.shared.core.slots.as_features` (used by above references)
  - `rasa.shared.core.training_data.structures.StoryStep.explicit_events`
  (i.e. slots needed for core training)

#### create

```python
 | @classmethod
 | create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> DomainForCoreTrainingProvider
```

Creates component (see parent class for full docstring).

#### provide

```python
 | provide(domain: Domain) -> Domain
```

Recreates the given domain but drops information that is irrelevant for core.

**Arguments**:

- `domain` - A domain.
  

**Returns**:

  A similar domain without information that is irrelevant for core training.

#### create\_pruned\_version

```python
 | @staticmethod
 | create_pruned_version(domain: Domain) -> Domain
```

Recreates the given domain but drops information that is irrelevant for core.

**Arguments**:

- `domain` - A domain.
  

**Returns**:

  A similar domain without information that is irrelevant for core training.


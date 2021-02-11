---
sidebar_label: single_state_featurizer
title: rasa.core.featurizers.single_state_featurizer
---

## SingleStateFeaturizer Objects

```python
class SingleStateFeaturizer()
```

Base class to transform the dialogue state into an ML format.

Subclasses of SingleStateFeaturizer will decide how a bot will
transform the dialogue state into a dictionary mapping an attribute
to its features. Possible attributes are: INTENT, TEXT, ACTION_NAME,
ACTION_TEXT, ENTITIES, SLOTS and ACTIVE_LOOP. Each attribute will be
featurized into a list of `rasa.utils.features.Features`.

#### \_\_init\_\_

```python
 | __init__() -> None
```

Initialize the single state featurizer.

#### prepare\_for\_training

```python
 | prepare_for_training(domain: Domain, interpreter: NaturalLanguageInterpreter, bilou_tagging: bool = False) -> None
```

Gets necessary information for featurization from domain.

**Arguments**:

- `domain` - An instance of :class:`rasa.shared.core.domain.Domain`.
- `interpreter` - The interpreter used to encode the state
- `bilou_tagging` - indicates whether BILOU tagging should be used or not

#### encode\_state

```python
 | encode_state(state: State, interpreter: NaturalLanguageInterpreter) -> Dict[Text, List["Features"]]
```

Encode the given state with the help of the given interpreter.

**Arguments**:

- `state` - The state to encode
- `interpreter` - The interpreter used to encode the state
  

**Returns**:

  A dictionary of state_type to list of features.

#### encode\_entities

```python
 | encode_entities(entity_data: Dict[Text, Any], interpreter: NaturalLanguageInterpreter, bilou_tagging: bool = False) -> Dict[Text, List["Features"]]
```

Encode the given entity data with the help of the given interpreter.

Produce numeric entity tags for tokens.

**Arguments**:

- `entity_data` - The dict containing the text and entity labels and locations
- `interpreter` - The interpreter used to encode the state
- `bilou_tagging` - indicates whether BILOU tagging should be used or not
  

**Returns**:

  A dictionary of entity type to list of features.

#### encode\_all\_actions

```python
 | encode_all_actions(domain: Domain, interpreter: NaturalLanguageInterpreter) -> List[Dict[Text, List["Features"]]]
```

Encode all action from the domain using the given interpreter.

**Arguments**:

- `domain` - The domain that contains the actions.
- `interpreter` - The interpreter used to encode the actions.
  

**Returns**:

  A list of encoded actions.


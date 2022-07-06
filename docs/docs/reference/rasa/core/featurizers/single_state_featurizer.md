---
sidebar_label: rasa.core.featurizers.single_state_featurizer
title: rasa.core.featurizers.single_state_featurizer
---
## SingleStateFeaturizer Objects

```python
class SingleStateFeaturizer()
```

Base class to transform the dialogue state into an ML format.

Subclasses of SingleStateFeaturizer will decide how a bot will
transform the dialogue state into a dictionary mapping an attribute
to its features. Possible attributes are: `INTENT`, `TEXT`, `ACTION_NAME`,
`ACTION_TEXT`, `ENTITIES`, `SLOTS` and `ACTIVE_LOOP`. Each attribute will be
featurized into a list of `rasa.utils.features.Features`.

#### \_\_init\_\_

```python
def __init__() -> None
```

Initialize the single state featurizer.

#### prepare\_for\_training

```python
def prepare_for_training(domain: Domain, bilou_tagging: bool = False) -> None
```

Gets necessary information for featurization from domain.

**Arguments**:

- `domain` - An instance of :class:`rasa.shared.core.domain.Domain`.
- `bilou_tagging` - indicates whether BILOU tagging should be used or not

#### encode\_state

```python
def encode_state(state: State, precomputations: Optional[MessageContainerForCoreFeaturization]) -> Dict[Text, List[Features]]
```

Encode the given state.

**Arguments**:

- `state` - The state to encode
- `precomputations` - Contains precomputed features and attributes.
  

**Returns**:

  A dictionary of state_type to list of features.

#### encode\_entities

```python
def encode_entities(entity_data: Dict[Text, Any], precomputations: Optional[MessageContainerForCoreFeaturization], bilou_tagging: bool = False) -> Dict[Text, List[Features]]
```

Encode the given entity data.

Produce numeric entity tags for tokens.

**Arguments**:

- `entity_data` - The dict containing the text and entity labels and locations
- `precomputations` - Contains precomputed features and attributes.
- `bilou_tagging` - indicates whether BILOU tagging should be used or not
  

**Returns**:

  A dictionary of entity type to list of features.

#### encode\_all\_labels

```python
def encode_all_labels(domain: Domain, precomputations: Optional[MessageContainerForCoreFeaturization]) -> List[Dict[Text, List[Features]]]
```

Encode all action from the domain.

**Arguments**:

- `domain` - The domain that contains the actions.
- `precomputations` - Contains precomputed features and attributes.
  

**Returns**:

  A list of encoded actions.

## IntentTokenizerSingleStateFeaturizer Objects

```python
class IntentTokenizerSingleStateFeaturizer(SingleStateFeaturizer)
```

A SingleStateFeaturizer for use with policies that predict intent labels.

#### encode\_all\_labels

```python
def encode_all_labels(domain: Domain, precomputations: Optional[MessageContainerForCoreFeaturization]) -> List[Dict[Text, List[Features]]]
```

Encodes all relevant labels from the domain using the given precomputations.

**Arguments**:

- `domain` - The domain that contains the labels.
- `precomputations` - Contains precomputed features and attributes.
  

**Returns**:

  A list of encoded labels.


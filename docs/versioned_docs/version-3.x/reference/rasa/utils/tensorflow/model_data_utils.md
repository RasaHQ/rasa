---
sidebar_label: rasa.utils.tensorflow.model_data_utils
title: rasa.utils.tensorflow.model_data_utils
---
#### featurize\_training\_examples

```python
featurize_training_examples(training_examples: List[Message], attributes: List[Text], entity_tag_specs: Optional[List["EntityTagSpec"]] = None, featurizers: Optional[List[Text]] = None, bilou_tagging: bool = False) -> Tuple[List[Dict[Text, List["Features"]]], Dict[Text, Dict[Text, List[int]]]]
```

Converts training data into a list of attribute to features.

Possible attributes are, for example, INTENT, RESPONSE, TEXT, ACTION_TEXT,
ACTION_NAME or ENTITIES.
Also returns sparse feature sizes for each attribute. It could look like this:
{TEXT: {FEATURE_TYPE_SEQUENCE: [16, 32], FEATURE_TYPE_SENTENCE: [16, 32]}}.

**Arguments**:

- `training_examples` - the list of training examples
- `attributes` - the attributes to consider
- `entity_tag_specs` - the entity specs
- `featurizers` - the featurizers to consider
- `bilou_tagging` - indicates whether BILOU tagging should be used or not
  

**Returns**:

  A list of attribute to features.
  A dictionary of attribute to feature sizes.

#### get\_tag\_ids

```python
get_tag_ids(example: Message, tag_spec: "EntityTagSpec", bilou_tagging: bool) -> "Features"
```

Creates a feature array containing the entity tag ids of the given example.

**Arguments**:

- `example` - the message
- `tag_spec` - entity tag spec
- `bilou_tagging` - indicates whether BILOU tagging should be used or not
  

**Returns**:

  A list of features.

#### convert\_to\_data\_format

```python
convert_to_data_format(features: Union[
        List[List[Dict[Text, List["Features"]]]], List[Dict[Text, List["Features"]]]
    ], fake_features: Optional[Dict[Text, List["Features"]]] = None, consider_dialogue_dimension: bool = True, featurizers: Optional[List[Text]] = None) -> Tuple[Data, Optional[Dict[Text, List["Features"]]]]
```

Converts the input into &quot;Data&quot; format.

&quot;features&quot; can, for example, be a dictionary of attributes (INTENT,
TEXT, ACTION_NAME, ACTION_TEXT, ENTITIES, SLOTS, FORM) to a list of features for
all dialogue turns in all training trackers.
For NLU training it would just be a dictionary of attributes (either INTENT or
RESPONSE, TEXT, and potentially ENTITIES) to a list of features for all training
examples.

The &quot;Data&quot; format corresponds to Dict[Text, Dict[Text, List[FeatureArray]]]. It&#x27;s
a dictionary of attributes (e.g. TEXT) to a dictionary of secondary attributes
(e.g. SEQUENCE or SENTENCE) to the list of actual features.

**Arguments**:

- `features` - a dictionary of attributes to a list of features for all
  examples in the training data
- `fake_features` - Contains default feature values for attributes
- `consider_dialogue_dimension` - If set to false the dialogue dimension will be
  removed from the resulting sequence features.
- `featurizers` - the featurizers to consider
  

**Returns**:

  Input in &quot;Data&quot; format and fake features


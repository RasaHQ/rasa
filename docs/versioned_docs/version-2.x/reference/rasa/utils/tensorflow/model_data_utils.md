---
sidebar_label: model_data_utils
title: rasa.utils.tensorflow.model_data_utils
---

#### surface\_attributes

```python
surface_attributes(tracker_state_features: List[List[Dict[Text, List["Features"]]]]) -> Dict[Text, List[List[List["Features"]]]]
```

Restructure the input.

**Arguments**:

- `tracker_state_features` - a dictionary of attributes (INTENT, TEXT, ACTION_NAME,
  ACTION_TEXT, ENTITIES, SLOTS, FORM) to a list of features for all
  dialogue turns in all training trackers
  

**Returns**:

  A dictionary of attributes to a list of features for all dialogue turns
  and all training trackers.

#### create\_zero\_features

```python
create_zero_features(tracker_features: List[List[List["Features"]]]) -> List["Features"]
```

Computes default feature values for an attribute;

**Arguments**:

- `tracker_features` - list containing all feature values encountered
  in the dataset for an attribute;

#### convert\_to\_data\_format

```python
convert_to_data_format(tracker_state_features: Union[
        List[List[Dict[Text, List["Features"]]]], List[Dict[Text, List["Features"]]]
    ], zero_state_features: Optional[Dict[Text, List["Features"]]] = None) -> Tuple[Data, Optional[Dict[Text, List["Features"]]]]
```

Converts the input into &quot;Data&quot; format.

**Arguments**:

- `tracker_state_features` - a dictionary of attributes (INTENT, TEXT, ACTION_NAME,
  ACTION_TEXT, ENTITIES, SLOTS, FORM) to a list of features for all
  dialogue turns in all training trackers
- `zero_state_features` - Contains default feature values for attributes
  

**Returns**:

  Input in &quot;Data&quot; format and zero state features

#### map\_tracker\_features

```python
map_tracker_features(tracker_features: List[List[List["Features"]]], zero_features: List["Features"]) -> Tuple[
    List[np.ndarray],
    Dict[Text, List[List["Features"]]],
    Dict[Text, List[List["Features"]]],
]
```

Create masks for all attributes of the given features and split the features
into sparse and dense features.

**Arguments**:

- `tracker_features` - all features
- `zero_features` - list of zero features
  

**Returns**:

  - a list of attribute masks
  - a map of attribute to dense features
  - a map of attribute to sparse features


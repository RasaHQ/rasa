---
sidebar_label: rasa.shared.nlu.training_data.training_data
title: rasa.shared.nlu.training_data.training_data
---
## TrainingData Objects

```python
class TrainingData()
```

Holds loaded intent and entity training data.

#### fingerprint

```python
 | fingerprint() -> Text
```

Fingerprint the training data.

**Returns**:

  hex string as a fingerprint of the training data.

#### label\_fingerprint

```python
 | label_fingerprint() -> Text
```

Fingerprints the labels in the training data.

**Returns**:

  hex string as a fingerprint of the training data labels.

#### merge

```python
 | merge(*others: Optional["TrainingData"]) -> "TrainingData"
```

Return merged instance of this data with other training data.

**Arguments**:

- `others` - other training data instances to merge this one with
  

**Returns**:

  Merged training data object. Merging is not done in place, this
  will be a new instance.

#### filter\_training\_examples

```python
 | filter_training_examples(condition: Callable[[Message], bool]) -> "TrainingData"
```

Filter training examples.

**Arguments**:

- `condition` - A function that will be applied to filter training examples.
  

**Returns**:

- `TrainingData` - A TrainingData with filtered training examples.

#### \_\_hash\_\_

```python
 | __hash__() -> int
```

Calculate hash for the training data object.

**Returns**:

  Hash of the training data object.

#### sanitize\_examples

```python
 | @staticmethod
 | sanitize_examples(examples: List[Message]) -> List[Message]
```

Makes sure the training data is clean.

Remove trailing whitespaces from intent and response annotations and drop
duplicate examples.

#### nlu\_examples

```python
 | @lazy_property
 | nlu_examples() -> List[Message]
```

Return examples which have come from NLU training data.

E.g. If the example came from a story or domain it is not included.

**Returns**:

  List of NLU training examples.

#### intent\_examples

```python
 | @lazy_property
 | intent_examples() -> List[Message]
```

Returns the list of examples that have intent.

#### response\_examples

```python
 | @lazy_property
 | response_examples() -> List[Message]
```

Returns the list of examples that have response.

#### entity\_examples

```python
 | @lazy_property
 | entity_examples() -> List[Message]
```

Returns the list of examples that have entities.

#### intents

```python
 | @lazy_property
 | intents() -> Set[Text]
```

Returns the set of intents in the training data.

#### action\_names

```python
 | @lazy_property
 | action_names() -> Set[Text]
```

Returns the set of action names in the training data.

#### retrieval\_intents

```python
 | @lazy_property
 | retrieval_intents() -> Set[Text]
```

Returns the total number of response types in the training data.

#### number\_of\_examples\_per\_intent

```python
 | @lazy_property
 | number_of_examples_per_intent() -> Dict[Text, int]
```

Calculates the number of examples per intent.

#### number\_of\_examples\_per\_response

```python
 | @lazy_property
 | number_of_examples_per_response() -> Dict[Text, int]
```

Calculates the number of examples per response.

#### entities

```python
 | @lazy_property
 | entities() -> Set[Text]
```

Returns the set of entity types in the training data.

#### entity\_roles

```python
 | @lazy_property
 | entity_roles() -> Set[Text]
```

Returns the set of entity roles in the training data.

#### entity\_groups

```python
 | @lazy_property
 | entity_groups() -> Set[Text]
```

Returns the set of entity groups in the training data.

#### entity\_roles\_groups\_used

```python
 | entity_roles_groups_used() -> bool
```

Checks if any entity roles or groups are used in the training data.

#### number\_of\_examples\_per\_entity

```python
 | @lazy_property
 | number_of_examples_per_entity() -> Dict[Text, int]
```

Calculates the number of examples per entity.

#### sort\_regex\_features

```python
 | sort_regex_features() -> None
```

Sorts regex features lexicographically by name+pattern

#### nlu\_as\_json

```python
 | nlu_as_json(**kwargs: Any) -> Text
```

Represent this set of training examples as json.

#### nlg\_as\_markdown

```python
 | nlg_as_markdown() -> Text
```

Generates the markdown representation of the response phrases (NLG) of
TrainingData.

#### nlg\_as\_yaml

```python
 | nlg_as_yaml() -> Text
```

Generates yaml representation of the response phrases (NLG) of TrainingData.

**Returns**:

  responses in yaml format as a string

#### nlu\_as\_markdown

```python
 | nlu_as_markdown() -> Text
```

Generates the markdown representation of the NLU part of TrainingData.

#### persist

```python
 | persist(dir_name: Text, filename: Text = DEFAULT_TRAINING_DATA_OUTPUT_PATH) -> Dict[Text, Any]
```

Persists this training data to disk and returns necessary
information to load it again.

#### sorted\_entities

```python
 | sorted_entities() -> List[Any]
```

Extract all entities from examples and sorts them by entity type.

#### sorted\_intent\_examples

```python
 | sorted_intent_examples() -> List[Message]
```

Sorts the intent examples by the name of the intent and then response.

#### validate

```python
 | validate() -> None
```

Ensures that the loaded training data is valid.

Checks that the data has a minimum of certain training examples.

#### train\_test\_split

```python
 | train_test_split(train_frac: float = 0.8, random_seed: Optional[int] = None) -> Tuple["TrainingData", "TrainingData"]
```

Split into a training and test dataset,
preserving the fraction of examples per intent.

#### split\_nlu\_examples

```python
 | split_nlu_examples(train_frac: float, random_seed: Optional[int] = None) -> Tuple[list, list]
```

Split the training data into a train and test set.

**Arguments**:

- `train_frac` - percentage of examples to add to the training set.
- `random_seed` - random seed used to shuffle examples.
  

**Returns**:

  Test and training examples.

#### is\_empty

```python
 | is_empty() -> bool
```

Checks if any training data was loaded.

#### contains\_no\_pure\_nlu\_data

```python
 | contains_no_pure_nlu_data() -> bool
```

Checks if any NLU training data was loaded.

#### has\_e2e\_examples

```python
 | has_e2e_examples() -> bool
```

Checks if there are any training examples from e2e stories.

#### list\_to\_str

```python
list_to_str(lst: List[Text], delim: Text = ", ", quote: Text = "'") -> Text
```

Converts list to a string.

**Arguments**:

- `lst` - The list to convert.
- `delim` - The delimiter that is used to separate list inputs.
- `quote` - The quote that is used to wrap list inputs.
  

**Returns**:

  The string.


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
def fingerprint() -> Text
```

Fingerprint the training data.

**Returns**:

  hex string as a fingerprint of the training data.

#### label\_fingerprint

```python
def label_fingerprint() -> Text
```

Fingerprints the labels in the training data.

**Returns**:

  hex string as a fingerprint of the training data labels.

#### merge

```python
def merge(*others: Optional["TrainingData"]) -> "TrainingData"
```

Return merged instance of this data with other training data.

**Arguments**:

- `others` - other training data instances to merge this one with
  

**Returns**:

  Merged training data object. Merging is not done in place, this
  will be a new instance.

#### filter\_training\_examples

```python
def filter_training_examples(condition: Callable[[Message], bool]) -> "TrainingData"
```

Filter training examples.

**Arguments**:

- `condition` - A function that will be applied to filter training examples.
  

**Returns**:

- `TrainingData` - A TrainingData with filtered training examples.

#### \_\_hash\_\_

```python
def __hash__() -> int
```

Calculate hash for the training data object.

**Returns**:

  Hash of the training data object.

#### sanitize\_examples

```python
@staticmethod
def sanitize_examples(examples: List[Message]) -> List[Message]
```

Makes sure the training data is clean.

Remove trailing whitespaces from intent and response annotations and drop
duplicate examples.

#### nlu\_examples

```python
@lazy_property
def nlu_examples() -> List[Message]
```

Return examples which have come from NLU training data.

E.g. If the example came from a story or domain it is not included.

**Returns**:

  List of NLU training examples.

#### intent\_examples

```python
@lazy_property
def intent_examples() -> List[Message]
```

Returns the list of examples that have intent.

#### response\_examples

```python
@lazy_property
def response_examples() -> List[Message]
```

Returns the list of examples that have response.

#### entity\_examples

```python
@lazy_property
def entity_examples() -> List[Message]
```

Returns the list of examples that have entities.

#### intents

```python
@lazy_property
def intents() -> Set[Text]
```

Returns the set of intents in the training data.

#### action\_names

```python
@lazy_property
def action_names() -> Set[Text]
```

Returns the set of action names in the training data.

#### retrieval\_intents

```python
@lazy_property
def retrieval_intents() -> Set[Text]
```

Returns the total number of response types in the training data.

#### number\_of\_examples\_per\_intent

```python
@lazy_property
def number_of_examples_per_intent() -> Dict[Text, int]
```

Calculates the number of examples per intent.

#### number\_of\_examples\_per\_response

```python
@lazy_property
def number_of_examples_per_response() -> Dict[Text, int]
```

Calculates the number of examples per response.

#### entities

```python
@lazy_property
def entities() -> Set[Text]
```

Returns the set of entity types in the training data.

#### entity\_roles

```python
@lazy_property
def entity_roles() -> Set[Text]
```

Returns the set of entity roles in the training data.

#### entity\_groups

```python
@lazy_property
def entity_groups() -> Set[Text]
```

Returns the set of entity groups in the training data.

#### entity\_roles\_groups\_used

```python
def entity_roles_groups_used() -> bool
```

Checks if any entity roles or groups are used in the training data.

#### number\_of\_examples\_per\_entity

```python
@lazy_property
def number_of_examples_per_entity() -> Dict[Text, int]
```

Calculates the number of examples per entity.

#### sort\_regex\_features

```python
def sort_regex_features() -> None
```

Sorts regex features lexicographically by name+pattern

#### nlu\_as\_json

```python
def nlu_as_json(**kwargs: Any) -> Text
```

Represent this set of training examples as json.

#### nlg\_as\_yaml

```python
def nlg_as_yaml() -> Text
```

Generates yaml representation of the response phrases (NLG) of TrainingData.

**Returns**:

  responses in yaml format as a string

#### nlu\_as\_yaml

```python
def nlu_as_yaml() -> Text
```

Generates YAML representation of NLU of TrainingData.

**Returns**:

  data in YAML format as a string

#### persist\_nlu

```python
def persist_nlu(filename: Text = DEFAULT_TRAINING_DATA_OUTPUT_PATH) -> None
```

Saves NLU to a file.

#### persist\_nlg

```python
def persist_nlg(filename: Text) -> None
```

Saves NLG to a file.

#### get\_nlg\_persist\_filename

```python
@staticmethod
def get_nlg_persist_filename(nlu_filename: Text) -> Text
```

Returns the full filename to persist NLG data.

#### persist

```python
def persist(dir_name: Text, filename: Text = DEFAULT_TRAINING_DATA_OUTPUT_PATH) -> Dict[Text, Any]
```

Persists this training data to disk and returns necessary
information to load it again.

#### sorted\_entities

```python
def sorted_entities() -> List[Any]
```

Extract all entities from examples and sorts them by entity type.

#### validate

```python
def validate() -> None
```

Ensures that the loaded training data is valid.

Checks that the data has a minimum of certain training examples.

#### train\_test\_split

```python
def train_test_split(train_frac: float = 0.8, random_seed: Optional[int] = None) -> Tuple["TrainingData", "TrainingData"]
```

Split into a training and test dataset,
preserving the fraction of examples per intent.

#### split\_nlu\_examples

```python
def split_nlu_examples(train_frac: float, random_seed: Optional[int] = None) -> Tuple[list, list]
```

Split the training data into a train and test set.

**Arguments**:

- `train_frac` - percentage of examples to add to the training set.
- `random_seed` - random seed used to shuffle examples.
  

**Returns**:

  Test and training examples.

#### is\_empty

```python
def is_empty() -> bool
```

Checks if any training data was loaded.

#### contains\_no\_pure\_nlu\_data

```python
def contains_no_pure_nlu_data() -> bool
```

Checks if any NLU training data was loaded.

#### has\_e2e\_examples

```python
def has_e2e_examples() -> bool
```

Checks if there are any training examples from e2e stories.

#### list\_to\_str

```python
def list_to_str(lst: List[Text], delim: Text = ", ", quote: Text = "'") -> Text
```

Converts list to a string.

**Arguments**:

- `lst` - The list to convert.
- `delim` - The delimiter that is used to separate list inputs.
- `quote` - The quote that is used to wrap list inputs.
  

**Returns**:

  The string.


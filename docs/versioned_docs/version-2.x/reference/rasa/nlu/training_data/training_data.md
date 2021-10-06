---
sidebar_label: rasa.nlu.training_data.training_data
title: rasa.nlu.training_data.training_data
---
## TrainingData Objects

```python
class TrainingData()
```

Holds loaded intent and entity training data.

#### merge

```python
 | merge(*others: "TrainingData") -> "TrainingData"
```

Return merged instance of this data with other training data.

#### filter\_training\_examples

```python
 | filter_training_examples(condition: Callable[[Message], bool]) -> "TrainingData"
```

Filter training examples.

**Arguments**:

- `condition` - A function that will be applied to filter training examples.
  

**Returns**:

- `TrainingData` - A TrainingData with filtered training examples.

#### sanitize\_examples

```python
 | @staticmethod
 | sanitize_examples(examples: List[Message]) -> List[Message]
```

Makes sure the training data is clean.

Remove trailing whitespaces from intent and response annotations and drop
duplicate examples.

#### intents

```python
 | @lazy_property
 | intents() -> Set[Text]
```

Returns the set of intents in the training data.

#### retrieval\_intents

```python
 | @lazy_property
 | retrieval_intents() -> Set[Text]
```

Returns the total number of response types in the training data

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

Sorts the intent examples by the name of the intent and then response

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
- `random_seed` - random seed
  

**Returns**:

  Test and training examples.

#### is\_empty

```python
 | is_empty() -> bool
```

Checks if any training data was loaded.

#### without\_empty\_e2e\_examples

```python
 | without_empty_e2e_examples() -> "TrainingData"
```

Removes training data examples from intent labels and action names which
were added for end-to-end training.

**Returns**:

  Itself but without training examples which don&#x27;t have a text or intent.


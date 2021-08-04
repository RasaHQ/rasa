---
sidebar_label: diet_classifier
title: rasa.nlu.classifiers.diet_classifier
---

## EntityTagSpec Objects

```python
class EntityTagSpec(NamedTuple)
```

Specification of an entity tag present in the training data.

## DIETClassifier Objects

```python
class DIETClassifier(IntentClassifier,  EntityExtractor)
```

DIET (Dual Intent and Entity Transformer) is a multi-task architecture for
intent classification and entity recognition.

The architecture is based on a transformer which is shared for both tasks.
A sequence of entity labels is predicted through a Conditional Random Field (CRF)
tagging layer on top of the transformer output sequence corresponding to the
input sequence of tokens. The transformer output for the ``__CLS__`` token and
intent labels are embedded into a single semantic vector space. We use the
dot-product loss to maximize the similarity with the target label and minimize
similarities with negative samples.

#### \_\_init\_\_

```python
 | __init__(component_config: Optional[Dict[Text, Any]] = None, index_label_id_mapping: Optional[Dict[int, Text]] = None, entity_tag_specs: Optional[List[EntityTagSpec]] = None, model: Optional[RasaModel] = None) -> None
```

Declare instance variables with default values.

#### label\_key

```python
 | @property
 | label_key() -> Optional[Text]
```

Return key if intent classification is activated.

#### label\_sub\_key

```python
 | @property
 | label_sub_key() -> Optional[Text]
```

Return sub key if intent classification is activated.

#### preprocess\_train\_data

```python
 | preprocess_train_data(training_data: TrainingData) -> RasaModelData
```

Prepares data for training.

Performs sanity checks on training data, extracts encodings for labels.

#### train

```python
 | train(training_data: TrainingData, config: Optional[RasaNLUModelConfig] = None, **kwargs: Any, ,) -> None
```

Train the embedding intent classifier on a data set.

#### process

```python
 | process(message: Message, **kwargs: Any) -> None
```

Return the most likely label and its similarity to the input.

#### persist

```python
 | persist(file_name: Text, model_dir: Text) -> Dict[Text, Any]
```

Persist this model into the passed directory.

Return the metadata necessary to load the model again.

#### load

```python
 | @classmethod
 | load(cls, meta: Dict[Text, Any], model_dir: Text = None, model_metadata: Metadata = None, cached_component: Optional["DIETClassifier"] = None, **kwargs: Any, ,) -> "DIETClassifier"
```

Loads the trained model from the provided directory.


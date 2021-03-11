---
sidebar_label: spacy_utils
title: rasa.nlu.utils.spacy_utils
---

## SpacyNLP Objects

```python
class SpacyNLP(Component)
```

#### load\_model

```python
 | @staticmethod
 | load_model(spacy_model_name: Text) -> "Language"
```

Try loading the model, catching the OSError if missing.

#### merge\_content\_lists

```python
 | @staticmethod
 | merge_content_lists(indexed_training_samples: List[Tuple[int, Text]], doc_lists: List[Tuple[int, "Doc"]]) -> List[Tuple[int, "Doc"]]
```

Merge lists with processed Docs back into their original order.

#### filter\_training\_samples\_by\_content

```python
 | @staticmethod
 | filter_training_samples_by_content(indexed_training_samples: List[Tuple[int, Text]]) -> Tuple[List[Tuple[int, Text]], List[Tuple[int, Text]]]
```

Separates empty training samples from content bearing ones.

#### process\_content\_bearing\_samples

```python
 | process_content_bearing_samples(samples_to_pipe: List[Tuple[int, Text]]) -> List[Tuple[int, "Doc"]]
```

Sends content bearing training samples to spaCy&#x27;s pipe.

#### process\_non\_content\_bearing\_samples

```python
 | process_non_content_bearing_samples(empty_samples: List[Tuple[int, Text]]) -> List[Tuple[int, "Doc"]]
```

Creates empty Doc-objects from zero-lengthed training samples strings.

#### ensure\_proper\_language\_model

```python
 | @staticmethod
 | ensure_proper_language_model(nlp: Optional["Language"]) -> None
```

Checks if the spacy language model is properly loaded.

Raises an exception if the model is invalid.


---
sidebar_label: rasa.nlu.utils._spacy_utils
title: rasa.nlu.utils._spacy_utils
---
## SpacyNLP Objects

```python
class SpacyNLP(Component)
```

The core component that links spaCy to related components in the pipeline.

#### load\_model

```python
@staticmethod
def load_model(spacy_model_name: Text) -> "Language"
```

Try loading the model, catching the OSError if missing.

#### provide\_context

```python
def provide_context() -> Dict[Text, Any]
```

Creates a context dictionary from spaCy nlp object.

#### doc\_for\_text

```python
def doc_for_text(text: Text) -> "Doc"
```

Makes a spaCy doc object from a string of text.

#### preprocess\_text

```python
def preprocess_text(text: Optional[Text]) -> Text
```

Processes the text before it is handled by spaCy.

#### merge\_content\_lists

```python
@staticmethod
def merge_content_lists(indexed_training_samples: List[Tuple[int, Text]], doc_lists: List[Tuple[int, "Doc"]]) -> List[Tuple[int, "Doc"]]
```

Merge lists with processed Docs back into their original order.

#### filter\_training\_samples\_by\_content

```python
@staticmethod
def filter_training_samples_by_content(indexed_training_samples: List[Tuple[int, Text]]) -> Tuple[List[Tuple[int, Text]], List[Tuple[int, Text]]]
```

Separates empty training samples from content bearing ones.

#### process\_content\_bearing\_samples

```python
def process_content_bearing_samples(samples_to_pipe: List[Tuple[int, Text]]) -> List[Tuple[int, "Doc"]]
```

Sends content bearing training samples to spaCy&#x27;s pipe.

#### process\_non\_content\_bearing\_samples

```python
def process_non_content_bearing_samples(empty_samples: List[Tuple[int, Text]]) -> List[Tuple[int, "Doc"]]
```

Creates empty Doc-objects from zero-lengthed training samples strings.

#### load

```python
@classmethod
def load(cls, meta: Dict[Text, Any], model_dir: Text, model_metadata: "Metadata" = None, cached_component: Optional["SpacyNLP"] = None, **kwargs: Any, ,) -> "SpacyNLP"
```

Loads trained component (see parent class for full docstring).

#### ensure\_proper\_language\_model

```python
@staticmethod
def ensure_proper_language_model(nlp: Optional["Language"]) -> None
```

Checks if the spacy language model is properly loaded.

Raises an exception if the model is invalid.


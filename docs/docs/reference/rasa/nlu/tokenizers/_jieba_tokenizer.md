---
sidebar_label: rasa.nlu.tokenizers._jieba_tokenizer
title: rasa.nlu.tokenizers._jieba_tokenizer
---
## JiebaTokenizer Objects

```python
class JiebaTokenizer(Tokenizer)
```

This tokenizer is a wrapper for Jieba (https://github.com/fxsjy/jieba).

#### \_\_init\_\_

```python
def __init__(component_config: Dict[Text, Any] = None) -> None
```

Construct a new intent classifier using the MITIE framework.

#### load\_custom\_dictionary

```python
@staticmethod
def load_custom_dictionary(path: Text) -> None
```

Load all the custom dictionaries stored in the path.

More information about the dictionaries file format can
be found in the documentation of jieba.
https://github.com/fxsjy/jieba#load-dictionary

#### load

```python
@classmethod
def load(cls, meta: Dict[Text, Any], model_dir: Text, model_metadata: Optional["Metadata"] = None, cached_component: Optional[Component] = None, **kwargs: Any, ,) -> "JiebaTokenizer"
```

Loads trained component (see parent class for full docstring).

#### persist

```python
def persist(file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]
```

Persist this model into the passed directory.


---
sidebar_label: jieba_tokenizer
title: rasa.nlu.tokenizers.jieba_tokenizer
---

## JiebaTokenizer Objects

```python
class JiebaTokenizer(Tokenizer)
```

#### \_\_init\_\_

```python
 | __init__(component_config: Dict[Text, Any] = None) -> None
```

Construct a new intent classifier using the MITIE framework.

#### load\_custom\_dictionary

```python
 | @staticmethod
 | load_custom_dictionary(path: Text) -> None
```

Load all the custom dictionaries stored in the path.

More information about the dictionaries file format can
be found in the documentation of jieba.
https://github.com/fxsjy/jieba#load-dictionary

#### persist

```python
 | persist(file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]
```

Persist this model into the passed directory.


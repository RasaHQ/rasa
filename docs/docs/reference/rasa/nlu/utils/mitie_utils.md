---
sidebar_label: rasa.nlu.utils.mitie_utils
title: rasa.nlu.utils.mitie_utils
---
## MitieNLP Objects

```python
class MitieNLP(Component)
```

#### \_\_init\_\_

```python
 | __init__(component_config: Optional[Dict[Text, Any]] = None, extractor: Optional["mitie.total_word_feature_extractor"] = None) -> None
```

Construct a new language model from the MITIE framework.

#### load

```python
 | @classmethod
 | load(cls, meta: Dict[Text, Any], model_dir: Text, model_metadata: Optional[Metadata] = None, cached_component: Optional["MitieNLP"] = None, **kwargs: Any, ,) -> "MitieNLP"
```

Loads trained component (see parent class for full docstring).


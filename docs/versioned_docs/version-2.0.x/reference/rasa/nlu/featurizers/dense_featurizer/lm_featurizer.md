---
sidebar_label: rasa.nlu.featurizers.dense_featurizer.lm_featurizer
title: rasa.nlu.featurizers.dense_featurizer.lm_featurizer
---

## LanguageModelFeaturizer Objects

```python
class LanguageModelFeaturizer(DenseFeaturizer)
```

Featurizer using transformer based language models.

Uses the output of HFTransformersNLP component to set the sequence and sentence
level representations for dense featurizable attributes of each message object.

#### process

```python
 | process(message: Message, **kwargs: Any) -> None
```

Sets the dense features from the language model doc to the incoming
message.


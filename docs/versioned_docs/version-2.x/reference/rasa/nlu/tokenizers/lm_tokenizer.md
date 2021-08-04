---
sidebar_label: rasa.nlu.tokenizers.lm_tokenizer
title: rasa.nlu.tokenizers.lm_tokenizer
---

## LanguageModelTokenizer Objects

```python
class LanguageModelTokenizer(Tokenizer)
```

Tokenizer using transformer based language models.

Uses the output of HFTransformersNLP component to set the tokens
for dense featurizable attributes of each message object.


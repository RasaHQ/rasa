---
sidebar_label: rasa.nlu.tokenizers.convert_tokenizer
title: rasa.nlu.tokenizers.convert_tokenizer
---

## ConveRTTokenizer Objects

```python
class ConveRTTokenizer(WhitespaceTokenizer)
```

Tokenizer using ConveRT model.
Loads the ConveRT(https://github.com/PolyAI-LDN/polyai-models#convert)
model from TFHub and computes sub-word tokens for dense
featurizable attributes of each message object.

#### \_\_init\_\_

```python
 | __init__(component_config: Dict[Text, Any] = None) -> None
```

Construct a new tokenizer using the WhitespaceTokenizer framework.

#### tokenize

```python
 | tokenize(message: Message, attribute: Text) -> List[Token]
```

Tokenize the text using the ConveRT model.
ConveRT adds a special char in front of (some) words and splits words into
sub-words. To ensure the entity start and end values matches the token values,
tokenize the text first using the whitespace tokenizer. If individual tokens
are split up into multiple tokens, add this information to the
respected tokens.


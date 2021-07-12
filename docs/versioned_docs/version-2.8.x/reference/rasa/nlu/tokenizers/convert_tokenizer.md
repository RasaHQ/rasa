---
sidebar_label: rasa.nlu.tokenizers.convert_tokenizer
title: rasa.nlu.tokenizers.convert_tokenizer
---
## ConveRTTokenizer Objects

```python
class ConveRTTokenizer(WhitespaceTokenizer)
```

This tokenizer is deprecated and will be removed in the future.

The ConveRTFeaturizer component now sets the sub-token information
for dense featurizable attributes of each message object.

#### \_\_init\_\_

```python
 | __init__(component_config: Dict[Text, Any] = None) -> None
```

Initializes ConveRTTokenizer with the ConveRT model.

**Arguments**:

- `component_config` - Configuration for the component.


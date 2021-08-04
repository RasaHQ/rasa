---
sidebar_label: rasa.nlu.utils.hugging_face.hf_transformers
title: rasa.nlu.utils.hugging_face.hf_transformers
---

## HFTransformersNLP Objects

```python
class HFTransformersNLP(Component)
```

Utility Component for interfacing between Transformers library and Rasa OS.

The transformers(https://github.com/huggingface/transformers) library
is used to load pre-trained language models like BERT, GPT-2, etc.
The component also tokenizes and featurizes dense featurizable attributes of each
message.

#### train

```python
 | train(training_data: TrainingData, config: Optional[RasaNLUModelConfig] = None, **kwargs: Any, ,) -> None
```

Compute tokens and dense features for each message in training data.

**Arguments**:

- `training_data` - NLU training data to be tokenized and featurized
- `config` - NLU pipeline config consisting of all components.

#### process

```python
 | process(message: Message, **kwargs: Any) -> None
```

Process an incoming message by computing its tokens and dense features.

**Arguments**:

- `message` - Incoming message object


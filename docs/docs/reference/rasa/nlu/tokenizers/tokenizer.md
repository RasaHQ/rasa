---
sidebar_label: rasa.nlu.tokenizers.tokenizer
title: rasa.nlu.tokenizers.tokenizer
---
## TokenizerGraphComponent Objects

```python
class TokenizerGraphComponent(GraphComponent,  abc.ABC)
```

Base class for tokenizers.

#### \_\_init\_\_

```python
 | __init__(config: Dict[Text, Any]) -> None
```

Construct a new tokenizer.

#### tokenize

```python
 | @abc.abstractmethod
 | tokenize(message: Message, attribute: Text) -> List[Token]
```

Tokenizes the text of the provided attribute of the incoming message.

#### process\_training\_data

```python
 | process_training_data(training_data: TrainingData) -> TrainingData
```

Tokenize all training data.

#### process

```python
 | process(messages: List[Message]) -> List[Message]
```

Tokenize the incoming messages.


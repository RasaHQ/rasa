---
sidebar_label: rasa.nlu.tokenizers.tokenizer
title: rasa.nlu.tokenizers.tokenizer
---
## Token Objects

```python
class Token()
```

Used by `Tokenizers` which split a single message into multiple `Token`s.

#### \_\_init\_\_

```python
 | __init__(text: Text, start: int, end: Optional[int] = None, data: Optional[Dict[Text, Any]] = None, lemma: Optional[Text] = None) -> None
```

Create a `Token`.

**Arguments**:

- `text` - The token text.
- `start` - The start index of the token within the entire message.
- `end` - The end index of the token within the entire message.
- `data` - Additional token data.
- `lemma` - An optional lemmatized version of the token text.

#### set

```python
 | set(prop: Text, info: Any) -> None
```

Set property value.

#### get

```python
 | get(prop: Text, default: Optional[Any] = None) -> Any
```

Returns token value.

#### fingerprint

```python
 | fingerprint() -> Text
```

Returns a stable hash for this Token.

## Tokenizer Objects

```python
class Tokenizer(GraphComponent,  abc.ABC)
```

Base class for tokenizers.

#### \_\_init\_\_

```python
 | __init__(config: Dict[Text, Any]) -> None
```

Construct a new tokenizer.

#### create

```python
 | @classmethod
 | create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> GraphComponent
```

Creates a new component (see parent class for full docstring).

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


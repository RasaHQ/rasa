---
sidebar_label: rasa.nlu.extractors.crf_entity_extractor
title: rasa.nlu.extractors.crf_entity_extractor
---
## CRFEntityExtractorOptions Objects

```python
class CRFEntityExtractorOptions(str,  Enum)
```

Features that can be used for the &#x27;CRFEntityExtractor&#x27;.

## CRFEntityExtractor Objects

```python
@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.ENTITY_EXTRACTOR, is_trainable=True
)
class CRFEntityExtractor(GraphComponent,  EntityExtractorMixin)
```

Implements conditional random fields (CRF) to do named entity recognition.

#### required\_components

```python
@classmethod
def required_components(cls) -> List[Type]
```

Components that should be included in the pipeline before this component.

#### get\_default\_config

```python
@staticmethod
def get_default_config() -> Dict[Text, Any]
```

The component&#x27;s default config (see parent class for full docstring).

#### \_\_init\_\_

```python
def __init__(config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, entity_taggers: Optional[Dict[Text, "CRF"]] = None) -> None
```

Creates an instance of entity extractor.

#### create

```python
@classmethod
def create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> CRFEntityExtractor
```

Creates a new untrained component (see parent class for full docstring).

#### required\_packages

```python
@staticmethod
def required_packages() -> List[Text]
```

Any extra python dependencies required for this component to run.

#### train

```python
def train(training_data: TrainingData) -> Resource
```

Trains the extractor on a data set.

#### process

```python
def process(messages: List[Message]) -> List[Message]
```

Augments messages with entities.

#### extract\_entities

```python
def extract_entities(message: Message) -> List[Dict[Text, Any]]
```

Extract entities from the given message using the trained model(s).

#### load

```python
@classmethod
def load(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext, **kwargs: Any, ,) -> CRFEntityExtractor
```

Loads trained component (see parent class for full docstring).

#### persist

```python
def persist() -> None
```

Persist this model into the passed directory.


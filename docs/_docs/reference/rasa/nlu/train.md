---
sidebar_label: rasa.nlu.train
title: rasa.nlu.train
---
## TrainingException Objects

```python
class TrainingException(Exception)
```

Exception wrapping lower level exceptions that may happen while training

**Attributes**:

- `failed_target_project` - name of the failed project
- `message` - explanation of why the request is invalid

#### load\_data\_from\_endpoint

```python
async load_data_from_endpoint(data_endpoint: EndpointConfig, language: Optional[Text] = "en") -> "TrainingData"
```

Load training data from a URL.

#### create\_persistor

```python
create_persistor(persistor: Optional[Text]) -> Optional["Persistor"]
```

Create a remote persistor to store the model if configured.

#### train

```python
async train(nlu_config: Union[Text, Dict, RasaNLUModelConfig], data: Union[Text, "TrainingDataImporter"], path: Optional[Text] = None, fixed_model_name: Optional[Text] = None, storage: Optional[Text] = None, component_builder: Optional[ComponentBuilder] = None, training_data_endpoint: Optional[EndpointConfig] = None, persist_nlu_training_data: bool = False, model_to_finetune: Optional[Interpreter] = None, **kwargs: Any, ,) -> Tuple[Trainer, Interpreter, Optional[Text]]
```

Loads the trainer and the data and runs the training of the model.


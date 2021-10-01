---
sidebar_label: rasa.nlu.components
title: rasa.nlu.components
---
#### validate\_requirements

```python
def validate_requirements(component_names: List[Optional[Text]]) -> None
```

Validates that all required importable python packages are installed.

**Raises**:

- `InvalidConfigException` - If one of the component names is `None`, likely
  indicates that a custom implementation is missing this property
  or that there is an invalid configuration file that we did not
  catch earlier.
  

**Arguments**:

- `component_names` - The list of component names.

#### validate\_component\_keys

```python
def validate_component_keys(component: "Component", component_config: Dict[Text, Any]) -> None
```

Validates that all keys for a component are valid.

**Arguments**:

- `component` - The component class
- `component_config` - The user-provided config for the component in the pipeline

#### validate\_empty\_pipeline

```python
def validate_empty_pipeline(pipeline: List["Component"]) -> None
```

Ensures the pipeline is not empty.

**Arguments**:

- `pipeline` - the list of the :class:`rasa.nlu.components.Component`.

#### validate\_only\_one\_tokenizer\_is\_used

```python
def validate_only_one_tokenizer_is_used(pipeline: List["Component"]) -> None
```

Validates that only one tokenizer is present in the pipeline.

**Arguments**:

- `pipeline` - the list of the :class:`rasa.nlu.components.Component`.

#### validate\_required\_components

```python
def validate_required_components(pipeline: List["Component"]) -> None
```

Validates that all required components are present in the pipeline.

**Arguments**:

- `pipeline` - The list of the :class:`rasa.nlu.components.Component`.

#### validate\_pipeline

```python
def validate_pipeline(pipeline: List["Component"]) -> None
```

Validates the pipeline.

**Arguments**:

- `pipeline` - The list of the :class:`rasa.nlu.components.Component`.

#### any\_components\_in\_pipeline

```python
def any_components_in_pipeline(components: Iterable[Text], pipeline: List["Component"]) -> bool
```

Check if any of the provided components are listed in the pipeline.

**Arguments**:

- `components` - Component class names to check.
- `pipeline` - A list of :class:`rasa.nlu.components.Component`s.
  

**Returns**:

  `True` if any of the `components` are in the `pipeline`, else `False`.

#### find\_components\_in\_pipeline

```python
def find_components_in_pipeline(components: Iterable[Text], pipeline: List["Component"]) -> Set[Text]
```

Finds those of the given components that are present in the pipeline.

**Arguments**:

- `components` - A list of str of component class names to check.
- `pipeline` - A list of :class:`rasa.nlu.components.Component`s.
  

**Returns**:

  A list of str of component class names that are present in the pipeline.

#### validate\_required\_components\_from\_data

```python
def validate_required_components_from_data(pipeline: List["Component"], data: TrainingData) -> None
```

Validates that all components are present in the pipeline based on data.

**Arguments**:

- `pipeline` - The list of the :class:`rasa.nlu.components.Component`s.
- `data` - The :class:`rasa.shared.nlu.training_data.training_data.TrainingData`.

#### warn\_of\_competing\_extractors

```python
def warn_of_competing_extractors(pipeline: List["Component"]) -> None
```

Warns the user when using competing extractors.

Competing extractors are e.g. `CRFEntityExtractor` and `DIETClassifier`.
Both of these look for the same entities based on the same training data
leading to ambiguity in the results.

**Arguments**:

- `pipeline` - The list of the :class:`rasa.nlu.components.Component`s.

#### warn\_of\_competition\_with\_regex\_extractor

```python
def warn_of_competition_with_regex_extractor(pipeline: List["Component"], data: TrainingData) -> None
```

Warns when regex entity extractor is competing with a general one.

This might be the case when the following conditions are all met:
* You are using a general entity extractor and the `RegexEntityExtractor`
* AND you have regex patterns for entity type A
* AND you have annotated text examples for entity type A

**Arguments**:

- `pipeline` - The list of the :class:`rasa.nlu.components.Component`s.
- `data` - The :class:`rasa.shared.nlu.training_data.training_data.TrainingData`.

## MissingArgumentError Objects

```python
class MissingArgumentError(ValueError)
```

Raised when not all parameters can be filled from the context / config.

**Attributes**:

- `message` - explanation of which parameter is missing

## UnsupportedLanguageError Objects

```python
class UnsupportedLanguageError(RasaException)
```

Raised when a component is created but the language is not supported.

**Attributes**:

- `component` - component name
- `language` - language that component doesn&#x27;t support

## ComponentMetaclass Objects

```python
class ComponentMetaclass(type)
```

Metaclass with `name` class property.

#### name

```python
@property
def name(cls) -> Text
```

The name property is a function of the class - its __name__.

## Component Objects

```python
class Component(, metaclass=ComponentMetaclass)
```

A component is a message processing unit in a pipeline.

Components are collected sequentially in a pipeline. Each component
is called one after another. This holds for
initialization, training, persisting and loading the components.
If a component comes first in a pipeline, its
methods will be called first.

E.g. to process an incoming message, the ``process`` method of
each component will be called. During the processing
(as well as the training, persisting and initialization)
components can pass information to other components.
The information is passed to other components by providing
attributes to the so called pipeline context. The
pipeline context contains all the information of the previous
components a component can use to do its own
processing. For example, a featurizer component can provide
features that are used by another component down
the pipeline to do intent classification.

#### name

```python
@property
def name() -> Text
```

Returns the name of the component to be used in the model configuration.

Component class name is used when integrating it in a
pipeline. E.g. `[ComponentA, ComponentB]`
will be a proper pipeline definition where `ComponentA`
is the name of the first component of the pipeline.

#### unique\_name

```python
@property
def unique_name() -> Text
```

Gets a unique name for the component in the pipeline.

The unique name can be used to distinguish components in
a pipeline, e.g. when the pipeline contains multiple
featurizers of the same type.

#### required\_components

```python
@classmethod
def required_components(cls) -> List[Type["Component"]]
```

Specifies which components need to be present in the pipeline.

Which components are required by this component.
Listed components should appear before the component itself in the pipeline.

**Returns**:

  The class names of the required components.

#### required\_packages

```python
@classmethod
def required_packages(cls) -> List[Text]
```

Specifies which python packages need to be installed.

E.g. ``[&quot;spacy&quot;]``. More specifically, these should be
importable python package names e.g. `sklearn` and not package
names in the dependencies sense e.g. `scikit-learn`

This list of requirements allows us to fail early during training
if a required package is not installed.

**Returns**:

  The list of required package names.

#### load

```python
@classmethod
def load(cls, meta: Dict[Text, Any], model_dir: Text, model_metadata: Optional["Metadata"] = None, cached_component: Optional["Component"] = None, **kwargs: Any, ,) -> "Component"
```

Loads this component from file.

After a component has been trained, it will be persisted by
calling `persist`. When the pipeline gets loaded again,
this component needs to be able to restore itself.
Components can rely on any context attributes that are
created by :meth:`components.Component.create`
calls to components previous to this one.

**Arguments**:

- `meta` - Any configuration parameter related to the model.
- `model_dir` - The directory to load the component from.
- `model_metadata` - The model&#x27;s :class:`rasa.nlu.model.Metadata`.
- `cached_component` - The cached component.
  

**Returns**:

  the loaded component

#### create

```python
@classmethod
def create(cls, component_config: Dict[Text, Any], config: RasaNLUModelConfig) -> "Component"
```

Creates this component (e.g. before a training is started).

Method can access all configuration parameters.

**Arguments**:

- `component_config` - The components configuration parameters.
- `config` - The model configuration parameters.
  

**Returns**:

  The created component.

#### provide\_context

```python
def provide_context() -> Optional[Dict[Text, Any]]
```

Initializes this component for a new pipeline.

This function will be called before the training
is started and before the first message is processed using
the interpreter. The component gets the opportunity to
add information to the context that is passed through
the pipeline during training and message parsing. Most
components do not need to implement this method.
It&#x27;s mostly used to initialize framework environments
like MITIE and spacy
(e.g. loading word vectors for the pipeline).

**Returns**:

  The updated component configuration.

#### train

```python
def train(training_data: TrainingData, config: Optional[RasaNLUModelConfig] = None, **kwargs: Any, ,) -> None
```

Trains this component.

This is the components chance to train itself provided
with the training data. The component can rely on
any context attribute to be present, that gets created
by a call to :meth:`rasa.nlu.components.Component.create`
of ANY component and
on any context attributes created by a call to
:meth:`rasa.nlu.components.Component.train`
of components previous to this one.

**Arguments**:

- `training_data` - The
  :class:`rasa.shared.nlu.training_data.training_data.TrainingData`.
- `config` - The model configuration parameters.

#### process

```python
def process(message: Message, **kwargs: Any) -> None
```

Processes an incoming message.

This is the components chance to process an incoming
message. The component can rely on
any context attribute to be present, that gets created
by a call to :meth:`rasa.nlu.components.Component.create`
of ANY component and
on any context attributes created by a call to
:meth:`rasa.nlu.components.Component.process`
of components previous to this one.

**Arguments**:

- `message` - The :class:`rasa.shared.nlu.training_data.message.Message` to
  process.

#### persist

```python
def persist(file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]
```

Persists this component to disk for future loading.

**Arguments**:

- `file_name` - The file name of the model.
- `model_dir` - The directory to store the model to.
  

**Returns**:

  An optional dictionary with any information about the stored model.

#### cache\_key

```python
@classmethod
def cache_key(cls, component_meta: Dict[Text, Any], model_metadata: "Metadata") -> Optional[Text]
```

This key is used to cache components.

If a component is unique to a model it should return None.
Otherwise, an instantiation of the
component will be reused for all models where the
metadata creates the same key.

**Arguments**:

- `component_meta` - The component configuration.
- `model_metadata` - The component&#x27;s :class:`rasa.nlu.model.Metadata`.
  

**Returns**:

  A unique caching key.

#### \_\_getstate\_\_

```python
def __getstate__() -> Any
```

Gets a copy of picklable parts of the component.

#### prepare\_partial\_processing

```python
def prepare_partial_processing(pipeline: List["Component"], context: Dict[Text, Any]) -> None
```

Sets the pipeline and context used for partial processing.

The pipeline should be a list of components that are
previous to this one in the pipeline and
have already finished their training (and can therefore
be safely used to process messages).

**Arguments**:

- `pipeline` - The list of components.
- `context` - The context of processing.

#### partially\_process

```python
def partially_process(message: Message) -> Message
```

Allows the component to process messages during
training (e.g. external training data).

The passed message will be processed by all components
previous to this one in the pipeline.

**Arguments**:

- `message` - The :class:`rasa.shared.nlu.training_data.message.Message` to
  process.
  

**Returns**:

  The processed :class:`rasa.shared.nlu.training_data.message.Message`.

#### can\_handle\_language

```python
@classmethod
def can_handle_language(cls, language: Hashable) -> bool
```

Check if component supports a specific language.

This method can be overwritten when needed. (e.g. dynamically
determine which language is supported.)

**Arguments**:

- `language` - The language to check.
  

**Returns**:

  `True` if component can handle specific language, `False` otherwise.

## ComponentBuilder Objects

```python
class ComponentBuilder()
```

Creates trainers and interpreters based on configurations.

Caches components for reuse.

#### load\_component

```python
def load_component(component_meta: Dict[Text, Any], model_dir: Text, model_metadata: "Metadata", **context: Any, ,) -> Optional[Component]
```

Loads a component.

Tries to retrieve a component from the cache, else calls
``load`` to create a new component.

**Arguments**:

  component_meta:
  The metadata of the component to load in the pipeline.
  model_dir:
  The directory to read the model from.
  model_metadata (Metadata):
  The model&#x27;s :class:`rasa.nlu.model.Metadata`.
  

**Returns**:

  The loaded component.

#### create\_component

```python
def create_component(component_config: Dict[Text, Any], cfg: RasaNLUModelConfig) -> Component
```

Creates a component.

Tries to retrieve a component from the cache,
calls `create` to create a new component.

**Arguments**:

- `component_config` - The component configuration.
- `cfg` - The model configuration.
  

**Returns**:

  The created component.


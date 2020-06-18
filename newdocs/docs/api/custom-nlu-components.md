# Custom NLU Components

You can create a custom component to perform a specific task which NLU doesn’t currently offer (for example, sentiment analysis).
Below is the specification of the `rasa.nlu.components.Component` class with the methods you’ll need to implement.

**NOTE**: There is a detailed tutorial on building custom components [here](https://blog.rasa.com/enhancing-rasa-nlu-with-custom-components/).

You can add a custom component to your pipeline by adding the module path.
So if you have a module called `sentiment`
containing a `SentimentAnalyzer` class:

> ```
> pipeline:
> - name: "sentiment.SentimentAnalyzer"
> ```

Also be sure to read the section on the Component Lifecycle.

To get started, you can use this skeleton that contains the most important
methods that you should implement:

```
import typing
from typing import Any, Optional, Text, Dict, List, Type

from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import Message, TrainingData

if typing.TYPE_CHECKING:
    from rasa.nlu.model import Metadata


class MyComponent(Component):
    """A new component"""

    # Which components are required by this component.
    # Listed components should appear before the component itself in the pipeline.
    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        """Specify which components need to be present in the pipeline."""

        return []

    # Defines the default configuration parameters of a component
    # these values can be overwritten in the pipeline configuration
    # of the model. The component should choose sensible defaults
    # and should be able to create reasonable results with the defaults.
    defaults = {}

    # Defines what language(s) this component can handle.
    # This attribute is designed for instance method: `can_handle_language`.
    # Default value is None which means it can handle all languages.
    # This is an important feature for backwards compatibility of components.
    language_list = None

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:
        super().__init__(component_config)

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Train this component.

        This is the components chance to train itself provided
        with the training data. The component can rely on
        any context attribute to be present, that gets created
        by a call to :meth:`components.Component.pipeline_init`
        of ANY component and
        on any context attributes created by a call to
        :meth:`components.Component.train`
        of components previous to this one."""
        pass

    def process(self, message: Message, **kwargs: Any) -> None:
        """Process an incoming message.

        This is the components chance to process an incoming
        message. The component can rely on
        any context attribute to be present, that gets created
        by a call to :meth:`components.Component.pipeline_init`
        of ANY component and
        on any context attributes created by a call to
        :meth:`components.Component.process`
        of components previous to this one."""
        pass

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this component to disk for future loading."""

        pass

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Optional[Text] = None,
        model_metadata: Optional["Metadata"] = None,
        cached_component: Optional["Component"] = None,
        **kwargs: Any,
    ) -> "Component":
        """Load this component from file."""

        if cached_component:
            return cached_component
        else:
            return cls(meta)
```

**NOTE**: If you create a custom tokenizer you should implement the methods of `rasa.nlu.tokenizers.tokenizer.Tokenizer`.
The `train` and `process` methods are already implemented and you simply need to overwrite the `tokenize`
method. `train` and `process` will automatically add a special token `__CLS__` to the end of list of tokens,
which is needed further down the pipeline.

**NOTE**: If you create a custom featurizer you should return a sequence of features.
E.g. your featurizer should return a matrix of size (number-of-tokens x feature-dimension).
The feature vector of the `__CLS__` token should contain features for the complete message.

## Component


### class rasa.nlu.components.Component(component_config=None)
A component is a message processing unit in a pipeline.

Components are collected sequentially in a pipeline. Each component
is called one after another. This holds for
initialization, training, persisting and loading the components.
If a component comes first in a pipeline, its
methods will be called first.

E.g. to process an incoming message, the `process` method of
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


#### classmethod required_packages()
Specify which python packages need to be installed to use this
component, e.g. `["spacy"]`.

This list of requirements allows us to fail early during training
if a required package is not installed.


* **Return type**

    `List`[`str`]



#### classmethod create(component_config, config)
Creates this component (e.g. before a training is started).

Method can access all configuration parameters.


* **Return type**

    `Component`



#### provide_context()
Initialize this component for a new pipeline

This function will be called before the training
is started and before the first message is processed using
the interpreter. The component gets the opportunity to
add information to the context that is passed through
the pipeline during training and message parsing. Most
components do not need to implement this method.
It’s mostly used to initialize framework environments
like MITIE and spacy
(e.g. loading word vectors for the pipeline).


* **Return type**

    `Optional`[`Dict`[`str`, `Any`]]



#### train(training_data, cfg, \*\*kwargs)
Train this component.

This is the components chance to train itself provided
with the training data. The component can rely on
any context attribute to be present, that gets created
by a call to `rasa.nlu.components.Component.create()`
of ANY component and
on any context attributes created by a call to
`rasa.nlu.components.Component.train()`
of components previous to this one.


* **Return type**

    `None`



#### process(message, \*\*kwargs)
Process an incoming message.

This is the components chance to process an incoming
message. The component can rely on
any context attribute to be present, that gets created
by a call to `rasa.nlu.components.Component.create()`
of ANY component and
on any context attributes created by a call to
`rasa.nlu.components.Component.process()`
of components previous to this one.


* **Return type**

    `None`



#### persist(file_name, model_dir)
Persist this component to disk for future loading.


* **Return type**

    `Optional`[`Dict`[`str`, `Any`]]



#### prepare_partial_processing(pipeline, context)
Sets the pipeline and context used for partial processing.

The pipeline should be a list of components that are
previous to this one in the pipeline and
have already finished their training (and can therefore
be safely used to process messages).


* **Return type**

    `None`



#### partially_process(message)
Allows the component to process messages during
training (e.g. external training data).

The passed message will be processed by all components
previous to this one in the pipeline.


* **Return type**

    `Message`



#### classmethod can_handle_language(language)
Check if component supports a specific language.

This method can be overwritten when needed. (e.g. dynamically
determine which language is supported.)


* **Return type**

    `bool`

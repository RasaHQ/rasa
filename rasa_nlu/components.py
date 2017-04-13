from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import logging
from builtins import object
import inspect

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Text
from typing import Tuple
from typing import Type

from rasa_nlu.config import RasaNLUConfig


def load_component(component_clz, context, config):
    # type: (Type[Component], Dict[Text, Any], Dict[Text, Any]) -> Optional[Component]
    """Calls a components load method to init it based on a previously persisted model."""

    if component_clz is not None:
        load_args = fill_args(component_clz.load_args(), context, config)
        return component_clz.load(*load_args)
    else:
        return None


def create_component(component_clz, config):
    # type: (Type[Component], Dict[Text, Any]) -> Optional[Component]
    """Calls a components load method to init it based on a previously persisted model."""

    if component_clz is not None:
        create_args = fill_args(component_clz.create_args(), context={}, config=config)
        return component_clz.create(*create_args)
    else:
        return None


def fill_args(arguments, context, config):
    # type: (List[Text], Dict[Text, Any], Dict[Text, Any]) -> List[Any]
    """Given a list of arguments, tries to look up these argument names in the config / context to fill the arguments"""

    filled = []
    for arg in arguments:
        if arg in context:
            filled.append(context[arg])
        elif arg in config:
            filled.append(config[arg])
        else:
            raise MissingArgumentError("Couldn't fill argument '{}' :(".format(arg))
    return filled


class MissingArgumentError(ValueError):
    """Raised when a function is called and not all parameters can be filled from the context / config.

    Attributes:
        message -- explanation of which parameter is missing
    """

    def __init__(self, message):
        # type: (Text) -> None
        super(MissingArgumentError, self).__init__(message)


class Component(object):
    """A component is a message processing unit in a pipeline.

    Components are collected sequentially in a pipeline. Each component is called one after another. This holds for
     initialization, training, persisting and loading the components. If a component comes first in a pipeline, its
     methods will be called first.

    E.g. to process an incoming message, the `process` method of each component will be called. During the processing
     (as well as the training, persisting and initialization) components can pass information to other components.
     The information is passed to other components by providing attributes to the so called pipeline context. The
     pipeline context contains all the information of the previous components a component can use to do its own
     processing. For example, a featurizer component can provide features that are used by another component down
     the pipeline to do intent classification."""

    # Name of the component to be used when integrating it in a pipeline. E.g. `[ComponentA, ComponentB]`
    # will be a proper pipeline definition where `ComponentA` is the name of the first component of the pipeline.
    name = ""

    # Defines what attributes the pipeline component will provide when called. The different keys indicate the
    # different functions (`pipeline_init`, `train`, `process`) that are able to update the pipelines context.
    # (mostly used to check if the pipeline is valid)
    context_provides = {
        "pipeline_init": [],
        "train": [],
        "process": [],
    }

    # Defines which of the attributes the component provides should be added to the final output json at the end of the
    # pipeline. Every attribute in `output_provides` should be part of the above `context_provides['process']`. As it
    # wouldn't make much sense to keep an attribute in the output that is not generated. Every other attribute provided
    # in the context during the process step will be removed from the output json.
    output_provides = []

    @classmethod
    def load(cls, *args):
        # type: (*Any) -> Component
        """Load this component from file.

        After a component got trained, it will be persisted by calling `persist`. When the pipeline gets loaded again,
         this component needs to be able to restore itself. Components can rely on any context attributes that are
         created by `pipeline_init` calls to components previous to this one."""
        return cls(*args)

    @classmethod
    def create(cls, *args):
        # type: (*Any) -> Component
        """Creates this component (e.g. before a training is started).

        Method can access all configuration parameters."""
        return cls(*args)

    def pipeline_init(self, *args):
        # type: (*Any) -> Optional[Dict[Text, Any]]
        """Initialize this component for a new pipeline

        This function will be called before the training is started and before the first message is processed using
        the interpreter. The component gets the opportunity to add information to the context that is passed through
        the pipeline during training and message parsing. Most components do not need to implement this method.
        It's mostly used to initialize framework environments like MITIE and spacy
        (e.g. loading word vectors for the pipeline)."""
        pass

    def train(self, *args):
        # type: (*Any) -> Optional[Dict[Text, Any]]
        """Train this component.

        This is the components chance to train itself provided with the training data. The component can rely on
        any context attribute to be present, that gets created by a call to `pipeline_init` of ANY component and
        on any context attributes created by a call to `train` of components previous to this one."""
        pass

    def process(self, *args):
        # type: (*Any) -> Optional[Dict[Text, Any]]
        """Process an incomming message.

       This is the components chance to process an incommng message. The component can rely on
       any context attribute to be present, that gets created by a call to `pipeline_init` of ANY component and
       on any context attributes created by a call to `process` of components previous to this one."""
        pass

    def persist(self, model_dir):
        # type: (Text) -> Optional[Dict[Text, Any]]
        """Persist this component to disk for future loading."""
        pass

    @classmethod
    def cache_key(cls, model_metadata):
        # type: (Metadata) -> Optional[Text]
        """This key is used to cache components.

        If a component is unique to a model it should return None. Otherwise, an instantiation of the
        component will be reused for all models where the metadata creates the same key."""
        from rasa_nlu.model import Metadata

        return None

    def pipeline_init_args(self):
        # type: () -> List[Text]
        return [arg for arg in inspect.getargspec(self.pipeline_init).args if arg not in ["self"]]

    @classmethod
    def create_args(cls):
        # type: () -> List[Text]
        return [arg for arg in inspect.getargspec(cls.create).args if arg not in ["cls"]]

    def train_args(self):
        # type: () -> List[Text]
        return [arg for arg in inspect.getargspec(self.train).args if arg not in ["self"]]

    def process_args(self):
        # type: () -> List[Text]
        return [arg for arg in inspect.getargspec(self.process).args if arg not in ["self"]]

    @classmethod
    def load_args(cls):
        # type: () -> List[Text]
        return [arg for arg in inspect.getargspec(cls.load).args if arg not in ["cls"]]

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class ComponentBuilder(object):
    """Creates trainers and interpreters based on configurations. Caches components for reuse."""

    def __init__(self, use_cache=True):
        self.use_cache = use_cache
        # Reuse nlp and featurizers where possible to save memory,
        # every component that implements a cache-key will be cached
        self.component_cache = {}

    def __get_cached_component(self, component_name, metadata):
        # type: (Text, Metadata) -> Tuple[Optional[Component], Optional[Text]]
        """Load a component from the cache, if it exists. Returns the component, if found, and the cache key."""
        from rasa_nlu import registry
        from rasa_nlu.model import Metadata

        component_class = registry.get_component_class(component_name)
        if component_class is None:
            raise Exception("Failed to find component class for '{}'. Unknown component name.".format(component_name))

        cache_key = component_class.cache_key(metadata)
        if cache_key is not None and self.use_cache and cache_key in self.component_cache:
            return self.component_cache[cache_key], cache_key
        else:
            return None, cache_key

    def __add_to_cache(self, component, cache_key):
        # type: (Component, Text) -> None
        """Add a component to the cache."""

        if cache_key is not None and self.use_cache:
            self.component_cache[cache_key] = component
            logging.info("Added '{}' to component cache. Key '{}'.".format(component.name, cache_key))

    def load_component(self, component_name, context, model_config, meta):
        # type: (Text, Dict[Text, Any], Dict[Text, Any], Metadata) -> Component
        """Tries to retrieve a component from the cache, calls `load` to create a new component."""
        from rasa_nlu import registry
        from rasa_nlu.model import Metadata

        try:
            component, cache_key = self.__get_cached_component(component_name, meta)
            if component is None:
                component = registry.load_component_by_name(component_name, context, model_config)
                if component is None:
                    raise Exception(
                        "Failed to load component '{}'. Unknown component name.".format(component_name))
                self.__add_to_cache(component, cache_key)
            return component
        except MissingArgumentError as e:
            raise Exception("Failed to load component '{}'. {}".format(component_name, e.message))

    def create_component(self, component_name, config):
        # type: (Text, RasaNLUConfig) -> Component

        """Tries to retrieve a component from the cache, calls `create` to create a new component."""
        from rasa_nlu import registry
        from rasa_nlu.model import Metadata

        try:
            component, cache_key = self.__get_cached_component(component_name, Metadata(config.as_dict(), None))
            if component is None:
                component = registry.create_component_by_name(component_name, config.as_dict())
                if component is None:
                    raise Exception(
                        "Failed to create component '{}'. Unknown component name.".format(component_name))
                self.__add_to_cache(component, cache_key)
            return component
        except MissingArgumentError as e:
            raise Exception("Failed to create component '{}'. {}".format(component_name, e.message))

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import logging
import os
from collections import defaultdict

import typing
from builtins import object
import inspect

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Text
from typing import Tuple

from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.training_data import Message

if typing.TYPE_CHECKING:
    from rasa_nlu.training_data import TrainingData
    from rasa_nlu.model import Metadata

logger = logging.getLogger(__name__)


def _read_dev_requirements(file_name):
    """Reads the dev requirements and groups the pinned versions into sections indicated by comments in the file.

    The dev requirements should be grouped by preceeding comments. The comment should start with `#` followed by
    the name of the requirement, e.g. `# sklearn`. All following lines till the next line starting with `#` will be
    required to be installed if the name `sklearn` is requested to be available."""

    # pragma: no cover
    try:
        import pkg_resources
        req_lines = pkg_resources.resource_string("rasa_nlu", "../" + file_name).split("\n")
    except Exception as e:
        logger.info("Couldn't read dev-requirements.txt. Error: {}".format(e))
        req_lines = []
    return _requirements_from_lines(req_lines)


def _requirements_from_lines(req_lines):
    requirements = defaultdict(list)
    current_name = None
    for req_line in req_lines:
        if req_line.startswith("#"):
            current_name = req_line[1:].strip(' \n')
        elif current_name is not None:
            requirements[current_name].append(req_line.strip(' \n'))
    return requirements


def find_unavailable_packages(package_names):
    # type: (List[Text]) -> Set[Text]
    """Tries to import all the package names and returns the packages where it failed."""
    import importlib

    failed_imports = set()
    for package in package_names:
        try:
            importlib.import_module(package)
        except ImportError:
            failed_imports.add(package)
    return failed_imports


def validate_requirements(component_names, dev_requirements_file="dev-requirements.txt"):
    # type: (List[Text], Text) -> None
    """Ensures that all required python packages are installed to instantiate and used the passed components."""
    from rasa_nlu import registry

    # Validate that all required packages are installed
    failed_imports = set()
    for component_name in component_names:
        component_class = registry.get_component_class(component_name)
        failed_imports.update(find_unavailable_packages(component_class.required_packages()))
    if failed_imports:  # pragma: no cover
        # if available, use the development file to figure out the correct version numbers for each requirement
        all_requirements = _read_dev_requirements(dev_requirements_file)
        if all_requirements:
            missing_requirements = [r for i in failed_imports for r in all_requirements[i]]
            raise Exception("Not all required packages are installed. " +
                            "Failed to find the following imports {}. ".format(", ".join(failed_imports)) +
                            "To use this pipeline, you need to install the missing dependencies, e.g. by running:\n\t" +
                            "> pip install {}".format(" ".join(missing_requirements)))
        else:
            raise Exception("Not all required packages are installed. " +
                            "To use this pipeline, you need to install the missing dependencies. " +
                            "Please install {}".format(", ".join(failed_imports)))


def validate_arguments(pipeline, context, allow_empty_pipeline=False):
    # type: (List[Component], Dict[Text, Any], bool) -> None
    """Validates a pipeline before it is run. Ensures, that all arguments are present to train the pipeline."""

    # Ensure the pipeline is not empty
    if not allow_empty_pipeline and len(pipeline) == 0:
        raise ValueError("Can not train an empty pipeline. " +
                         "Make sure to specify a proper pipeline in the configuration using the `pipeline` key." +
                         "The `backend` configuration key is NOT supported anymore.")

    provided_properties = set(context.keys())

    for component in pipeline:
        for r in component.requires:
            if r not in provided_properties:
                raise Exception("Failed to validate at component '{}'. Missing property: '{}'".format(
                    component.name, r))
        provided_properties.update(component.provides)


class MissingArgumentError(ValueError):
    """Raised when a function is called and not all parameters can be filled from the context / config.

    Attributes:
        message -- explanation of which parameter is missing
    """

    def __init__(self, message):
        # type: (Text) -> None
        super(MissingArgumentError, self).__init__(message)
        self.message = message

    def __str__(self):
        return self.message


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

    # Defines what attributes the pipeline component will provide when called. The listed attributes
    # should be set by the component on the message object during test and train, e.g.
    # ```message.set("entities", [...])```
    provides = []

    # Which attributes on a message are required by this component. e.g. if requires contains "tokens", than a
    # previous component in the pipeline needs to have "tokens" within the above described `provides` property.
    requires = []

    def __init__(self):
        self.partial_processing_pipeline = None
        self.partial_processing_context = None

    def __getstate__(self):
        d = self.__dict__.copy()
        # these properties should not be pickled
        if "partial_processing_context" in d:
            del d["partial_processing_context"]
        if "partial_processing_pipeline" in d:
            del d["partial_processing_pipeline"]
        return d

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        """Specify which python packages need to be installed to use this component, e.g. `["spacy", "numpy"]`.

        This list of requirements allows us to fail early during training if a required package is not installed."""
        return []

    @classmethod
    def load(cls, model_dir=None, model_metadata=None, cached_component=None, **kwargs):
        # type: (Text, Metadata, Optional[Component], **Any) -> Component
        """Load this component from file.

        After a component got trained, it will be persisted by calling `persist`. When the pipeline gets loaded again,
         this component needs to be able to restore itself. Components can rely on any context attributes that are
         created by `pipeline_init` calls to components previous to this one."""
        return cached_component if cached_component else cls()

    @classmethod
    def create(cls, config):
        # type: (RasaNLUConfig) -> Component
        """Creates this component (e.g. before a training is started).

        Method can access all configuration parameters."""
        return cls()

    def provide_context(self):
        # type: () -> Optional[Dict[Text, Any]]
        """Initialize this component for a new pipeline

        This function will be called before the training is started and before the first message is processed using
        the interpreter. The component gets the opportunity to add information to the context that is passed through
        the pipeline during training and message parsing. Most components do not need to implement this method.
        It's mostly used to initialize framework environments like MITIE and spacy
        (e.g. loading word vectors for the pipeline)."""
        pass

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUConfig, **Any) -> None
        """Train this component.

        This is the components chance to train itself provided with the training data. The component can rely on
        any context attribute to be present, that gets created by a call to `pipeline_init` of ANY component and
        on any context attributes created by a call to `train` of components previous to this one."""
        pass

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
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

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def prepare_partial_processing(self, pipeline, context):
        """Sets the pipeline and context used for partial processing.

        The pipeline should be a list of components that are previous to this one in the pipeline and
        have already finished their training (and can therefore be safely used to process messages)."""

        self.partial_processing_pipeline = pipeline
        self.partial_processing_context = context

    def partially_process(self, message):
        """Allows the component to process messages during training (e.g. external training data).

        The passed message will be processed by all components previous to this one in the pipeline."""

        if self.partial_processing_context is not None:
            for component in self.partial_processing_pipeline:
                component.process(message, **self.partial_processing_context)
        else:
            logger.info("Failed to run partial processing due to missing pipeline.")
        return message


class ComponentBuilder(object):
    """Creates trainers and interpreters based on configurations. Caches components for reuse."""

    def __init__(self, use_cache=True):
        self.use_cache = use_cache
        # Reuse nlp and featurizers where possible to save memory,
        # every component that implements a cache-key will be cached
        self.component_cache = {}

    def __get_cached_component(self, component_name, model_metadata):
        # type: (Text, Metadata) -> Tuple[Optional[Component], Optional[Text]]
        """Load a component from the cache, if it exists. Returns the component, if found, and the cache key."""
        from rasa_nlu import registry
        from rasa_nlu.model import Metadata

        component_class = registry.get_component_class(component_name)
        cache_key = component_class.cache_key(model_metadata)
        if cache_key is not None and self.use_cache and cache_key in self.component_cache:
            return self.component_cache[cache_key], cache_key
        else:
            return None, cache_key

    def __add_to_cache(self, component, cache_key):
        # type: (Component, Text) -> None
        """Add a component to the cache."""

        if cache_key is not None and self.use_cache:
            self.component_cache[cache_key] = component
            logger.info("Added '{}' to component cache. Key '{}'.".format(component.name, cache_key))

    def load_component(self, component_name, model_dir, model_metadata, **context):
        # type: (Text, Text, Metadata, **Any) -> Component
        """Tries to retrieve a component from the cache, calls `load` to create a new component."""
        from rasa_nlu import registry
        from rasa_nlu.model import Metadata

        try:
            cached_component, cache_key = self.__get_cached_component(component_name, model_metadata)
            component = registry.load_component_by_name(component_name, model_dir,
                                                        model_metadata, cached_component, **context)
            if not cached_component:
                # If the component wasn't in the cache, let us add it if possible
                self.__add_to_cache(component, cache_key)
            return component
        except MissingArgumentError as e:  # pragma: no cover
            raise Exception("Failed to load component '{}'. {}".format(component_name, e))

    def create_component(self, component_name, config):
        # type: (Text, RasaNLUConfig) -> Component
        """Tries to retrieve a component from the cache, calls `create` to create a new component."""
        from rasa_nlu import registry
        from rasa_nlu.model import Metadata

        try:
            component, cache_key = self.__get_cached_component(component_name, Metadata(config.as_dict(), None))
            if component is None:
                component = registry.create_component_by_name(component_name, config)
                self.__add_to_cache(component, cache_key)
            return component
        except MissingArgumentError as e:  # pragma: no cover
            raise Exception("Failed to create component '{}'. {}".format(component_name, e))

import logging
import typing
from typing import Any, Dict, Hashable, List, Optional, Set, Text, Tuple

from rasa_nlu.config import RasaNLUModelConfig, override_defaults

if typing.TYPE_CHECKING:
    from rasa_nlu.training_data import TrainingData
    from rasa_nlu.model import Metadata
    from rasa_nlu.training_data import Message

logger = logging.getLogger(__name__)


def find_unavailable_packages(package_names: List[Text]) -> Set[Text]:
    """Tries to import all the package names and returns
    the packages where it failed."""
    import importlib

    failed_imports = set()
    for package in package_names:
        try:
            importlib.import_module(package)
        except ImportError:
            failed_imports.add(package)
    return failed_imports


def validate_requirements(component_names: List[Text]) -> None:
    """Ensures that all required python packages are installed to
    instantiate and used the passed components."""
    from rasa_nlu import registry

    # Validate that all required packages are installed
    failed_imports = set()
    for component_name in component_names:
        component_class = registry.get_component_class(component_name)
        failed_imports.update(find_unavailable_packages(
            component_class.required_packages()))
    if failed_imports:  # pragma: no cover
        # if available, use the development file to figure out the correct
        # version numbers for each requirement
        raise Exception("Not all required packages are installed. " +
                        "To use this pipeline, you need to install the "
                        "missing dependencies. " +
                        "Please install {}".format(", ".join(failed_imports)))


def validate_arguments(pipeline: List['Component'],
                       context: Dict[Text, Any],
                       allow_empty_pipeline: bool = False) -> None:
    """Validates a pipeline before it is run. Ensures, that all
    arguments are present to train the pipeline."""

    # Ensure the pipeline is not empty
    if not allow_empty_pipeline and len(pipeline) == 0:
        raise ValueError("Can not train an empty pipeline. "
                         "Make sure to specify a proper pipeline in "
                         "the configuration using the `pipeline` key." +
                         "The `backend` configuration key is "
                         "NOT supported anymore.")

    provided_properties = set(context.keys())

    for component in pipeline:
        for r in component.requires:
            if r not in provided_properties:
                raise Exception("Failed to validate at component "
                                "'{}'. Missing property: '{}'"
                                "".format(component.name, r))
        provided_properties.update(component.provides)


class MissingArgumentError(ValueError):
    """Raised when a function is called and not all parameters can be
    filled from the context / config.

    Attributes:
        message -- explanation of which parameter is missing
    """

    def __init__(self, message: Text) -> None:
        super(MissingArgumentError, self).__init__(message)
        self.message = message

    def __str__(self) -> Text:
        return self.message


class UnsupportedLanguageError(Exception):
    """Raised when a component is created but the language is not supported.

    Attributes:
        component -- component name
        language -- language that component doesn't support
    """

    def __init__(self, component: Text, language: Text) -> None:
        self.component = component
        self.language = language

        super(UnsupportedLanguageError, self).__init__(component, language)

    def __str__(self) -> Text:
        return ("component {} does not support language {}"
                "".format(self.component, self.language))


class ComponentMetaclass(type):
    """Metaclass with `name` class property"""

    @property
    def name(cls):
        """The name property is a function of the class - its __name__."""

        return cls.__name__


class Component(object, metaclass=ComponentMetaclass):
    """A component is a message processing unit in a pipeline.

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
    the pipeline to do intent classification."""

    # Component class name is used when integrating it in a
    # pipeline. E.g. ``[ComponentA, ComponentB]``
    # will be a proper pipeline definition where ``ComponentA``
    # is the name of the first component of the pipeline.
    @property
    def name(self):
        """Access the class's property name from an instance."""

        return type(self).name

    # Defines what attributes the pipeline component will
    # provide when called. The listed attributes
    # should be set by the component on the message object
    # during test and train, e.g.
    # ```message.set("entities", [...])```
    provides = []

    # Which attributes on a message are required by this
    # component. e.g. if requires contains "tokens", than a
    # previous component in the pipeline needs to have "tokens"
    # within the above described `provides` property.
    requires = []

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

    def __init__(self,
                 component_config: Optional[Dict[Text, Any]] = None) -> None:

        if not component_config:
            component_config = {}

        # makes sure the name of the configuration is part of the config
        # this is important for e.g. persistence
        component_config["name"] = self.name

        self.component_config = override_defaults(self.defaults,
                                                  component_config)

        self.partial_processing_pipeline = None
        self.partial_processing_context = None

    @classmethod
    def required_packages(cls) -> List[Text]:
        """Specify which python packages need to be installed to use this
        component, e.g. ``["spacy"]``.

        This list of requirements allows us to fail early during training
        if a required package is not installed."""
        return []

    @classmethod
    def load(cls,
             meta: Dict[Text, Any],
             model_dir: Optional[Text] = None,
             model_metadata: Optional['Metadata'] = None,
             cached_component: Optional['Component'] = None,
             **kwargs: Any
             ) -> 'Component':
        """Load this component from file.

        After a component has been trained, it will be persisted by
        calling `persist`. When the pipeline gets loaded again,
        this component needs to be able to restore itself.
        Components can rely on any context attributes that are
        created by :meth:`components.Component.pipeline_init`
        calls to components previous
        to this one."""
        if cached_component:
            return cached_component
        else:
            return cls(meta)

    @classmethod
    def create(cls,
               component_config: Dict[Text, Any],
               config: RasaNLUModelConfig) -> 'Component':
        """Creates this component (e.g. before a training is started).

        Method can access all configuration parameters."""

        # Check language supporting
        language = config.language
        if not cls.can_handle_language(language):
            # check failed
            raise UnsupportedLanguageError(cls.name, language)

        return cls(component_config)

    def provide_context(self) -> Optional[Dict[Text, Any]]:
        """Initialize this component for a new pipeline

        This function will be called before the training
        is started and before the first message is processed using
        the interpreter. The component gets the opportunity to
        add information to the context that is passed through
        the pipeline during training and message parsing. Most
        components do not need to implement this method.
        It's mostly used to initialize framework environments
        like MITIE and spacy
        (e.g. loading word vectors for the pipeline)."""
        pass

    def train(self,
              training_data: 'TrainingData',
              cfg: RasaNLUModelConfig,
              **kwargs: Any) -> None:
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

    def process(self, message: 'Message', **kwargs: Any) -> None:
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

    def persist(self,
                file_name: Text,
                model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this component to disk for future loading."""

        pass

    @classmethod
    def cache_key(cls,
                  component_meta: Dict[Text, Any],
                  model_metadata: 'Metadata') -> Optional[Text]:
        """This key is used to cache components.

        If a component is unique to a model it should return None.
        Otherwise, an instantiation of the
        component will be reused for all models where the
        metadata creates the same key."""

        return None

    def __getstate__(self) -> Any:
        d = self.__dict__.copy()
        # these properties should not be pickled
        if "partial_processing_context" in d:
            del d["partial_processing_context"]
        if "partial_processing_pipeline" in d:
            del d["partial_processing_pipeline"]
        return d

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def prepare_partial_processing(self,
                                   pipeline: List['Component'],
                                   context: Dict[Text, Any]) -> None:
        """Sets the pipeline and context used for partial processing.

        The pipeline should be a list of components that are
        previous to this one in the pipeline and
        have already finished their training (and can therefore
        be safely used to process messages)."""

        self.partial_processing_pipeline = pipeline
        self.partial_processing_context = context

    def partially_process(self, message: 'Message') -> 'Message':
        """Allows the component to process messages during
        training (e.g. external training data).

        The passed message will be processed by all components
        previous to this one in the pipeline."""

        if self.partial_processing_context is not None:
            for component in self.partial_processing_pipeline:
                component.process(message, **self.partial_processing_context)
        else:
            logger.info("Failed to run partial processing due "
                        "to missing pipeline.")
        return message

    @classmethod
    def can_handle_language(cls, language: Hashable) -> bool:
        """Check if component supports a specific language.

        This method can be overwritten when needed. (e.g. dynamically
        determine which language is supported.)"""

        # if language_list is set to `None` it means: support all languages
        if language is None or cls.language_list is None:
            return True

        return language in cls.language_list


class ComponentBuilder(object):
    """Creates trainers and interpreters based on configurations.

    Caches components for reuse.
    """

    def __init__(self, use_cache: bool = True) -> None:
        self.use_cache = use_cache
        # Reuse nlp and featurizers where possible to save memory,
        # every component that implements a cache-key will be cached
        self.component_cache = {}

    def __get_cached_component(self,
                               component_meta: Dict[Text, Any],
                               model_metadata: 'Metadata'
                               ) -> Tuple[Optional[Component], Optional[Text]]:
        """Load a component from the cache, if it exists.

        Returns the component, if found, and the cache key.
        """

        from rasa_nlu import registry
        # try to get class name first, else create by name
        component_name = component_meta.get('class', component_meta['name'])
        component_class = registry.get_component_class(component_name)
        cache_key = component_class.cache_key(component_meta, model_metadata)
        if (cache_key is not None and
                self.use_cache and
                cache_key in self.component_cache):
            return self.component_cache[cache_key], cache_key
        else:
            return None, cache_key

    def __add_to_cache(self,
                       component: Component,
                       cache_key: Optional[Text]) -> None:
        """Add a component to the cache."""

        if cache_key is not None and self.use_cache:
            self.component_cache[cache_key] = component
            logger.info("Added '{}' to component cache. Key '{}'."
                        "".format(component.name, cache_key))

    def load_component(self,
                       component_meta: Dict[Text, Any],
                       model_dir: Text,
                       model_metadata: 'Metadata',
                       **context: Any) -> Component:
        """Tries to retrieve a component from the cache, else calls
        ``load`` to create a new component.

        Args:
            component_meta (dict):
                the metadata of the component to load in the pipeline
            model_dir (str):
                the directory to read the model from
            model_metadata (Metadata):
                the model's :class:`rasa_nlu.models.Metadata`

        Returns:
            Component: the loaded component.
        """

        from rasa_nlu import registry

        try:
            cached_component, cache_key = self.__get_cached_component(
                component_meta, model_metadata)
            component = registry.load_component_by_meta(
                component_meta, model_dir, model_metadata,
                cached_component, **context)
            if not cached_component:
                # If the component wasn't in the cache,
                # let us add it if possible
                self.__add_to_cache(component, cache_key)
            return component
        except MissingArgumentError as e:  # pragma: no cover
            raise Exception("Failed to load component from file `{}`. "
                            "{}".format(component_meta.get("file"), e))

    def create_component(self,
                         component_config: Dict[Text, Any],
                         cfg: RasaNLUModelConfig) -> Component:
        """Tries to retrieve a component from the cache,
        calls `create` to create a new component."""
        from rasa_nlu import registry
        from rasa_nlu.model import Metadata

        try:
            component, cache_key = self.__get_cached_component(
                component_config, Metadata(cfg.as_dict(), None))
            if component is None:
                component = registry.create_component_by_config(
                    component_config, cfg)
                self.__add_to_cache(component, cache_key)
            return component
        except MissingArgumentError as e:  # pragma: no cover
            raise Exception("Failed to create component `{}`. "
                            "{}".format(component_config['name'], e))

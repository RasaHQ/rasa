import inspect

from typing import Optional
from typing import Type


def load_component(component_clz, context, config):
    # type: (Type[Component], dict, dict) -> Optional[Component]
    """Calls a components load method to init it based on a previously persisted model."""

    if component_clz is not None:
        load_args = fill_args(component_clz.load_args(), context, config)
        return component_clz.load(*load_args)
    else:
        return None


def init_component(component, context, config):
    # type: (Component, dict, dict) -> None
    """Initializes a component using the attributes from the context and configuration."""

    args = fill_args(component.pipeline_init_args(), context, config)
    updates = component.pipeline_init(*args)
    if updates:
        context.update(updates)


def fill_args(arguments, context, config):
    # type: ([str], dict, dict) -> [object]
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


class MissingArgumentError(Exception):
    """Raised when a function is called and not all parameters can be filled from the context / config.

    Attributes:
        message -- explanation of which parameter is missing
    """

    def __init__(self, message):
        # type: (str) -> None
        self.message = message


class Component(object):
    # Name of the component to be used when integrating it in a pipeline. E.g. `[ComponentA, ComponentB]`
    # will be a proper pipeline definition where `ComponentA` is the name of the first component of the pipeline.
    name = ""

    # Defines what attributes the pipeline component will provide when called
    # (mostly used to check if the pipeline is valid)
    context_provides = []  # type: [str]

    @classmethod
    def load(cls, *args):
        # type: (...) -> 'cls'
        return cls()

    def pipeline_init(self, *args):
        # type: (...) -> Optional[dict]
        pass

    def train(self, *args):
        # type: (...) -> Optional[dict]
        pass

    def process(self, *args):
        # type: (...) -> Optional[dict]
        pass

    def persist(self, model_dir):
        # type: (str) -> Optional[dict]
        pass

    @classmethod
    def cache_key(cls, model_metadata):
        # type: (Metadata) -> Optional[str]
        """This key is used to cache components.

        If a model is unique to a model it should return None. Otherwise, an instantiation of the
        component will be reused for all models where the metadata creates the same key."""
        from rasa_nlu.model import Metadata

        return None

    def pipeline_init_args(self):
        # type: () -> [str]
        return filter(lambda arg: arg not in ["self"], inspect.getargspec(self.pipeline_init).args)

    def train_args(self):
        # type: () -> [str]
        return filter(lambda arg: arg not in ["self"], inspect.getargspec(self.train).args)

    def process_args(self):
        # type: () -> [str]
        return filter(lambda arg: arg not in ["self"], inspect.getargspec(self.process).args)

    @classmethod
    def load_args(cls):
        # type: () -> [str]
        return filter(lambda arg: arg not in ["cls"], inspect.getargspec(cls.load).args)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

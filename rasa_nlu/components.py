import inspect


def get_component_class(component_name):
    # resolve component name to class
    from rasa_nlu.pipeline import registered_components
    return registered_components.get(component_name)


def load_component_instance(component_name, context, config):
    # load component from file
    component_clz = get_component_class(component_name)
    if component_clz is not None:
        load_args = fill_args(component_clz.load_args(), context, config)
        return component_clz.load(*load_args)
    else:
        return None


def init_component(component, context, config):
    # init component with context
    args = fill_args(component.pipeline_init_args(), context, config)
    updates = component.pipeline_init(*args)
    if updates:
        context.update(updates)


def fill_args(arguments, context, config):
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
        self.message = message


class Component(object):
    name = ""

    context_provides = []

    def pipeline_init(self, *args):
        pass

    def train(self, *args):
        pass

    def process(self, *args):
        pass

    def persist(self, model_dir):
        pass

    @classmethod
    def cache_key(cls, model_metadata):
        """This key is used to cache components.

        If a model is unique to a model it should return None. Otherwise, an instantiation of the
        component will be reused for all models where the metadata creates the same key.

        :type model_metadata: rasa_nlu.model.Metadata
        :rtype: None or str
        """
        return None

    @classmethod
    def load(cls, *args):
        return cls()

    def pipeline_init_args(self):
        return filter(lambda arg: arg not in ["self"], inspect.getargspec(self.pipeline_init).args)

    def train_args(self):
        return filter(lambda arg: arg not in ["self"], inspect.getargspec(self.train).args)

    def process_args(self):
        return filter(lambda arg: arg not in ["self"], inspect.getargspec(self.process).args)

    @classmethod
    def load_args(cls):
        return filter(lambda arg: arg not in ["cls"], inspect.getargspec(cls.load).args)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__
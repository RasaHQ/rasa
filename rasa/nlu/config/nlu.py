import logging

from rasa.nlu.config.exceptions import InvalidConfigError
from rasa.nlu.utils import json_to_string

logger = logging.getLogger(__name__)


class RasaNLUModelConfig:
    def __init__(self, configuration_values=None):
        """Create a model configuration, optionally overriding
        defaults with a dictionary ``configuration_values``.
        """
        if not configuration_values:
            configuration_values = {}

        self.language = "en"
        self.pipeline = []
        self.data = None

        self.override(configuration_values)

        if self.__dict__["pipeline"] is None:
            # replaces None with empty list
            self.__dict__["pipeline"] = []
        elif isinstance(self.__dict__["pipeline"], str):
            from rasa.nlu.components import registry  # pytype: disable=pyi-error

            template_name = self.__dict__["pipeline"]
            new_names = {
                "spacy_sklearn": "pretrained_embeddings_spacy",
                "tensorflow_embedding": "supervised_embeddings",
            }
            if template_name in new_names:
                logger.warning(
                    "You have specified the pipeline template "
                    "'{}' which has been renamed to '{}'. "
                    "Please update your code as it will no "
                    "longer work with future versions of "
                    "Rasa NLU.".format(template_name, new_names[template_name])
                )
                template_name = new_names[template_name]

            pipeline = registry.pipeline_template(template_name)

            if pipeline:
                # replaces the template with the actual components
                self.__dict__["pipeline"] = pipeline
            else:
                known_templates = ", ".join(
                    registry.registered_pipeline_templates.keys()
                )

                raise InvalidConfigError(
                    "No pipeline specified and unknown "
                    "pipeline template '{}' passed. Known "
                    "pipeline templates: {}"
                    "".format(template_name, known_templates)
                )

        for key, value in self.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        return self.__dict__[key]

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __delitem__(self, key):
        del self.__dict__[key]

    def __contains__(self, key):
        return key in self.__dict__

    def __len__(self):
        return len(self.__dict__)

    def __getstate__(self):
        return self.as_dict()

    def __setstate__(self, state):
        self.override(state)

    def items(self):
        return list(self.__dict__.items())

    def as_dict(self):
        return dict(list(self.items()))

    def view(self):
        return json_to_string(self.__dict__, indent=4)

    def for_component(self, index, defaults=None):
        from rasa.nlu.config.manager import ConfigManager  # pytype: disable=pyi-error

        return ConfigManager.component_config_from_pipeline(
            index, self.pipeline, defaults
        )

    @property
    def component_names(self):
        if self.pipeline:
            return [c.get("name") for c in self.pipeline]
        else:
            return []

    def set_component_attr(self, index, **kwargs):
        try:
            self.pipeline[index].update(kwargs)
        except IndexError:
            logger.warning(
                "Tried to set configuration value for component "
                "number {} which is not part of the pipeline."
                "".format(index)
            )

    def override(self, config):
        if config:
            self.__dict__.update(config)

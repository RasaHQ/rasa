import copy
import logging
import os
import ruamel.yaml as yaml
from typing import Any, Dict, List, Optional, Text

from rasa_nlu import utils
from rasa_nlu.utils import json_to_string

# Describes where to search for the config file if no location is specified
DEFAULT_CONFIG_LOCATION = "config.yml"

DEFAULT_CONFIG = {
    "language": "en",
    "pipeline": [],
    "data": None,
}

logger = logging.getLogger(__name__)


class InvalidConfigError(ValueError):
    """Raised if an invalid configuration is encountered."""

    def __init__(self, message: Text) -> None:
        super(InvalidConfigError, self).__init__(message)


def load(filename: Optional[Text] = None,
         **kwargs: Any) -> 'RasaNLUModelConfig':
    if filename is None and os.path.isfile(DEFAULT_CONFIG_LOCATION):
        filename = DEFAULT_CONFIG_LOCATION

    if filename is not None:
        try:
            file_config = utils.read_yaml_file(filename)
        except yaml.parser.ParserError as e:
            raise InvalidConfigError("Failed to read configuration file "
                                     "'{}'. Error: {}".format(filename, e))

        if kwargs:
            file_config.update(kwargs)
        return RasaNLUModelConfig(file_config)
    else:
        return RasaNLUModelConfig(kwargs)


def override_defaults(defaults: Optional[Dict[Text, Any]],
                      custom: Optional[Dict[Text, Any]]
                      ) -> Dict[Text, Any]:
    if defaults:
        cfg = copy.deepcopy(defaults)
    else:
        cfg = {}

    if custom:
        cfg.update(custom)
    return cfg


def make_path_absolute(path: Text) -> Text:
    if path and not os.path.isabs(path):
        return os.path.join(os.getcwd(), path)
    else:
        return path


def component_config_from_pipeline(
        index: int,
        pipeline: List[Dict[Text, Any]],
        defaults: Optional[Dict[Text, Any]] = None
) -> Dict[Text, Any]:
    try:
        c = pipeline[index]
        return override_defaults(defaults, c)
    except IndexError:
        logger.warning("Tried to get configuration value for component "
                       "number {} which is not part of the pipeline. "
                       "Returning `defaults`."
                       "".format(index))
        return override_defaults(defaults, {})


class RasaNLUModelConfig(object):
    DEFAULT_PROJECT_NAME = "default"

    def __init__(self, configuration_values=None):
        """Create a model configuration, optionally overridding
        defaults with a dictionary ``configuration_values``.
        """
        if not configuration_values:
            configuration_values = {}

        self.override(DEFAULT_CONFIG)
        self.override(configuration_values)

        if self.__dict__['pipeline'] is None:
            # replaces None with empty list
            self.__dict__['pipeline'] = []
        elif isinstance(self.__dict__['pipeline'], str):
            from rasa_nlu import registry

            template_name = self.__dict__['pipeline']
            new_names = {"spacy_sklearn": "pretrained_embeddings_spacy",
                         "tensorflow_embedding": "supervised_embeddings"}
            if template_name in new_names:
                logger.warning("You have specified the pipeline template "
                               "'{}' which has been renamed to '{}'. "
                               "Please update your code as it will no "
                               "longer work with future versions of "
                               "Rasa NLU.".format(template_name,
                                                  new_names[template_name]))
                template_name = new_names[template_name]

            pipeline = registry.pipeline_template(template_name)

            if pipeline:
                # replaces the template with the actual components
                self.__dict__['pipeline'] = pipeline
            else:
                known_templates = ", ".join(
                    registry.registered_pipeline_templates.keys())

                raise InvalidConfigError("No pipeline specified and unknown "
                                         "pipeline template '{}' passed. Known "
                                         "pipeline templates: {}"
                                         "".format(template_name,
                                                   known_templates))

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
        return component_config_from_pipeline(index, self.pipeline, defaults)

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
            logger.warning("Tried to set configuration value for component "
                           "number {} which is not part of the pipeline."
                           "".format(index))

    def override(self, config):
        if config:
            self.__dict__.update(config)

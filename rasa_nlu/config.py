from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import six
from builtins import object
# Describes where to search for the config file if no location is specified
from typing import Text

from rasa_nlu import utils
from rasa_nlu.utils import json_to_string

DEFAULT_CONFIG_LOCATION = "config.yml"

DEFAULT_CONFIG = {
    "language": "en",
    "pipeline": [],
}


class InvalidConfigError(ValueError):
    """Raised if an invalid configuration is encountered."""

    def __init__(self, message):
        # type: (Text) -> None
        super(InvalidConfigError, self).__init__(message)


def load(filename=None):
    if filename is None and os.path.isfile(DEFAULT_CONFIG_LOCATION):
        filename = DEFAULT_CONFIG_LOCATION

    if filename is not None:
        try:
            file_config = utils.read_yaml_file(filename)
        except ValueError as e:
            raise InvalidConfigError("Failed to read configuration file "
                                     "'{}'. Error: {}".format(filename, e))
        return RasaNLUModelConfig(file_config)
    else:
        return RasaNLUModelConfig()


class RasaNLUModelConfig(object):
    DEFAULT_PROJECT_NAME = "default"

    def __init__(self, configuration_values=None):

        if not configuration_values:
            configuration_values = {}

        self.override(DEFAULT_CONFIG)
        self.override(configuration_values)

        if isinstance(self.__dict__['pipeline'], six.string_types):
            from rasa_nlu import registry

            template_name = self.__dict__['pipeline']
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

    def make_paths_absolute(self, config, keys):
        abs_path_config = dict(config)
        for key in keys:
            if (key in abs_path_config
                    and abs_path_config[key] is not None
                    and not os.path.isabs(abs_path_config[key])):
                abs_path_config[key] = os.path.join(os.getcwd(),
                                                    abs_path_config[key])
        return abs_path_config

    # noinspection PyCompatibility
    def make_unicode(self, config):
        # TODO: i think this can be removed I don't think there is a case
        # where we would parse the config and it would turn out to be a str
        if six.PY2:
            # Sometimes (depending on the source of the config value)
            # an argument will be str instead of unicode
            # to unify that and ease further usage of the config,
            # we convert everything to unicode
            for k, v in config.items():
                if type(v) is str:
                    config[k] = unicode(v, "utf-8")
        return config

    def override(self, config):
        abs_path_config = self.make_paths_absolute(config,
                                                   ["path", "response_log"])
        unicode_config = self.make_unicode(abs_path_config)
        self.__dict__.update(unicode_config)

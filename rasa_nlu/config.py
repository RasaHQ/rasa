from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import object
import io
import json
import os
import warnings
import six


# Describes where to search for the configuration file if the location is not set by the user
DEFAULT_CONFIG_LOCATION = "config.json"


DEFAULT_CONFIG = {
    "config": DEFAULT_CONFIG_LOCATION,
    "data": None,
    "emulate": None,
    "language": "en",
    "log_file": None,
    "log_level": 'INFO',
    "mitie_file": os.path.join("data", "total_word_feature_extractor.dat"),
    "spacy_model_name": None,
    "num_threads": 1,
    "fine_tune_spacy_ner": False,
    "path": "models",
    "port": 5000,
    "server_model_dirs": None,
    "token": None,
    "max_number_of_ngrams": 7,
    "pipeline": [],
    "response_log": "logs",
    "duckling_processing_mode": "append",
    "luis_data_tokenizer": None,
    "entity_crf_BILOU_flag": True,
    "entity_crf_features": [
        ["low", "title", "upper", "pos", "pos2"],
        ["bias", "low", "word3", "word2", "upper", "title", "digit", "pos", "pos2"],
        ["low", "title", "upper", "pos", "pos2"]]
}


class InvalidConfigError(ValueError):
    """Raised if an invalid configuration is encountered."""

    def __init__(self, message):
        # type: (str) -> None
        super(InvalidConfigError, self).__init__(message)


class RasaNLUConfig(object):

    def __init__(self, filename=None, env_vars=None, cmdline_args=None):

        if filename is None and os.path.isfile(DEFAULT_CONFIG_LOCATION):
            filename = DEFAULT_CONFIG_LOCATION

        self.override(DEFAULT_CONFIG)
        if filename is not None:
            try:
                with io.open(filename, encoding='utf-8') as f:
                    file_config = json.loads(f.read())
            except ValueError as e:
                raise InvalidConfigError("Failed to read configuration file '{}'. Error: {}".format(filename, e))
            self.override(file_config)

        if env_vars is not None:
            env_config = self.format_env_vars(env_vars)
            self.override(env_config)

        if cmdline_args is not None:
            cmdline_config = {k: v for k, v in list(cmdline_args.items()) if v is not None}
            self.override(cmdline_config)

        if isinstance(self.__dict__['pipeline'], six.string_types):
            from rasa_nlu import registry
            if self.__dict__['pipeline'] in registry.registered_pipeline_templates:
                self.__dict__['pipeline'] = registry.registered_pipeline_templates[self.__dict__['pipeline']]
            else:
                warnings.warn("No pipeline specified and unknown pipeline template " +
                              "'{}' passed.".format(self.__dict__['pipeline']))

        for key, value in self.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __delitem__(self, key):
        del self.__dict__[key]

    def __contains__(self, key):
        return key in self.__dict__

    def __len__(self):
        return len(self.__dict__)

    def items(self):
        return list(self.__dict__.items())

    def as_dict(self):
        return dict(list(self.items()))

    def view(self):
        return json.dumps(self.__dict__, indent=4)

    def format_env_vars(self, env_vars):
        keys = [key for key in env_vars.keys() if "RASA" in key]
        return {key.split('RASA_')[1].lower(): env_vars[key] for key in keys}

    def is_set(self, key):
        return key in self.__dict__ and self[key] is not None

    def make_paths_absolute(self, config, keys):
        abs_path_config = dict(config)
        for key in keys:
            if key in abs_path_config and not os.path.isabs(abs_path_config[key]):
                abs_path_config[key] = os.path.join(os.getcwd(), abs_path_config[key])
        return abs_path_config

    # noinspection PyCompatibility
    def make_unicode(self, config):
        if six.PY2:
            # Sometimes (depending on the source of the config value) an argument will be str instead of unicode
            # to unify that and ease further usage of the config, we convert everything to unicode
            for k, v in config.items():
                if type(v) is str:
                    config[k] = unicode(v, "utf-8")
        return config

    def override(self, config):
        abs_path_config = self.make_unicode(self.make_paths_absolute(config, ["path", "response_log"]))
        self.__dict__.update(abs_path_config)

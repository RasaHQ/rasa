import codecs
import json
import os
import logging


# Describes where to search for the configuration file if the location is not set by the user
from rasa_nlu.utils.mitie import MITIE_BACKEND_NAME

DEFAULT_CONFIG_LOCATION = "config.json"


class RasaNLUConfig(object):

    def __init__(self, filename=None, env_vars=None, cmdline_args=None):

        defaults = {
          "backend": "mitie",
          "config": DEFAULT_CONFIG_LOCATION,
          "data": None,
          "emulate": None,
          "language": "en",
          "log_file": None,
          "log_level": logging.INFO,
          "mitie_file": os.path.join("data", "total_word_feature_extractor.dat"),
          "num_threads": 1,
          "fine_tune_spacy_ner": False,
          "path": os.path.join(os.getcwd(), "models"),
          "port": 5000,
          "server_model_dirs": None,
          "token": None,
          "response_log": os.path.join(os.getcwd(), "logs")
        }

        if filename is None and os.path.isfile(DEFAULT_CONFIG_LOCATION):
            filename = DEFAULT_CONFIG_LOCATION

        self.override(defaults)
        if filename is not None:
            with codecs.open(filename, encoding='utf-8') as f:
                file_config = json.loads(f.read())
            self.override(file_config)

        if env_vars is not None:
            env_config = self.format_env_vars(env_vars)
            self.override(env_config)

        if cmdline_args is not None:
            cmdline_config = {k: v for k, v in cmdline_args.items() if v is not None}
            self.override(cmdline_config)

        for key, value in self.items():
            setattr(self, key, value)

        self.validate()

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
        return self.__dict__.items()

    def view(self):
        return json.dumps(self.__dict__, indent=4)

    def format_env_vars(self, env_vars):
        keys = [key for key in env_vars.keys() if "RASA" in key]
        return {key.split('RASA_')[1].lower(): env_vars[key] for key in keys}

    def is_set(self, key):
        return key in self.__dict__ and self[key] is not None

    def override(self, new_dict):
        self.__dict__.update(new_dict)

    def validate(self):
        if self.backend == MITIE_BACKEND_NAME:
            if not self.is_set("mitie_file"):
                raise ValueError("backend set to 'mitie' but mitie_file not specified")

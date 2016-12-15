import codecs
import json
import os
import logging


class RasaNLUConfig(object):

    def __init__(self, filename=None, env_vars=None, cmdline_args=None):

        defaults = {
          "backend": "mitie",
          "config": "config.json",
          "data": None,
          "emulate": None,
          "language": "en",
          "log_file": None,
          "log_level": logging.INFO,
          "mitie_file": "./data/total_word_feature_extractor.dat",
          "path": os.getcwd(),
          "port": 5000,
          "server_model_dir": None,
          "token": None,
          "write": os.path.join(os.getcwd(), "rasa_nlu_log.json")
        }

        self.override(defaults)
        if filename is not None:
            file_config = json.loads(codecs.open(filename, encoding='utf-8').read())
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
        if self.backend == "mitie":
            if not self.is_set("mitie_file"):
                raise ValueError("backend set to 'mitie' but mitie_file not specified")
            if self.language != "en":
                raise ValueError("backend set to 'mitie' but language not set to 'en'.")

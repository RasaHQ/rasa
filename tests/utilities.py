from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os

from builtins import object
import tempfile
import pytest
import json

from rasa_nlu import registry
from rasa_nlu.project import Project
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Interpreter, Metadata
from rasa_nlu.train import do_train
from rasa_nlu.utils import json_to_string

slowtest = pytest.mark.slowtest


def base_test_conf(pipeline_template):
    # 'response_log': temp_log_file_dir(),
    # 'port': 5022,
    # "path": tempfile.mkdtemp(),
    # "data": "./data/test/demo-rasa-small.json"

    return RasaNLUModelConfig({"pipeline": pipeline_template})


def write_file_config(file_config):
    with tempfile.NamedTemporaryFile("w+",
                                     suffix="_tmp_config_file.json",
                                     delete=False) as f:
        f.write(json_to_string(file_config))
        f.flush()
        return f


def interpreter_for(component_builder, data, path, config):
    (trained, _, path) = do_train(config, data, path,
                                  component_builder=component_builder)
    interpreter = Interpreter.load(path, component_builder)
    return interpreter


def temp_log_file_dir():
    return tempfile.mkdtemp(suffix="_rasa_nlu_logs")


class ResponseTest(object):
    def __init__(self, endpoint, expected_response, payload=None):
        self.endpoint = endpoint
        self.expected_response = expected_response
        self.payload = payload

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import object
import tempfile
import pytest
import json

from rasa_nlu import registry
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.data_router import DataRouter
from rasa_nlu.model import Interpreter
from rasa_nlu.train import do_train

slowtest = pytest.mark.slowtest


def base_test_conf(pipeline_template):
    return RasaNLUConfig(cmdline_args={
        'response_log': temp_log_file_dir(),
        'port': 5022,
        "pipeline": registry.registered_pipeline_templates.get(pipeline_template, []),
        "path": tempfile.mkdtemp(),
        "data": "./data/test/demo-rasa-small.json"
    })


def write_file_config(file_config):
    with tempfile.NamedTemporaryFile("w+", suffix="_tmp_config_file.json", delete=False) as f:
        f.write(json.dumps(file_config))
        f.flush()
        return f


def interpreter_for(component_builder, config):
    (trained, path) = run_train(config, component_builder)
    interpreter = load_interpreter_for_model(config, path, component_builder)
    return interpreter


def temp_log_file_dir():
    return tempfile.mkdtemp(suffix="_rasa_nlu_logs")


def run_train(config, component_builder):
    (trained, _, path) = do_train(config, component_builder)
    return trained, path


def load_interpreter_for_model(config, persisted_path, component_builder):
    metadata = DataRouter.read_model_metadata(persisted_path, config)
    return Interpreter.create(metadata, config, component_builder)


class ResponseTest(object):
    def __init__(self, endpoint, expected_response, payload=None):
        self.endpoint = endpoint
        self.expected_response = expected_response
        self.payload = payload

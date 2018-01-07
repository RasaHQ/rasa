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
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Interpreter, Metadata
from rasa_nlu.train import do_train
from rasa_nlu.utils import json_to_string

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
        f.write(json_to_string(file_config))
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
    def read_model_metadata(model_dir, config):
        if model_dir is None:
            data = Project._default_model_metadata()
            return Metadata(data, model_dir)
        else:
            if not os.path.isabs(model_dir):
                model_dir = os.path.join(config['path'], model_dir)

            # download model from S3 if needed
            if not os.path.isdir(model_dir):
                Project._load_model_from_cloud(model_dir, config)

            return Metadata.load(model_dir)

    metadata = read_model_metadata(persisted_path, config)
    return Interpreter.create(metadata, config, component_builder)


class ResponseTest(object):
    def __init__(self, endpoint, expected_response, payload=None):
        self.endpoint = endpoint
        self.expected_response = expected_response
        self.payload = payload

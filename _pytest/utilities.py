import tempfile

from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.data_router import DataRouter
from rasa_nlu.train import do_train


def base_test_conf(backend):
    return {
        'write': temp_log_file_location(),
        'port': 5022,
        "backend": backend,
        "path": tempfile.mkdtemp(),
        "data": "./data/examples/rasa/demo-rasa.json"
    }


def interpreter_for(nlp, config):
    (trained, path) = run_train(config)
    interpreter = load_interpreter_for_model(nlp, config, path)
    return interpreter


def temp_log_file_location():
    return tempfile.mkstemp(suffix="_rasa_nlu_logs.json")[1]


def run_train(_config):
    config = RasaNLUConfig(cmdline_args=_config)
    (trained, path) = do_train(config)
    return trained, path


def load_interpreter_for_model(nlp, config, persisted_path):
    metadata = DataRouter.read_model_metadata(persisted_path, config)
    return DataRouter.create_interpreter(metadata, nlp)


class ResponseTest(object):
    def __init__(self, endpoint, expected_response, payload=None):
        self.endpoint = endpoint
        self.expected_response = expected_response
        self.payload = payload

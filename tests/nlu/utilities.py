import tempfile

from ruamel.yaml import YAML


def write_file_config(file_config):
    with tempfile.NamedTemporaryFile(
        "w+", suffix="_tmp_config_file.yml", delete=False
    ) as f:
        with YAML(typ="safe", pure=True, output=f) as yaml:
            yaml.dump(file_config)

        f.flush()
        return f


class ResponseTest:
    def __init__(self, endpoint, expected_response, payload=None):
        self.endpoint = endpoint
        self.expected_response = expected_response
        self.payload = payload

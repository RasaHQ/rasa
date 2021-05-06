import tempfile
import ruamel.yaml as yaml


def write_file_config(file_config):
    with tempfile.NamedTemporaryFile(
        "w+", suffix="_tmp_config_file.yml", delete=False
    ) as f:
        f.write(yaml.safe_dump(file_config))
        f.flush()
        return f


def verify_sequence_not_none(sequence, sentence):
    if sequence and sentence:
        return sequence.features, sentence.features
    if sequence and not sentence:
        return sequence.features, None
    if not sequence and sentence:
        return None, sentence.features


class ResponseTest:
    def __init__(self, endpoint, expected_response, payload=None):
        self.endpoint = endpoint
        self.expected_response = expected_response
        self.payload = payload

import tempfile
import ruamel.yaml as yaml


def write_file_config(file_config):
    with tempfile.NamedTemporaryFile(
        "w+", suffix="_tmp_config_file.yml", delete=False
    ) as f:
        f.write(yaml.safe_dump(file_config))
        f.flush()
        return f


# check if os sequences e sentences is loaded correctly
def get_feature_vectors(sequence, sentence):
    return (
        sequence.features if sequence else None,
        sentence.features if sentence else None,
    )


class ResponseTest:
    def __init__(self, endpoint, expected_response, payload=None):
        self.endpoint = endpoint
        self.expected_response = expected_response
        self.payload = payload

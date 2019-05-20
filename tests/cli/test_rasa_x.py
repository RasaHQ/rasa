import rasa.utils.io as io_utils
from rasa.cli import x


def test_x_help(run):
    output = run("x", "--help")

    help_text = """usage: rasa x [-h] [-v] [-vv] [--quiet] [-m MODEL] [--no-prompt]
              [--production] [--data DATA] [--log-file LOG_FILE]
              [--endpoints ENDPOINTS] [-p PORT] [-t AUTH_TOKEN]
              [--cors [CORS [CORS ...]]] [--enable-api]
              [--remote-storage REMOTE_STORAGE] [--credentials CREDENTIALS]
              [--connector CONNECTOR] [--jwt-secret JWT_SECRET]
              [--jwt-method JWT_METHOD]"""

    lines = help_text.split("\n")

    for i, line in enumerate(lines):
        assert output.outlines[i] == line


def test_prepare_credentials_for_rasa_x_if_rasa_channel_not_given(tmpdir_factory):
    directory = tmpdir_factory.mktemp("directory")
    credentials_path = str(directory / "credentials.yml")

    io_utils.write_yaml_file({}, credentials_path)

    tmp_credentials = x._prepare_credentials_for_rasa_x(
        credentials_path, "http://localhost:5002"
    )

    actual = io_utils.read_yaml_file(tmp_credentials)

    assert actual["rasa"]["url"] == "http://localhost:5002"


def test_prepare_credentials_if_already_valid(tmpdir_factory):
    directory = tmpdir_factory.mktemp("directory")
    credentials_path = str(directory / "credentials.yml")

    credentials = {
        "rasa": {"url": "my-custom-url"},
        "another-channel": {"url": "some-url"},
    }
    io_utils.write_yaml_file(credentials, credentials_path)

    x._prepare_credentials_for_rasa_x(credentials_path)

    actual = io_utils.read_yaml_file(credentials_path)

    assert actual == credentials

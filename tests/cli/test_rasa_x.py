def test_x_help(run):
    help = run("x", "--help")

    help_text = """usage: rasa x [-h] [-v] [-vv] [--quiet] [-m MODEL] [--no-prompt]
              [--production] [--nlg NLG]
              [--model-endpoint-url MODEL_ENDPOINT_URL]
              [--project-path PROJECT_PATH] [--data-path DATA_PATH]
              [--log-file LOG_FILE] [--endpoints ENDPOINTS] [-p PORT]
              [-t AUTH_TOKEN] [--cors [CORS [CORS ...]]] [--enable-api]
              [--remote-storage REMOTE_STORAGE] [--credentials CREDENTIALS]
              [--connector CONNECTOR] [--jwt-secret JWT_SECRET]
              [--jwt-method JWT_METHOD]"""

    lines = help_text.split("\n")

    for i, line in enumerate(lines):
        assert help.outlines[i] == line

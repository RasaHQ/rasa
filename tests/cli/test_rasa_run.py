def test_run_help(run):
    help, _ = run("run", "--help")

    help_text = """usage: rasa run [-h] [-v] [-vv] [--quiet] [--log-file LOG_FILE]
                [--endpoints ENDPOINTS] [-p PORT] [-t AUTH_TOKEN]
                [--cors [CORS [CORS ...]]] [--enable-api]
                [--remote-storage REMOTE_STORAGE] [--credentials CREDENTIALS]
                [--connector CONNECTOR] [--jwt-secret JWT_SECRET]
                [--jwt-method JWT_METHOD] [-m MODEL]
                [model-as-positional-argument] {actions} ..."""

    lines = help_text.split("\n")

    for i, line in enumerate(lines):
        assert help.outlines[i] == line


def test_run_action_help(run):
    help, _ = run("run", "actions", "--help")

    help_text = """usage: rasa run [model-as-positional-argument] actions [-h] [-v] [-vv]
                                                       [--quiet] [-p PORT]
                                                       [--cors [CORS [CORS ...]]]
                                                       [--actions ACTIONS]"""

    lines = help_text.split("\n")

    for i, line in enumerate(lines):
        assert help.outlines[i] == line

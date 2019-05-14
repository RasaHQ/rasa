def test_interactive_help(run):
    help = run("interactive", "--help")

    help_text = """usage: rasa interactive [-h] [-v] [-vv] [--quiet] [-m MODEL] [-c CONFIG]
                        [-d DOMAIN] [--data DATA [DATA ...]] [--out OUT]
                        [--force] [--skip-visualization] [--log-file LOG_FILE]
                        [--endpoints ENDPOINTS] [-p PORT] [-t AUTH_TOKEN]
                        [--cors [CORS [CORS ...]]] [--enable-api]
                        [--remote-storage REMOTE_STORAGE]
                        [--credentials CREDENTIALS] [--connector CONNECTOR]
                        [--jwt-secret JWT_SECRET] [--jwt-method JWT_METHOD]
                        [model-as-positional-argument] {core} ..."""

    lines = help_text.split("\n")

    for i, line in enumerate(lines):
        assert help.outlines[i] == line


def test_interactive_core_help(run):
    help = run("interactive", "core", "--help")

    help_text = """usage: rasa interactive [model-as-positional-argument] core
       [-h] [-v] [-vv] [--quiet] [-m MODEL] [-c CONFIG] [-d DOMAIN]
       [-s STORIES] [--out OUT] [--augmentation AUGMENTATION] [--debug-plots]
       [--dump-stories] [--skip-visualization] [--log-file LOG_FILE]
       [--endpoints ENDPOINTS] [-p PORT] [-t AUTH_TOKEN]
       [--cors [CORS [CORS ...]]] [--enable-api]
       [--remote-storage REMOTE_STORAGE] [--credentials CREDENTIALS]
       [--connector CONNECTOR] [--jwt-secret JWT_SECRET]
       [--jwt-method JWT_METHOD]
       [model-as-positional-argument]"""

    lines = help_text.split("\n")

    for i, line in enumerate(lines):
        assert help.outlines[i] == line

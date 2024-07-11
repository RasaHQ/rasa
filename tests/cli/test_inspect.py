from typing import Callable

from pytest import RunResult


def test_rasa_inspect_help(run: Callable[..., RunResult]) -> None:
    help_text = """usage: rasa inspect [-h] [-v] [-vv] [--quiet]
                    [--logging-config-file LOGGING_CONFIG_FILE] [-m MODEL]
                    [--log-file LOG_FILE] [--use-syslog]
                    [--syslog-address SYSLOG_ADDRESS]
                    [--syslog-port SYSLOG_PORT]
                    [--syslog-protocol SYSLOG_PROTOCOL]
                    [--endpoints ENDPOINTS] [-i INTERFACE] [-p PORT]
                    [--response-timeout RESPONSE_TIMEOUT]
                    [--request-timeout REQUEST_TIMEOUT]
                    [--remote-storage REMOTE_STORAGE]
                    [--ssl-certificate SSL_CERTIFICATE]
                    [--ssl-keyfile SSL_KEYFILE] [--ssl-ca-file SSL_CA_FILE]
                    [--ssl-password SSL_PASSWORD] [--jwt-secret JWT_SECRET]
                    [--jwt-method JWT_METHOD]
                    [--jwt-private-key JWT_PRIVATE_KEY]
                    [model-as-positional-argument]"""
    lines = help_text.split("\n")

    output = run("inspect", "--help")
    printed_help = {line.strip() for line in output.outlines}
    for line in lines:
        assert line.strip() in printed_help

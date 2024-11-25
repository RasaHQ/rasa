import argparse


def test_rasa_inspect_default_args(
    inspect_parser: argparse.ArgumentParser,
) -> None:
    """Tests default settings for `rasa inspect` CLI command."""
    args = inspect_parser.parse_args(["inspect"])

    assert args.model == "models"
    assert args.log_file is None
    assert not args.use_syslog
    assert args.syslog_address == "localhost"
    assert args.syslog_port == 514
    assert args.syslog_protocol == "UDP"
    assert args.endpoints == "endpoints.yml"
    assert args.interface == "0.0.0.0"
    assert args.port == 5005
    assert args.auth_token is None
    assert args.cors is None
    assert args.response_timeout == 3600
    assert args.request_timeout == 300
    assert args.remote_storage is None
    assert args.ssl_certificate is None
    assert args.ssl_keyfile is None
    assert args.ssl_ca_file is None
    assert args.ssl_password is None
    assert args.jwt_secret is None
    assert args.jwt_method == "HS256"
    assert args.jwt_private_key is None
    assert args.skip_yaml_validation == []

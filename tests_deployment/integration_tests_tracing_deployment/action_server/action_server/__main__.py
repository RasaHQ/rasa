import argparse
import logging

from action_server.config import configure_tracing

DEFAULT_SERVER_PORT = "5055"


def create_argument_parser():
    """Parse all the command line arguments for the nlg server script."""

    parser = argparse.ArgumentParser(description="starts the nlg endpoint")
    parser.add_argument(
        "-p",
        "--port",
        default=DEFAULT_SERVER_PORT,
        type=int,
        help="port to run the server at",
    )
    parser.add_argument(
        "--endpoints",
        default="endpoints.yml",
        help="endpoints file for tracing configuration",
    )

    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    arg_parser = create_argument_parser()
    cmdline_args = arg_parser.parse_args()
    configure_tracing(cmdline_args.endpoints)

    from action_server.run import run_app

    run_app(cmdline_args.port)

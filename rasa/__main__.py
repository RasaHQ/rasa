import argparse

# Check TensorFlow availability without importing the module
import importlib.util
import os
import platform
import sys
from typing import List, Optional

import structlog
from rasa_sdk import __version__ as rasa_sdk_version

import rasa.telemetry
import rasa.utils.io
import rasa.utils.licensing
from rasa import version
from rasa.cli import (
    data,
    evaluate,
    export,
    interactive,
    llm_fine_tuning,
    run,
    scaffold,
    shell,
    telemetry,
    test,
    train,
    visualize,
    x,
)
from rasa.cli.arguments.default_arguments import add_logging_options
from rasa.cli.utils import (
    check_if_studio_command,
    parse_last_positional_argument_as_model_path,
    warn_if_rasa_plus_package_installed,
)
from rasa.constants import MINIMUM_COMPATIBLE_VERSION
from rasa.exceptions import DetailedRasaException
from rasa.plugin import plugin_manager
from rasa.shared.exceptions import RasaException
from rasa.utils.common import configure_logging_and_warnings
from rasa.utils.log_utils import configure_structlog

TENSORFLOW_AVAILABLE = importlib.util.find_spec("tensorflow") is not None

# Only import TensorFlow modules if TensorFlow is available
if TENSORFLOW_AVAILABLE:
    import rasa.utils.tensorflow.environment as tf_env
else:
    tf_env = None  # type: ignore[assignment]

structlogger = structlog.get_logger()


def create_argument_parser() -> argparse.ArgumentParser:
    """Parse all the command line arguments for the training script."""
    parser = argparse.ArgumentParser(
        prog="rasa",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Rasa command line interface. Rasa allows you to build "
        "your own conversational assistants 🤖. The 'rasa' command "
        "allows you to easily run most common commands like "
        "creating a new bot, training or evaluating models.",
    )

    parser.add_argument(
        "--version",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Print installed Rasa version",
    )

    parent_parser = argparse.ArgumentParser(add_help=False)
    add_logging_options(parent_parser)
    parent_parsers = [parent_parser]

    subparsers = parser.add_subparsers(help="Rasa commands")

    scaffold.add_subparser(subparsers, parents=parent_parsers)
    run.add_subparser(subparsers, parents=parent_parsers)
    shell.add_subparser(subparsers, parents=parent_parsers)
    train.add_subparser(subparsers, parents=parent_parsers)
    interactive.add_subparser(subparsers, parents=parent_parsers)
    telemetry.add_subparser(subparsers, parents=parent_parsers)
    test.add_subparser(subparsers, parents=parent_parsers)
    visualize.add_subparser(subparsers, parents=parent_parsers)
    data.add_subparser(subparsers, parents=parent_parsers)
    export.add_subparser(subparsers, parents=parent_parsers)
    x.add_subparser(subparsers, parents=parent_parsers)
    evaluate.add_subparser(subparsers, parents=parent_parsers)
    llm_fine_tuning.add_subparser(subparsers, parent_parsers)
    plugin_manager().hook.refine_cli(
        subparsers=subparsers, parent_parsers=parent_parsers
    )

    return parser


def print_version() -> None:
    """Prints version information of rasa tooling and python."""
    from rasa.utils.licensing import get_license_expiration_date

    print(f"Rasa Pro Version  :         {version.__version__}")
    print(f"Minimum Compatible Version: {MINIMUM_COMPATIBLE_VERSION}")
    print(f"Rasa SDK Version  :         {rasa_sdk_version}")
    print(f"Python Version    :         {platform.python_version()}")
    print(f"Operating System  :         {platform.platform()}")
    print(f"Python Path       :         {sys.executable}")
    print(f"License Expires   :         {get_license_expiration_date()}")


def main(raw_arguments: Optional[List[str]] = None) -> None:
    """Run as standalone python application.

    Args:
        raw_arguments: Arguments to parse. If not provided,
            arguments will be taken from the command line.
    """
    rasa.utils.licensing.validate_license_from_env()

    warn_if_rasa_plus_package_installed()
    parse_last_positional_argument_as_model_path()
    arg_parser = create_argument_parser()
    cmdline_arguments = arg_parser.parse_args(raw_arguments)
    log_level = getattr(cmdline_arguments, "loglevel", None)
    logging_config_file = getattr(cmdline_arguments, "logging_config_file", None)
    configure_logging_and_warnings(
        log_level, logging_config_file, warn_only_once=True, filter_repeated_logs=True
    )
    # TODO: we shouldn't configure colored logging, since we are using structlog
    # for logging - should be removed as part of logs cleanup
    rasa.utils.io.configure_colored_logging(log_level)
    configure_structlog(log_level)

    # Only setup TensorFlow environment if TensorFlow is available
    if TENSORFLOW_AVAILABLE and tf_env is not None:
        tf_env.setup_tf_environment()
        tf_env.check_deterministic_ops()

    # insert current path in syspath so custom modules are found
    sys.path.insert(1, os.getcwd())

    try:
        if hasattr(cmdline_arguments, "func"):
            is_studio_command = check_if_studio_command()

            if not is_studio_command:
                result = plugin_manager().hook.configure_commandline(
                    cmdline_arguments=cmdline_arguments
                )
                endpoints_file = result[0] if result else None

            rasa.telemetry.initialize_telemetry()
            rasa.telemetry.initialize_error_reporting()
            if not is_studio_command:
                plugin_manager().hook.init_telemetry(endpoints_file=endpoints_file)
                plugin_manager().hook.init_managers(endpoints_file=endpoints_file)

            cmdline_arguments.func(cmdline_arguments)
        elif hasattr(cmdline_arguments, "version"):
            print_version()
        else:
            # user has not provided a subcommand, let's print the help
            structlogger.error("cli.no_command", event_info="No command specified.")
            arg_parser.print_help()
            sys.exit(1)
    except DetailedRasaException as exc:
        structlogger.error(
            exc.code,
            event_info=exc.info,
            **exc.ctx,
        )
        sys.exit(1)
    except RasaException as exc:
        # these are exceptions we expect to happen (e.g. invalid training data format)
        # it doesn't make sense to print a stacktrace for these if we are not in
        # debug mode
        structlogger.debug(
            "cli.exception.details",
            event_info="Failed to run CLI command due to an exception.",
            exc_info=exc,
        )
        structlogger.error("cli.exception.rasa_exception", event_info=f"{exc}")
        sys.exit(1)
    except Exception as exc:
        structlogger.error(
            "cli.exception.general_exception",
            event_info=f"{exc.__class__.__name__}: {exc}",
            exc_info=exc,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

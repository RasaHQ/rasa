import argparse
import logging
import os
import platform
import sys
import importlib

from rasa_sdk import __version__ as rasa_sdk_version
from rasa.constants import MINIMUM_COMPATIBLE_VERSION
from rasa.utils.log_utils import configure_structlog

import rasa.telemetry
import rasa.utils.io
import rasa.utils.tensorflow.environment as tf_env
from rasa import version
from rasa.cli import (
    data,
    export,
    interactive,
    run,
    scaffold,
    shell,
    telemetry,
    test,
    train,
    visualize,
    x,
    evaluate,
)
from rasa.cli.arguments.default_arguments import add_logging_options
from rasa.cli.utils import parse_last_positional_argument_as_model_path
from rasa.plugin import plugin_manager
from rasa.shared.exceptions import RasaException
from rasa.shared.utils.cli import print_error
from rasa.utils.common import configure_logging_and_warnings

logger = logging.getLogger(__name__)


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
    plugin_manager().hook.refine_cli(
        subparsers=subparsers, parent_parsers=parent_parsers
    )

    return parser


def print_version() -> None:
    """Prints version information of rasa tooling and python."""
    print(f"Rasa Version      :         {version.__version__}")
    print(f"Minimum Compatible Version: {MINIMUM_COMPATIBLE_VERSION}")
    print(f"Rasa SDK Version  :         {rasa_sdk_version}")
    print(f"Python Version    :         {platform.python_version()}")
    print(f"Operating System  :         {platform.platform()}")
    print(f"Python Path       :         {sys.executable}")

    result = plugin_manager().hook.get_version_info()
    if result:
        print(f"\t{result[0][0]}  :         {result[0][1]}")

def _load_custom_logger_util(spec: str):
    """
    Lädt eine Logger-Setup-Funktion aus 'modul[:funktion]'.
    - Wenn keine Funktion angegeben ist, wird 'configure_unified_structlog' erwartet.
    - Die Funktion muss das Signatur-Pattern (log_level: Optional[int]) -> None unterstützen.
    """
    if not spec:
        return None
    if ":" in spec:
        module_name, func_name = spec.split(":", 1)
    else:
        module_name, func_name = spec, "configure_unified_structlog"

    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        raise RuntimeError(f"Could not import logger util module '{module_name}': {e}") from e
    try:
        fn = getattr(mod, func_name)
    except AttributeError as e:
        raise RuntimeError(f"Logger util function '{func_name}' not found in module '{module_name}'.") from e
    return fn

def main() -> None:
    """Run as standalone python application."""
    parse_last_positional_argument_as_model_path()
    arg_parser = create_argument_parser()
    cmdline_arguments = arg_parser.parse_args()

    log_level = getattr(cmdline_arguments, "loglevel", None)
    logging_config_file = getattr(cmdline_arguments, "logging_config_file", None)
    configure_logging_and_warnings(
        log_level, logging_config_file, warn_only_once=True, filter_repeated_logs=True
    )

    tf_env.setup_tf_environment()
    tf_env.check_deterministic_ops()

    # insert current path in syspath so custom modules are found
    sys.path.insert(1, os.getcwd())

    try:
        if hasattr(cmdline_arguments, "func"):
            rasa.utils.io.configure_colored_logging(log_level)

            result = plugin_manager().hook.configure_commandline(
                cmdline_arguments=cmdline_arguments
            )
            endpoints_file = result[0] if result else None

            rasa.telemetry.initialize_telemetry()
            rasa.telemetry.initialize_error_reporting()
            plugin_manager().hook.init_telemetry(endpoints_file=endpoints_file)
            plugin_manager().hook.init_managers(endpoints_file=endpoints_file)
            plugin_manager().hook.init_anonymization_pipeline(
                endpoints_file=endpoints_file
            )
            # === Logging-Setup auswählen ===
            custom_logger_spec = getattr(cmdline_arguments, "logger_util", None)
            if custom_logger_spec:
                # Nutzer wünscht explizit: Custom Logger statt Rasa-Default
                try:
                    init_fn = _load_custom_logger_util(custom_logger_spec)
                    # Wichtig: log_level (von CLI) weiterreichen
                    init_fn(log_level)
                    logger.info(
                        "Using custom logger util.",
                        extra={"extra_data": {"logger_util": custom_logger_spec}},
                    )
                except Exception as e:
                    # Fehler klar anzeigen und mit Exit abbrechen (keine halb-konfigurierte Logs)
                    print_error(f"Logger initialization failed: {e}")
                    sys.exit(1)
            else:
                # Fallback: Rasa-Default
                configure_structlog(log_level)

            cmdline_arguments.func(cmdline_arguments)
        elif hasattr(cmdline_arguments, "version"):
            print_version()
        else:
            # user has not provided a subcommand, let's print the help
            logger.error("No command specified.")
            arg_parser.print_help()
            sys.exit(1)
    except RasaException as e:
        # these are exceptions we expect to happen (e.g. invalid training data format)
        # it doesn't make sense to print a stacktrace for these if we are not in
        # debug mode
        logger.debug("Failed to run CLI command due to an exception.", exc_info=e)
        print_error(f"{e.__class__.__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

import argparse
import textwrap
from typing import List

from rasa import telemetry
from rasa.cli import SubParsersAction
import rasa.cli.utils
from rasa.shared.constants import DOCS_URL_TELEMETRY
import rasa.shared.utils.cli
from rasa.utils import licensing


def add_subparser(
    subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]
) -> None:
    """Add all telemetry tracking parsers.

    Args:
        subparsers: subparser we are going to attach to
        parents: Parent parsers, needed to ensure tree structure in argparse
    """
    telemetry_parser = subparsers.add_parser(
        "telemetry",
        parents=parents,
        help="Configuration of Rasa Pro telemetry reporting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    telemetry_subparsers = telemetry_parser.add_subparsers()
    telemetry_disable_parser = telemetry_subparsers.add_parser(
        "disable",
        parents=parents,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Disable Rasa Pro Telemetry reporting.",
    )
    telemetry_disable_parser.set_defaults(func=disable_telemetry)

    telemetry_enable_parser = telemetry_subparsers.add_parser(
        "enable",
        parents=parents,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Enable Rasa Pro Telemetry reporting.",
    )
    telemetry_enable_parser.set_defaults(func=enable_telemetry)
    telemetry_parser.set_defaults(func=inform_about_telemetry)


def inform_about_telemetry(_: argparse.Namespace) -> None:
    """Inform user about telemetry tracking."""
    is_enabled = telemetry.is_telemetry_enabled()
    if is_enabled:
        rasa.shared.utils.cli.print_success(
            "Telemetry reporting is currently enabled for this installation."
        )
    else:
        rasa.shared.utils.cli.print_success(
            "Telemetry reporting is currently disabled for this installation."
        )

    print(
        textwrap.dedent(
            """
            Rasa uses telemetry to report anonymous usage information. This information
            is essential to help improve Rasa Pro for all users."""
        )
    )
    if licensing.is_champion_server_license():
        print(
            "\nYou are using a developer license, which requires telemetry "
            "reporting to be enabled."
        )
    elif not is_enabled:
        print("\nYou can enable telemetry reporting using")
        rasa.shared.utils.cli.print_info("\n\trasa telemetry enable")
    else:
        print("\nYou can disable telemetry reporting using:")
        rasa.shared.utils.cli.print_info("\n\trasa telemetry disable")

    rasa.shared.utils.cli.print_success(
        "\nYou can find more information about telemetry reporting at "
        "" + DOCS_URL_TELEMETRY
    )


def disable_telemetry(_: argparse.Namespace) -> None:
    """Disable telemetry tracking."""
    if licensing.is_champion_server_license():
        rasa.shared.utils.cli.print_error(
            "You are using a developer license, which requires telemetry "
            "reporting to be enabled."
        )
        return

    telemetry.track_telemetry_disabled()
    telemetry.toggle_telemetry_reporting(is_enabled=False)
    rasa.shared.utils.cli.print_success("Disabled telemetry reporting.")


def enable_telemetry(_: argparse.Namespace) -> None:
    """Enable telemetry tracking."""
    telemetry.toggle_telemetry_reporting(is_enabled=True)
    rasa.shared.utils.cli.print_success("Enabled telemetry reporting.")

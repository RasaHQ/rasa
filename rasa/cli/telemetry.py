import argparse
from typing import List

from rasa import telemetry
import rasa.cli.utils
from rasa.shared.utils.cli import print_info, print_success
from rasa.shared.constants import DOCS_URL_TELEMETRY


# noinspection PyProtectedMember
def add_subparser(
    subparsers: argparse._SubParsersAction,  # skipcq:PYL-W0212
    parents: List[argparse.ArgumentParser],
) -> None:
    """Add all telemetry tracking parsers.

    Args:
        subparsers: subparser we are going to attach to
        parents: Parent parsers, needed to ensure tree structure in argparse
    """

    telemetry_parser = subparsers.add_parser(
        "telemetry",
        parents=parents,
        help="Configuration of Rasa Open Source telemetry reporting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    telemetry_subparsers = telemetry_parser.add_subparsers()
    telemetry_disable_parser = telemetry_subparsers.add_parser(
        "disable",
        parents=parents,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Disable Rasa Open Source Telemetry reporting.",
    )
    telemetry_disable_parser.set_defaults(func=disable_telemetry)

    telemetry_disable_parser = telemetry_subparsers.add_parser(
        "enable",
        parents=parents,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Enable Rasa Open Source Telemetry reporting.",
    )
    telemetry_disable_parser.set_defaults(func=enable_telemetry)
    telemetry_parser.set_defaults(func=inform_about_telemetry)


def inform_about_telemetry(_: argparse.Namespace) -> None:
    """Inform user about telemetry tracking."""
    print_success("TODO: some information about telemetry")

    print("\nYou can enable telemetry reporting using")
    print_info("\n\trasa telemetry enable")
    print("\nand disable telemetry reporting using:")
    print_info("\n\trasa telemetry disable")

    print_success(
        "\nYou can find more information about telemetry reporting at "
        "" + DOCS_URL_TELEMETRY
    )


def disable_telemetry(_: argparse.Namespace) -> None:
    """Disable telemetry tracking."""
    rasa.cli.utils.run_in_loop(telemetry.track_telemetry_disabled())
    telemetry.toggle_telemetry_reporting(is_enabled=False)
    print_success("Disabled telemetry reporting.")


def enable_telemetry(_: argparse.Namespace) -> None:
    """Enable telemetry tracking."""
    telemetry.toggle_telemetry_reporting(is_enabled=True)
    print_success("Enabled telemetry reporting.")

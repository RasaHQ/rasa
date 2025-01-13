import argparse
from pathlib import Path
from typing import List, Text

from rasa.cli import SubParsersAction
from rasa.cli.arguments.default_arguments import add_domain_param
from rasa.markers.upload import upload
from rasa.shared.core.domain import Domain
from rasa.shared.utils.cli import print_error_and_exit


def add_subparser(
    subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]
) -> None:
    """Add all marker parsers.

    Args:
        subparsers: subparser we are going to attach to
        parents: Parent parsers, needed to ensure tree structure in argparse
    """
    markers_parser = subparsers.add_parser(
        "markers",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Rasa Studio commands.",
    )
    marker_subparser = markers_parser.add_subparsers()

    _add_upload_subparser(marker_subparser, parents)


def _add_upload_subparser(
    marker_sub_parsers: SubParsersAction, parents: List[argparse.ArgumentParser]
) -> None:
    marker_upload_sub_parser = marker_sub_parsers.add_parser(
        "upload",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Upload Markers to Rasa Pro Services",
    )

    marker_upload_sub_parser.set_defaults(func=_upload_markers)

    marker_upload_sub_parser.add_argument(
        "--config",
        default="markers.yml",
        type=Path,
        help="The marker configuration file(s) containing marker definitions. "
        "This can be a single YAML file, or a directory that contains several "
        "files with marker definitions in it. The content of these files will "
        "be read and merged together.",
    )

    marker_upload_sub_parser.add_argument(
        "--rasa-pro-services-url",
        default="",
        type=Text,
        help="The URL of the Rasa Pro Services instance to upload markers to."
        "Specified URL should not contain a trailing slash.",
    )

    add_domain_param(marker_upload_sub_parser)


def _upload_markers(args: argparse.Namespace) -> None:
    markers_path = args.config
    domain_path = args.domain
    url = args.rasa_pro_services_url

    domain = Domain.load(domain_path) if domain_path else None

    if domain is None:
        print_error_and_exit("No domain specified. Skipping validation.")

    upload(url=url, domain=domain, markers_path=markers_path)

import argparse
from typing import List

import questionary
from rasa.cli import SubParsersAction

import rasa.cli.studio.download
import rasa.cli.studio.train
import rasa.cli.studio.upload
from rasa.studio.auth import StudioAuth
from rasa.studio.config import StudioConfig


def add_subparser(
    subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]
) -> None:
    studio_parser = subparsers.add_parser(
        "studio",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Rasa Studio commands.",
    )
    studio_subparsers = studio_parser.add_subparsers()

    rasa.cli.studio.train.add_subparser(studio_subparsers, parents)
    rasa.cli.studio.upload.add_subparser(studio_subparsers, parents)
    rasa.cli.studio.download.add_subparser(studio_subparsers, parents)

    _add_config_subparser(studio_subparsers, parents)
    _add_login_subparser(studio_subparsers, parents)


def _add_config_subparser(
    studio_sub_parsers: SubParsersAction, parents: List[argparse.ArgumentParser]
) -> None:
    studio_sub_parsers.add_parser(
        "config",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Configure communication parameters for Rasa Studio",
    ).set_defaults(func=_create_studio_config)


def _add_login_subparser(
    studio_sub_parsers: SubParsersAction, parents: List[argparse.ArgumentParser]
) -> None:
    login_parser = studio_sub_parsers.add_parser(
        "login",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Login to Rasa Studio",
    )
    login_parser.set_defaults(func=_studio_login)

    login_parser.add_argument(
        "--username",
        type=str,
        help="Username for Rasa Studio",
    )

    login_parser.add_argument(
        "--password",
        type=str,
        help="Password for Rasa Studio",
    )


def _studio_login(args: argparse.Namespace) -> None:
    """Login to Rasa Studio.

    Args:
        args: Commandline arguments
    """
    username = args.username
    password = args.password

    studio_config = StudioConfig.read_config()
    studio_auth = StudioAuth(studio_config)
    studio_auth.login(username, password)


def _create_studio_config(_: argparse.Namespace) -> None:
    authentication_server_url = questionary.text(
        "Please enter an URL path to the Authentication Server",
        default="https://auth.rasa.com",
    ).ask()

    studio_url = questionary.text(
        "Please enter an URL path to the Studio",
        default="https://studio.rasa.com/graphql/",
    ).ask()

    realm_name = questionary.text("Please enter Realm Name", default="Studio").ask()

    client_id = questionary.text(
        "Please enter client ID",
    ).ask()

    if not authentication_server_url.endswith("/"):
        authentication_server_url += "/"

    StudioConfig(
        authentication_server_url=authentication_server_url,
        studio_url=studio_url,
        client_id=client_id,
        realm_name=realm_name,
    ).write_config()

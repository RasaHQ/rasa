import argparse
from typing import List, Optional
from urllib.parse import ParseResult, urlparse

import questionary
from rasa.cli import SubParsersAction

import rasa.cli.studio.download
import rasa.cli.studio.train
import rasa.cli.studio.upload
from rasa.studio.auth import StudioAuth
from rasa.studio.config import StudioConfig


DEFAULT_REALM_NAME = "rasa-studio"

DEFAULT_CLIENT_ID = "admin-cli"


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
    studio_config_parser = studio_sub_parsers.add_parser(
        "config",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Configure communication parameters for Rasa Studio",
    )

    studio_config_parser.set_defaults(func=create_and_store_studio_config)

    # add advanced configuration flag to trigger
    # advanced configuration setup for authentication settings
    studio_config_parser.add_argument(
        "--advanced",
        action="store_true",
        default=False,
        help="Configure additional authentication parameters for Rasa Studio",
    )


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
    studio_config = StudioConfig.read_config()
    studio_auth = StudioAuth(studio_config)

    # show the user the studio url they are logging into
    # urlparse will always return parseresult if a string is passed in as url
    parsed_url: ParseResult = urlparse(studio_config.studio_url)  # type: ignore[assignment]
    studio_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

    rasa.shared.utils.cli.print_info(
        f"Trying to log in to Rasa Studio at {studio_url} ..."
    )

    try:
        username = args.username
        password = args.password

        if not username:
            username = questionary.text("Please enter your username").unsafe_ask()

        if not password:
            password = questionary.password("Please enter your password").unsafe_ask()

        studio_auth.login(username, password)
    except KeyboardInterrupt:
        rasa.shared.utils.cli.print_error("Login to Rasa Studio aborted.")
    except Exception as e:
        rasa.shared.utils.cli.print_error(f"Failed to login to Rasa Studio. {e}")


def _configure_studio_url() -> Optional[str]:
    """Configure the Rasa Studio URL.

    Returns:
        The configured Rasa Studio URL
    """
    studio_url = questionary.text(
        "Please provide the Rasa Studio URL",
    ).unsafe_ask()

    if not studio_url.endswith("/"):
        studio_url += "/"

    # prepend with https if no protocol is provided
    if not studio_url.startswith("http"):
        return "https://" + studio_url
    return studio_url


def _get_advanced_config(studio_url: str) -> tuple:
    """Get the advanced configuration values for Rasa Studio."""
    keycloak_url = questionary.text(
        "Please provide your Rasa Studio Keycloak URL",
        default=studio_url + "auth/",
    ).unsafe_ask()

    realm_name = questionary.text(
        "Please enter Realm Name", default=DEFAULT_REALM_NAME
    ).unsafe_ask()

    client_id = questionary.text(
        "Please enter client ID", default=DEFAULT_CLIENT_ID
    ).unsafe_ask()

    return keycloak_url, realm_name, client_id


def _get_default_config(studio_url: str) -> tuple:
    """Get the default configuration values for Rasa Studio."""
    keycloak_url = studio_url + "auth/"
    realm_name = DEFAULT_REALM_NAME
    client_id = DEFAULT_CLIENT_ID

    rasa.shared.utils.cli.print_info(
        f"Using default values for "
        f"Keycloak URL: {keycloak_url}, "
        f"Realm Name: '{realm_name}', "
        f"Client ID: '{client_id}'. "
        f"You can use '--advanced' to configure these settings."
    )

    return keycloak_url, realm_name, client_id


def _create_studio_config(
    studio_url: str, keycloak_url: str, realm_name: str, client_id: str
) -> StudioConfig:
    """Create a StudioConfig object with the provided parameters."""
    return StudioConfig(
        authentication_server_url=keycloak_url,
        studio_url=studio_url + "api/graphql/",
        client_id=client_id,
        realm_name=realm_name,
    )


def _check_studio_auth(studio_auth: StudioAuth) -> bool:
    """Check if the Rasa Studio authentication server is reachable."""
    if studio_auth.health_check():
        rasa.shared.utils.cli.print_info(
            "Tried configuration and successfully reached Rasa Studio."
        )
        return True
    return False


def _prompt_store_config_anyways() -> bool:
    """Prompt the user to store the configuration."""
    should_store_config = questionary.confirm(
        "Do you want to store the configuration anyway?"
    ).unsafe_ask()

    return should_store_config


def _configure_studio_config(args: argparse.Namespace) -> StudioConfig:
    """Configure the Rasa Studio connection settings."""
    studio_url = _configure_studio_url()

    # check if the user wants to configure advanced authentication settings
    if args.advanced:
        keycloak_url, realm_name, client_id = _get_advanced_config(studio_url)
    else:
        keycloak_url, realm_name, client_id = _get_default_config(studio_url)

    # create a configuration and auth object to try to reach the studio
    studio_config = _create_studio_config(
        studio_url, keycloak_url, realm_name, client_id
    )
    studio_auth = StudioAuth(studio_config)

    if _check_studio_auth(studio_auth):
        return studio_config

    rasa.shared.utils.cli.print_error(
        f"Failed to reach Rasa Studio authentication server at "
        f"{studio_config.authentication_server_url}."
    )

    if _prompt_store_config_anyways():
        return studio_config

    rasa.shared.utils.cli.print_info(
        "The config was not stored, you can change the values "
        "again or abort with Ctrl-C."
    )

    return _configure_studio_config(args)


def create_and_store_studio_config(args: argparse.Namespace) -> None:
    """Create and store the Rasa Studio configuration.

    Args:
        args: Commandline arguments
    """
    try:
        rasa.shared.utils.cli.print_info(
            "Configuring Rasa Studio connection settings ..."
        )
        studio_config = _configure_studio_config(args)

        if studio_config:
            studio_config.write_config()

            rasa.shared.utils.cli.print_success(
                "Successfully configured Rasa Pro to connect to Studio."
            )
        else:
            rasa.shared.utils.cli.print_error(
                "Failed to configure Rasa Pro to connect to Studio."
            )
    except KeyboardInterrupt:
        rasa.shared.utils.cli.print_error("Configuration of Rasa Studio aborted.")

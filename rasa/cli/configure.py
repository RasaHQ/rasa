import os

import questionary


def add_subparser(subparsers):
    config_parser = subparsers.add_parser(
        'configure',
        help='Configure a Rasa bot')
    config_parser.set_defaults(func=run)
    config_parser.add_argument('section',
                               choices=["channel"],
                               help="which element of the bot you want to "
                                    "configure")


def configure_channel(channel):
    from rasa_core.utils import print_error, print_success
    import rasa_core.utils

    credentials_file = questionary.text(
        "Please enter a path where to store the credentials file",
        default="credentials.yml").ask()

    if channel == "facebook":
        fb_config = questionary.form(
            verify=questionary.text(
                "Facebook verification string (choosen during "
                "webhook creation)"),
            secret=questionary.text(
                "Facebook application secret"),
            access_token=questionary.text(
                "Facebook access token"),
        ).ask()

        credentials = {
            "verify": fb_config["verify"],
            "secret": fb_config["secret"],
            "page-access-token": fb_config["access_token"]}

        rasa_core.utils.dump_obj_as_yaml_to_file(
            credentials_file,
            {"facebook": credentials}
        )
        print_success("Created facebook configuration and added it to '{}'."
                      "".format(os.path.abspath(credentials_file)))
    else:
        print_error("Pieee...Rumble...ERROR! Configuration of this channel "
                    "not yet supported.")
        exit(1)


def run(args):
    if args.section == "channel":
        channel = questionary.select("Which channel do you want to add",
                                     choices=["facebook",
                                              "slack",
                                              "rasa"]).ask()
        configure_channel(channel)

import argparse
import os
from typing import List, Text

import rasa.train
from rasa.cli.shell import shell
from rasa.cli.utils import create_output_path, print_success
from rasa.constants import DEFAULT_CONFIG_PATH, DEFAULT_DATA_PATH, DEFAULT_DOMAIN_PATH


# noinspection PyProtectedMember
def add_subparser(
    subparsers: argparse._SubParsersAction, parents: List[argparse.ArgumentParser]
):
    scaffold_parser = subparsers.add_parser(
        "init", parents=parents, help="Create a new project from a initial_project"
    )
    scaffold_parser.add_argument(
        "--no_prompt",
        "--no-prompt",
        action="store_true",
        help="Automatic yes or default options to prompts and uppressed warnings",
    )
    scaffold_parser.set_defaults(func=run)


def print_train_or_instructions(args: argparse.Namespace, path: Text) -> None:
    import questionary

    print_success("Finished creating project structure.")

    should_train = questionary.confirm(
        "Do you want me to train an initial model for the bot? 💪🏽"
    ).skip_if(args.no_prompt, default=True)

    if should_train:
        print_success("Training an initial model...")
        config = os.path.join(path, DEFAULT_CONFIG_PATH)
        training_files = os.path.join(path, DEFAULT_DATA_PATH)
        domain = os.path.join(path, DEFAULT_DOMAIN_PATH)
        output = os.path.join(path, create_output_path())

        args.model = rasa.train(domain, config, training_files, output)

        print_run_or_instructions(args, path)

    else:
        print_success(
            "No problem 👍🏼. You can also train me later by going "
            "to the project directory and running 'rasa train'."
            "".format(path)
        )


def print_run_or_instructions(args: argparse.Namespace, path: Text) -> None:
    from rasa.core import constants
    import questionary

    should_run = (
        questionary.confirm(
            "Do you want to speak to the trained bot on the command line? 🤖"
        )
        .skip_if(args.no_prompt, default=False)
        .ask()
    )

    if should_run:
        # provide defaults for command line arguments
        attributes = [
            "endpoints",
            "credentials",
            "cors",
            "auth_token",
            "jwt_secret",
            "jwt_method",
            "enable_api",
        ]
        for a in attributes:
            setattr(args, a, None)

        args.port = constants.DEFAULT_SERVER_PORT

        shell(args)
    else:
        print_success(
            "Ok 👍🏼. If you want to speak to the bot later, "
            "change into the project directory and run 'rasa shell'."
            "".format(path)
        )


def init_project(args: argparse.Namespace, path: Text) -> None:
    _create_initial_project(path)
    print ("Created project directory at '{}'.".format(os.path.abspath(path)))
    print_train_or_instructions(args, path)


def _create_initial_project(path: Text) -> None:
    from distutils.dir_util import copy_tree

    copy_tree(scaffold_path(), path)


def scaffold_path() -> Text:
    import pkg_resources

    return pkg_resources.resource_filename(__name__, "initial_project")


def print_cancel() -> None:
    print_success(
        "Ok. Then I stop here. If you need me again, simply type 'rasa init' 🙋🏽‍♀️"
    )
    exit(0)


def _ask_create_path(path: Text) -> None:
    import questionary

    should_create = questionary.confirm(
        "Path '{}' does not exist 🧐. Should I create it?".format(path)
    ).ask()
    if should_create:
        os.makedirs(path)
    else:
        print_success(
            "Ok. Then I stop here. If you need me again, simply type "
            "'rasa init' 🙋🏽‍♀️"
        )
        exit(0)


def _ask_overwrite(path: Text) -> None:
    import questionary

    overwrite = questionary.confirm(
        "Directory '{}' is not empty. Continue?".format(os.path.abspath(path))
    ).ask()
    if not overwrite:
        print_cancel()


def run(args: argparse.Namespace) -> None:
    import questionary

    print_success("Welcome to Rasa! 🤖\n")
    print (
        "To get started quickly, I can assist you to create an "
        "initial project.\n"
        "If you need some help to get from this template to a "
        "bad ass contextual assistant, checkout our quickstart guide"
        "here: https://rasa.com/docs/core/quickstart \n\n"
        "Now let's start! 👇🏽\n"
    )

    path = (
        questionary.text(
            "Please enter a folder path where I should create "
            "the initial project [default: current directory]",
            default=".",
        )
        .skip_if(args.no_prompt, default=".")
        .ask()
    )

    if not os.path.isdir(path):
        _ask_create_path(path)

    if path is None or not os.path.isdir(path):
        print_cancel()

    if not args.no_prompt and len(os.listdir(path)) > 0:
        _ask_overwrite(path)

    init_project(args, path)

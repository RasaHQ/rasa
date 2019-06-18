import argparse
import os
from typing import List, Text

import rasa.train
from rasa.cli.shell import shell
from rasa.cli.utils import create_output_path, print_success
from rasa.constants import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_DOMAIN_PATH,
    DOCS_BASE_URL,
)


# noinspection PyProtectedMember
def add_subparser(
    subparsers: argparse._SubParsersAction, parents: List[argparse.ArgumentParser]
):
    scaffold_parser = subparsers.add_parser(
        "init",
        parents=parents,
        help="Creates a new project, with example training data, actions, and config files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    scaffold_parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Automatically choose default options for prompts and suppress warnings.",
    )
    scaffold_parser.set_defaults(func=run)


def print_train_or_instructions(args: argparse.Namespace, path: Text) -> None:
    import questionary

    print_success("Finished creating project structure.")

    should_train = questionary.confirm(
        "Do you want to train an initial model? 💪🏽"
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
            "No problem 👍🏼. You can also train a model later by going "
            "to the project directory and running 'rasa train'."
            "".format(path)
        )


def print_run_or_instructions(args: argparse.Namespace, path: Text) -> None:
    from rasa.core import constants
    import questionary

    should_run = (
        questionary.confirm(
            "Do you want to speak to the trained assistant on the command line? 🤖"
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
        if args.no_prompt:
            print (
                "If you want to speak to the assistant, "
                "run 'rasa shell' at any time inside "
                "the project directory."
                "".format(path)
            )
        else:
            print_success(
                "Ok 👍🏼. "
                "If you want to speak to the assistant, "
                "run 'rasa shell' at any time inside "
                "the project directory."
                "".format(path)
            )


def init_project(args: argparse.Namespace, path: Text) -> None:
    create_initial_project(path)
    print ("Created project directory at '{}'.".format(os.path.abspath(path)))
    print_train_or_instructions(args, path)


def create_initial_project(path: Text) -> None:
    from distutils.dir_util import copy_tree

    copy_tree(scaffold_path(), path)


def scaffold_path() -> Text:
    import pkg_resources
    # bf: changed scaffold folder to bf project
    return pkg_resources.resource_filename(__name__, "bf_initial_project")


def print_cancel() -> None:
    print_success("Ok. You can continue setting up by running 'rasa init' 🙋🏽‍♀️")
    exit(0)


def _ask_create_path(path: Text) -> None:
    import questionary

    should_create = questionary.confirm(
        "Path '{}' does not exist 🧐. Create path?".format(path)
    ).ask()
    if should_create:
        os.makedirs(path)
    else:
        print_success("Ok. You can continue setting up by running " "'rasa init' 🙋🏽‍♀️")
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
    if args.no_prompt:
        print (
            "To get started quickly, an "
            "initial project will be created.\n"
            "If you need some help, check out "
            "the documentation at {}.\n".format(DOCS_BASE_URL)
        )
    else:
        print (
            "To get started quickly, an "
            "initial project will be created.\n"
            "If you need some help, check out "
            "the documentation at {}.\n"
            "Now let's start! 👇🏽\n".format(DOCS_BASE_URL)
        )

    path = (
        questionary.text(
            "Please enter a path where the project will be "
            "created [default: current directory]",
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

import argparse
import os
import sys
from collections import defaultdict
from enum import Enum
from typing import List, Text

from rasa import telemetry
from rasa.api import train
from rasa.cli import SubParsersAction
from rasa.cli.inspect import inspect
from rasa.shared.constants import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_DOMAIN_PATH,
    DEFAULT_MODELS_PATH,
    DOCS_BASE_URL,
)
from rasa.shared.utils.cli import print_error_and_exit, print_success


class ProjectTemplateName(Enum):
    """Enum of the different project templates."""

    DEFAULT = "default"
    TUTORIAL = "tutorial"
    CALM = "calm"

    def __str__(self) -> str:
        return self.value


template_domain_path = defaultdict(lambda: DEFAULT_DOMAIN_PATH)
template_domain_path[ProjectTemplateName.CALM] = "domain"


def add_subparser(
    subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]
) -> None:
    """Add all init parsers.

    Args:
        subparsers: subparser we are going to attach to
        parents: Parent parsers, needed to ensure tree structure in argparse
    """
    scaffold_parser = subparsers.add_parser(
        "init",
        parents=parents,
        help="Creates a new project, with example training data, "
        "actions, and config files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    scaffold_parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Automatically choose default options for prompts and suppress warnings.",
    )
    scaffold_parser.add_argument(
        "--init-dir",
        default=None,
        help="Directory where your project should be initialized.",
    )
    scaffold_parser.add_argument(
        "--template",
        type=ProjectTemplateName,
        choices=list(ProjectTemplateName),
        default=ProjectTemplateName.DEFAULT,
        help="Select the template to use for the project.",
    )
    scaffold_parser.set_defaults(func=run)


def print_train_or_instructions(args: argparse.Namespace) -> None:
    """Train a model if the user wants to."""
    import questionary

    print_success("Finished creating project structure.")

    should_train = (
        questionary.confirm("Do you want to train an initial model? 💪🏽")
        .skip_if(args.no_prompt, default=True)
        .ask()
    )

    if should_train:
        print_success("Training an initial model...")
        training_result = train(
            template_domain_path[args.template],
            DEFAULT_CONFIG_PATH,
            DEFAULT_DATA_PATH,
            DEFAULT_MODELS_PATH,
        )
        args.model = training_result.model

        print_run_or_instructions(args)

    else:
        print_success(
            "No problem 👍🏼. You can also train a model later by going "
            "to the project directory and running 'rasa train'."
        )


def print_run_or_instructions(args: argparse.Namespace) -> None:
    import questionary

    from rasa.core import constants

    should_run = (
        questionary.confirm("Do you want to speak to the trained assistant? 🤖")
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
            "remote_storage",
        ]
        for a in attributes:
            setattr(args, a, None)

        args.port = constants.DEFAULT_SERVER_PORT

        inspect(args)
    else:
        if args.no_prompt:
            print(
                "If you want to speak to the assistant, "
                "run 'rasa shell' at any time inside "
                "the project directory."
            )
        else:
            print_success(
                "Ok 👍🏼. "
                "If you want to speak to the assistant, "
                "run 'rasa shell' at any time inside "
                "the project directory."
            )


def init_project(args: argparse.Namespace, path: Text) -> None:
    """Inits project."""
    os.chdir(path)
    create_initial_project(".", args.template)
    print(f"Created project directory at '{os.getcwd()}'.")
    print_train_or_instructions(args)


def create_initial_project(
    path: Text, template: ProjectTemplateName = ProjectTemplateName.DEFAULT
) -> None:
    """Creates directory structure and templates for initial project."""
    from distutils.dir_util import copy_tree

    copy_tree(scaffold_path(template), path)


def scaffold_path(template: ProjectTemplateName) -> Text:
    import importlib_resources

    import rasa.cli.project_templates

    template_module = rasa.cli.project_templates.__name__
    return str(importlib_resources.files(template_module).joinpath(template.value))


def print_cancel() -> None:
    print_success("Ok. You can continue setting up by running 'rasa init' 🙋🏽‍♀️")
    sys.exit(0)


def _ask_create_path(path: Text) -> None:
    import questionary

    should_create = questionary.confirm(
        f"Path '{path}' does not exist 🧐. Create path?"
    ).ask()

    if should_create:
        try:
            os.makedirs(path)
        except (PermissionError, OSError, FileExistsError) as e:
            print_error_and_exit(
                f"Failed to create project path at '{path}'. " f"Error: {e}"
            )
    else:
        print_success(
            "Ok, will exit for now. You can continue setting up by "
            "running 'rasa init' again 🙋🏽‍♀️"
        )
        sys.exit(0)


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
    print(
        "By installing and using this software, you agree to be bound by the terms "
        "and conditions of the Developer Terms available at "
        "https://rasa.com/developer-terms. "
        "Please review the Developer Terms carefully before proceeding.\n"
    )
    if args.no_prompt:
        print(
            f"To get started quickly, an "
            f"initial project will be created.\n"
            f"If you need some help, check out "
            f"the documentation at {DOCS_BASE_URL}.\n"
        )
    else:
        print(
            f"To get started quickly, an "
            f"initial project will be created.\n"
            f"If you need some help, check out "
            f"the documentation at {DOCS_BASE_URL}.\n"
            f"Now let's start! 👇🏽\n"
        )

    if args.init_dir is not None:
        path = args.init_dir
    else:
        path = (
            questionary.text(
                "Please enter a path where the project will be "
                "created [default: current directory]"
            )
            .skip_if(args.no_prompt, default="")
            .ask()
        )
        # set the default directory. we can't use the `default` property
        # in questionary as we want to avoid showing the "." in the prompt as the
        # initial value. users tend to overlook it and it leads to invalid
        # paths like: ".C:\mydir".
        # Can't use `if not path` either, as `None` will be handled differently (abort)
        if path == "":
            path = "."

    if args.no_prompt and not os.path.isdir(path):
        print_error_and_exit(f"Project init path '{path}' not found.")

    if path and not os.path.isdir(path):
        _ask_create_path(path)

    if path is None or not os.path.isdir(path):
        print_cancel()

    if not args.no_prompt and len(os.listdir(path)) > 0:
        _ask_overwrite(path)

    telemetry.track_project_init(path)

    init_project(args, path)

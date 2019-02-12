import os
from distutils.dir_util import copy_tree

import questionary
from rasa.cli import train
from rasa_core.utils import print_success
from rasa_core import constants

from rasa.cli.shell import shell
from rasa.cli.train import create_default_output_path


def add_subparser(subparsers, parents):
    scaffold_parser = subparsers.add_parser(
        "init",
        parents=parents,
        help="Create a new project from a initial_project")
    scaffold_parser.set_defaults(func=run)


def scaffold_path():
    import pkg_resources
    return pkg_resources.resource_filename(__name__, "initial_project")


def print_train_or_instructions(args, path):
    print_success("Your bot initial_project is ready to go!")
    should_train = questionary.confirm("Do you want me to train an initial "
                                       "model for the bot?").ask()
    if should_train:
        args.config = os.path.join(path, "config.yml")
        args.stories = os.path.join(path, "data/core")
        args.domain = os.path.join(path, "domain.yml")
        args.nlu = os.path.join(path, "data/nlu")
        args.out = os.path.join(path, create_default_output_path())

        args.model = train.train(args)

        print_run_or_instructions(args)

    else:
        print("Great. To train your bot, run `cd {} && rasa train`."
              "".format(path))


def print_run_or_instructions(args):
    should_run = questionary.confirm("Do you want me to run the trained model "
                                     "in the command line?").ask()

    if should_run:
        # provide defaults for command line arguments
        attributes = ["endpoints", "credentials", "cors", "auth_token",
                      "jwt_secret", "jwt_method", "enable_api"]
        for a in attributes:
            setattr(args, a, None)

        args.port = constants.DEFAULT_SERVER_PORT

        shell(args)
    else:
        print("Great. To run your bot, run `cd {} && rasa shell`."
              "".format(args.model))


def init_project(args, path):
    if not os.path.isdir(path):
        os.makedirs(path)

    copy_tree(scaffold_path(), path)

    print("Created project directory at '{}'.".format(os.path.abspath(path)))
    print_train_or_instructions(args, path)


def run(args):
    path = questionary.text("Please enter a folder path for the bot "
                            "[default: current directory]", ".").ask()
    if path is None or not os.path.isdir(path):
        print("Path '{}' is not a valid directory. Aborted creation."
              "".format(path))
        exit(1)
    else:
        if len(os.listdir(path)) > 0:
            overwrite = questionary.confirm(
                "Directory '{}' is not empty. Continue?"
                "".format(os.path.abspath(path))).ask()
            if not overwrite:
                print("Aborted creation.")
                exit(1)
        init_project(args, path)

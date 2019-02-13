import os
from distutils.dir_util import copy_tree

import questionary
from rasa.cli import train
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
    from rasa_core.utils import print_success

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

        print_run_or_instructions(args, path)

    else:
        print("Great. To train your bot, run `cd {} && rasa train`."
              "".format(path))


def print_run_or_instructions(args, path):
    from rasa_core import constants

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
              "".format(path))


def init_project(args, path):
    copy_tree(scaffold_path(), path)

    print("Created project directory at '{}'.".format(os.path.abspath(path)))
    print_train_or_instructions(args, path)


def run(args):
    path = questionary.text("Please enter a folder path for the bot "
                            "[default: current directory]", ".").ask()

    if not os.path.isdir(path):
        should_create = questionary.confirm("Path '{}' does not exist. Should "
                                            "I create it?".format(path)).ask()
        if should_create:
            os.makedirs(path)
        else:
            print("Ok. Then I stop here.")
            exit(0)

    if path is None or not os.path.isdir(path):
        print("Path '{}' is not a valid directory. Aborted creation."
              "".format(path))
        exit(1)

    if len(os.listdir(path)) > 0:
        overwrite = questionary.confirm(
            "Directory '{}' is not empty. Continue?"
            "".format(os.path.abspath(path))).ask()
        if not overwrite:
            print("Aborted creation.")
            exit(1)

    init_project(args, path)

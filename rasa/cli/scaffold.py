import os
from distutils.dir_util import copy_tree

import questionary
from rasa.cli import train
from rasa_core.utils import print_success


def add_subparser(subparsers):
    scaffold_parser = subparsers.add_parser(
        'scaffold',
        help='Create a new project from a scaffold')
    scaffold_parser.set_defaults(func=run)


def scaffold_path():
    import pkg_resources
    return pkg_resources.resource_filename(__name__, "scaffold")


def print_instructions(args, path):
    print_success("Your bot scaffold is ready to go!")
    should_train = questionary.confirm("Do you want me to train an initial "
                                       "model for the bot?").ask()
    if should_train:
        default_model_path = os.path.join(path, "model.tar.gz")
        output_path = questionary.text("Please enter a folder where the "
                                       "trained model gets stored",
                                       default=default_model_path
                                       ).ask()
        train.train(os.path.join(path, "config.yml"),
                    os.path.join(path, "nlu.md"),
                    os.path.join(path, "domain.yml"),
                    os.path.join(path, "stories.md"),
                    output_path)
    else:
        print("Great. To train your bot, run `rasa train {}`".format(path))


def init_project(args, path):
    if not os.path.isdir(path):
        os.makedirs(path)

    copy_tree(scaffold_path(), path)

    print("Created project directory at '{}'".format(os.path.abspath(path)))
    print_instructions(args, path)


def run(args):
    path = questionary.text("Please enter a folder path for the bot").ask()
    if path is None:
        print("Aborted creation.")
        exit(1)
    else:
        if os.path.exists(path):
            overwrite = questionary.confirm(
                "Path '{}' already exists. Overwrite?"
                "".format(os.path.abspath(path))).ask()
            if not overwrite:
                print("Aborted creation.")
                exit(1)
        init_project(args, path)

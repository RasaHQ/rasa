import argparse
from typing import List

import rasa.cli.run


# noinspection PyProtectedMember
def add_subparser(
    subparsers: argparse._SubParsersAction, parents: List[argparse.ArgumentParser]
):
    shell_parser = subparsers.add_parser(
        "shell",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Speak to a trained model on the command line",
    )
    rasa.cli.run.add_run_arguments(shell_parser)
    shell_parser.set_defaults(func=shell)


def shell(args: argparse.Namespace):
    args.connector = "cmdline"
    rasa.cli.run.run(args)

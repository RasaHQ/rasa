"""Argparse registration for the ``rasa tools`` command family."""

import argparse
from typing import List

from rasa.cli import SubParsersAction


def add_subparser(
    subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]
) -> None:
    """Add all tools parsers.

    Args:
        subparsers: subparser we are going to attach to
        parents: Parent parsers, needed to ensure tree structure in argparse
    """
    from rasa.cli.arguments.tools import (
        set_tools_docs_arguments,
        set_tools_init_arguments,
        set_tools_run_arguments,
        set_tools_skills_arguments,
        set_tools_status_arguments,
    )
    from rasa.cli.tools.docs import docs_tools
    from rasa.cli.tools.init import init_tools
    from rasa.cli.tools.run import run_tools
    from rasa.cli.tools.skills import skills_tools
    from rasa.cli.tools.status import status_tools

    tools_parser = subparsers.add_parser(
        "tools",
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="Commands for Rasa developer tools.",
    )
    tools_parser.set_defaults(func=lambda _: tools_parser.print_help(None))

    tools_subparsers = tools_parser.add_subparsers()

    init_parser = tools_subparsers.add_parser(
        "init",
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="Interactive setup wizard for Rasa Tools.",
    )
    init_parser.set_defaults(func=init_tools)
    set_tools_init_arguments(init_parser)

    init_subparsers = init_parser.add_subparsers()

    init_skills_parser = init_subparsers.add_parser(
        "skills",
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="Fetch and install Rasa agent skills.",
    )
    init_skills_parser.set_defaults(func=skills_tools)
    set_tools_skills_arguments(init_skills_parser)

    init_docs_parser = init_subparsers.add_parser(
        "docs",
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="Fetch offline Rasa documentation files.",
    )
    init_docs_parser.set_defaults(func=docs_tools)
    set_tools_docs_arguments(init_docs_parser)

    run_parser = tools_subparsers.add_parser(
        "run",
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="Start the Rasa MCP server.",
    )
    run_parser.set_defaults(func=run_tools)
    set_tools_run_arguments(run_parser)

    status_parser = tools_subparsers.add_parser(
        "status",
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="Show current Rasa Tools configuration and readiness.",
    )
    status_parser.set_defaults(func=status_tools)
    set_tools_status_arguments(status_parser)

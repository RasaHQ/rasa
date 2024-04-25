import argparse
from typing import Text


def remove_argument_from_parser(
    parser: argparse.ArgumentParser, argument: Text
) -> None:
    """Remove argument from parse.

    A public API for this use case doesn't exist; so we use a private one.
    We should be able to detect breaking changes of this API through unit-testing.

    Inspired by this Stackoverflow answer: https://stackoverflow.com/a/32809642

    Args:
        parser: an argument parser instance
        argument: the argument to remove, e.g. -f or --foo
    """
    for action in parser._actions:
        for option in action.option_strings:
            if option == argument:
                parser._handle_conflict_resolve(
                    None,
                    [(option, action)],
                )
                return

    raise LookupError(f"Cannot find argument {argument}")

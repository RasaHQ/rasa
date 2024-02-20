import argparse
import io
from typing import Text, Tuple

import pytest

from rasa.utils.cli import remove_argument_from_parser


@pytest.mark.parametrize(
    "argument,lookup_argument",
    [
        (("--model",), "--model"),
        (("-m",), "-m"),
        (("-m", "--model"), "--model"),
        (("-m", "--model"), "-m"),
    ],
)
def test_remove_argument_from_parser(
    argument: Tuple[Text], lookup_argument: Text
) -> None:
    parser = argparse.ArgumentParser("hello")
    parser.add_argument(*argument)

    remove_argument_from_parser(parser, lookup_argument)

    buffer = io.StringIO()
    parser.print_help(buffer)
    buffer.getvalue() == """usage: hello [-h]

options:
-h, --help  show this help message and exit"""


def test_remove_argument_from_parser_unknown() -> None:
    parser = argparse.ArgumentParser("hello")
    parser.add_argument("--foo")

    with pytest.raises(LookupError):
        remove_argument_from_parser(parser, "--bar")

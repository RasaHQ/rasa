import argparse
from typing import Any, List, Text

import pytest

from rasa.cli import SubParsersAction, e2e_test, test
from rasa.cli.e2e_test import DEFAULT_E2E_OUTPUT_TESTS_PATH
from rasa.nlu.persistor import RemoteStorageType


@pytest.fixture
def rasa_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog="rasa",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Rasa command line interface. Rasa allows you to build "
        "your own conversational assistants 🤖. The 'rasa' command "
        "allows you to easily run most common commands like "
        "creating a new bot, training or evaluating models.",
    )


@pytest.fixture
def parent_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(add_help=False)


@pytest.fixture
def parent_parsers(
    parent_parser: argparse.ArgumentParser,
) -> List[argparse.ArgumentParser]:
    return [parent_parser]


@pytest.fixture
def subparsers(rasa_parser: argparse.ArgumentParser) -> SubParsersAction:
    return rasa_parser.add_subparsers(help="Rasa commands")


@pytest.fixture
def e2e_test_parser(
    subparsers: SubParsersAction, parent_parsers: List[argparse.ArgumentParser]
) -> None:
    test.add_subparser(subparsers, parents=parent_parsers)
    return e2e_test.add_subparser(subparsers, parents=parent_parsers)


@pytest.mark.parametrize(
    "args, expected_result",
    [
        (
            ["test", "e2e", "tests/data/test_e2e_stories/"],
            argparse.Namespace(
                fail_fast=False,
                remote_storage=None,
                **{"path-to-test-cases": "tests/data/test_e2e_stories/"},
            ),
        ),
        (
            ["test", "e2e", "tests/data/test_e2e_stories/", "--fail-fast"],
            argparse.Namespace(endpoints="endpoints.yml"),
        ),
        (
            [
                "test",
                "e2e",
                "tests/data/test_e2e_stories/",
                "--fail-fast",
            ],
            argparse.Namespace(
                fail_fast=True,
                remote_storage=None,
                **{"path-to-test-cases": "tests/data/test_e2e_stories/"},
            ),
        ),
        (
            [
                "test",
                "e2e",
                "tests/data/test_e2e_stories/",
                "--e2e-results",
            ],
            argparse.Namespace(
                fail_fast=False,
                remote_storage=None,
                e2e_results=DEFAULT_E2E_OUTPUT_TESTS_PATH,
                **{
                    "path-to-test-cases": "tests/data/test_e2e_stories/",
                },
            ),
        ),
        (
            [
                "test",
                "e2e",
                "tests/data/test_e2e_stories/",
                "--remote-storage",
                "gcs",
            ],
            argparse.Namespace(
                fail_fast=False,
                remote_storage=RemoteStorageType.GCS,
                **{"path-to-test-cases": "tests/data/test_e2e_stories/"},
            ),
        ),
        (
            [
                "test",
                "e2e",
                "tests/data/test_e2e_stories/",
                "--endpoints",
                "tests/data/test_endpoints/endpoints.yml",
            ],
            argparse.Namespace(
                fail_fast=False,
                remote_storage=None,
                endpoints="tests/data/test_endpoints/endpoints.yml",
                **{"path-to-test-cases": "tests/data/test_e2e_stories/"},
            ),
        ),
        (
            [
                "test",
                "e2e",
                "tests/data/test_e2e_stories/",
                "--model",
                "tests/data/test_models/test_moodbot.tar.gz",
            ],
            argparse.Namespace(
                fail_fast=False,
                remote_storage=None,
                model="tests/data/test_models/test_moodbot.tar.gz",
                **{"path-to-test-cases": "tests/data/test_e2e_stories/"},
            ),
        ),
    ],
)
def test_e2e_cli(
    args: List[Text],
    expected_result: argparse.Namespace,
    rasa_parser: argparse.ArgumentParser,
    e2e_test_parser: Any,
) -> None:
    arguments = rasa_parser.parse_args(args)

    for attribute in expected_result.__dict__:
        assert getattr(arguments, attribute) == getattr(expected_result, attribute)

import argparse
import filecmp
import os
from pathlib import Path
from unittest.mock import Mock
import pytest
from collections import namedtuple
from typing import Callable, Text

from _pytest.monkeypatch import MonkeyPatch
from _pytest.pytester import RunResult
from rasa.cli import data
from rasa.shared.core.domain import Domain
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.validator import Validator
import rasa.shared.utils.io


def test_data_split_nlu(run_in_simple_project: Callable[..., RunResult]):
    responses_yml = (
        "responses:\n"
        "  chitchat/ask_name:\n"
        "  - text: my name is Sara, Rasa's documentation bot!\n"
        "  chitchat/ask_weather:\n"
        "  - text: the weather is great!\n"
    )

    with open("data/responses.yml", "w") as f:
        f.write(responses_yml)

    run_in_simple_project(
        "data",
        "split",
        "nlu",
        "-u",
        "data/nlu.yml",
        "--training-fraction",
        "0.75",
        "--random-seed",
        "12345",
    )

    folder = Path("train_test_split")
    assert folder.exists()

    nlu_files = [folder / "test_data.yml", folder / "training_data.yml"]
    nlg_files = [folder / "nlg_test_data.yml", folder / "nlg_training_data.yml"]
    for yml_file in nlu_files:
        assert yml_file.exists(), f"{yml_file} file does not exist"
        nlu_data = rasa.shared.utils.io.read_yaml_file(yml_file)
        assert "version" in nlu_data
        assert nlu_data.get("nlu")

    for yml_file in nlg_files:
        assert yml_file.exists(), f"{yml_file} file does not exist"


def test_data_convert_nlu(run_in_simple_project: Callable[..., RunResult]):
    run_in_simple_project(
        "data",
        "convert",
        "nlu",
        "--data",
        "data/nlu.yml",
        "--out",
        "out_nlu_data.json",
        "-f",
        "json",
    )

    assert os.path.exists("out_nlu_data.json")


def test_data_split_help(run: Callable[..., RunResult]):
    output = run("data", "split", "nlu", "--help")

    help_text = """usage: rasa data split nlu [-h] [-v] [-vv] [--quiet] [-u NLU]
                           [--training-fraction TRAINING_FRACTION]
                           [--random-seed RANDOM_SEED] [--out OUT]"""

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = set(output.outlines)
    for line in lines:
        assert line in printed_help


def test_data_convert_help(run: Callable[..., RunResult]):
    output = run("data", "convert", "nlu", "--help")

    help_text = """usage: rasa data convert nlu [-h] [-v] [-vv] [--quiet] -f {json,md,yaml}
                             --data DATA --out OUT [-l LANGUAGE]"""

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = set(output.outlines)
    for line in lines:
        assert line in printed_help


def test_data_validate_help(run: Callable[..., RunResult]):
    output = run("data", "validate", "--help")

    help_text = """usage: rasa data validate [-h] [-v] [-vv] [--quiet]
                          [--max-history MAX_HISTORY] [--fail-on-warnings]"""

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = set(output.outlines)
    for line in lines:
        assert line in printed_help


def _text_is_part_of_output_error(text: Text, output: RunResult) -> bool:
    found_info_string = False
    for line in output.errlines:
        if text in line:
            found_info_string = True
    return found_info_string


def test_data_validate_stories_with_max_history_zero(monkeypatch: MonkeyPatch):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="Rasa commands")
    data.add_subparser(subparsers, parents=[])

    args = parser.parse_args(["data", "validate", "stories", "--max-history", 0])

    async def mock_from_importer(importer: TrainingDataImporter) -> Validator:
        return Mock()

    monkeypatch.setattr("rasa.validator.Validator.from_importer", mock_from_importer)

    with pytest.raises(argparse.ArgumentTypeError):
        data.validate_files(args)


def test_validate_files_exit_early():
    with pytest.raises(SystemExit) as pytest_e:
        args = {
            "domain": "data/test_domains/duplicate_intents.yml",
            "data": None,
            "max_history": None,
        }
        data.validate_files(namedtuple("Args", args.keys())(*args.values()))

    assert pytest_e.type == SystemExit
    assert pytest_e.value.code == 1


def test_rasa_data_convert_nlu_to_yaml(
    run_in_simple_project: Callable[..., RunResult],
):
    converted_data_folder = "converted_data"
    os.mkdir(converted_data_folder)

    converted_single_file_folder = "converted_single_file"
    os.mkdir(converted_single_file_folder)

    simple_nlu_md = """
    ## intent:greet
    - hey
    - hello
    """

    os.mkdir("data/nlu")
    with open("data/nlu/nlu.md", "w") as f:
        f.write(simple_nlu_md)

    run_in_simple_project(
        "data",
        "convert",
        "nlu",
        "-f",
        "yaml",
        "--data",
        "data",
        "--out",
        converted_data_folder,
    )

    run_in_simple_project(
        "data",
        "convert",
        "nlu",
        "-f",
        "yaml",
        "--data",
        "data/nlu/nlu.md",
        "--out",
        converted_single_file_folder,
    )

    assert filecmp.cmp(
        Path(converted_data_folder) / "nlu_converted.yml",
        Path(converted_single_file_folder) / "nlu_converted.yml",
    )


def test_rasa_data_convert_stories_to_yaml(
    run_in_simple_project: Callable[..., RunResult],
):
    converted_data_folder = "converted_data"
    os.mkdir(converted_data_folder)

    converted_single_file_folder = "converted_single_file"
    os.mkdir(converted_single_file_folder)

    simple_story_md = """
    ## happy path
    * greet OR goodbye
        - utter_greet
        - form{"name": null}
    """

    with open("data/stories.md", "w") as f:
        f.write(simple_story_md)

    run_in_simple_project(
        "data",
        "convert",
        "core",
        "-f",
        "yaml",
        "--data",
        "data",
        "--out",
        converted_data_folder,
    )

    run_in_simple_project(
        "data",
        "convert",
        "core",
        "-f",
        "yaml",
        "--data",
        "data/stories.md",
        "--out",
        converted_single_file_folder,
    )

    assert filecmp.cmp(
        Path(converted_data_folder) / "stories_converted.yml",
        Path(converted_single_file_folder) / "stories_converted.yml",
    )


def test_rasa_data_convert_nlg_to_yaml(run_in_simple_project: Callable[..., RunResult]):
    converted_data_folder = "converted_data"
    os.mkdir(converted_data_folder)

    converted_single_file_folder = "converted_single_file"
    os.mkdir(converted_single_file_folder)

    simple_nlg_md = (
        "## ask name\n"
        "* chitchat/ask_name\n"
        "- my name is Sara, Rasa's documentation bot!\n"
    )

    with open("data/responses.md", "w") as f:
        f.write(simple_nlg_md)

    run_in_simple_project(
        "data",
        "convert",
        "nlg",
        "-f",
        "yaml",
        "--data",
        "data",
        "--out",
        converted_data_folder,
    )

    run_in_simple_project(
        "data",
        "convert",
        "nlg",
        "-f",
        "yaml",
        "--data",
        "data/responses.md",
        "--out",
        converted_single_file_folder,
    )

    assert filecmp.cmp(
        Path(converted_data_folder) / "responses_converted.yml",
        Path(converted_single_file_folder) / "responses_converted.yml",
    )


def test_rasa_data_convert_nlu_lookup_tables_to_yaml(
    run_in_simple_project: Callable[..., RunResult]
):
    converted_data_folder = "converted_data"
    os.mkdir(converted_data_folder)

    simple_nlu_md = """
    ## lookup:products.txt
      data/nlu/lookups/products.txt
    """

    os.mkdir("data/nlu")
    with open("data/nlu/nlu.md", "w") as f:
        f.write(simple_nlu_md)

    simple_lookup_table_txt = "core\n nlu\n x\n"
    os.mkdir("data/nlu/lookups")
    with open("data/nlu/lookups/products.txt", "w") as f:
        f.write(simple_lookup_table_txt)

    run_in_simple_project(
        "data",
        "convert",
        "nlu",
        "-f",
        "yaml",
        "--data",
        "data",
        "--out",
        converted_data_folder,
    )

    assert len(os.listdir(converted_data_folder)) == 1


def test_convert_config(
    run: Callable[..., RunResult], tmp_path: Path, default_domain_path: Text
):
    deprecated_config = {
        "policies": [{"name": "MappingPolicy"}],
        "pipeline": {"name": "WhitespaceTokenizer"},
    }
    config_file = tmp_path / "config.yml"
    rasa.shared.utils.io.write_yaml(deprecated_config, config_file)

    domain = Domain.empty()
    domain_file = tmp_path / "domain.yml"
    domain.persist(domain_file)

    result = run(
        "data",
        "convert",
        "config",
        "--config",
        str(config_file),
        "--domain",
        str(domain_file),
    )

    assert result.ret == 0
    # TODO: Validate the actual migration 😀


def test_convert_config_with_invalid_config(run: Callable[..., RunResult]):
    result = run("data", "convert", "config", "--config", "invalid path")

    assert result.ret == 1


def test_convert_config_with_missing_nlu_pipeline_config(
    run_in_simple_project: Callable[..., RunResult], tmp_path: Path
):
    deprecated_config = {"policies": [{"name": "FallbackPolicy"}]}
    config_file = tmp_path / "config.yml"
    rasa.shared.utils.io.write_yaml(deprecated_config, config_file)

    result = run_in_simple_project(
        "data", "convert", "config", "--config", str(config_file)
    )

    assert result.ret == 1


def test_convert_config_with_missing_nlu_pipeline_config_if_no_fallbacks(
    run_in_simple_project: Callable[..., RunResult], tmp_path: Path
):
    deprecated_config = {"policies": [{"name": "MappingPolicy"}]}
    config_file = tmp_path / "config.yml"
    rasa.shared.utils.io.write_yaml(deprecated_config, config_file)

    result = run_in_simple_project(
        "data", "convert", "config", "--config", str(config_file)
    )

    assert result.ret == 0


def test_convert_config_with_form_policy_present(
    run_in_simple_project: Callable[..., RunResult], tmp_path: Path
):
    deprecated_config = {
        "policies": [{"name": "MappingPolicy"}, {"name": "FormPolicy"}],
        "pipeline": {"name": "WhitespaceTokenizer"},
    }
    config_file = tmp_path / "config.yml"
    rasa.shared.utils.io.write_yaml(deprecated_config, config_file)

    result = run_in_simple_project(
        "data", "convert", "config", "--config", str(config_file)
    )

    assert result.ret == 1


def test_convert_config_with_customized_deny_suggestion_intent(
    run_in_simple_project: Callable[..., RunResult], tmp_path: Path
):
    deprecated_config = {
        "policies": [
            {"name": "MappingPolicy"},
            {
                "name": "TwoStageFallbackPolicy",
                "deny_suggestion_intent_name": "something else",
            },
        ],
        "pipeline": {"name": "WhitespaceTokenizer"},
    }
    config_file = tmp_path / "config.yml"
    rasa.shared.utils.io.write_yaml(deprecated_config, config_file)

    result = run_in_simple_project(
        "data", "convert", "config", "--config", str(config_file)
    )

    assert result.ret == 1


def test_convert_config_with_invalid_domain_path(run: Callable[..., RunResult]):
    result = run("data", "convert", "config", "--domain", "invalid path")

    assert result.ret == 1


def test_convert_config_with_default_rules_directory(
    run: Callable[..., RunResult], tmp_path: Path
):
    result = run("data", "convert", "config", "--out", str(tmp_path))

    assert result.ret == 1

import argparse
import filecmp
import os
from pathlib import Path
from unittest.mock import Mock
import pytest
from collections import namedtuple
from typing import Callable, Text, Dict, List

from _pytest.monkeypatch import MonkeyPatch
from _pytest.pytester import RunResult, Testdir
from rasa.cli import data
from rasa.core.constants import (
    DEFAULT_NLU_FALLBACK_AMBIGUITY_THRESHOLD,
    DEFAULT_NLU_FALLBACK_THRESHOLD,
    DEFAULT_CORE_FALLBACK_THRESHOLD,
)
from rasa.shared.constants import LATEST_TRAINING_DATA_FORMAT_VERSION, DEFAULT_DATA_PATH
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

    help_text = """usage: rasa data convert nlu [-h] [-v] [-vv] [--quiet] [-f {json,md,yaml}]
                             --data DATA [--out OUT] [-l LANGUAGE]"""

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

    args = parser.parse_args(
        [
            "data",
            "validate",
            "stories",
            "--data",
            "examples/moodbot/data",
            "--max-history",
            0,
        ]
    )

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


def test_rasa_data_convert_test_stories_to_yaml(
    run_in_simple_project: Callable[..., RunResult]
):
    converted_data_folder = "converted_data"
    os.mkdir(converted_data_folder)

    simple_test_story_md = """
    ## happy path
    * greet: hi how are you?
        - utter_greet
    """

    with open("tests/test_stories.md", "w") as f:
        f.write(simple_test_story_md)

    run_in_simple_project(
        "data",
        "convert",
        "core",
        "-f",
        "yaml",
        "--data",
        "tests",
        "--out",
        converted_data_folder,
    )

    assert (Path(converted_data_folder) / "test_stories_converted.yml").exists()


def test_rasa_data_convert_nlg_to_yaml(
    run_in_simple_project: Callable[..., RunResult], run: Callable[..., RunResult]
):
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


def test_rasa_data_convert_responses(
    run_in_simple_project: Callable[..., RunResult], run: Callable[..., RunResult]
):
    converted_data_folder = "converted_data"
    os.mkdir(converted_data_folder)
    expected_data_folder = "expected_data"
    os.mkdir(expected_data_folder)

    domain = (
        "version: '2.0'\n"
        "session_config:\n"
        "  session_expiration_time: 60\n"
        "  carry_over_slots_to_new_session: true\n"
        "actions:\n"
        "- respond_chitchat\n"
        "- utter_greet\n"
        "- utter_cheer_up"
    )

    expected_domain = (
        "version: '2.0'\n"
        "session_config:\n"
        "  session_expiration_time: 60\n"
        "  carry_over_slots_to_new_session: true\n"
        "actions:\n"
        "- utter_chitchat\n"
        "- utter_greet\n"
        "- utter_cheer_up\n"
    )

    with open(f"{expected_data_folder}/domain.yml", "w") as f:
        f.write(expected_domain)

    with open("domain.yml", "w") as f:
        f.write(domain)

    stories = (
        "stories:\n"
        "- story: sad path\n"
        "  steps:\n"
        "  - intent: greet\n"
        "  - action: utter_greet\n"
        "  - intent: mood_unhappy\n"
        "- story: chitchat\n"
        "  steps:\n"
        "  - intent: chitchat\n"
        "  - action: respond_chitchat\n"
    )

    expected_stories = (
        'version: "2.0"\n'
        "stories:\n"
        "- story: sad path\n"
        "  steps:\n"
        "  - intent: greet\n"
        "  - action: utter_greet\n"
        "  - intent: mood_unhappy\n"
        "- story: chitchat\n"
        "  steps:\n"
        "  - intent: chitchat\n"
        "  - action: utter_chitchat\n"
    )

    with open(f"{expected_data_folder}/stories.yml", "w") as f:
        f.write(expected_stories)

    with open("data/stories.yml", "w") as f:
        f.write(stories)

    run_in_simple_project(
        "data",
        "convert",
        "responses",
        "--data",
        "data",
        "--out",
        converted_data_folder,
    )

    assert filecmp.cmp(
        Path(converted_data_folder) / "domain_converted.yml",
        Path(expected_data_folder) / "domain.yml",
    )

    assert filecmp.cmp(
        Path(converted_data_folder) / "stories_converted.yml",
        Path(expected_data_folder) / "stories.yml",
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
        "policies": [{"name": "MappingPolicy"}, {"name": "FallbackPolicy"}],
        "pipeline": [{"name": "WhitespaceTokenizer"}],
    }
    config_file = tmp_path / "config.yml"
    rasa.shared.utils.io.write_yaml(deprecated_config, config_file)

    domain = Domain.from_dict(
        {
            "intents": [{"greet": {"triggers": "action_greet"}}, "leave"],
            "actions": ["action_greet"],
        }
    )
    domain_file = tmp_path / "domain.yml"
    domain.persist(domain_file)

    rules_file = tmp_path / "rules.yml"

    result = run(
        "data",
        "convert",
        "config",
        "--config",
        str(config_file),
        "--domain",
        str(domain_file),
        "--out",
        str(rules_file),
    )

    assert result.ret == 0
    new_config = rasa.shared.utils.io.read_yaml_file(config_file)
    new_domain = rasa.shared.utils.io.read_yaml_file(domain_file)
    new_rules = rasa.shared.utils.io.read_yaml_file(rules_file)

    assert new_config == {
        "policies": [
            {
                "name": "RulePolicy",
                "core_fallback_action_name": "action_default_fallback",
                "core_fallback_threshold": DEFAULT_CORE_FALLBACK_THRESHOLD,
            }
        ],
        "pipeline": [
            {"name": "WhitespaceTokenizer"},
            {
                "name": "FallbackClassifier",
                "ambiguity_threshold": DEFAULT_NLU_FALLBACK_AMBIGUITY_THRESHOLD,
                "threshold": DEFAULT_NLU_FALLBACK_THRESHOLD,
            },
        ],
    }
    assert new_domain["intents"] == ["greet", "leave"]
    assert new_rules == {
        "rules": [
            {
                "rule": "Rule to map `greet` intent to `action_greet` "
                "(automatic conversion)",
                "steps": [{"intent": "greet"}, {"action": "action_greet"}],
            },
            {
                "rule": "Rule to handle messages with low NLU confidence "
                "(automated conversion from 'FallbackPolicy')",
                "steps": [
                    {"intent": "nlu_fallback"},
                    {"action": "action_default_fallback"},
                ],
            },
        ],
        "version": LATEST_TRAINING_DATA_FORMAT_VERSION,
    }

    domain_backup = tmp_path / "domain.yml.bak"
    assert domain_backup.exists()

    config_backup = tmp_path / "config.yml.bak"
    assert config_backup.exists()


def test_convert_config_if_nothing_to_migrate(
    run_in_simple_project: Callable[..., RunResult], tmp_path: Path
):
    result = run_in_simple_project("data", "convert", "config")

    assert result.ret == 1

    output = "\n".join(result.outlines)
    assert "No policies were found which need migration" in output


def test_convert_config_with_output_file_containing_data(
    run_in_simple_project: Callable[..., RunResult], tmp_path: Path, testdir: Testdir
):
    deprecated_config = {
        "policies": [{"name": "FallbackPolicy"}],
        "pipeline": [{"name": "WhitespaceTokenizer"}],
    }
    config_file = tmp_path / "config.yml"
    rasa.shared.utils.io.write_yaml(deprecated_config, config_file)

    output_file = testdir.tmpdir / DEFAULT_DATA_PATH / "rules.yml"
    # the default project already contains rules training data
    assert output_file.exists()
    existing_rules = rasa.shared.utils.io.read_yaml_file(output_file)["rules"]
    assert existing_rules

    result = run_in_simple_project(
        "data", "convert", "config", "--config", str(config_file)
    )

    assert result.ret == 0

    new_rules = rasa.shared.utils.io.read_yaml_file(output_file)["rules"]
    expected_new_rule = {
        "rule": "Rule to handle messages with low NLU confidence "
        "(automated conversion from 'FallbackPolicy')",
        "steps": [{"intent": "nlu_fallback"}, {"action": "action_default_fallback"}],
    }

    assert expected_new_rule in new_rules
    assert all(existing in new_rules for existing in existing_rules)

    backup_file = testdir.tmpdir / DEFAULT_DATA_PATH / "rules.yml.bak"
    assert backup_file.exists()


def test_convert_config_with_invalid_config_path(run: Callable[..., RunResult]):
    result = run("data", "convert", "config", "--config", "invalid path")

    assert result.ret == 1

    output = "\n".join(result.outlines)
    assert "Please provide a valid path" in output


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

    output = "\n".join(result.outlines)
    assert "The model configuration has to include an NLU pipeline" in output


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


@pytest.mark.parametrize(
    "policy_config, expected_error_message",
    [
        (
            [{"name": "FallbackPolicy"}, {"name": "TwoStageFallbackPolicy"}],
            "two configured policies for handling fallbacks",
        )
    ],
)
def test_convert_config_with_invalid_policy_config(
    run_in_simple_project: Callable[..., RunResult],
    tmp_path: Path,
    policy_config: List[Dict],
    expected_error_message: Text,
):
    deprecated_config = {
        "policies": policy_config,
        "pipeline": [{"name": "WhitespaceTokenizer"}],
    }
    config_file = tmp_path / "config.yml"
    rasa.shared.utils.io.write_yaml(deprecated_config, config_file)

    result = run_in_simple_project(
        "data", "convert", "config", "--config", str(config_file)
    )

    assert result.ret == 1

    output = "\n".join(result.outlines)
    assert expected_error_message in output


def test_warning_for_form_migration(
    run_in_simple_project: Callable[..., RunResult], tmp_path: Path
):
    deprecated_config = {
        "policies": [{"name": "FallbackPolicy"}, {"name": "FormPolicy"}],
        "pipeline": [{"name": "WhitespaceTokenizer"}],
    }
    config_file = tmp_path / "config.yml"
    rasa.shared.utils.io.write_yaml(deprecated_config, config_file)

    result = run_in_simple_project(
        "data", "convert", "config", "--config", str(config_file)
    )

    assert result.ret == 0

    output = "\n".join(result.outlines)
    assert "you have to migrate the 'FormPolicy' manually" in output


@pytest.mark.parametrize(
    "two_stage_config, expected_error_message",
    [
        (
            {"deny_suggestion_intent_name": "something else"},
            "has to use the intent 'out_of_scope'",
        ),
        (
            {"fallback_nlu_action_name": "something else"},
            "has to use the action 'action_default_fallback",
        ),
    ],
)
def test_convert_config_with_two_stage_fallback_policy(
    run_in_simple_project: Callable[..., RunResult],
    tmp_path: Path,
    two_stage_config: Dict,
    expected_error_message: Text,
):
    deprecated_config = {
        "policies": [
            {"name": "MappingPolicy"},
            {"name": "TwoStageFallbackPolicy", **two_stage_config},
        ],
        "pipeline": [{"name": "WhitespaceTokenizer"}],
    }
    config_file = tmp_path / "config.yml"
    rasa.shared.utils.io.write_yaml(deprecated_config, config_file)

    result = run_in_simple_project(
        "data", "convert", "config", "--config", str(config_file)
    )

    assert result.ret == 1

    output = "\n".join(result.outlines)
    assert expected_error_message in output


def test_convert_config_with_invalid_domain_path(run: Callable[..., RunResult]):
    result = run("data", "convert", "config", "--domain", "invalid path")

    assert result.ret == 1

    output = "\n".join(result.outlines)
    assert "path to a valid model configuration" in output


def test_convert_config_with_default_rules_directory(
    run_in_simple_project: Callable[..., RunResult], tmp_path: Path
):
    deprecated_config = {
        "policies": [{"name": "FallbackPolicy"}],
        "pipeline": [{"name": "WhitespaceTokenizer"}],
    }
    config_file = tmp_path / "config.yml"
    rasa.shared.utils.io.write_yaml(deprecated_config, config_file)

    result = run_in_simple_project(
        "data",
        "convert",
        "config",
        "--config",
        str(config_file),
        "--out",
        str(tmp_path),
    )

    assert result.ret == 1

    output = "\n".join(result.outlines)
    assert "needs to be the path to a file" in output

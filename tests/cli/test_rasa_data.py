import os
from pathlib import Path
from typing import Callable

from _pytest.fixtures import FixtureRequest
from _pytest.pytester import RunResult
from rasa.shared.nlu.training_data.formats import RasaYAMLReader
import rasa.shared.utils.io

from tests.cli.conftest import RASA_EXE


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


def test_data_convert_nlu_json(run_in_simple_project: Callable[..., RunResult]):

    result = run_in_simple_project(
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

    assert "NLU data in Rasa JSON format is deprecated" in str(result.stderr)
    assert os.path.exists("out_nlu_data.json")


def test_data_convert_nlu_yml(
    run: Callable[..., RunResult], tmp_path: Path, request: FixtureRequest
):

    target_file = tmp_path / "out.yml"

    # The request rootdir is required as the `testdir` fixture in `run` changes the
    # working directory
    test_data_dir = Path(request.config.rootdir, "data", "examples", "rasa")
    source_file = (test_data_dir / "demo-rasa.json").absolute()
    result = run(
        "data",
        "convert",
        "nlu",
        "--data",
        str(source_file),
        "--out",
        str(target_file),
        "-f",
        "yaml",
    )

    assert result.ret == 0
    assert target_file.exists()

    actual_data = RasaYAMLReader().read(target_file)
    expected = RasaYAMLReader().read(test_data_dir / "demo-rasa.yml")

    assert len(actual_data.training_examples) == len(expected.training_examples)
    assert len(actual_data.entity_synonyms) == len(expected.entity_synonyms)
    assert len(actual_data.regex_features) == len(expected.regex_features)
    assert len(actual_data.lookup_tables) == len(expected.lookup_tables)
    assert actual_data.entities == expected.entities


def test_data_split_help(run: Callable[..., RunResult]):
    output = run("data", "split", "nlu", "--help")

    help_text = f"""usage: {RASA_EXE} data split nlu [-h] [-v] [-vv] [--quiet]\n
                           [--logging-config-file LOGGING_CONFIG_FILE]\n
                           [-u NLU] [--training-fraction TRAINING_FRACTION]\n
                           [--random-seed RANDOM_SEED] [--out OUT]"""

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = set(output.outlines)
    for line in lines:
        assert line in printed_help


def test_data_convert_help(run: Callable[..., RunResult]):
    output = run("data", "convert", "nlu", "--help")

    help_text = f"""usage: {RASA_EXE} data convert nlu [-h] [-v] [-vv] [--quiet]\n
                           [--logging-config-file LOGGING_CONFIG_FILE]\n
                           [-f {"{json,yaml}"}] [--data DATA [DATA ...]]"""

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = {line.strip() for line in output.outlines}
    for line in lines:
        assert line.strip() in printed_help


def test_data_validate_help(run: Callable[..., RunResult]):
    output = run("data", "validate", "--help")

    help_text = f"""usage: {RASA_EXE} data validate [-h] [-v] [-vv] [--quiet]
                          [--logging-config-file LOGGING_CONFIG_FILE]
                          [--max-history MAX_HISTORY] [-c CONFIG]
                          [--fail-on-warnings] [-d DOMAIN]
                          [--data DATA [DATA ...]]
                          {{stories}} ..."""

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = {line.strip() for line in output.outlines}
    for line in lines:
        assert line.strip() in printed_help


def test_data_migrate_help(run: Callable[..., RunResult]):
    output = run("data", "migrate", "--help")
    printed_help = set(output.outlines)

    help_text = f"""usage: {RASA_EXE} data migrate [-h] [-v] [-vv] [--quiet]
                          [--logging-config-file LOGGING_CONFIG_FILE]
                          [-d DOMAIN] [--out OUT]"""
    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = {line.strip() for line in output.outlines}
    for line in lines:
        assert line.strip() in printed_help


def test_data_validate_default_options(
    run_in_simple_project: Callable[..., RunResult], request: FixtureRequest
):
    test_data_dir = Path(request.config.rootdir, "data", "test_moodbot", "data")
    test_domain = Path(request.config.rootdir, "data", "test_moodbot", "domain.yml")
    test_config = Path(request.config.rootdir, "data", "test_moodbot", "config.yml")

    result = run_in_simple_project(
        "data",
        "validate",
        "--data",
        str(test_data_dir),
        "--domain",
        str(test_domain),
        "-c",
        str(test_config),
    )

    # should not raise any errors
    assert result.ret == 0


def test_data_validate_not_used_warning(
    run_in_simple_project: Callable[..., RunResult], request: FixtureRequest
):
    test_data_dir = Path(request.config.rootdir, "data", "test_validation", "data")
    test_domain = Path(request.config.rootdir, "data", "test_validation", "domain.yml")
    test_config = Path(request.config.rootdir, "data", "test_moodbot", "config.yml")

    result = run_in_simple_project(
        "data",
        "validate",
        "--data",
        str(test_data_dir),
        "--domain",
        str(test_domain),
        "-c",
        str(test_config),
    )

    for warning in [
        "The intent 'goodbye' is not used in any story or rule.",
        "The utterance 'utter_chatter' is not used in any story or rule.",
    ]:
        assert warning in str(result.stderr)


def test_data_validate_failed_to_load_domain(
    run_in_simple_project_with_no_domain: Callable[..., RunResult]
):
    result = run_in_simple_project_with_no_domain(
        "data",
        "validate",
        "--domain",
        "not-existing-domain.yml",
    )

    assert "The path 'not-existing-domain.yml' does not exist." in str(result.outlines)
    assert result.ret == 1


def test_data_split_stories(run_in_simple_project: Callable[..., RunResult]):
    stories_yml = (
        "stories:\n"
        "- story: story 1\n"
        "  steps:\n"
        "  - intent: intent_a\n"
        "  - action: utter_a\n"
        "- story: story 2\n"
        "  steps:\n"
        "  - intent: intent_a\n"
        "  - action: utter_a\n"
    )

    Path("data/stories.yml").write_text(stories_yml)
    run_in_simple_project(
        "data", "split", "stories", "--random-seed", "123", "--training-fraction", "0.5"
    )

    folder = Path("train_test_split")
    assert folder.exists()

    train_file = folder / "train_stories.yml"
    assert train_file.exists()
    test_file = folder / "test_stories.yml"
    assert test_file.exists()

    train_data = rasa.shared.utils.io.read_yaml_file(train_file)
    assert len(train_data.get("stories", [])) == 1
    assert train_data["stories"][0].get("story") == "story 1"
    test_data = rasa.shared.utils.io.read_yaml_file(test_file)
    assert len(test_data.get("stories", [])) == 1
    assert test_data["stories"][0].get("story") == "story 2"

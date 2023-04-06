import os
import shutil
from pathlib import Path
from shutil import copyfile

from pytest import Testdir, Pytester, ExitCode
from _pytest.pytester import RunResult

from rasa.core.constants import (
    CONFUSION_MATRIX_STORIES_FILE,
    STORIES_WITH_WARNINGS_FILE,
)
from rasa.constants import RESULTS_FILE
from rasa.shared.constants import DEFAULT_RESULTS_PATH
from rasa.shared.utils.io import list_files, write_yaml, write_text_file
from typing import Callable

from tests.cli.conftest import RASA_EXE


def test_test_core(run_in_simple_project: Callable[..., RunResult]):
    run_in_simple_project("test", "core", "--stories", "data")

    assert os.path.exists("results")


def test_test_core_no_plot(run_in_simple_project: Callable[..., RunResult]):
    run_in_simple_project("test", "core", "--no-plot")

    assert not os.path.exists(f"results/{CONFUSION_MATRIX_STORIES_FILE}")


def test_test_core_warnings(run_in_simple_project_with_model: Callable[..., RunResult]):
    write_yaml(
        {
            "language": "en",
            "pipeline": [],
            "policies": [
                {"name": "MemoizationPolicy", "max_history": 3},
                {"name": "UnexpecTEDIntentPolicy", "max_history": 5, "epochs": 1},
                {
                    "name": "TEDPolicy",
                    "max_history": 5,
                    "epochs": 1,
                    "constrain_similarities": True,
                },
                {"name": "RulePolicy"},
            ],
        },
        "config.yml",
    )

    simple_test_story_yaml = """
version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
stories:
- story: unlikely path
  steps:
  - user: |
      very terrible
    intent: mood_unhappy
  - action: utter_cheer_up
  - action: utter_did_that_help
  - intent: affirm
  - action: utter_happy
"""
    with open("tests/test_stories.yml", "w") as f:
        f.write(simple_test_story_yaml)

    run_in_simple_project_with_model("test", "core", "--no-warnings")
    assert not os.path.exists(f"results/{STORIES_WITH_WARNINGS_FILE}")

    run_in_simple_project_with_model("test", "core")
    assert os.path.exists(f"results/{STORIES_WITH_WARNINGS_FILE}")


def test_test_core_with_no_model(run_in_simple_project: Callable[..., RunResult]):
    assert not os.path.exists("models")

    output = run_in_simple_project("test", "core")

    assert (
        "No model provided. Please make sure to specify the model to test with"
        in output.outlines[7]
    )


def test_test(run_in_simple_project_with_model: Callable[..., RunResult]):
    write_yaml(
        {
            "pipeline": "KeywordIntentClassifier",
            "policies": [{"name": "MemoizationPolicy"}],
        },
        "config2.yml",
    )

    run_in_simple_project_with_model("test")

    assert os.path.exists("results")
    assert os.path.exists("results/intent_histogram.png")
    assert os.path.exists("results/intent_confusion_matrix.png")


def test_test_with_no_user_utterance(
    run_in_simple_project_with_model: Callable[..., RunResult]
):
    write_yaml(
        {"pipeline": "KeywordIntentClassifier", "policies": [{"name": "TEDPolicy"}]},
        "config.yml",
    )

    simple_test_story_yaml = """
stories:
- story: happy path 1
  steps:
  - intent: greet
  - action: utter_greet
  - intent: mood_great
  - action: utter_happy
"""
    with open("tests/test_story_no_utterance.yaml", "w") as f:
        f.write(simple_test_story_yaml)

    run_in_simple_project_with_model("test", "--fail-on-prediction-errors")
    assert os.path.exists("results")
    assert not os.path.exists("results/failed_test_stories.yml")


def test_test_no_plot(run_in_simple_project: Callable[..., RunResult]):
    run_in_simple_project("test", "--no-plot")

    assert not os.path.exists("results/intent_histogram.png")
    assert not os.path.exists("results/intent_confusion_matrix.png")
    assert not os.path.exists("results/story_confmat.pdf")


def test_test_nlu(run_in_simple_project_with_model: Callable[..., RunResult]):
    run_in_simple_project_with_model("test", "nlu", "--nlu", "data", "--successes")

    assert os.path.exists("results/intent_histogram.png")
    assert os.path.exists("results/intent_confusion_matrix.png")
    assert os.path.exists("results/intent_successes.json")


def test_test_nlu_no_plot(run_in_simple_project: Callable[..., RunResult]):
    run_in_simple_project("test", "nlu", "--no-plot")

    assert not os.path.exists("results/intent_histogram.png")
    assert not os.path.exists("results/intent_confusion_matrix.png")


def test_test_nlu_cross_validation(run_in_simple_project: Callable[..., RunResult]):
    run_in_simple_project(
        "test", "nlu", "--cross-validation", "-c", "config.yml", "-f", "2", "-r", "1"
    )

    assert os.path.exists("results/intent_histogram.png")
    assert os.path.exists("results/intent_confusion_matrix.png")


def test_test_nlu_cross_validation_with_autoconfig(
    testdir: Testdir, moodbot_nlu_data_path: Path
):
    os.environ["LOG_LEVEL"] = "ERROR"
    config_path = str(testdir.tmpdir / "config.yml")
    nlu_path = str(testdir.tmpdir / "nlu.yml")
    shutil.copy(str(moodbot_nlu_data_path), nlu_path)
    write_yaml(
        {
            "assistant_id": "placeholder_default",
            "language": "en",
            "pipeline": None,
            "policies": None,
        },
        config_path,
    )
    args = [
        shutil.which(RASA_EXE),
        "test",
        "nlu",
        "--cross-validation",
        "-c",
        "config.yml",
        "--nlu",
        "nlu.yml",
    ]

    # we don't wanna run the cross validation for real, just want to see that it does
    # not crash
    try:
        run_result = testdir.run(*args, timeout=8.0)
        # we'll only get here if the run fails due to an exception
        assert run_result.ret != ExitCode.TESTS_FAILED
    except Pytester.TimeoutExpired:
        pass


def test_test_nlu_comparison(run_in_simple_project: Callable[..., RunResult]):
    write_yaml({"pipeline": "KeywordIntentClassifier"}, "config.yml")
    write_yaml({"pipeline": "KeywordIntentClassifier"}, "config2.yml")

    # TODO: Loading still needs fixing
    run_in_simple_project(
        "test",
        "nlu",
        "--config",
        "config.yml",
        "config2.yml",
        "--run",
        "2",
        "--percentages",
        "75",
        "25",
    )

    assert os.path.exists("results/run_1")
    assert os.path.exists("results/run_2")


def test_test_core_comparison(
    run_in_simple_project_with_model: Callable[..., RunResult]
):
    files = list_files("models")
    copyfile(files[0], "models/copy-model.tar.gz")

    run_in_simple_project_with_model(
        "test",
        "core",
        "-m",
        files[0],
        "models/copy-model.tar.gz",
        "--stories",
        "data/stories.yml",
    )

    assert os.path.exists(os.path.join(DEFAULT_RESULTS_PATH, RESULTS_FILE))


def test_test_core_comparison_after_train(
    run_in_simple_project: Callable[..., RunResult],
    trained_rasa_model: str,
    tmp_path: Path,
):
    path = Path(tmp_path / "comparison_models")
    path.mkdir()

    run_one = Path(path / "run_1")
    run_one.mkdir()
    shutil.copy(trained_rasa_model, run_one)

    run_two = Path(path / "run_2")
    run_two.mkdir()
    shutil.copy(trained_rasa_model, run_two)

    write_text_file("[1]", path / "num_stories.json")

    run_in_simple_project(
        "test",
        "core",
        "-m",
        str(path),
        "--stories",
        "data/stories",
        "--evaluate-model-directory",
    )

    assert os.path.exists(os.path.join(DEFAULT_RESULTS_PATH, RESULTS_FILE))
    assert os.path.exists(
        os.path.join(DEFAULT_RESULTS_PATH, "core_model_comparison_graph.pdf")
    )


def test_test_help(run: Callable[..., RunResult]):
    output = run("test", "--help")

    help_text = f"""usage: {RASA_EXE} test [-h] [-v] [-vv] [--quiet]
                 [--logging-config-file LOGGING_CONFIG_FILE] [-m MODEL]
                 [-s STORIES] [--max-stories MAX_STORIES]
                 [--endpoints ENDPOINTS] [--fail-on-prediction-errors]
                 [--url URL] [--evaluate-model-directory] [-u NLU]
                 [-c CONFIG [CONFIG ...]] [-d DOMAIN] [--cross-validation]
                 [-f FOLDS] [-r RUNS] [-p PERCENTAGES [PERCENTAGES ...]]
                 [--no-plot] [--successes] [--no-errors] [--no-warnings]
                 [--out OUT]
                 {{core,nlu}} ..."""

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = {line.strip() for line in output.outlines}
    for line in lines:
        assert line.strip() in printed_help


def test_test_nlu_help(run: Callable[..., RunResult]):
    output = run("test", "nlu", "--help")

    help_text = f"""usage: {RASA_EXE} test nlu [-h] [-v] [-vv] [--quiet]
                     [--logging-config-file LOGGING_CONFIG_FILE] [-m MODEL]
                     [-u NLU] [--out OUT] [-c CONFIG [CONFIG ...]] [-d DOMAIN]
                     [--cross-validation] [-f FOLDS] [-r RUNS]
                     [-p PERCENTAGES [PERCENTAGES ...]] [--no-plot]
                     [--successes] [--no-errors] [--no-warnings]"""

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = {line.strip() for line in output.outlines}
    for line in lines:
        assert line.strip() in printed_help


def test_test_core_help(run: Callable[..., RunResult]):
    output = run("test", "core", "--help")

    help_text = f"""usage: {RASA_EXE} test core [-h] [-v] [-vv] [--quiet]
                      [--logging-config-file LOGGING_CONFIG_FILE]
                      [-m MODEL [MODEL ...]] [-s STORIES]
                      [--max-stories MAX_STORIES] [--out OUT] [--e2e]
                      [--endpoints ENDPOINTS] [--fail-on-prediction-errors]
                      [--url URL] [--evaluate-model-directory] [--no-plot]
                      [--successes] [--no-errors] [--no-warnings]"""

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = {line.strip() for line in output.outlines}
    for line in lines:
        assert line.strip() in printed_help

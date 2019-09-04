import os
from shutil import copyfile
from rasa.constants import DEFAULT_RESULTS_PATH, RESULTS_FILE
from rasa.utils.io import list_files, write_yaml_file


def test_test_core(run_in_default_project):
    run_in_default_project("test", "core", "--stories", "data")

    assert os.path.exists("results")


def test_test(run_in_default_project):
    run_in_default_project("test")

    assert os.path.exists("results")
    assert os.path.exists("results/hist.png")
    assert os.path.exists("results/confmat.png")


def test_test_nlu(run_in_default_project):
    run_in_default_project("test", "nlu", "--nlu", "data", "--successes")

    assert os.path.exists("results/hist.png")
    assert os.path.exists("results/confmat.png")
    assert os.path.exists("results/intent_successes.json")


def test_test_nlu_cross_validation(run_in_default_project):
    run_in_default_project(
        "test", "nlu", "--cross-validation", "-c", "config.yml", "-f", "2"
    )

    assert os.path.exists("results/hist.png")
    assert os.path.exists("results/confmat.png")


def test_test_nlu_comparison(run_in_default_project):
    copyfile("config.yml", "nlu-config.yml")

    run_in_default_project(
        "test", "nlu", "-c", "config.yml", "nlu-config.yml", "--run", "2"
    )

    assert os.path.exists("results/run_1")
    assert os.path.exists("results/run_2")


def test_test_core_comparison(run_in_default_project):
    files = list_files("models")
    copyfile(files[0], "models/copy-model.tar.gz")

    run_in_default_project(
        "test",
        "core",
        "-m",
        files[0],
        "models/copy-model.tar.gz",
        "--stories",
        "data/stories.md",
    )

    assert os.path.exists(os.path.join(DEFAULT_RESULTS_PATH, RESULTS_FILE))


def test_test_core_comparison_after_train(run_in_default_project):
    write_yaml_file(
        {
            "language": "en",
            "pipeline": "supervised_embeddings",
            "policies": [{"name": "KerasPolicy"}],
        },
        "config_1.yml",
    )

    write_yaml_file(
        {
            "language": "en",
            "pipeline": "supervised_embeddings",
            "policies": [{"name": "MemoizationPolicy"}],
        },
        "config_2.yml",
    )
    run_in_default_project(
        "train",
        "core",
        "-c",
        "config_1.yml",
        "config_2.yml",
        "--stories",
        "data/stories.md",
        "--runs",
        "2",
        "--percentages",
        "25",
        "75",
        "--augmentation",
        "5",
        "--out",
        "comparison_models",
    )

    assert os.path.exists("comparison_models")
    assert os.path.exists("comparison_models/run_1")
    assert os.path.exists("comparison_models/run_2")

    run_in_default_project(
        "test",
        "core",
        "-m",
        "comparison_models",
        "--stories",
        "data/stories",
        "--evaluate-model-directory",
    )

    assert os.path.exists(os.path.join(DEFAULT_RESULTS_PATH, RESULTS_FILE))
    assert os.path.exists(
        os.path.join(DEFAULT_RESULTS_PATH, "core_model_comparison_graph.pdf")
    )


def test_test_help(run):
    output = run("test", "--help")

    help_text = """usage: rasa test [-h] [-v] [-vv] [--quiet] [-m MODEL] [-s STORIES]
                 [--max-stories MAX_STORIES] [--e2e] [--endpoints ENDPOINTS]
                 [--fail-on-prediction-errors] [--url URL]
                 [--evaluate-model-directory] [-u NLU] [--out OUT]
                 [--successes] [--no-errors] [--histogram HISTOGRAM]
                 [--confmat CONFMAT] [-c CONFIG [CONFIG ...]]
                 [--cross-validation] [-f FOLDS] [-r RUNS]
                 [-p PERCENTAGES [PERCENTAGES ...]]
                 {core,nlu} ..."""

    lines = help_text.split("\n")

    for i, line in enumerate(lines):
        assert output.outlines[i] == line


def test_test_nlu_help(run):
    output = run("test", "nlu", "--help")

    help_text = """usage: rasa test nlu [-h] [-v] [-vv] [--quiet] [-m MODEL] [-u NLU] [--out OUT]
                     [--successes] [--no-errors] [--histogram HISTOGRAM]
                     [--confmat CONFMAT] [-c CONFIG [CONFIG ...]]
                     [--cross-validation] [-f FOLDS] [-r RUNS]
                     [-p PERCENTAGES [PERCENTAGES ...]]"""

    lines = help_text.split("\n")

    for i, line in enumerate(lines):
        assert output.outlines[i] == line


def test_test_core_help(run):
    output = run("test", "core", "--help")

    help_text = """usage: rasa test core [-h] [-v] [-vv] [--quiet] [-m MODEL [MODEL ...]]
                      [-s STORIES] [--max-stories MAX_STORIES] [--out OUT]
                      [--e2e] [--endpoints ENDPOINTS]
                      [--fail-on-prediction-errors] [--url URL]
                      [--evaluate-model-directory]"""

    lines = help_text.split("\n")

    for i, line in enumerate(lines):
        assert output.outlines[i] == line

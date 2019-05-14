import os

from rasa.nlu.utils import list_files


def test_train(run_in_default_project):

    result, temp_dir = run_in_default_project(
        "train",
        "-c",
        "config.yml",
        "-d",
        "domain.yml",
        "--data",
        "data",
        "--out",
        "train_models",
        "--force",
    )

    assert os.path.exists(os.path.join(temp_dir, "train_models"))
    files = list_files(os.path.join(temp_dir, "train_models"))
    assert len(files) == 1


def test_train_core(run_in_default_project):

    result, temp_dir = run_in_default_project(
        "train",
        "core",
        "-c",
        "config.yml",
        "-d",
        "domain.yml",
        "--stories",
        "data",
        "--out",
        "train_models",
    )

    assert os.path.exists(os.path.join(temp_dir, "train_models"))
    files = list_files(os.path.join(temp_dir, "train_models"))
    assert len(files) == 1
    assert os.path.basename(files[0]).startswith("core-")


def test_train_nlu(run_in_default_project):

    result, temp_dir = run_in_default_project(
        "train",
        "nlu",
        "-c",
        "config.yml",
        "--nlu",
        "data/nlu.md",
        "--out",
        "train_models",
    )

    assert os.path.exists(os.path.join(temp_dir, "train_models"))
    files = list_files(os.path.join(temp_dir, "train_models"))
    assert len(files) == 1
    assert os.path.basename(files[0]).startswith("nlu-")


def test_train_help(run):
    help, _ = run("train", "--help")

    help_text = """usage: rasa train [-h] [-v] [-vv] [--quiet] [--data DATA [DATA ...]]
                  [-c CONFIG] [-d DOMAIN] [--out OUT]
                  [--augmentation AUGMENTATION] [--debug-plots]
                  [--dump-stories] [--force]
                  {core,nlu} ..."""

    lines = help_text.split("\n")

    for i, line in enumerate(lines):
        assert help.outlines[i] == line


def test_train_nlu_help(run):
    help, _ = run("train", "nlu", "--help")

    help_text = """usage: rasa train nlu [-h] [-v] [-vv] [--quiet] [-c CONFIG] [--out OUT]
                      [-u NLU]"""

    lines = help_text.split("\n")

    for i, line in enumerate(lines):
        assert help.outlines[i] == line


def test_train_core_help(run):
    help, _ = run("train", "core", "--help")

    help_text = """usage: rasa train core [-h] [-v] [-vv] [--quiet] [-s STORIES] [-d DOMAIN]
                       [-c CONFIG [CONFIG ...]] [--out OUT]
                       [--augmentation AUGMENTATION] [--debug-plots]
                       [--dump-stories] [--force]
                       [--percentages [PERCENTAGES [PERCENTAGES ...]]]
                       [--runs RUNS]"""

    lines = help_text.split("\n")

    for i, line in enumerate(lines):
        assert help.outlines[i] == line

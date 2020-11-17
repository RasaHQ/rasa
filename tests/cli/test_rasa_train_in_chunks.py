import os

from typing import Callable
from _pytest.pytester import RunResult

import rasa.shared.utils.io
from rasa import model
from rasa.nlu.model import Metadata
from rasa.shared.nlu.training_data import training_data


def test_train_in_chunks(run_in_simple_project: Callable[..., RunResult]):
    temp_dir = os.getcwd()

    run_in_simple_project(
        "train-in-chunks",
        "-c",
        "config.yml",
        "-d",
        "domain.yml",
        "--data",
        "data",
        "--out",
        "train_models",
        "--fixed-model-name",
        "test-model",
    )

    assert os.path.exists(os.path.join(temp_dir, "train_models"))
    files = rasa.shared.utils.io.list_files(os.path.join(temp_dir, "train_models"))
    assert len(files) == 1
    assert os.path.basename(files[0]) == "test-model.tar.gz"
    model_dir = model.get_model("train_models")
    assert model_dir is not None
    metadata = Metadata.load(os.path.join(model_dir, "nlu"))
    assert metadata.get("training_data") is None
    assert not os.path.exists(
        os.path.join(model_dir, "nlu", training_data.DEFAULT_TRAINING_DATA_OUTPUT_PATH)
    )


def test_train_in_chunks_skip_on_model_not_changed(
    run_in_simple_project_with_model: Callable[..., RunResult]
):
    temp_dir = os.getcwd()

    assert os.path.exists(os.path.join(temp_dir, "models"))
    files = rasa.shared.utils.io.list_files(os.path.join(temp_dir, "models"))
    assert len(files) == 1

    file_name = files[0]
    run_in_simple_project_with_model("train-in-chunks")

    assert os.path.exists(os.path.join(temp_dir, "models"))
    files = rasa.shared.utils.io.list_files(os.path.join(temp_dir, "models"))
    assert len(files) == 1
    assert file_name == files[0]


def test_train_in_chunks_force(
    run_in_simple_project_with_model: Callable[..., RunResult]
):
    temp_dir = os.getcwd()

    assert os.path.exists(os.path.join(temp_dir, "models"))
    files = rasa.shared.utils.io.list_files(os.path.join(temp_dir, "models"))
    assert len(files) == 1

    run_in_simple_project_with_model("train-in-chunks", "--force")

    assert os.path.exists(os.path.join(temp_dir, "models"))
    files = rasa.shared.utils.io.list_files(os.path.join(temp_dir, "models"))
    assert len(files) == 2


def test_train_core_in_chunks(run_in_simple_project: Callable[..., RunResult]):
    temp_dir = os.getcwd()

    run_in_simple_project(
        "train-in-chunks",
        "core" "-c",
        "config.yml",
        "-d",
        "domain.yml",
        "--data",
        "data",
        "--out",
        "train_models",
        "--fixed-model-name",
        "test-model",
    )

    assert os.path.exists(os.path.join(temp_dir, "train_models"))
    files = rasa.shared.utils.io.list_files(os.path.join(temp_dir, "train_models"))
    assert len(files) == 1
    assert os.path.basename(files[0]) == "test-model.tar.gz"
    model_dir = model.get_model("train_models")
    assert model_dir is not None
    metadata = Metadata.load(os.path.join(model_dir, "nlu"))
    assert metadata.get("training_data") is None
    assert not os.path.exists(
        os.path.join(model_dir, "nlu", training_data.DEFAULT_TRAINING_DATA_OUTPUT_PATH)
    )


def test_train_nlu_in_chunks(run_in_simple_project: Callable[..., RunResult]):
    temp_dir = os.getcwd()

    run_in_simple_project(
        "train-in-chunks",
        "nlu" "-c",
        "config.yml",
        "-d",
        "domain.yml",
        "--data",
        "data",
        "--out",
        "train_models",
        "--fixed-model-name",
        "test-model",
    )

    assert os.path.exists(os.path.join(temp_dir, "train_models"))
    files = rasa.shared.utils.io.list_files(os.path.join(temp_dir, "train_models"))
    assert len(files) == 1
    assert os.path.basename(files[0]) == "test-model.tar.gz"
    model_dir = model.get_model("train_models")
    assert model_dir is not None
    metadata = Metadata.load(os.path.join(model_dir, "nlu"))
    assert metadata.get("training_data") is None
    assert not os.path.exists(
        os.path.join(model_dir, "nlu", training_data.DEFAULT_TRAINING_DATA_OUTPUT_PATH)
    )


def test_train_in_chunks_help(run):
    output = run("train-in-chunks", "--help")

    help_text = """usage: rasa train-in-chunks [-h] [-v] [-vv] [--quiet] [--data DATA [DATA ...]]
                            [-c CONFIG] [-d DOMAIN] [--out OUT]
                            [--augmentation AUGMENTATION]
                            [--num-threads NUM_THREADS]
                            [--fixed-model-name FIXED_MODEL_NAME] [--force]"""

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = set(output.outlines)
    for line in lines:
        assert line in printed_help


def test_train_in_chunks_core_help(run):
    output = run("train-in-chunks", "core", "--help")

    help_text = """usage: rasa train-in-chunks core [-h] [-v] [-vv] [--quiet] [--data DATA [DATA ...]]
                            [-c CONFIG] [-d DOMAIN] [--out OUT]
                            [--augmentation AUGMENTATION]
                            [--num-threads NUM_THREADS]
                            [--fixed-model-name FIXED_MODEL_NAME] [--force]"""

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = set(output.outlines)
    for line in lines:
        assert line in printed_help


def test_train_in_chunks_nlu_help(run):
    output = run("train-in-chunks", "nlu", "--help")

    help_text = """usage: rasa train-in-chunks nlu [-h] [-v] [-vv] [--quiet] [--data DATA [DATA ...]]
                            [-c CONFIG] [-d DOMAIN] [--out OUT]
                            [--augmentation AUGMENTATION]
                            [--num-threads NUM_THREADS]
                            [--fixed-model-name FIXED_MODEL_NAME] [--force]"""

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = set(output.outlines)
    for line in lines:
        assert line in printed_help

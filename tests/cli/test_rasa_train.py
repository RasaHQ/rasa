import os
import shutil
import tempfile

import pytest

from rasa import model

from rasa.cli.train import _get_valid_config
from rasa.constants import (
    CONFIG_MANDATORY_KEYS_CORE,
    CONFIG_MANDATORY_KEYS,
    CONFIG_MANDATORY_KEYS_NLU,
)
from rasa.nlu.utils import list_files


def test_train(run_in_default_project):
    temp_dir = os.getcwd()

    run_in_default_project(
        "train",
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
    files = list_files(os.path.join(temp_dir, "train_models"))
    assert len(files) == 1
    assert os.path.basename(files[0]) == "test-model.tar.gz"


def test_train_no_domain_exists(run_in_default_project):

    os.remove("domain.yml")
    run_in_default_project(
        "train",
        "-c",
        "config.yml",
        "--data",
        "data",
        "--out",
        "train_models_no_domain",
        "--fixed-model-name",
        "nlu-model-only",
    )

    assert os.path.exists("train_models_no_domain")
    files = list_files("train_models_no_domain")
    assert len(files) == 1

    trained_model_path = "train_models_no_domain/nlu-model-only.tar.gz"
    unpacked = model.unpack_model(trained_model_path)

    metadata_path = os.path.join(unpacked, "nlu", "metadata.json")
    assert os.path.exists(metadata_path)


def test_train_skip_on_model_not_changed(run_in_default_project):
    temp_dir = os.getcwd()

    assert os.path.exists(os.path.join(temp_dir, "models"))
    files = list_files(os.path.join(temp_dir, "models"))
    assert len(files) == 1

    file_name = files[0]

    run_in_default_project("train")

    assert os.path.exists(os.path.join(temp_dir, "models"))
    files = list_files(os.path.join(temp_dir, "models"))
    assert len(files) == 1
    assert file_name == files[0]


def test_train_force(run_in_default_project):
    temp_dir = os.getcwd()

    assert os.path.exists(os.path.join(temp_dir, "models"))
    files = list_files(os.path.join(temp_dir, "models"))
    assert len(files) == 1

    run_in_default_project("train", "--force")

    assert os.path.exists(os.path.join(temp_dir, "models"))
    files = list_files(os.path.join(temp_dir, "models"))
    assert len(files) == 2


def test_train_with_only_nlu_data(run_in_default_project):
    temp_dir = os.getcwd()

    assert os.path.exists(os.path.join(temp_dir, "data/stories.md"))
    os.remove(os.path.join(temp_dir, "data/stories.md"))
    shutil.rmtree(os.path.join(temp_dir, "models"))

    run_in_default_project("train", "--fixed-model-name", "test-model")

    assert os.path.exists(os.path.join(temp_dir, "models"))
    files = list_files(os.path.join(temp_dir, "models"))
    assert len(files) == 1
    assert os.path.basename(files[0]) == "test-model.tar.gz"


def test_train_with_only_core_data(run_in_default_project):
    temp_dir = os.getcwd()

    assert os.path.exists(os.path.join(temp_dir, "data/nlu.md"))
    os.remove(os.path.join(temp_dir, "data/nlu.md"))
    shutil.rmtree(os.path.join(temp_dir, "models"))

    run_in_default_project("train", "--fixed-model-name", "test-model")

    assert os.path.exists(os.path.join(temp_dir, "models"))
    files = list_files(os.path.join(temp_dir, "models"))
    assert len(files) == 1
    assert os.path.basename(files[0]) == "test-model.tar.gz"


def test_train_core(run_in_default_project):
    run_in_default_project(
        "train",
        "core",
        "-c",
        "config.yml",
        "-d",
        "domain.yml",
        "--stories",
        "data",
        "--out",
        "train_rasa_models",
        "--fixed-model-name",
        "rasa-model",
    )

    assert os.path.exists("train_rasa_models/rasa-model.tar.gz")
    assert os.path.isfile("train_rasa_models/rasa-model.tar.gz")


def test_train_core_no_domain_exists(run_in_default_project):

    os.remove("domain.yml")
    run_in_default_project(
        "train",
        "core",
        "--config",
        "config.yml",
        "--domain",
        "domain1.yml",
        "--stories",
        "data",
        "--out",
        "train_rasa_models_no_domain",
        "--fixed-model-name",
        "rasa-model",
    )

    assert not os.path.exists("train_rasa_models_no_domain/rasa-model.tar.gz")
    assert not os.path.isfile("train_rasa_models_no_domain/rasa-model.tar.gz")


def count_rasa_temp_files():
    count = 0
    for entry in os.scandir(tempfile.gettempdir()):
        if not entry.is_dir():
            continue

        try:
            for f in os.listdir(entry.path):
                if f.endswith("_nlu.md") or f.endswith("_stories.md"):
                    count += 1
        except PermissionError:
            # Ignore permission errors
            pass

    return count


def test_train_core_temp_files(run_in_default_project):
    count = count_rasa_temp_files()
    run_in_default_project("train", "core")
    assert count == count_rasa_temp_files()


def test_train_nlu(run_in_default_project):
    run_in_default_project(
        "train",
        "nlu",
        "-c",
        "config.yml",
        "--nlu",
        "data/nlu.md",
        "--out",
        "train_models",
    )

    assert os.path.exists("train_models")
    files = list_files("train_models")
    assert len(files) == 1
    assert os.path.basename(files[0]).startswith("nlu-")


def test_train_nlu_temp_files(run_in_default_project):
    count = count_rasa_temp_files()
    run_in_default_project("train", "nlu")
    assert count == count_rasa_temp_files()


def test_train_help(run):
    output = run("train", "--help")

    help_text = """usage: rasa train [-h] [-v] [-vv] [--quiet] [--data DATA [DATA ...]]
                  [-c CONFIG] [-d DOMAIN] [--out OUT]
                  [--augmentation AUGMENTATION] [--debug-plots]
                  [--dump-stories] [--fixed-model-name FIXED_MODEL_NAME]
                  [--force]
                  {core,nlu} ..."""

    lines = help_text.split("\n")

    for i, line in enumerate(lines):
        assert output.outlines[i] == line


def test_train_nlu_help(run):
    output = run("train", "nlu", "--help")

    help_text = """usage: rasa train nlu [-h] [-v] [-vv] [--quiet] [-c CONFIG] [--out OUT]
                      [-u NLU] [--fixed-model-name FIXED_MODEL_NAME]"""

    lines = help_text.split("\n")

    for i, line in enumerate(lines):
        assert output.outlines[i] == line


def test_train_core_help(run):
    output = run("train", "core", "--help")

    help_text = """usage: rasa train core [-h] [-v] [-vv] [--quiet] [-s STORIES] [-d DOMAIN]
                       [-c CONFIG [CONFIG ...]] [--out OUT]
                       [--augmentation AUGMENTATION] [--debug-plots]
                       [--dump-stories] [--force]
                       [--fixed-model-name FIXED_MODEL_NAME]
                       [--percentages [PERCENTAGES [PERCENTAGES ...]]]
                       [--runs RUNS]"""

    lines = help_text.split("\n")

    for i, line in enumerate(lines):
        assert output.outlines[i] == line


@pytest.mark.parametrize(
    "parameters",
    [
        {
            "config_data": {"language": "en", "pipeline": "supervised"},
            "default_config": {
                "language": "en",
                "pipeline": "supervised",
                "policies": ["KerasPolicy", "FallbackPolicy"],
            },
            "mandatory_keys": CONFIG_MANDATORY_KEYS_CORE,
            "error": True,
        },
        {
            "config_data": {},
            "default_config": {
                "language": "en",
                "pipeline": "supervised",
                "policies": ["KerasPolicy", "FallbackPolicy"],
            },
            "mandatory_keys": CONFIG_MANDATORY_KEYS,
            "error": True,
        },
        {
            "config_data": {
                "policies": ["KerasPolicy", "FallbackPolicy"],
                "imports": "other-folder",
            },
            "default_config": {
                "language": "en",
                "pipeline": "supervised",
                "policies": ["KerasPolicy", "FallbackPolicy"],
            },
            "mandatory_keys": CONFIG_MANDATORY_KEYS_NLU,
            "error": True,
        },
        {
            "config_data": None,
            "default_config": {
                "pipeline": "supervised",
                "policies": ["KerasPolicy", "FallbackPolicy"],
            },
            "mandatory_keys": CONFIG_MANDATORY_KEYS_NLU,
            "error": True,
        },
        {
            "config_data": None,
            "default_config": {
                "language": "en",
                "pipeline": "supervised",
                "policies": ["KerasPolicy", "FallbackPolicy"],
            },
            "mandatory_keys": CONFIG_MANDATORY_KEYS,
            "error": False,
        },
        {
            "config_data": None,
            "default_config": {"language": "en", "pipeline": "supervised"},
            "mandatory_keys": CONFIG_MANDATORY_KEYS_CORE,
            "error": True,
        },
        {
            "config_data": None,
            "default_config": None,
            "mandatory_keys": CONFIG_MANDATORY_KEYS,
            "error": True,
        },
    ],
)
def test_get_valid_config(parameters):
    import rasa.utils.io

    config_path = None
    if parameters["config_data"] is not None:
        config_path = os.path.join(tempfile.mkdtemp(), "config.yml")
        rasa.utils.io.write_yaml_file(parameters["config_data"], config_path)

    default_config_path = None
    if parameters["default_config"] is not None:
        default_config_path = os.path.join(tempfile.mkdtemp(), "default-config.yml")
        rasa.utils.io.write_yaml_file(parameters["default_config"], default_config_path)

    if parameters["error"]:
        with pytest.raises(SystemExit):
            _get_valid_config(config_path, parameters["mandatory_keys"])

    else:
        config_path = _get_valid_config(
            config_path, parameters["mandatory_keys"], default_config_path
        )

        config_data = rasa.utils.io.read_yaml_file(config_path)

        for k in parameters["mandatory_keys"]:
            assert k in config_data


def test_get_valid_config_with_non_existing_file():
    with pytest.raises(SystemExit):
        _get_valid_config("non-existing-file.yml", CONFIG_MANDATORY_KEYS)

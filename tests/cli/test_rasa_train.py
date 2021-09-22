import os
import tempfile
from pathlib import Path

from _pytest.capture import CaptureFixture
import pytest
from typing import Callable
from _pytest.pytester import RunResult
from _pytest.tmpdir import TempPathFactory

import rasa.shared.utils.io
from rasa.constants import NUMBER_OF_TRAINING_STORIES_FILE
from rasa.core.policies import Policy
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.storage.resource import Resource
from rasa.nlu.components import Component
from rasa.shared.core.domain import Domain
from rasa.model_training import (
    CODE_NEEDS_TO_BE_RETRAINED,
    CODE_FORCED_TRAINING,
)

# noinspection PyProtectedMember
from rasa.cli.train import _get_valid_config
from rasa.shared.constants import (
    CONFIG_MANDATORY_KEYS_CORE,
    CONFIG_MANDATORY_KEYS_NLU,
    CONFIG_MANDATORY_KEYS,
    LATEST_TRAINING_DATA_FORMAT_VERSION,
)
from rasa.shared.nlu.training_data.training_data import (
    DEFAULT_TRAINING_DATA_OUTPUT_PATH,
)
import rasa.utils.io


def test_train(run_in_simple_project: Callable[..., RunResult], tmp_path: Path):
    temp_dir = os.getcwd()

    run_in_simple_project(
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

    models_dir = Path(temp_dir, "train_models")
    assert models_dir.is_dir()

    models = list(models_dir.glob("*"))
    assert len(models) == 1

    model = models[0]
    assert model.name == "test-model.tar.gz"

    _, metadata = LocalModelStorage.from_model_archive(tmp_path, model)
    assert metadata.model_id
    assert (
        metadata.domain.as_dict() == Domain.load(Path(temp_dir, "domain.yml")).as_dict()
    )


def test_train_finetune(
    run_in_simple_project: Callable[..., RunResult], capsys: CaptureFixture
):
    run_in_simple_project(
        "train", "--finetune",
    )

    output = capsys.readouterr().out
    assert "No model for finetuning found" in output


def test_train_persist_nlu_data(
    run_in_simple_project: Callable[..., RunResult], tmp_path: Path
):
    temp_dir = os.getcwd()

    run_in_simple_project(
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
        "--persist-nlu-data",
    )

    models_dir = Path(temp_dir, "train_models")
    assert models_dir.is_dir()

    models = list(models_dir.glob("*"))
    assert len(models) == 1

    model = models[0]
    assert model.name == "test-model.tar.gz"

    storage, _ = LocalModelStorage.from_model_archive(tmp_path, model)

    with storage.read_from(Resource("nlu_training_data_provider")) as directory:
        assert (directory / DEFAULT_TRAINING_DATA_OUTPUT_PATH).exists()


def test_train_no_domain_exists(
    run_in_simple_project: Callable[..., RunResult], tmp_path: Path
) -> None:

    os.remove("domain.yml")
    run_in_simple_project(
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

    model_file = Path("train_models_no_domain", "nlu-model-only.tar.gz")
    assert model_file.is_file()

    _, metadata = LocalModelStorage.from_model_archive(tmp_path, model_file)

    assert not any(
        issubclass(component.uses, Policy)
        for component in metadata.train_schema.nodes.values()
    )
    assert not any(
        issubclass(component.uses, Policy)
        for component in metadata.predict_schema.nodes.values()
    )


def test_train_skip_on_model_not_changed(
    run_in_simple_project_with_model: Callable[..., RunResult],
    tmp_path_factory: TempPathFactory,
):
    temp_dir = os.getcwd()

    models_dir = Path(temp_dir, "models")
    model_files = list(models_dir.glob("*"))
    assert len(model_files) == 1
    old_model = model_files[0]

    run_in_simple_project_with_model("train")

    model_files = list(sorted(models_dir.glob("*")))
    assert len(model_files) == 2

    new_model = model_files[1]
    assert old_model != new_model

    old_dir = tmp_path_factory.mktemp("old")
    _, old_metadata = LocalModelStorage.from_model_archive(old_dir, old_model)

    new_dir = tmp_path_factory.mktemp("new")
    _, new_metadata = LocalModelStorage.from_model_archive(new_dir, new_model)

    assert old_metadata.model_id != new_metadata.model_id
    assert old_metadata.trained_at < new_metadata.trained_at
    assert old_metadata.domain.as_dict() == new_metadata.domain.as_dict()

    assert rasa.utils.io.are_directories_equal(old_dir, new_dir)


def test_train_force(
    run_in_simple_project_with_model: Callable[..., RunResult],
    tmp_path_factory: TempPathFactory,
):
    temp_dir = os.getcwd()

    assert os.path.exists(os.path.join(temp_dir, "models"))
    files = rasa.shared.utils.io.list_files(os.path.join(temp_dir, "models"))
    assert len(files) == 1

    run_in_simple_project_with_model("train", "--force")

    assert os.path.exists(os.path.join(temp_dir, "models"))
    files = rasa.shared.utils.io.list_files(os.path.join(temp_dir, "models"))
    assert len(files) == 2

    old_dir = tmp_path_factory.mktemp("old")
    _ = LocalModelStorage.from_model_archive(old_dir, files[0])

    new_dir = tmp_path_factory.mktemp("new")
    _ = LocalModelStorage.from_model_archive(new_dir, files[1])

    assert not rasa.utils.io.are_directories_equal(old_dir, new_dir)


def test_train_dry_run(run_in_simple_project_with_model: Callable[..., RunResult]):
    temp_dir = os.getcwd()

    assert os.path.exists(os.path.join(temp_dir, "models"))
    files = rasa.shared.utils.io.list_files(os.path.join(temp_dir, "models"))
    assert len(files) == 1

    output = run_in_simple_project_with_model("train", "--dry-run")

    assert [s for s in output.outlines if "No training of components required" in s]
    assert output.ret == 0


def test_train_dry_run_failure(run_in_simple_project: Callable[..., RunResult]):
    temp_dir = os.getcwd()

    domain = (
        "version: '" + LATEST_TRAINING_DATA_FORMAT_VERSION + "'\n"
        "session_config:\n"
        "  session_expiration_time: 60\n"
        "  carry_over_slots_to_new_session: true\n"
        "actions:\n"
        "- utter_greet\n"
        "- utter_cheer_up"
    )

    with open(os.path.join(temp_dir, "domain.yml"), "w") as f:
        f.write(domain)

    output = run_in_simple_project("train", "--dry-run")

    assert not any([s for s in output.outlines if "No training required." in s])
    assert (output.ret & CODE_NEEDS_TO_BE_RETRAINED == CODE_NEEDS_TO_BE_RETRAINED) and (
        output.ret & CODE_FORCED_TRAINING != CODE_FORCED_TRAINING
    )


def test_train_dry_run_force(
    run_in_simple_project_with_model: Callable[..., RunResult]
):
    temp_dir = os.getcwd()

    assert os.path.exists(os.path.join(temp_dir, "models"))
    files = rasa.shared.utils.io.list_files(os.path.join(temp_dir, "models"))
    assert len(files) == 1

    output = run_in_simple_project_with_model("train", "--dry-run", "--force")

    assert [s for s in output.outlines if "The training was forced." in s]
    assert output.ret == CODE_FORCED_TRAINING


def test_train_with_only_nlu_data(run_in_simple_project: Callable[..., RunResult]):
    temp_dir = Path.cwd()

    for core_file in ["stories.yml", "rules.yml"]:
        assert (temp_dir / "data" / core_file).exists()
        (temp_dir / "data" / core_file).unlink()

    run_in_simple_project("train", "--fixed-model-name", "test-model")

    assert os.path.exists(os.path.join(temp_dir, "models"))
    files = rasa.shared.utils.io.list_files(os.path.join(temp_dir, "models"))
    assert len(files) == 1
    assert os.path.basename(files[0]) == "test-model.tar.gz"


def test_train_with_only_core_data(run_in_simple_project: Callable[..., RunResult]):
    temp_dir = os.getcwd()

    assert os.path.exists(os.path.join(temp_dir, "data/nlu.yml"))
    os.remove(os.path.join(temp_dir, "data/nlu.yml"))

    run_in_simple_project("train", "--fixed-model-name", "test-model")

    assert os.path.exists(os.path.join(temp_dir, "models"))
    files = rasa.shared.utils.io.list_files(os.path.join(temp_dir, "models"))
    assert len(files) == 1
    assert os.path.basename(files[0]) == "test-model.tar.gz"


def test_train_core(run_in_simple_project: Callable[..., RunResult]):
    run_in_simple_project(
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


def test_train_core_no_domain_exists(run_in_simple_project: Callable[..., RunResult]):
    os.remove("domain.yml")
    run_in_simple_project(
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

    assert not list(Path("train_rasa_models_no_domain").glob("*"))


def test_train_core_compare(
    run_in_simple_project: Callable[..., RunResult], tmp_path: Path
):
    run_in_simple_project(
        "train",
        "core",
        "-c",
        "config.yml",
        "config.yml",
        "-d",
        "domain.yml",
        "--stories",
        "data",
        "--out",
        str(tmp_path),
        "--runs",
        "2",
        "--percentages",
        "50",
        "100",
    )

    for run in range(1, 2):
        assert (tmp_path / f"run_{run}" / "config__percentage__50.tar.gz").exists()
        assert (tmp_path / f"run_{run}" / "config__percentage__100.tar.gz").exists()

    num_stories = rasa.shared.utils.io.read_yaml_file(
        tmp_path / NUMBER_OF_TRAINING_STORIES_FILE
    )
    assert num_stories == [3, 0]


def test_train_nlu(run_in_simple_project: Callable[..., RunResult], tmp_path: Path):
    run_in_simple_project(
        "train",
        "nlu",
        "-c",
        "config.yml",
        "--nlu",
        "data/nlu.yml",
        "--out",
        "train_models",
    )

    model_dir = Path("train_models")
    assert model_dir.is_dir()

    models = list(model_dir.glob("*.tar.gz"))
    assert len(models) == 1

    model_file = models[0]
    assert model_file.name.startswith("nlu-")

    _, metadata = LocalModelStorage.from_model_archive(tmp_path, model_file)

    assert not any(
        issubclass(component.uses, Component)
        for component in metadata.train_schema.nodes.values()
    )
    assert not any(
        issubclass(component.uses, Component)
        for component in metadata.predict_schema.nodes.values()
    )


def test_train_nlu_persist_nlu_data(
    run_in_simple_project: Callable[..., RunResult], tmp_path: Path
) -> None:
    run_in_simple_project(
        "train",
        "nlu",
        "-c",
        "config.yml",
        "--nlu",
        "data/nlu.yml",
        "--out",
        "train_models",
        "--persist-nlu-data",
    )

    models_dir = Path("train_models")
    assert models_dir.is_dir()

    models = list(models_dir.glob("*"))
    assert len(models) == 1

    model = models[0]
    assert model.name.startswith("nlu-")

    storage, _ = LocalModelStorage.from_model_archive(tmp_path, model)

    with storage.read_from(Resource("nlu_training_data_provider")) as directory:
        assert (directory / DEFAULT_TRAINING_DATA_OUTPUT_PATH).exists()


def test_train_help(run: Callable[..., RunResult]):
    output = run("train", "--help")

    help_text = """usage: rasa train [-h] [-v] [-vv] [--quiet] [--data DATA [DATA ...]]
                  [-c CONFIG] [-d DOMAIN] [--out OUT] [--dry-run]
                  [--augmentation AUGMENTATION] [--debug-plots]
                  [--num-threads NUM_THREADS]
                  [--fixed-model-name FIXED_MODEL_NAME] [--persist-nlu-data]
                  [--force] [--finetune [FINETUNE]]
                  [--epoch-fraction EPOCH_FRACTION]
                  {core,nlu} ..."""

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = set(output.outlines)

    for line in lines:
        assert line in printed_help


def test_train_nlu_help(run: Callable[..., RunResult]):
    output = run("train", "nlu", "--help")

    help_text = """usage: rasa train nlu [-h] [-v] [-vv] [--quiet] [-c CONFIG] [-d DOMAIN]
                      [--out OUT] [-u NLU] [--num-threads NUM_THREADS]
                      [--fixed-model-name FIXED_MODEL_NAME]
                      [--persist-nlu-data] [--finetune [FINETUNE]]
                      [--epoch-fraction EPOCH_FRACTION]"""

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = set(output.outlines)
    for line in lines:
        assert line in printed_help


def test_train_core_help(run: Callable[..., RunResult]):
    output = run("train", "core", "--help")

    help_text = """usage: rasa train core [-h] [-v] [-vv] [--quiet] [-s STORIES] [-d DOMAIN]
                       [-c CONFIG [CONFIG ...]] [--out OUT]
                       [--augmentation AUGMENTATION] [--debug-plots] [--force]
                       [--fixed-model-name FIXED_MODEL_NAME]
                       [--percentages [PERCENTAGES [PERCENTAGES ...]]]
                       [--runs RUNS] [--finetune [FINETUNE]]
                       [--epoch-fraction EPOCH_FRACTION]"""

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = set(output.outlines)
    for line in lines:
        assert line in printed_help


@pytest.mark.parametrize(
    "parameters",
    [
        {
            "config_data": {"language": "en", "pipeline": "supervised"},
            "default_config": {
                "language": "en",
                "pipeline": "supervised",
                "policies": ["TEDPolicy", "FallbackPolicy"],
            },
            "mandatory_keys": CONFIG_MANDATORY_KEYS_CORE,
            "error": False,
        },
        {
            "config_data": {
                "language": "en",
                "pipeline": "supervised",
                "policies": None,
            },
            "default_config": {
                "language": "en",
                "pipeline": "supervised",
                "policies": ["TEDPolicy", "FallbackPolicy"],
            },
            "mandatory_keys": CONFIG_MANDATORY_KEYS_CORE,
            "error": False,
        },
        {
            "config_data": {},
            "default_config": {
                "language": "en",
                "pipeline": "supervised",
                "policies": ["TEDPolicy", "FallbackPolicy"],
            },
            "mandatory_keys": CONFIG_MANDATORY_KEYS,
            "error": True,
        },
        {
            "config_data": {
                "policies": ["TEDPolicy", "FallbackPolicy"],
                "imports": "other-folder",
            },
            "default_config": {
                "language": "en",
                "pipeline": "supervised",
                "policies": ["TEDPolicy", "FallbackPolicy"],
            },
            "mandatory_keys": CONFIG_MANDATORY_KEYS_NLU,
            "error": True,
        },
        {
            "config_data": None,
            "default_config": {
                "pipeline": "supervised",
                "policies": ["TEDPolicy", "FallbackPolicy"],
            },
            "mandatory_keys": CONFIG_MANDATORY_KEYS_NLU,
            "error": True,
        },
        {
            "config_data": None,
            "default_config": {
                "language": "en",
                "pipeline": "supervised",
                "policies": ["TEDPolicy", "FallbackPolicy"],
            },
            "mandatory_keys": CONFIG_MANDATORY_KEYS,
            "error": False,
        },
        {
            "config_data": None,
            "default_config": {"language": "en", "pipeline": "supervised"},
            "mandatory_keys": CONFIG_MANDATORY_KEYS_CORE,
            "error": False,
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
    config_path = None
    if parameters["config_data"] is not None:
        config_path = os.path.join(tempfile.mkdtemp(), "config.yml")
        rasa.shared.utils.io.write_yaml(parameters["config_data"], config_path)

    default_config_path = None
    if parameters["default_config"] is not None:
        default_config_path = os.path.join(tempfile.mkdtemp(), "default-config.yml")
        rasa.shared.utils.io.write_yaml(
            parameters["default_config"], default_config_path
        )

    if parameters["error"]:
        with pytest.raises(SystemExit):
            _get_valid_config(config_path, parameters["mandatory_keys"])

    else:
        config_path = _get_valid_config(
            config_path, parameters["mandatory_keys"], default_config_path
        )

        config_data = rasa.shared.utils.io.read_yaml_file(config_path)

        for k in parameters["mandatory_keys"]:
            assert k in config_data


def test_get_valid_config_with_non_existing_file():
    with pytest.raises(SystemExit):
        _get_valid_config("non-existing-file.yml", CONFIG_MANDATORY_KEYS)


def test_train_nlu_finetune_with_model(
    run_in_simple_project_with_model: Callable[..., RunResult]
):
    temp_dir = os.getcwd()

    files = rasa.shared.utils.io.list_files(os.path.join(temp_dir, "models"))
    assert len(files) == 1

    model_name = os.path.relpath(files[0])
    output = run_in_simple_project_with_model("train", "--finetune", model_name)
    assert any(
        "Your Rasa model is trained and saved at" in line for line in output.outlines
    )

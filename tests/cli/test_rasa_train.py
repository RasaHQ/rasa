import argparse
import os
import sys
from pathlib import Path
from typing import Callable, List, Union
from unittest.mock import patch

import pytest
from _pytest.capture import CaptureFixture
from _pytest.pytester import RunResult
from _pytest.tmpdir import TempPathFactory
from pytest import MonkeyPatch

import rasa.shared.utils.io
import rasa.utils.io
from rasa.cli.train import _check_nlg_endpoint_validity, run_training
from rasa.constants import NUMBER_OF_TRAINING_STORIES_FILE
from rasa.core.policies.policy import Policy
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.storage.resource import Resource
from rasa.model_training import (
    CODE_FORCED_TRAINING,
    CODE_NEEDS_TO_BE_RETRAINED,
    TrainingResult,
)
from rasa.shared.constants import (
    LATEST_TRAINING_DATA_FORMAT_VERSION,
    OPENAI_API_KEY_ENV_VAR,
)
from rasa.shared.core.domain import Domain
from rasa.shared.nlu.training_data.training_data import (
    DEFAULT_TRAINING_DATA_OUTPUT_PATH,
)
from rasa.shared.utils.yaml import read_yaml_file
from tests.cli.conftest import RASA_EXE


@pytest.mark.parametrize(
    "optional_arguments",
    [
        ["--endpoints", "endpoints.yml"],
        ["--endpoints", "non_existent_endpoints.yml"],
        [],
    ],
)
def test_train(
    run_in_simple_project: Callable[..., RunResult],
    tmp_path: Path,
    optional_arguments: List,
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
        *optional_arguments,
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
    output = run_in_simple_project("train", "--finetune")
    logs = output.outlines + output.errlines
    assert [log for log in logs if "No model for finetuning found" in log]


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
        "--skip-validation",
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

    logs = output.outlines + output.errlines
    assert [log for log in logs if "No training of components required" in log]
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

    logs = output.outlines + output.errlines
    assert not any([log for log in logs if "No training required." in log])
    assert (output.ret & CODE_NEEDS_TO_BE_RETRAINED == CODE_NEEDS_TO_BE_RETRAINED) and (
        output.ret & CODE_FORCED_TRAINING != CODE_FORCED_TRAINING
    )


def test_train_dry_run_force(
    run_in_simple_project_with_model: Callable[..., RunResult],
):
    temp_dir = os.getcwd()

    assert os.path.exists(os.path.join(temp_dir, "models"))
    files = rasa.shared.utils.io.list_files(os.path.join(temp_dir, "models"))
    assert len(files) == 1

    output = run_in_simple_project_with_model("train", "--dry-run", "--force")

    logs = output.outlines + output.errlines
    assert [log for log in logs if "The training was forced." in log]
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

    num_stories = read_yaml_file(tmp_path / NUMBER_OF_TRAINING_STORIES_FILE)
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
        issubclass(component.uses, Policy)
        for component in metadata.train_schema.nodes.values()
    )
    assert not any(
        issubclass(component.uses, Policy)
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

    help_text = f"""usage: {RASA_EXE} train [-h] [-v] [-vv] [--quiet]
                  [--logging-config-file LOGGING_CONFIG_FILE]
                  [--data DATA [DATA ...]] [-c CONFIG] [-d DOMAIN] [--out OUT]
                  [--dry-run] [--skip-validation]
                  [--fail-on-validation-warnings]
                  [--validation-max-history VALIDATION_MAX_HISTORY]
                  [--augmentation AUGMENTATION] [--debug-plots]
                  [--num-threads NUM_THREADS]
                  [--fixed-model-name FIXED_MODEL_NAME] [--persist-nlu-data]
                  [--force] [--finetune [FINETUNE]]
                  [--epoch-fraction EPOCH_FRACTION] [--endpoints ENDPOINTS]
                  {{core,nlu}} ..."""

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = {line.strip() for line in output.outlines}
    for line in lines:
        assert line.strip() in printed_help


def test_train_nlu_help(run: Callable[..., RunResult]):
    output = run("train", "nlu", "--help")

    help_text = f"""usage: {RASA_EXE} train nlu [-h] [-v] [-vv] [--quiet]
                      [--logging-config-file LOGGING_CONFIG_FILE] [-c CONFIG]
                      [-d DOMAIN] [--out OUT] [-u NLU]
                      [--num-threads NUM_THREADS]
                      [--fixed-model-name FIXED_MODEL_NAME]
                      [--persist-nlu-data] [--finetune [FINETUNE]]
                      [--epoch-fraction EPOCH_FRACTION]"""

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = {line.strip() for line in output.outlines}
    for line in lines:
        assert line.strip() in printed_help


def test_train_core_help(run: Callable[..., RunResult]):
    output = run("train", "core", "--help")

    if sys.version_info.minor >= 9:
        # This is required because `argparse` behaves differently on
        # Python 3.9 and above. The difference is the changed formatting of help
        # output for CLI arguments with `nargs="*"
        help_text = f"""usage: {RASA_EXE} train core [-h] [-v] [-vv] [--quiet]
                       [--logging-config-file LOGGING_CONFIG_FILE]
                       [-s STORIES] [-d DOMAIN] [-c CONFIG [CONFIG ...]]
                       [--out OUT] [--augmentation AUGMENTATION]
                       [--debug-plots] [--force]
                       [--fixed-model-name FIXED_MODEL_NAME]
                       [--percentages [PERCENTAGES ...]] [--runs RUNS]
                       [--finetune [FINETUNE]]
                       [--epoch-fraction EPOCH_FRACTION]"""
    else:
        help_text = f"""usage: {RASA_EXE} train core [-h] [-v] [-vv] [--quiet]
                       [--logging-config-file LOGGING_CONFIG_FILE]
                       [-s STORIES] [-d DOMAIN] [-c CONFIG [CONFIG ...]]
                       [--out OUT] [--augmentation AUGMENTATION]
                       [--debug-plots] [--force]
                       [--fixed-model-name FIXED_MODEL_NAME]
                       [--percentages [PERCENTAGES [PERCENTAGES ...]]]
                       [--runs RUNS] [--finetune [FINETUNE]]
                       [--epoch-fraction EPOCH_FRACTION]"""

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = {line.strip() for line in output.outlines}
    for line in lines:
        assert line.strip() in printed_help


def test_train_nlu_finetune_with_model(
    run_in_simple_project_with_model: Callable[..., RunResult],
):
    temp_dir = os.getcwd()

    files = rasa.shared.utils.io.list_files(os.path.join(temp_dir, "models"))
    assert len(files) == 1

    model_name = os.path.relpath(files[0])
    output = run_in_simple_project_with_model("train", "--finetune", model_name)
    logs = output.outlines + output.errlines
    assert [log for log in logs if "Your Rasa model is trained and saved at" in log]


def test_train_validation_debug_messages(
    run_in_simple_project: Callable[..., RunResult], request: pytest.FixtureRequest
):
    test_data_dir = Path(request.config.rootdir, "data", "test_validation", "data")
    test_domain = Path(request.config.rootdir, "data", "test_validation", "domain.yml")

    result = run_in_simple_project(
        "train",
        "--data",
        str(test_data_dir),
        "--domain",
        str(test_domain),
        "-c",
        "config.yml",
        "--debug",
    )
    log_output = "\n".join(result.outlines + result.errlines)

    assert result.ret == 0
    message = "The intent 'goodbye' is not used in any story, rule or flow."
    assert message in log_output


def test_train_validation_fail_on_warnings(
    run_in_simple_project_with_warnings: Callable[..., RunResult],
    request: pytest.FixtureRequest,
):
    test_data_dir = Path(request.config.rootdir, "data", "test_moodbot", "data")
    test_domain = Path(request.config.rootdir, "data", "test_domains", "default.yml")

    result = run_in_simple_project_with_warnings(
        "train",
        "--fail-on-validation-warnings",
        "--data",
        str(test_data_dir),
        "--domain",
        str(test_domain),
        "-c",
        "config.yml",
    )

    logs = result.outlines + result.errlines
    assert [log for log in logs if "Project validation completed with errors." in log]
    assert result.ret == 1


def test_train_succeeds_by_falling_back_to_default_domain(
    run_in_simple_project: Callable[..., RunResult],
):
    result = run_in_simple_project(
        "train",
        "--domain",
        "not_existing_domain.yml",
    )
    logs = result.outlines + result.errlines

    assert [log for log in logs if "Your Rasa model is trained and saved" in log]
    assert result.ret == 0


def test_train_validation_max_history_1(
    run_in_simple_project_with_warnings: Callable[..., RunResult],
    request: pytest.FixtureRequest,
):
    test_data_dir = Path(
        request.config.rootdir,
        "data",
        "test_yaml_stories",
        "stories_conflicting_at_1.yml",
    )
    test_domain = Path(request.config.rootdir, "data", "test_domains", "default.yml")

    result = run_in_simple_project_with_warnings(
        "train",
        "--validation-max-history",
        "1",
        "--data",
        str(test_data_dir),
        "--domain",
        str(test_domain),
        "-c",
        "config.yml",
    )

    logs = result.outlines + result.errlines

    assert [log for log in logs if "Found story structure conflict" in log]
    assert result.ret == 0


def test_train_validation_max_history_2(
    run_in_simple_project_with_warnings: Callable[..., RunResult],
    request: pytest.FixtureRequest,
):
    test_data_dir = Path(
        request.config.rootdir,
        "data",
        "test_yaml_stories",
        "stories_conflicting_at_1.yml",
    )
    test_domain = Path(request.config.rootdir, "data", "test_domains", "default.yml")

    result = run_in_simple_project_with_warnings(
        "train",
        "--validation-max-history",
        "2",
        "--data",
        str(test_data_dir),
        "--domain",
        str(test_domain),
        "-c",
        "config.yml",
    )

    assert "Story structure conflict" not in str(result.errlines)
    assert result.ret == 0


def test_train_validate_nlg_config_valid(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "my key")
    args = argparse.Namespace(
        domain="data/test_domains/default.yml",
        config="data/test_config/config_defaults.yml",
        data=["data/test_moodbot/data"],
        endpoints="data/test_nlg/endpoint_with_valid_nlg.yml",
        skip_validation=True,
        out="models",
        force=False,
        fixed_model_name=None,
        persist_nlu_data=False,
        epoch_fraction=1.0,
        dry_run=False,
        finetune=None,
        remote_storage=None,
    )

    with patch("rasa.api.train", return_value=TrainingResult(0)):
        run_training(args)


def test_train_validate_nlg_config_invalid() -> None:
    args = argparse.Namespace(
        domain="data/test_domains/default.yml",
        config="data/test_config/config_defaults.yml",
        data=["data/test_moodbot/data"],
        endpoints="data/test_nlg/endpoint_with_invalid_nlg.yml",
        remote_storage=None,
    )

    with pytest.raises(SystemExit):
        run_training(args)


@pytest.mark.parametrize(
    "endpoint_path, expected_error",
    [
        ("data/test_nlg/endpoint_with_valid_nlg.yml", False),
        ("data/test_nlg/endpoint_with_invalid_nlg.yml", True),
    ],
)
def test_train_check_nlg_endpoint_validity(
    endpoint_path: Union[Path, str], expected_error: bool, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "mock key in test_rasa_train")
    if expected_error:
        with pytest.raises(SystemExit):
            _check_nlg_endpoint_validity(endpoint=endpoint_path)
    else:
        _check_nlg_endpoint_validity(endpoint=endpoint_path)

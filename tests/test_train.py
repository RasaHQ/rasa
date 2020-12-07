import sys
import tempfile
import os
from pathlib import Path
from typing import Text, Dict, Any
from unittest.mock import Mock

import pytest
from _pytest.capture import CaptureFixture
from _pytest.monkeypatch import MonkeyPatch

import rasa.model
import rasa.core
import rasa.nlu
from rasa.nlu.classifiers.diet_classifier import DIETClassifier
import rasa.shared.importers.autoconfig as autoconfig
import rasa.shared.utils.io
from rasa.core.agent import Agent
from rasa.core.interpreter import RasaNLUInterpreter
from rasa.nlu.model import Interpreter

from rasa.train import train_core, train_nlu, train
from rasa.utils.tensorflow.constants import EPOCHS
from tests.conftest import DEFAULT_CONFIG_PATH, DEFAULT_NLU_DATA, AsyncMock
from tests.core.conftest import DEFAULT_DOMAIN_PATH_WITH_SLOTS, DEFAULT_STORIES_FILE
from tests.test_model import _fingerprint


@pytest.mark.parametrize(
    "parameters",
    [
        {"model_name": "test-1234", "prefix": None},
        {"model_name": None, "prefix": "core-"},
        {"model_name": None, "prefix": None},
    ],
)
def test_package_model(trained_rasa_model: Text, parameters: Dict):
    output_path = tempfile.mkdtemp()
    train_path = rasa.model.unpack_model(trained_rasa_model)

    model_path = rasa.model.package_model(
        _fingerprint(),
        output_path,
        train_path,
        parameters["model_name"],
        parameters["prefix"],
    )

    assert os.path.exists(model_path)

    file_name = os.path.basename(model_path)

    if parameters["model_name"]:
        assert parameters["model_name"] in file_name

    if parameters["prefix"]:
        assert parameters["prefix"] in file_name

    assert file_name.endswith(".tar.gz")


def count_temp_rasa_files(directory: Text) -> int:
    return len(
        [
            entry
            for entry in os.listdir(directory)
            if not any(
                [
                    # Ignore the following files/directories:
                    entry == "__pycache__",  # Python bytecode
                    entry.endswith(".py")  # Temp .py files created by TF
                    # Anything else is considered to be created by Rasa
                ]
            )
        ]
    )


def test_train_temp_files(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    default_domain_path: Text,
    default_stories_file: Text,
    default_stack_config: Text,
    default_nlu_data: Text,
):
    (tmp_path / "training").mkdir()
    (tmp_path / "models").mkdir()

    monkeypatch.setattr(tempfile, "tempdir", tmp_path / "training")
    output = str(tmp_path / "models")

    train(
        default_domain_path,
        default_stack_config,
        [default_stories_file, default_nlu_data],
        output=output,
        force_training=True,
    )

    assert count_temp_rasa_files(tempfile.tempdir) == 0

    # After training the model, try to do it again. This shouldn't try to train
    # a new model because nothing has been changed. It also shouldn't create
    # any temp files.
    train(
        default_domain_path,
        default_stack_config,
        [default_stories_file, default_nlu_data],
        output=output,
    )

    assert count_temp_rasa_files(tempfile.tempdir) == 0


def test_train_core_temp_files(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    default_domain_path: Text,
    default_stories_file: Text,
    default_stack_config: Text,
):
    (tmp_path / "training").mkdir()
    (tmp_path / "models").mkdir()

    monkeypatch.setattr(tempfile, "tempdir", tmp_path / "training")

    train_core(
        default_domain_path,
        default_stack_config,
        default_stories_file,
        output=str(tmp_path / "models"),
    )

    assert count_temp_rasa_files(tempfile.tempdir) == 0


def test_train_nlu_temp_files(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    default_stack_config: Text,
    default_nlu_data: Text,
):
    (tmp_path / "training").mkdir()
    (tmp_path / "models").mkdir()

    monkeypatch.setattr(tempfile, "tempdir", tmp_path / "training")

    train_nlu(default_stack_config, default_nlu_data, output=str(tmp_path / "models"))

    assert count_temp_rasa_files(tempfile.tempdir) == 0


def test_train_nlu_wrong_format_error_message(
    capsys: CaptureFixture,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    default_stack_config: Text,
    incorrect_nlu_data: Text,
):
    (tmp_path / "training").mkdir()
    (tmp_path / "models").mkdir()

    monkeypatch.setattr(tempfile, "tempdir", tmp_path / "training")

    train_nlu(default_stack_config, incorrect_nlu_data, output=str(tmp_path / "models"))

    captured = capsys.readouterr()
    assert "Please verify the data format" in captured.out


def test_train_nlu_with_responses_no_domain_warns(tmp_path: Path):
    data_path = "data/test_nlu_no_responses/nlu_no_responses.yml"

    with pytest.warns(UserWarning) as records:
        train_nlu(
            "data/test_config/config_response_selector_minimal.yml",
            data_path,
            output=str(tmp_path / "models"),
        )

    assert any(
        "You either need to add a response phrase or correct the intent"
        in record.message.args[0]
        for record in records
    )


def test_train_nlu_with_responses_and_domain_no_warns(tmp_path: Path):
    data_path = "data/test_nlu_no_responses/nlu_no_responses.yml"
    domain_path = "data/test_nlu_no_responses/domain_with_only_responses.yml"

    with pytest.warns(None) as records:
        train_nlu(
            "data/test_config/config_response_selector_minimal.yml",
            data_path,
            output=str(tmp_path / "models"),
            domain=domain_path,
        )

    assert not any(
        "You either need to add a response phrase or correct the intent"
        in record.message.args[0]
        for record in records
    )


def test_train_nlu_no_nlu_file_error_message(
    capsys: CaptureFixture,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    default_stack_config: Text,
):
    (tmp_path / "training").mkdir()
    (tmp_path / "models").mkdir()

    monkeypatch.setattr(tempfile, "tempdir", tmp_path / "training")

    train_nlu(default_stack_config, "", output=str(tmp_path / "models"))

    captured = capsys.readouterr()
    assert "No NLU data given" in captured.out


def test_trained_interpreter_passed_to_core_training(
    monkeypatch: MonkeyPatch, tmp_path: Path, unpacked_trained_rasa_model: Text
):
    # Skip actual NLU training and return trained interpreter path from fixture
    # Patching is bit more complicated as we have a module `train` and function
    # with the same name 😬
    monkeypatch.setattr(
        sys.modules["rasa.train"],
        "_train_nlu_with_validated_data",
        AsyncMock(return_value=unpacked_trained_rasa_model),
    )

    # Mock the actual Core training
    _train_core = AsyncMock()
    monkeypatch.setattr(rasa.core, "train", _train_core)

    train(
        DEFAULT_DOMAIN_PATH_WITH_SLOTS,
        DEFAULT_CONFIG_PATH,
        [DEFAULT_STORIES_FILE, DEFAULT_NLU_DATA],
        str(tmp_path),
    )

    _train_core.assert_called_once()
    _, _, kwargs = _train_core.mock_calls[0]
    assert isinstance(kwargs["interpreter"], RasaNLUInterpreter)


def test_interpreter_of_old_model_passed_to_core_training(
    monkeypatch: MonkeyPatch, tmp_path: Path, trained_rasa_model: Text
):
    # NLU isn't retrained
    monkeypatch.setattr(
        rasa.model.FingerprintComparisonResult,
        rasa.model.FingerprintComparisonResult.should_retrain_nlu.__name__,
        lambda _: False,
    )

    # An old model with an interpreter exists
    monkeypatch.setattr(
        rasa.model, rasa.model.get_latest_model.__name__, lambda _: trained_rasa_model
    )

    # Mock the actual Core training
    _train_core = AsyncMock()
    monkeypatch.setattr(rasa.core, "train", _train_core)

    train(
        DEFAULT_DOMAIN_PATH_WITH_SLOTS,
        DEFAULT_CONFIG_PATH,
        [DEFAULT_STORIES_FILE, DEFAULT_NLU_DATA],
        str(tmp_path),
    )

    _train_core.assert_called_once()
    _, _, kwargs = _train_core.mock_calls[0]
    assert isinstance(kwargs["interpreter"], RasaNLUInterpreter)


def test_load_interpreter_returns_none_for_none():
    from rasa.train import _load_interpreter

    assert _load_interpreter(None) is None


def test_interpreter_from_previous_model_returns_none_for_none():
    from rasa.train import _interpreter_from_previous_model

    assert _interpreter_from_previous_model(None) is None


def test_train_core_autoconfig(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    default_domain_path: Text,
    default_stories_file: Text,
    default_stack_config: Text,
):
    monkeypatch.setattr(tempfile, "tempdir", tmp_path)

    # mock function that returns configuration
    mocked_get_configuration = Mock()
    monkeypatch.setattr(autoconfig, "get_configuration", mocked_get_configuration)

    # skip actual core training
    monkeypatch.setattr(
        sys.modules["rasa.train"], "_train_core_with_validated_data", AsyncMock()
    )

    # do training
    train_core(
        default_domain_path,
        default_stack_config,
        default_stories_file,
        output="test_train_core_temp_files_models",
    )

    mocked_get_configuration.assert_called_once()
    _, args, _ = mocked_get_configuration.mock_calls[0]
    assert args[1] == autoconfig.TrainingType.CORE


def test_train_nlu_autoconfig(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    default_stack_config: Text,
    default_nlu_data: Text,
):
    monkeypatch.setattr(tempfile, "tempdir", tmp_path)

    # mock function that returns configuration
    mocked_get_configuration = Mock()
    monkeypatch.setattr(autoconfig, "get_configuration", mocked_get_configuration)

    monkeypatch.setattr(
        sys.modules["rasa.train"], "_train_nlu_with_validated_data", AsyncMock()
    )

    # do training
    train_nlu(
        default_stack_config,
        default_nlu_data,
        output="test_train_nlu_temp_files_models",
    )

    mocked_get_configuration.assert_called_once()
    _, args, _ = mocked_get_configuration.mock_calls[0]
    assert args[1] == autoconfig.TrainingType.NLU


@pytest.mark.parametrize("use_latest_model", [True, False])
def test_model_finetuning(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    default_domain_path: Text,
    default_stories_file: Text,
    default_stack_config: Text,
    default_nlu_data: Text,
    trained_rasa_model: Text,
    use_latest_model: bool,
):
    mocked_nlu_training = AsyncMock(return_value="")
    monkeypatch.setattr(rasa.nlu, rasa.nlu.train.__name__, mocked_nlu_training)

    mocked_core_training = AsyncMock()
    monkeypatch.setattr(rasa.core, rasa.core.train.__name__, mocked_core_training)

    (tmp_path / "models").mkdir()
    output = str(tmp_path / "models")

    if use_latest_model:
        trained_rasa_model = str(Path(trained_rasa_model).parent)

    train(
        default_domain_path,
        default_stack_config,
        [default_stories_file, default_nlu_data],
        output=output,
        force_training=True,
        model_to_finetune=trained_rasa_model,
        finetuning_epoch_fraction=1,
    )

    mocked_core_training.assert_called_once()
    _, kwargs = mocked_core_training.call_args
    assert isinstance(kwargs["model_to_finetune"], Agent)

    mocked_nlu_training.assert_called_once()
    _, kwargs = mocked_nlu_training.call_args
    assert isinstance(kwargs["model_to_finetune"], Interpreter)


@pytest.mark.parametrize("use_latest_model", [True, False])
def test_model_finetuning_core(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    default_domain_path: Text,
    default_stories_file: Text,
    default_stack_config: Text,
    trained_rasa_model: Text,
    use_latest_model: bool,
):
    mocked_core_training = AsyncMock()
    monkeypatch.setattr(rasa.core, rasa.core.train.__name__, mocked_core_training)

    (tmp_path / "models").mkdir()
    output = str(tmp_path / "models")

    if use_latest_model:
        trained_rasa_model = str(Path(trained_rasa_model).parent)

    train_core(
        default_domain_path,
        default_stack_config,
        default_stories_file,
        output=output,
        model_to_finetune=trained_rasa_model,
        finetuning_epoch_fraction=1,
    )

    mocked_core_training.assert_called_once()
    _, kwargs = mocked_core_training.call_args
    assert isinstance(kwargs["model_to_finetune"], Agent)


@pytest.mark.parametrize("use_latest_model", [True, False])
def test_model_finetuning_nlu(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    default_domain_path: Text,
    default_nlu_data: Text,
    trained_moodbot_path: Text,
    use_latest_model: bool,
):
    mocked_nlu_training = AsyncMock(return_value="")
    monkeypatch.setattr(rasa.nlu, rasa.nlu.train.__name__, mocked_nlu_training)

    mock_interpreter_create = Mock(wraps=Interpreter.create)
    monkeypatch.setattr(Interpreter, "create", mock_interpreter_create)

    mock_DIET_load = Mock(wraps=DIETClassifier.load)
    monkeypatch.setattr(DIETClassifier, "load", mock_DIET_load)

    (tmp_path / "models").mkdir()
    output = str(tmp_path / "models")

    if use_latest_model:
        trained_moodbot_path = str(Path(trained_moodbot_path).parent)

    # Typically models will be fine-tuned with a smaller number of epochs than training
    # from scratch.
    # Fine-tuning will use the number of epochs in the new config.
    old_config = rasa.shared.utils.io.read_yaml_file("examples/moodbot/config.yml")
    old_config["pipeline"][-1][EPOCHS] = 10
    new_config_path = tmp_path / "new_config.yml"
    rasa.shared.utils.io.write_yaml(old_config, new_config_path)

    train_nlu(
        str(new_config_path),
        "examples/moodbot/data/nlu.yml",
        output=output,
        model_to_finetune=trained_moodbot_path,
        finetuning_epoch_fraction=0.5,
    )

    assert mock_interpreter_create.call_args[1]["should_finetune"]

    mocked_nlu_training.assert_called_once()
    _, nlu_train_kwargs = mocked_nlu_training.call_args
    model_to_finetune = nlu_train_kwargs["model_to_finetune"]
    assert isinstance(model_to_finetune, Interpreter)

    _, diet_kwargs = mock_DIET_load.call_args
    assert diet_kwargs["should_finetune"] is True

    new_diet_metadata = model_to_finetune.model_metadata.metadata["pipeline"][-1]
    assert new_diet_metadata["name"] == "DIETClassifier"
    assert new_diet_metadata[EPOCHS] == 5


def test_model_finetuning_nlu_with_default_epochs(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    default_domain_path: Text,
    default_nlu_data: Text,
    trained_moodbot_path: Text,
):
    mocked_nlu_training = AsyncMock(return_value="")
    monkeypatch.setattr(rasa.nlu, rasa.nlu.train.__name__, mocked_nlu_training)

    (tmp_path / "models").mkdir()
    output = str(tmp_path / "models")

    # Providing a new config with no epochs will mean the default amount are used
    # and then scaled by `finetuning_epoch_fraction`.
    old_config = rasa.shared.utils.io.read_yaml_file("examples/moodbot/config.yml")
    del old_config["pipeline"][-1][EPOCHS]
    new_config_path = tmp_path / "new_config.yml"
    rasa.shared.utils.io.write_yaml(old_config, new_config_path)

    train_nlu(
        str(new_config_path),
        "examples/moodbot/data/nlu.yml",
        output=output,
        model_to_finetune=trained_moodbot_path,
        finetuning_epoch_fraction=0.5,
    )

    mocked_nlu_training.assert_called_once()
    _, nlu_train_kwargs = mocked_nlu_training.call_args
    model_to_finetune = nlu_train_kwargs["model_to_finetune"]
    new_diet_metadata = model_to_finetune.model_metadata.metadata["pipeline"][-1]
    assert new_diet_metadata["name"] == "DIETClassifier"
    assert new_diet_metadata[EPOCHS] == DIETClassifier.defaults[EPOCHS] * 0.5


@pytest.mark.parametrize("model_to_fine_tune", ["invalid-path-to-model", "."])
def test_model_finetuning_with_invalid_model(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    default_domain_path: Text,
    default_stories_file: Text,
    default_stack_config: Text,
    default_nlu_data: Text,
    model_to_fine_tune: Text,
    capsys: CaptureFixture,
):
    mocked_nlu_training = AsyncMock(return_value="")
    monkeypatch.setattr(rasa.nlu, rasa.nlu.train.__name__, mocked_nlu_training)

    mocked_core_training = AsyncMock()
    monkeypatch.setattr(rasa.core, rasa.core.train.__name__, mocked_core_training)

    (tmp_path / "models").mkdir()
    output = str(tmp_path / "models")

    train(
        default_domain_path,
        default_stack_config,
        [default_stories_file, default_nlu_data],
        output=output,
        force_training=True,
        model_to_finetune=model_to_fine_tune,
        finetuning_epoch_fraction=1,
    )

    mocked_core_training.assert_called_once()
    _, kwargs = mocked_core_training.call_args
    assert kwargs["model_to_finetune"] is None

    mocked_nlu_training.assert_called_once()
    _, kwargs = mocked_nlu_training.call_args
    assert kwargs["model_to_finetune"] is None

    output = capsys.readouterr().out
    assert "No Core model for finetuning found" in output
    assert "No NLU model for finetuning found" in output


@pytest.mark.parametrize("model_to_fine_tune", ["invalid-path-to-model", "."])
def test_model_finetuning_with_invalid_model_core(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    default_domain_path: Text,
    default_stories_file: Text,
    default_stack_config: Text,
    model_to_fine_tune: Text,
    capsys: CaptureFixture,
):
    mocked_core_training = AsyncMock()
    monkeypatch.setattr(rasa.core, rasa.core.train.__name__, mocked_core_training)

    (tmp_path / "models").mkdir()
    output = str(tmp_path / "models")

    train_core(
        default_domain_path,
        default_stack_config,
        default_stories_file,
        output=output,
        model_to_finetune=model_to_fine_tune,
        finetuning_epoch_fraction=1,
    )

    mocked_core_training.assert_called_once()

    _, kwargs = mocked_core_training.call_args
    assert kwargs["model_to_finetune"] is None

    assert "No Core model for finetuning found" in capsys.readouterr().out


@pytest.mark.parametrize("model_to_fine_tune", ["invalid-path-to-model", "."])
def test_model_finetuning_with_invalid_model_nlu(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    default_domain_path: Text,
    default_stack_config: Text,
    default_nlu_data: Text,
    model_to_fine_tune: Text,
    capsys: CaptureFixture,
):
    mocked_nlu_training = AsyncMock(return_value="")
    monkeypatch.setattr(rasa.nlu, rasa.nlu.train.__name__, mocked_nlu_training)

    (tmp_path / "models").mkdir()
    output = str(tmp_path / "models")

    train_nlu(
        default_stack_config,
        default_nlu_data,
        domain=default_domain_path,
        output=output,
        model_to_finetune=model_to_fine_tune,
        finetuning_epoch_fraction=1,
    )

    mocked_nlu_training.assert_called_once()

    _, kwargs = mocked_nlu_training.call_args
    assert kwargs["model_to_finetune"] is None

    assert "No NLU model for finetuning found" in capsys.readouterr().out

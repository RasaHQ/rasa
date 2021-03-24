import logging
import secrets
import sys
import tempfile
import os
from pathlib import Path
from typing import Text, Dict, Any
from unittest.mock import Mock

import pytest
from _pytest.capture import CaptureFixture
from _pytest.logging import LogCaptureFixture
from _pytest.monkeypatch import MonkeyPatch

from rasa.core.policies.ted_policy import TEDPolicy
import rasa.model
import rasa.core
import rasa.core.train
import rasa.nlu
from rasa.nlu.classifiers.diet_classifier import DIETClassifier
import rasa.shared.importers.autoconfig as autoconfig
import rasa.shared.utils.io
from rasa.core.agent import Agent
from rasa.core.interpreter import RasaNLUInterpreter
from rasa.nlu.model import Interpreter

from rasa.model_training import train_core, train_nlu, train, dry_run_result
from rasa.utils.tensorflow.constants import EPOCHS
from tests.conftest import AsyncMock
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
    domain_path: Text,
    stories_path: Text,
    stack_config_path: Text,
    nlu_data_path: Text,
):
    (tmp_path / "training").mkdir()
    (tmp_path / "models").mkdir()

    monkeypatch.setattr(tempfile, "tempdir", tmp_path / "training")
    output = str(tmp_path / "models")

    train(
        domain_path,
        stack_config_path,
        [stories_path, nlu_data_path],
        output=output,
        force_training=True,
    )

    assert count_temp_rasa_files(tempfile.tempdir) == 0

    # After training the model, try to do it again. This shouldn't try to train
    # a new model because nothing has been changed. It also shouldn't create
    # any temp files.
    train(
        domain_path, stack_config_path, [stories_path, nlu_data_path], output=output,
    )

    assert count_temp_rasa_files(tempfile.tempdir) == 0


def test_train_core_temp_files(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    domain_path: Text,
    stories_path: Text,
    stack_config_path: Text,
):
    (tmp_path / "training").mkdir()
    (tmp_path / "models").mkdir()

    monkeypatch.setattr(tempfile, "tempdir", tmp_path / "training")

    train_core(
        domain_path, stack_config_path, stories_path, output=str(tmp_path / "models"),
    )

    assert count_temp_rasa_files(tempfile.tempdir) == 0


def test_train_nlu_temp_files(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    stack_config_path: Text,
    nlu_data_path: Text,
):
    (tmp_path / "training").mkdir()
    (tmp_path / "models").mkdir()

    monkeypatch.setattr(tempfile, "tempdir", tmp_path / "training")

    train_nlu(stack_config_path, nlu_data_path, output=str(tmp_path / "models"))

    assert count_temp_rasa_files(tempfile.tempdir) == 0


def test_train_nlu_wrong_format_error_message(
    capsys: CaptureFixture,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    stack_config_path: Text,
    incorrect_nlu_data_path: Text,
):
    (tmp_path / "training").mkdir()
    (tmp_path / "models").mkdir()

    monkeypatch.setattr(tempfile, "tempdir", tmp_path / "training")

    train_nlu(
        stack_config_path, incorrect_nlu_data_path, output=str(tmp_path / "models")
    )

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
    stack_config_path: Text,
):
    (tmp_path / "training").mkdir()
    (tmp_path / "models").mkdir()

    monkeypatch.setattr(tempfile, "tempdir", tmp_path / "training")

    train_nlu(stack_config_path, "", output=str(tmp_path / "models"))

    captured = capsys.readouterr()
    assert "No NLU data given" in captured.out


def test_trained_interpreter_passed_to_core_training(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    unpacked_trained_rasa_model: Text,
    nlu_data_path: Text,
    stories_path: Text,
    config_path: Text,
    domain_path: Text,
):
    # Skip actual NLU training and return trained interpreter path from fixture
    # Patching is bit more complicated as we have a module `train` and function
    # with the same name 😬
    monkeypatch.setattr(
        rasa.model_training,
        "_train_nlu_with_validated_data",
        AsyncMock(return_value=unpacked_trained_rasa_model),
    )

    # Mock the actual Core training
    _train_core = mock_core_training(monkeypatch)

    train(
        domain_path, config_path, [stories_path, nlu_data_path], str(tmp_path),
    )

    _train_core.assert_called_once()
    _, _, kwargs = _train_core.mock_calls[0]
    assert isinstance(kwargs["interpreter"], RasaNLUInterpreter)


def test_interpreter_of_old_model_passed_to_core_training(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    trained_rasa_model: Text,
    domain_path: Text,
    config_path: Text,
    stories_path: Text,
    nlu_data_path: Text,
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
    _train_core = mock_core_training(monkeypatch)

    train(
        domain_path, config_path, [stories_path, nlu_data_path], str(tmp_path),
    )

    _train_core.assert_called_once()
    _, _, kwargs = _train_core.mock_calls[0]
    assert isinstance(kwargs["interpreter"], RasaNLUInterpreter)


def test_load_interpreter_returns_none_for_none():
    from rasa.model_training import _load_interpreter

    assert _load_interpreter(None) is None


def test_interpreter_from_previous_model_returns_none_for_none():
    from rasa.model_training import _interpreter_from_previous_model

    assert _interpreter_from_previous_model(None) is None


def test_train_core_autoconfig(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    domain_path: Text,
    stories_path: Text,
    stack_config_path: Text,
):
    monkeypatch.setattr(tempfile, "tempdir", tmp_path)

    # mock function that returns configuration
    mocked_get_configuration = Mock()
    monkeypatch.setattr(autoconfig, "get_configuration", mocked_get_configuration)

    # skip actual core training
    monkeypatch.setattr(
        rasa.model_training, "_train_core_with_validated_data", AsyncMock()
    )

    # do training
    train_core(
        domain_path,
        stack_config_path,
        stories_path,
        output="test_train_core_temp_files_models",
    )

    mocked_get_configuration.assert_called_once()
    _, args, _ = mocked_get_configuration.mock_calls[0]
    assert args[1] == autoconfig.TrainingType.CORE


def test_train_nlu_autoconfig(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    stack_config_path: Text,
    nlu_data_path: Text,
):
    monkeypatch.setattr(tempfile, "tempdir", tmp_path)

    # mock function that returns configuration
    mocked_get_configuration = Mock()
    monkeypatch.setattr(autoconfig, "get_configuration", mocked_get_configuration)

    monkeypatch.setattr(
        rasa.model_training, "_train_nlu_with_validated_data", AsyncMock()
    )

    # do training
    train_nlu(
        stack_config_path, nlu_data_path, output="test_train_nlu_temp_files_models",
    )

    mocked_get_configuration.assert_called_once()
    _, args, _ = mocked_get_configuration.mock_calls[0]
    assert args[1] == autoconfig.TrainingType.NLU


def mock_async(monkeypatch: MonkeyPatch, target: Any, name: Text) -> Mock:
    mock = Mock()

    async def mock_async_func(*args: Any, **kwargs: Any) -> None:
        mock(*args, **kwargs)

    monkeypatch.setattr(target, name, mock_async_func)
    return mock


def mock_core_training(monkeypatch: MonkeyPatch) -> Mock:
    return mock_async(monkeypatch, rasa.core.train, rasa.core.train.train.__name__)


def mock_nlu_training(monkeypatch: MonkeyPatch) -> Mock:
    return mock_async(monkeypatch, rasa.nlu.train, rasa.nlu.train.train.__name__)


def new_model_path_in_same_dir(old_model_path: Text) -> Text:
    return str(Path(old_model_path).parent / (secrets.token_hex(8) + ".tar.gz"))


class TestE2e:
    def test_e2e_gives_experimental_warning(
        self,
        monkeypatch: MonkeyPatch,
        trained_e2e_model: Text,
        domain_path: Text,
        stack_config_path: Text,
        e2e_stories_path: Text,
        nlu_data_path: Text,
        caplog: LogCaptureFixture,
    ):
        mock_nlu_training(monkeypatch)
        mock_core_training(monkeypatch)

        with caplog.at_level(logging.WARNING):
            train(
                domain_path,
                stack_config_path,
                [e2e_stories_path, nlu_data_path],
                output=new_model_path_in_same_dir(trained_e2e_model),
            )

        assert any(
            [
                "The end-to-end training is currently experimental" in record.message
                for record in caplog.records
            ]
        )

    def test_models_not_retrained_if_no_new_data(
        self,
        monkeypatch: MonkeyPatch,
        trained_e2e_model: Text,
        domain_path: Text,
        stack_config_path: Text,
        e2e_stories_path: Text,
        nlu_data_path: Text,
    ):
        mocked_nlu_training = mock_nlu_training(monkeypatch)
        mocked_core_training = mock_core_training(monkeypatch)

        train(
            domain_path,
            stack_config_path,
            [e2e_stories_path, nlu_data_path],
            output=new_model_path_in_same_dir(trained_e2e_model),
        )

        mocked_core_training.assert_not_called()
        mocked_nlu_training.assert_not_called()

    def test_retrains_nlu_and_core_if_new_e2e_example(
        self,
        monkeypatch: MonkeyPatch,
        trained_e2e_model: Text,
        domain_path: Text,
        stack_config_path: Text,
        e2e_stories_path: Text,
        nlu_data_path: Text,
        tmp_path: Path,
    ):
        stories_yaml = rasa.shared.utils.io.read_yaml_file(e2e_stories_path)
        stories_yaml["stories"][1]["steps"].append({"user": "new message!"})

        new_stories_file = tmp_path / "new_stories.yml"
        rasa.shared.utils.io.write_yaml(stories_yaml, new_stories_file)

        mocked_nlu_training = mock_nlu_training(monkeypatch)
        mocked_core_training = mock_core_training(monkeypatch)

        new_model_path = train(
            domain_path,
            stack_config_path,
            [new_stories_file, nlu_data_path],
            output=new_model_path_in_same_dir(trained_e2e_model),
        ).model
        os.remove(new_model_path)

        mocked_core_training.assert_called_once()
        mocked_nlu_training.assert_called_once()

    def test_retrains_only_core_if_new_e2e_example_seen_before(
        self,
        monkeypatch: MonkeyPatch,
        trained_e2e_model: Text,
        domain_path: Text,
        stack_config_path: Text,
        e2e_stories_path: Text,
        nlu_data_path: Text,
        tmp_path: Path,
    ):
        stories_yaml = rasa.shared.utils.io.read_yaml_file(e2e_stories_path)
        stories_yaml["stories"][1]["steps"].append({"user": "Yes"})

        new_stories_file = new_stories_file = tmp_path / "new_stories.yml"
        rasa.shared.utils.io.write_yaml(stories_yaml, new_stories_file)

        mocked_nlu_training = mock_nlu_training(monkeypatch)
        mocked_core_training = mock_core_training(monkeypatch)

        new_model_path = train(
            domain_path,
            stack_config_path,
            [new_stories_file, nlu_data_path],
            output=new_model_path_in_same_dir(trained_e2e_model),
        ).model
        os.remove(new_model_path)

        mocked_core_training.assert_called_once()
        mocked_nlu_training.assert_not_called()

    def test_nlu_and_core_trained_if_no_nlu_data_but_e2e_stories(
        self,
        monkeypatch: MonkeyPatch,
        domain_path: Text,
        stack_config_path: Text,
        e2e_stories_path: Text,
        tmp_path: Path,
    ):
        mocked_nlu_training = mock_nlu_training(monkeypatch)
        mocked_core_training = mock_core_training(monkeypatch)

        output = self.make_tmp_model_dir(tmp_path)
        train(
            domain_path, stack_config_path, [e2e_stories_path], output=output,
        )

        mocked_core_training.assert_called_once()
        mocked_nlu_training.assert_called_once()

    @staticmethod
    def make_tmp_model_dir(tmp_path: Path) -> Text:
        (tmp_path / "models").mkdir()
        output = str(tmp_path / "models")
        return output

    def test_new_nlu_data_retrains_core_if_there_are_e2e_stories(
        self,
        monkeypatch: MonkeyPatch,
        trained_e2e_model: Text,
        domain_path: Text,
        stack_config_path: Text,
        e2e_stories_path: Text,
        nlu_data_path: Text,
        tmp_path: Path,
    ):
        nlu_yaml = rasa.shared.utils.io.read_yaml_file(nlu_data_path)
        nlu_yaml["nlu"][0]["examples"] += "- surprise!\n"

        new_nlu_file = tmp_path / "new_nlu.yml"
        rasa.shared.utils.io.write_yaml(nlu_yaml, new_nlu_file)

        mocked_nlu_training = mock_nlu_training(monkeypatch)
        mocked_core_training = mock_core_training(monkeypatch)

        new_model_path = train(
            domain_path,
            stack_config_path,
            [e2e_stories_path, new_nlu_file],
            output=new_model_path_in_same_dir(trained_e2e_model),
        ).model
        os.remove(new_model_path)

        mocked_core_training.assert_called_once()
        mocked_nlu_training.assert_called_once()

    def test_new_nlu_data_does_not_retrain_core_if_there_are_no_e2e_stories(
        self,
        monkeypatch: MonkeyPatch,
        trained_simple_rasa_model: Text,
        domain_path: Text,
        stack_config_path: Text,
        simple_stories_path: Text,
        nlu_data_path: Text,
        tmp_path: Path,
    ):
        nlu_yaml = rasa.shared.utils.io.read_yaml_file(nlu_data_path)
        nlu_yaml["nlu"][0]["examples"] += "- surprise!\n"

        new_nlu_file = tmp_path / "new_nlu.yml"
        rasa.shared.utils.io.write_yaml(nlu_yaml, new_nlu_file)

        mocked_nlu_training = mock_nlu_training(monkeypatch)
        mocked_core_training = mock_core_training(monkeypatch)

        new_model_path = train(
            domain_path,
            stack_config_path,
            [simple_stories_path, new_nlu_file],
            output=new_model_path_in_same_dir(trained_simple_rasa_model),
        ).model
        os.remove(new_model_path)

        mocked_core_training.assert_not_called()
        mocked_nlu_training.assert_called_once()

    def test_training_core_with_e2e_fails_gracefully(
        self,
        capsys: CaptureFixture,
        monkeypatch: MonkeyPatch,
        tmp_path: Path,
        domain_path: Text,
        stack_config_path: Text,
        e2e_stories_path: Text,
    ):

        mocked_nlu_training = mock_nlu_training(monkeypatch)
        mocked_core_training = mock_core_training(monkeypatch)

        output = self.make_tmp_model_dir(tmp_path)
        train_core(
            domain_path, stack_config_path, e2e_stories_path, output=output,
        )

        mocked_core_training.assert_not_called()
        mocked_nlu_training.assert_not_called()

        captured = capsys.readouterr()
        assert (
            "Stories file contains e2e stories. "
            "Please train using `rasa train` so that the NLU model is also trained."
        ) in captured.out


@pytest.mark.timeout(300)
@pytest.mark.parametrize("use_latest_model", [True, False])
def test_model_finetuning(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    domain_path: Text,
    stories_path: Text,
    stack_config_path: Text,
    nlu_data_path: Text,
    trained_rasa_model: Text,
    use_latest_model: bool,
):
    mocked_nlu_training = mock_nlu_training(monkeypatch)
    mocked_core_training = mock_core_training(monkeypatch)

    (tmp_path / "models").mkdir()
    output = str(tmp_path / "models")

    if use_latest_model:
        trained_rasa_model = str(Path(trained_rasa_model).parent)

    train(
        domain_path,
        stack_config_path,
        [stories_path, nlu_data_path],
        output=output,
        force_training=True,
        model_to_finetune=trained_rasa_model,
        finetuning_epoch_fraction=0.1,
    )

    mocked_core_training.assert_called_once()
    _, kwargs = mocked_core_training.call_args
    assert isinstance(kwargs["model_to_finetune"], Agent)

    mocked_nlu_training.assert_called_once()
    _, kwargs = mocked_nlu_training.call_args
    assert isinstance(kwargs["model_to_finetune"], Interpreter)


@pytest.mark.timeout(300)
@pytest.mark.parametrize("use_latest_model", [True, False])
def test_model_finetuning_core(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    trained_moodbot_path: Text,
    use_latest_model: bool,
):
    mocked_core_training = mock_core_training(monkeypatch)
    mock_agent_load = Mock(wraps=Agent.load)
    monkeypatch.setattr(Agent, "load", mock_agent_load)

    (tmp_path / "models").mkdir()
    output = str(tmp_path / "models")

    if use_latest_model:
        trained_moodbot_path = str(Path(trained_moodbot_path).parent)

    # Typically models will be fine-tuned with a smaller number of epochs than training
    # from scratch.
    # Fine-tuning will use the number of epochs in the new config.
    old_config = rasa.shared.utils.io.read_yaml_file("data/test_moodbot/config.yml")
    old_config["policies"][0]["epochs"] = 10
    new_config_path = tmp_path / "new_config.yml"
    rasa.shared.utils.io.write_yaml(old_config, new_config_path)

    old_stories = rasa.shared.utils.io.read_yaml_file(
        "data/test_moodbot/data/stories.yml"
    )
    old_stories["stories"].append(
        {"story": "new story", "steps": [{"intent": "greet"}]}
    )
    new_stories_path = tmp_path / "new_stories.yml"
    rasa.shared.utils.io.write_yaml(old_stories, new_stories_path)

    train_core(
        "data/test_moodbot/domain.yml",
        str(new_config_path),
        str(new_stories_path),
        output=output,
        model_to_finetune=trained_moodbot_path,
        finetuning_epoch_fraction=0.2,
    )

    mocked_core_training.assert_called_once()
    _, kwargs = mocked_core_training.call_args
    model_to_finetune = kwargs["model_to_finetune"]
    assert isinstance(model_to_finetune, Agent)

    ted = model_to_finetune.policy_ensemble.policies[0]
    assert ted.config[EPOCHS] == 2
    assert ted.finetune_mode


def test_model_finetuning_core_with_default_epochs(
    tmp_path: Path, monkeypatch: MonkeyPatch, trained_moodbot_path: Text,
):
    mocked_core_training = mock_core_training(monkeypatch)
    (tmp_path / "models").mkdir()
    output = str(tmp_path / "models")

    # Providing a new config with no epochs will mean the default amount are used
    # and then scaled by `finetuning_epoch_fraction`.
    old_config = rasa.shared.utils.io.read_yaml_file("data/test_moodbot/config.yml")
    del old_config["policies"][0]["epochs"]
    new_config_path = tmp_path / "new_config.yml"
    rasa.shared.utils.io.write_yaml(old_config, new_config_path)

    train_core(
        "data/test_moodbot/domain.yml",
        str(new_config_path),
        "data/test_moodbot/data/stories.yml",
        output=output,
        model_to_finetune=trained_moodbot_path,
        finetuning_epoch_fraction=2,
    )

    mocked_core_training.assert_called_once()
    _, kwargs = mocked_core_training.call_args
    model_to_finetune = kwargs["model_to_finetune"]

    ted = model_to_finetune.policy_ensemble.policies[0]
    assert ted.config[EPOCHS] == TEDPolicy.defaults[EPOCHS] * 2


def test_model_finetuning_core_new_domain_label(
    tmp_path: Path, monkeypatch: MonkeyPatch, trained_moodbot_path: Text,
):
    mocked_core_training = mock_core_training(monkeypatch)

    (tmp_path / "models").mkdir()
    output = str(tmp_path / "models")

    # Simulate addition to training data
    old_domain = rasa.shared.utils.io.read_yaml_file("data/test_moodbot/domain.yml")
    old_domain["intents"].append("a_new_one")
    new_domain_path = tmp_path / "new_domain.yml"
    rasa.shared.utils.io.write_yaml(old_domain, new_domain_path)

    with pytest.raises(SystemExit):
        train_core(
            domain=str(new_domain_path),
            config="data/test_moodbot/config.yml",
            stories="data/test_moodbot/data/stories.yml",
            output=output,
            model_to_finetune=trained_moodbot_path,
        )

    mocked_core_training.assert_not_called()


def test_model_finetuning_new_domain_label_stops_all_training(
    tmp_path: Path, monkeypatch: MonkeyPatch, trained_moodbot_path: Text,
):
    mocked_core_training = mock_core_training(monkeypatch)
    mocked_nlu_training = mock_nlu_training(monkeypatch)

    (tmp_path / "models").mkdir()
    output = str(tmp_path / "models")

    old_domain = rasa.shared.utils.io.read_yaml_file("data/test_moodbot/domain.yml")
    old_domain["intents"].append("a_new_one")
    new_domain_path = tmp_path / "new_domain.yml"
    rasa.shared.utils.io.write_yaml(old_domain, new_domain_path)

    with pytest.raises(SystemExit):
        train(
            domain=str(new_domain_path),
            config="data/test_moodbot/config.yml",
            training_files=[
                "data/test_moodbot/data/stories.yml",
                "data/test_moodbot/data/nlu.yml",
            ],
            output=output,
            model_to_finetune=trained_moodbot_path,
        )

    mocked_core_training.assert_not_called()
    mocked_nlu_training.assert_not_called()


@pytest.mark.timeout(300)
@pytest.mark.parametrize("use_latest_model", [True, False])
def test_model_finetuning_nlu(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    trained_nlu_moodbot_path: Text,
    use_latest_model: bool,
):
    mocked_nlu_training = mock_nlu_training(monkeypatch)

    mock_interpreter_create = Mock(wraps=Interpreter.create)
    monkeypatch.setattr(Interpreter, "create", mock_interpreter_create)

    mock_DIET_load = Mock(wraps=DIETClassifier.load)
    monkeypatch.setattr(DIETClassifier, "load", mock_DIET_load)

    (tmp_path / "models").mkdir()
    output = str(tmp_path / "models")

    if use_latest_model:
        trained_nlu_moodbot_path = str(Path(trained_nlu_moodbot_path).parent)

    # Typically models will be fine-tuned with a smaller number of epochs than training
    # from scratch.
    # Fine-tuning will use the number of epochs in the new config.
    old_config = rasa.shared.utils.io.read_yaml_file("data/test_moodbot/config.yml")
    old_config["pipeline"][-1][EPOCHS] = 10
    new_config_path = tmp_path / "new_config.yml"
    rasa.shared.utils.io.write_yaml(old_config, new_config_path)

    old_nlu = rasa.shared.utils.io.read_yaml_file("data/test_moodbot/data/nlu.yml")
    old_nlu["nlu"][-1]["examples"] = "-something else"
    new_nlu_path = tmp_path / "new_nlu.yml"
    rasa.shared.utils.io.write_yaml(old_nlu, new_nlu_path)

    train_nlu(
        str(new_config_path),
        str(new_nlu_path),
        domain="data/test_moodbot/domain.yml",
        output=output,
        model_to_finetune=trained_nlu_moodbot_path,
        finetuning_epoch_fraction=0.2,
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
    assert new_diet_metadata[EPOCHS] == 2


def test_model_finetuning_nlu_new_label(
    tmp_path: Path, monkeypatch: MonkeyPatch, trained_nlu_moodbot_path: Text,
):
    mocked_nlu_training = mock_nlu_training(monkeypatch)

    (tmp_path / "models").mkdir()
    output = str(tmp_path / "models")

    old_nlu = rasa.shared.utils.io.read_yaml_file("data/test_moodbot/data/nlu.yml")
    old_nlu["nlu"].append({"intent": "a_new_one", "examples": "-blah"})
    new_nlu_path = tmp_path / "new_nlu.yml"
    rasa.shared.utils.io.write_yaml(old_nlu, new_nlu_path)

    with pytest.raises(SystemExit):
        train_nlu(
            "data/test_moodbot/config.yml",
            str(new_nlu_path),
            domain="data/test_moodbot/domain.yml",
            output=output,
            model_to_finetune=trained_nlu_moodbot_path,
        )

    mocked_nlu_training.assert_not_called()


def test_model_finetuning_nlu_new_entity(
    tmp_path: Path, monkeypatch: MonkeyPatch, trained_nlu_moodbot_path: Text,
):
    mocked_nlu_training = mock_nlu_training(monkeypatch)

    (tmp_path / "models").mkdir()
    output = str(tmp_path / "models")

    old_nlu = rasa.shared.utils.io.read_yaml_file("data/test_moodbot/data/nlu.yml")
    old_nlu["nlu"][-1]["examples"] = "-[blah](something)"
    new_nlu_path = tmp_path / "new_nlu.yml"
    rasa.shared.utils.io.write_yaml(old_nlu, new_nlu_path)

    with pytest.raises(SystemExit):
        train_nlu(
            "data/test_moodbot/config.yml",
            str(new_nlu_path),
            domain="data/test_moodbot/domain.yml",
            output=output,
            model_to_finetune=trained_nlu_moodbot_path,
        )

    mocked_nlu_training.assert_not_called()


def test_model_finetuning_nlu_new_label_already_in_domain(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    trained_rasa_model: Text,
    nlu_data_path: Text,
    config_path: Text,
    domain_path: Text,
):
    mocked_nlu_training = mock_nlu_training(monkeypatch)

    (tmp_path / "models").mkdir()
    output = str(tmp_path / "models")

    old_nlu = rasa.shared.utils.io.read_yaml_file(nlu_data_path)
    # This intent exists in `domain_path` but not yet in the nlu data
    old_nlu["nlu"].append({"intent": "why", "examples": "whyy??"})
    new_nlu_path = tmp_path / "new_nlu.yml"
    rasa.shared.utils.io.write_yaml(old_nlu, new_nlu_path)

    with pytest.raises(SystemExit):
        train_nlu(
            config_path,
            str(new_nlu_path),
            domain=domain_path,
            output=output,
            model_to_finetune=trained_rasa_model,
        )

    mocked_nlu_training.assert_not_called()


def test_model_finetuning_nlu_new_label_to_domain_only(
    tmp_path: Path, monkeypatch: MonkeyPatch, trained_nlu_moodbot_path: Text,
):
    mocked_nlu_training = mock_nlu_training(monkeypatch)

    (tmp_path / "models").mkdir()
    output = str(tmp_path / "models")

    old_domain = rasa.shared.utils.io.read_yaml_file("data/test_moodbot/domain.yml")
    old_domain["intents"].append("a_new_one")
    new_domain_path = tmp_path / "new_domain.yml"
    rasa.shared.utils.io.write_yaml(old_domain, new_domain_path)

    train_nlu(
        "data/test_moodbot/config.yml",
        "data/test_moodbot/data/nlu.yml",
        domain=str(new_domain_path),
        output=output,
        model_to_finetune=trained_nlu_moodbot_path,
    )

    mocked_nlu_training.assert_called()


@pytest.mark.timeout(200)
def test_model_finetuning_nlu_with_default_epochs(
    tmp_path: Path, monkeypatch: MonkeyPatch, trained_nlu_moodbot_path: Text,
):
    mocked_nlu_training = mock_nlu_training(monkeypatch)

    (tmp_path / "models").mkdir()
    output = str(tmp_path / "models")

    # Providing a new config with no epochs will mean the default amount are used
    # and then scaled by `finetuning_epoch_fraction`.
    old_config = rasa.shared.utils.io.read_yaml_file("data/test_moodbot/config.yml")
    del old_config["pipeline"][-1][EPOCHS]
    new_config_path = tmp_path / "new_config.yml"
    rasa.shared.utils.io.write_yaml(old_config, new_config_path)

    train_nlu(
        str(new_config_path),
        "data/test_moodbot/data/nlu.yml",
        output=output,
        model_to_finetune=trained_nlu_moodbot_path,
        finetuning_epoch_fraction=0.1,
    )

    mocked_nlu_training.assert_called_once()
    _, nlu_train_kwargs = mocked_nlu_training.call_args
    model_to_finetune = nlu_train_kwargs["model_to_finetune"]
    new_diet_metadata = model_to_finetune.model_metadata.metadata["pipeline"][-1]
    assert new_diet_metadata["name"] == "DIETClassifier"
    assert new_diet_metadata[EPOCHS] == DIETClassifier.defaults[EPOCHS] * 0.1


@pytest.mark.parametrize("model_to_fine_tune", ["invalid-path-to-model", "."])
def test_model_finetuning_with_invalid_model(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    domain_path: Text,
    stories_path: Text,
    stack_config_path: Text,
    nlu_data_path: Text,
    model_to_fine_tune: Text,
    capsys: CaptureFixture,
):
    mocked_nlu_training = mock_nlu_training(monkeypatch)

    mocked_core_training = mock_core_training(monkeypatch)
    (tmp_path / "models").mkdir()
    output = str(tmp_path / "models")

    with pytest.raises(SystemExit):
        train(
            domain_path,
            stack_config_path,
            [stories_path, nlu_data_path],
            output=output,
            force_training=True,
            model_to_finetune=model_to_fine_tune,
            finetuning_epoch_fraction=1,
        )

    mocked_core_training.assert_not_called()
    mocked_nlu_training.assert_not_called()
    output = capsys.readouterr().out
    assert "No NLU model for finetuning found" in output


@pytest.mark.parametrize("model_to_fine_tune", ["invalid-path-to-model", "."])
def test_model_finetuning_with_invalid_model_core(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    domain_path: Text,
    stories_path: Text,
    stack_config_path: Text,
    model_to_fine_tune: Text,
    capsys: CaptureFixture,
):
    mocked_core_training = mock_core_training(monkeypatch)
    (tmp_path / "models").mkdir()
    output = str(tmp_path / "models")

    with pytest.raises(SystemExit):
        train_core(
            domain_path,
            stack_config_path,
            stories_path,
            output=output,
            model_to_finetune=model_to_fine_tune,
            finetuning_epoch_fraction=1,
        )

    mocked_core_training.assert_not_called()

    assert "No Core model for finetuning found" in capsys.readouterr().out


@pytest.mark.parametrize("model_to_fine_tune", ["invalid-path-to-model", "."])
def test_model_finetuning_with_invalid_model_nlu(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    domain_path: Text,
    stack_config_path: Text,
    nlu_data_path: Text,
    model_to_fine_tune: Text,
    capsys: CaptureFixture,
):
    mocked_nlu_training = mock_nlu_training(monkeypatch)

    (tmp_path / "models").mkdir()
    output = str(tmp_path / "models")

    with pytest.raises(SystemExit):
        train_nlu(
            stack_config_path,
            nlu_data_path,
            domain=domain_path,
            output=output,
            model_to_finetune=model_to_fine_tune,
            finetuning_epoch_fraction=1,
        )

    mocked_nlu_training.assert_not_called()

    assert "No NLU model for finetuning found" in capsys.readouterr().out


@pytest.mark.parametrize(
    "result, code, texts_count",
    [
        (
            rasa.model.FingerprintComparisonResult(
                core=False, nlu=False, nlg=False, force_training=True
            ),
            0b1000,
            1,
        ),
        (
            rasa.model.FingerprintComparisonResult(
                core=True, nlu=True, nlg=True, force_training=True
            ),
            0b1000,
            1,
        ),
        (
            rasa.model.FingerprintComparisonResult(
                core=False, nlu=False, nlg=True, force_training=False
            ),
            0b0100,
            1,
        ),
        (
            rasa.model.FingerprintComparisonResult(
                core=True, nlu=True, nlg=True, force_training=False
            ),
            0b0111,
            3,
        ),
        (
            rasa.model.FingerprintComparisonResult(
                core=False, nlu=False, nlg=False, force_training=False
            ),
            0,
            1,
        ),
    ],
)
def test_dry_run_result(
    result: rasa.model.FingerprintComparisonResult, code: int, texts_count: int,
):
    result_code, texts = dry_run_result(result)
    assert result_code == code
    assert len(texts) == texts_count

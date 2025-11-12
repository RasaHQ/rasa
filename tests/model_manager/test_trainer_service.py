import base64
import os
import subprocess
from pathlib import Path
from typing import Dict, Text
from unittest import mock

import pytest
from pytest import MonkeyPatch

import rasa.model_manager.config
from rasa.model_manager.trainer_service import (
    TrainingSession,
    cache_for_assistant_path,
    complete_training,
    persist_rasa_cache,
    prepare_training_directory,
    run_training,
    seed_training_directory_with_rasa_cache,
    terminate_training,
    train_path,
    update_training_status,
    write_encoded_data_to_file,
    write_training_data_to_files,
)
from rasa.shared.constants import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_ENDPOINTS_PATH,
    DEFAULT_PROMPTS_PATH,
)
from rasa.shared.utils.yaml import read_yaml
from rasa.studio.prompts import (
    COMMAND_GENERATOR_NAME,
    CONTEXTUAL_RESPONSE_REPHRASER_NAME,
    ENTERPRISE_SEARCH_NAME,
)


def _encode(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("utf-8")


@pytest.fixture
def training_session() -> TrainingSession:
    process = mock.Mock(spec=subprocess.Popen)
    process.returncode = 0
    return TrainingSession(
        training_id="test_training_id",
        assistant_id="test_assistant_id",
        client_id="test_client_id",
        progress=0,
        status="running",
        process=process,
        model_name="test_model_name",
        log_id="test_42",
    )


@pytest.fixture
def config_with_cg_and_es() -> str:
    return """recipe: default.v1
language: en
pipeline:
- name: SingleStepLLMCommandGenerator
  llm:
    provider: groq
    model: llama3-8b-8192

policies:
- name: FlowPolicy
- name: EnterpriseSearchPolicy

assistant_id: 20240418-073244-narrow-archive
"""


def test_train_path() -> None:
    training_id = "test_training_id"
    expected_path = Path("trainings") / "test_training_id"
    assert train_path(training_id).endswith(str(expected_path))


def test_cache_for_assistant_path() -> None:
    assistant_id = "test_assistant_id"
    expected_path = Path("caches") / assistant_id
    assert cache_for_assistant_path(assistant_id).endswith(str(expected_path))


def test_write_encoded_data_to_file(tmp_path: Path) -> None:
    encoded_data = base64.b64encode(b"test data")
    file_path = tmp_path / "test_file.txt"
    write_encoded_data_to_file(encoded_data, str(file_path))
    with open(file_path, "r") as f:
        assert f.read() == "test data"


def test_terminate_training(training_session: TrainingSession) -> None:
    terminate_training(training_session)
    training_session.process.terminate.assert_called_once()  # type: ignore[attr-defined]
    assert training_session.status == "stopped"


def test_terminate_on_stopped_training() -> None:
    training_session = TrainingSession(
        training_id="test_training_id",
        assistant_id="test_assistant_id",
        client_id="test_client_id",
        progress=0,
        status="stopped",
        process=mock.Mock(spec=subprocess.Popen),
        model_name="test_model_name",
        log_id="test_42",
    )
    terminate_training(training_session)
    # check that the process was not terminated again
    assert training_session.process.terminate.call_count == 0  # type: ignore[attr-defined]


def test_update_training_status_on_ended_process(
    training_session: TrainingSession,
) -> None:
    training_session.process.poll.return_value = 0  # type: ignore[attr-defined]
    update_training_status(training_session)
    assert training_session.status == "done"


def test_update_training_status_on_running_process(
    training_session: TrainingSession,
) -> None:
    training_session.process.poll.return_value = None  # type: ignore[attr-defined]
    update_training_status(training_session)
    assert training_session.status == "running"


def test_complete_training(training_session: TrainingSession) -> None:
    complete_training(training_session)
    assert training_session.status == "done"
    assert training_session.progress == 100


def test_seed_training_directory_with_rasa_cache(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setattr(
        rasa.model_manager.config, "SERVER_BASE_WORKING_DIRECTORY", str(tmp_path)
    )
    assistant_id = "test_assistant_id"
    training_base_path = train_path(assistant_id)
    cache_path = cache_for_assistant_path(assistant_id)
    os.makedirs(cache_path)
    seed_training_directory_with_rasa_cache(training_base_path, assistant_id)
    assert os.path.exists(f"{training_base_path}/.rasa")


def test_persist_rasa_cache(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    # patch the "SERVER_BASE_WORKING_DIRECTORY" import in trainer_service
    monkeypatch.setattr(
        rasa.model_manager.config, "SERVER_BASE_WORKING_DIRECTORY", str(tmp_path)
    )
    assistant_id = "test_assistant_id_2"
    training_base_path = train_path(assistant_id)
    cache_path = cache_for_assistant_path(assistant_id)
    os.makedirs(f"{training_base_path}/.rasa")
    persist_rasa_cache(assistant_id, training_base_path)
    assert os.path.exists(cache_path)


def test_persist_rasa_cache_if_no_cache_exists(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setattr(
        rasa.model_manager.config, "SERVER_BASE_WORKING_DIRECTORY", str(tmp_path)
    )
    assistant_id = "test_assistant_id_3"
    training_base_path = train_path(assistant_id)
    cache_path = cache_for_assistant_path(assistant_id)
    persist_rasa_cache(assistant_id, training_base_path)
    assert not os.path.exists(cache_path)


def test_write_training_data_to_files(tmp_path: Path) -> None:
    prompts = {COMMAND_GENERATOR_NAME: "custom prompt"}
    encoded_training_data = {
        "domain": base64.b64encode(b"domain data").decode("utf-8"),
        "credentials": base64.b64encode(b"credentials data").decode("utf-8"),
        "endpoints": base64.b64encode(b"nlg:").decode("utf-8"),
        "flows": base64.b64encode(b"flows data").decode("utf-8"),
        "config": base64.b64encode(b"pipeline: []").decode("utf-8"),
        "stories": base64.b64encode(b"stories data").decode("utf-8"),
        "rules": base64.b64encode(b"rules data").decode("utf-8"),
        "nlu": base64.b64encode(b"nlu data").decode("utf-8"),
        "prompts": prompts,
    }

    training_base_path = str(tmp_path / "training")
    write_training_data_to_files(encoded_training_data, training_base_path)
    with open(f"{training_base_path}/domain.yml", "r") as f:
        assert f.read() == "domain data"
    with open(f"{training_base_path}/credentials.yml", "r") as f:
        assert f.read() == "credentials data"
    with open(f"{training_base_path}/endpoints.yml", "r") as f:
        assert f.read() == "nlg:"
    with open(f"{training_base_path}/data/flows.yml", "r") as f:
        assert f.read() == "flows data"
    with open(f"{training_base_path}/config.yml", "r") as f:
        assert f.read() == "pipeline: []\n"
    with open(f"{training_base_path}/data/stories.yml", "r") as f:
        assert f.read() == "stories data"
    with open(f"{training_base_path}/data/rules.yml", "r") as f:
        assert f.read() == "rules data"
    with open(f"{training_base_path}/data/nlu.yml", "r") as f:
        assert f.read() == "nlu data"

    with open(
        f"{training_base_path}/{DEFAULT_PROMPTS_PATH}/{COMMAND_GENERATOR_NAME}.jinja2",
        "r",
    ) as f:
        assert f.read() == prompts[COMMAND_GENERATOR_NAME]


def test_write_training_data_handles_missing_keys(tmp_path: Path) -> None:
    encoded_training_data: Dict[str, str] = {
        # nothing provided, still all files need to exist
    }

    training_base_path = str(tmp_path / "training")
    write_training_data_to_files(encoded_training_data, training_base_path)
    with open(f"{training_base_path}/domain.yml", "r") as f:
        assert f.read() == ""
    with open(f"{training_base_path}/credentials.yml", "r") as f:
        assert f.read() == ""
    with open(f"{training_base_path}/endpoints.yml", "r") as f:
        assert f.read() == ""
    with open(f"{training_base_path}/data/flows.yml", "r") as f:
        assert f.read() == ""
    with open(f"{training_base_path}/config.yml", "r") as f:
        assert f.read() == ""
    with open(f"{training_base_path}/data/stories.yml", "r") as f:
        assert f.read() == ""
    with open(f"{training_base_path}/data/rules.yml", "r") as f:
        assert f.read() == ""
    with open(f"{training_base_path}/data/nlu.yml", "r") as f:
        assert f.read() == ""

    prompts_dir = Path(f"{training_base_path}/data/{DEFAULT_PROMPTS_PATH}")
    assert not prompts_dir.exists()


def test_prepare_training_directory(tmp_path: Path) -> None:
    training_base_path = str(tmp_path / "training")
    assistant_id = "test_assistant_id"
    data = {
        "domain": base64.b64encode(b"domain data").decode("utf-8"),
        "config": base64.b64encode(b"config data").decode("utf-8"),
    }
    prepare_training_directory(training_base_path, assistant_id, data)
    with open(f"{training_base_path}/domain.yml", "r") as f:
        assert f.read() == "domain data"
    with open(f"{training_base_path}/config.yml", "r") as f:
        assert f.read() == "config data"


@pytest.mark.flaky(reruns=2, reruns_delay=5)
@pytest.mark.timeout(120, func_only=True)
def test_run_training(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        rasa.model_manager.config, "SERVER_BASE_WORKING_DIRECTORY", str(tmp_path)
    )

    training_id = "test_training_id"
    assistant_id = "test_assistant_id"
    client_id = "test_client_id"
    data = {
        "domain": base64.b64encode(b"domain data").decode("utf-8"),
        "config": base64.b64encode(b"config data").decode("utf-8"),
    }
    session = run_training(training_id, assistant_id, client_id, data)
    assert session.training_id == training_id
    assert session.assistant_id == assistant_id
    assert session.client_id == client_id
    assert session.status == "running"
    assert session.process is not None

    # let's wait for the process to finish
    session.process.wait()
    update_training_status(session)
    # training data is invalid, so training should fail
    assert session.status == "error"


def test_custom_prompt_is_written_and_added_to_endpoints(
    tmp_path: Path, config_with_cg_and_es: Text, monkeypatch: MonkeyPatch
) -> None:
    prompts_dict = {
        CONTEXTUAL_RESPONSE_REPHRASER_NAME: "rephraser prompt",
        COMMAND_GENERATOR_NAME: "command generator prompt",
        ENTERPRISE_SEARCH_NAME: "enterprise search prompt",
    }
    encoded_training_data: Dict[str, str] = {
        "endpoints": _encode("nlg:"),
        "config": _encode(config_with_cg_and_es),
        "prompts": prompts_dict,
    }

    write_training_data_to_files(encoded_training_data, str(tmp_path))

    # `prompts` directory has been created
    prompt_dir = tmp_path / DEFAULT_PROMPTS_PATH
    assert prompt_dir.exists()

    # Prompts have been written to their files
    for prompt_name in prompts_dict.keys():
        prompt_file = prompt_dir / f"{prompt_name}.jinja2"
        assert prompt_file.exists()
        assert prompt_file.read_text(encoding="utf-8") == prompts_dict[prompt_name]

    # endpoints.yml has been updated with the prompt path
    endpoints_file = (tmp_path / DEFAULT_ENDPOINTS_PATH).read_text()
    endpoints = read_yaml(endpoints_file)
    assert (
        Path(endpoints["nlg"]["prompt"]).as_posix()
        == f"{DEFAULT_PROMPTS_PATH}/{CONTEXTUAL_RESPONSE_REPHRASER_NAME}.jinja2"
    )

    # config.yml has been updated with the prompt paths
    config_file = (tmp_path / DEFAULT_CONFIG_PATH).read_text()
    config = read_yaml(config_file)
    assert (
        Path(config["pipeline"][0]["prompt_template"]).as_posix()
        == f"{DEFAULT_PROMPTS_PATH}/{COMMAND_GENERATOR_NAME}.jinja2"
    )
    assert (
        Path(config["policies"][1]["prompt"]).as_posix()
        == f"{DEFAULT_PROMPTS_PATH}/{ENTERPRISE_SEARCH_NAME}.jinja2"
    )


def test_default_prompt_is_ignored(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    encoded_training_data: Dict[str, str] = {
        "endpoints": _encode("nlg:"),
        "prompts": {},
    }

    write_training_data_to_files(encoded_training_data, str(tmp_path))

    # 1. No prompt file has been created
    prompt_dir = tmp_path / DEFAULT_PROMPTS_PATH
    assert not prompt_dir.exists()

    # 2. endpoints.yml is unmodified (still only contains "nlg:")
    endpoints_file = tmp_path / DEFAULT_ENDPOINTS_PATH
    assert endpoints_file.read_text(encoding="utf-8").strip() == "nlg:"

import asyncio
import io
import json
import os
import shutil
import subprocess
import uuid
import zipfile
from http import HTTPStatus
from pathlib import Path
from typing import Dict, Text
from unittest import mock
from unittest.mock import MagicMock

import pytest
from pytest import MonkeyPatch
from sanic import Sanic
from sanic_testing.testing import SanicASGITestClient

import rasa
from rasa.cli.project_templates.defaults import RasaDefaults
from rasa.cli.scaffold import ProjectTemplateName, scaffold_path
from rasa.model_manager import config
from rasa.model_manager.model_api import (
    external_blueprint,
    internal_blueprint,
    running_bots,
    trainings,
)
from rasa.model_manager.runner_service import BotSession, BotSessionStatus
from rasa.model_manager.trainer_service import TrainingSession, TrainingSessionStatus
from rasa.model_manager.utils import models_base_path
from rasa.studio.upload import CALMUserData


@pytest.fixture
def app(monkeypatch: MonkeyPatch, tmp_path: Path) -> Sanic:
    monkeypatch.setattr(config, "SERVER_BASE_WORKING_DIRECTORY", str(tmp_path))
    app = Sanic("test_app")
    app.blueprint(internal_blueprint())
    app.blueprint(external_blueprint())
    return app


@pytest.fixture
def client(app: Sanic) -> SanicASGITestClient:
    return app.asgi_client


@pytest.fixture
def training_id() -> str:
    return uuid.uuid4().hex


@pytest.fixture
def training_session(training_id: str) -> MagicMock:
    training_session = MagicMock(spec=TrainingSession)
    training_session.configure_mock(
        training_id=training_id,
        status=TrainingSessionStatus.RUNNING,
        progress=50,
        assistant_id="assistant_1",
        client_id="client_1",
        process=MagicMock(spec=subprocess.Popen, returncode=0),
        model_name=None,
        logs=None,
        log_id="test_42",
    )
    return training_session


def setup_logs_path(tmp_path: Path, action_id: str) -> Path:
    logs_parent_path = tmp_path / "logs"
    logs_parent_path.mkdir(exist_ok=True)
    logs_path = logs_parent_path / f"{action_id}.txt"
    logs_path.touch(exist_ok=True)

    with open(logs_path, "w") as f:
        f.write(f"test logs for {action_id}")

    return logs_path


async def test_health_endpoint(client: SanicASGITestClient) -> None:
    _, response = await client.get("/")
    assert response.status == 200
    assert response.json == {"status": "ok", "bots": [], "trainings": []}


async def test_start_training(client: SanicASGITestClient, training_id: str) -> None:
    data = {
        "id": training_id,
        "assistant_id": "assistant_1",
        "client_id": "client_1",
    }
    _, response = await client.post("/training", json=data)
    assert response.status == 200
    assert response.json.get("training_id") == training_id
    assert response.json.get("model_name") is not None


async def test_start_training_conflict(
    client: SanicASGITestClient, training_id: str, training_session: MagicMock
) -> None:
    trainings[training_id] = training_session

    data = {"id": training_id, "assistant_id": "assistant_1", "client_id": "client_1"}
    _, response = await client.post("/training", json=data)
    assert response.status == 409
    assert response.json == {"message": "Training with this id already exists"}


async def test_start_training_parallel_requests(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    """Test that the training endpoint respects the parallel training limit."""
    monkeypatch.setattr(config, "SERVER_BASE_WORKING_DIRECTORY", str(tmp_path))
    monkeypatch.setattr(config, "MAX_PARALLEL_TRAININGS", 2)
    app = Sanic("test_parallel")

    bp = internal_blueprint()
    # we need to remove the before server stop listener
    # which updates the status of previously started trainings to "stopped",
    # because the Sanic testing client calls this listener after the request is done,
    # see here: https://github.com/sanic-org/sanic-testing/blob/66b72d22979f594b8ea45283c7a2f7db6f13c5e3/sanic_testing/testing.py#L394 # noqa: E501
    monkeypatch.setattr(bp, "_future_listeners", [])
    app.blueprint(bp)
    client = app.asgi_client

    data_1 = {
        "id": uuid.uuid4().hex,
        "assistant_id": "test_assistant_1",
        "model_name": "test_model_name_1",
        "client_id": "test_client",
    }
    data_2 = {
        "id": uuid.uuid4().hex,
        "assistant_id": "test_assistant_2",
        "model_name": "test_model_name_2",
        "client_id": "test_client",
    }
    data_3 = {
        "id": uuid.uuid4().hex,
        "assistant_id": "test_assistant_3",
        "model_name": "test_model_name_3",
        "client_id": "client_3",
    }

    data_4 = {
        "id": uuid.uuid4().hex,
        "assistant_id": "test_assistant_4",
        "model_name": "test_model_name_4",
        "client_id": "client_4",
    }

    tasks = [
        asyncio.create_task(client.post("/training", json=data))
        for data in [data_1, data_2, data_3, data_4]
    ]
    responses = await asyncio.gather(*tasks)
    assert responses[0][1].status == HTTPStatus.OK
    assert responses[1][1].status == HTTPStatus.OK
    assert responses[2][1].status == HTTPStatus.TOO_MANY_REQUESTS
    assert responses[3][1].status == HTTPStatus.TOO_MANY_REQUESTS


async def test_start_training_but_not_enough_diskspace(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(config, "SERVER_BASE_WORKING_DIRECTORY", str(tmp_path))
    monkeypatch.setattr(config, "MIN_REQUIRED_DISCSPACE_MB", 1000000000000)
    app = Sanic("test_discspace")
    bp = internal_blueprint()
    app.blueprint(bp)
    client = app.asgi_client

    data = {
        "id": uuid.uuid4().hex,
        "assistant_id": "test_assistant_1",
        "model_name": "test_model_name_1",
        "client_id": "test_client",
    }
    _, response = await client.post("/training", json=data)
    assert response.status == HTTPStatus.INSUFFICIENT_STORAGE
    assert "Please free up some space" in response.json.get("message", "")


async def test_get_training(
    client: SanicASGITestClient, training_id: str, training_session: MagicMock
) -> None:
    trainings[training_id] = training_session
    _, response = await client.get(f"/training/{training_id}")
    assert response.status == 200
    assert response.json == {
        "training_id": training_id,
        "assistant_id": "assistant_1",
        "client_id": "client_1",
        "progress": 50,
        "status": "running",
        "model_name": None,
        "logs": None,
    }


async def test_get_training_not_found(client: SanicASGITestClient) -> None:
    _, response = await client.get("/training/non_existent_id")
    assert response.status == 404
    assert response.json == {"message": "Training not found"}


async def test_get_training_stream_terminal_state(
    client: SanicASGITestClient,
    training_id: str,
) -> None:
    done_session = MagicMock(spec=TrainingSession)
    done_session.configure_mock(
        training_id=training_id,
        progress=100,
        status=TrainingSessionStatus.DONE,
        log_id="test_42",
        process=MagicMock(spec=subprocess.Popen),
    )
    trainings[training_id] = done_session

    _, response = await client.get(f"/training/{training_id}?stream_response=true")

    assert response.status == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    events = [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert len(events) == 1
    assert events[0] == {
        "training_id": training_id,
        "status": "done",
        "progress": 100,
        "logs": None,
    }


async def test_get_training_stream_status_transition(
    client: SanicASGITestClient,
    training_id: str,
    monkeypatch: MonkeyPatch,
) -> None:
    session = MagicMock(spec=TrainingSession)
    session.configure_mock(
        training_id=training_id,
        progress=50,
        status=TrainingSessionStatus.RUNNING,
        log_id="test_42",
        process=MagicMock(spec=subprocess.Popen),
    )
    trainings[training_id] = session

    async def advance_to_done(_delay: float) -> None:
        session.status = TrainingSessionStatus.DONE
        session.progress = 100

    monkeypatch.setattr(asyncio, "sleep", advance_to_done)

    _, response = await client.get(f"/training/{training_id}?stream_response=true")

    assert response.status == 200
    events = [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert len(events) == 2
    assert events[0] == {
        "training_id": training_id,
        "status": "running",
        "progress": 50,
    }
    assert events[1] == {
        "training_id": training_id,
        "status": "done",
        "progress": 100,
        "logs": None,
    }


async def test_stop_training(
    client: SanicASGITestClient, training_id: str, training_session: MagicMock
) -> None:
    trainings[training_id] = training_session
    _, response = await client.delete(f"/training/{training_id}")
    assert response.status == 200
    assert response.json == {"training_id": training_id}


async def test_stop_training_not_found(client: SanicASGITestClient) -> None:
    _, response = await client.delete("/training/non_existent_id")
    assert response.status == 404
    assert response.json == {"message": "Training session not found"}


def _make_zip(files: Dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


async def test_start_training_multipart_missing_zip(
    client: SanicASGITestClient, training_id: str
) -> None:
    _, response = await client.post(
        "/training",
        data={
            "id": training_id,
            "assistant_id": "assistant_1",
            "client_id": "client_1",
        },
        files={"not_zip": ("dummy", b"dummy", "text/plain")},
    )
    assert response.status == 400
    assert response.json == {"message": "zip file is required"}


@mock.patch("rasa.model_manager.model_api.run_training")
async def test_start_training_with_zip(
    mock_run_training: mock.Mock,
    client: SanicASGITestClient,
    training_id: str,
) -> None:
    mock_session = MagicMock(spec=TrainingSession)
    mock_session.configure_mock(
        training_id=training_id,
        model_name="test_model",
        status=TrainingSessionStatus.RUNNING,
        process=MagicMock(spec=subprocess.Popen),
    )
    mock_run_training.return_value = mock_session

    zip_bytes = _make_zip({"config.yml": b"pipeline: []"})

    _, response = await client.post(
        "/training",
        data={
            "id": training_id,
            "assistant_id": "assistant_1",
            "client_id": "client_1",
        },
        files={"zip": ("archive.zip", zip_bytes, "application/zip")},
    )
    assert response.status == 200
    assert response.json.get("training_id") == training_id
    assert response.json.get("model_name") == "test_model"
    _, call_kwargs = mock_run_training.call_args
    assert call_kwargs["zip_bytes"] == zip_bytes
    assert call_kwargs["encoded_training_data"] is None


async def test_start_bot(
    client: SanicASGITestClient, trained_rasa_model_with_flows: str
) -> None:
    # create models base path
    os.makedirs(models_base_path(), exist_ok=True)
    model_name = os.path.basename(trained_rasa_model_with_flows).split(".")[0]
    # move the model to the correct directory (a made up training directory)
    shutil.copy(trained_rasa_model_with_flows, models_base_path())

    data = {
        "deployment_id": "deployment_1",
        "model_name": model_name,
        "encoded_configs": {
            "credentials": "",
            "endpoints": "",
        },
    }
    _, response = await client.post("/bot", json=data)
    assert response.status == 200
    assert response.json.get("deployment_id") == "deployment_1"
    assert response.json.get("status") == "queued"
    assert response.json.get("url").startswith("http://mockserver:")


async def test_start_bot_conflict(client: SanicASGITestClient) -> None:
    running_bots["deployment_1"] = MagicMock()
    data = {"deployment_id": "deployment_1", "model_path": "/path/to/model"}
    _, response = await client.post("/bot", json=data)
    assert response.status == 409
    assert response.json == {"message": "Bot with this deployment id already exists"}


async def test_start_bot_parallel_requests(
    monkeypatch: MonkeyPatch, tmp_path: Path, trained_rasa_model_with_flows: str
) -> None:
    """Test that the training endpoint respects the parallel training limit."""
    monkeypatch.setattr(config, "SERVER_BASE_WORKING_DIRECTORY", str(tmp_path))
    monkeypatch.setattr(config, "MAX_PARALLEL_BOT_RUNS", 1)
    app = Sanic("test_parallel_runs")

    bp = internal_blueprint()
    # we need to remove the before server stop listener
    # which updates the status of previously started trainings to "stopped",
    # because the Sanic testing client calls this listener after the request is done,
    # see here: https://github.com/sanic-org/sanic-testing/blob/66b72d22979f594b8ea45283c7a2f7db6f13c5e3/sanic_testing/testing.py#L394 # noqa: E501
    monkeypatch.setattr(bp, "_future_listeners", [])
    app.blueprint(bp)
    client = app.asgi_client

    # create models base path
    os.makedirs(models_base_path(), exist_ok=True)
    model_name = os.path.basename(trained_rasa_model_with_flows).split(".")[0]
    shutil.copy(trained_rasa_model_with_flows, models_base_path())

    tasks = [
        asyncio.create_task(
            client.post(
                "/bot",
                json={
                    "deployment_id": f"deployment_test_{i}",
                    "model_name": model_name,
                    "encoded_configs": {
                        "credentials": "",
                        "endpoints": "",
                    },
                },
            )
        )
        for i in range(3)
    ]
    responses = await asyncio.gather(*tasks)
    assert responses[0][1].status == HTTPStatus.OK
    assert responses[1][1].status == HTTPStatus.TOO_MANY_REQUESTS
    assert responses[2][1].status == HTTPStatus.TOO_MANY_REQUESTS


async def test_start_bot_but_not_enough_diskspace(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(config, "SERVER_BASE_WORKING_DIRECTORY", str(tmp_path))
    monkeypatch.setattr(config, "MIN_REQUIRED_DISCSPACE_MB", 1000000000000)
    app = Sanic("test_discspace")
    bp = internal_blueprint()
    app.blueprint(bp)
    client = app.asgi_client

    _, response = await client.post(
        "/bot",
        json={
            "deployment_id": "deployment_test_no_space",
            "model_name": "test_no_storage",
            "encoded_configs": {
                "credentials": "",
                "endpoints": "",
            },
        },
    )
    assert response.status == HTTPStatus.INSUFFICIENT_STORAGE
    assert "Please free up some space" in response.json.get("message", "")


async def test_get_bot(client: SanicASGITestClient) -> None:
    # Create a MagicMock with all required attributes to avoid serialization issues
    running_bots["deployment_1"] = MagicMock()
    running_bots["deployment_1"].deployment_id = "deployment_1"
    running_bots["deployment_1"].status = BotSessionStatus.RUNNING
    running_bots["deployment_1"].url = "http://localhost:8000"
    running_bots["deployment_1"].log_id = "test_42"
    running_bots["deployment_1"].returncode = None
    running_bots["deployment_1"].process = MagicMock()
    _, response = await client.get("/bot/deployment_1")
    assert response.status == 200
    assert response.json == {
        "deployment_id": "deployment_1",
        "status": "running",
        "url": "http://localhost:8000",
        "returncode": None,
        "logs": None,
    }


async def test_get_bot_not_found(client: SanicASGITestClient) -> None:
    _, response = await client.get("/bot/non_existent_id")
    assert response.status == 404
    assert response.json == {"message": "Bot not found"}


async def test_get_bot_stream_terminal_running(
    client: SanicASGITestClient,
) -> None:
    bot = MagicMock(spec=BotSession)
    bot.configure_mock(
        deployment_id="deployment_1",
        status=BotSessionStatus.RUNNING,
        url="http://localhost:8000",
        internal_url="http://localhost:12345",
        returncode=None,
        log_id="test_42",
        process=MagicMock(spec=subprocess.Popen, returncode=0),
    )
    running_bots["deployment_1"] = bot

    _, response = await client.get("/bot/deployment_1?stream_response=true")

    assert response.status == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    events = [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert len(events) == 1
    assert events[0] == {
        "deployment_id": "deployment_1",
        "status": "running",
        "url": "http://localhost:8000",
        "internal_url": "http://localhost:12345",
        "returncode": None,
        "logs": None,
    }


async def test_get_bot_stream_terminal_stopped(
    client: SanicASGITestClient,
) -> None:
    bot = MagicMock(spec=BotSession)
    bot.configure_mock(
        deployment_id="deployment_1",
        status=BotSessionStatus.STOPPED,
        url="http://localhost:8000",
        internal_url="http://localhost:12345",
        returncode=1,
        log_id="test_42",
        process=MagicMock(spec=subprocess.Popen, returncode=1),
    )
    running_bots["deployment_1"] = bot

    _, response = await client.get("/bot/deployment_1?stream_response=true")

    assert response.status == 200
    events = [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert len(events) == 1
    assert events[0] == {
        "deployment_id": "deployment_1",
        "status": "stopped",
        "url": "http://localhost:8000",
        "internal_url": "http://localhost:12345",
        "returncode": 1,
        "logs": None,
    }


async def test_get_bot_stream_status_transition(
    client: SanicASGITestClient,
    monkeypatch: MonkeyPatch,
) -> None:
    bot = MagicMock(spec=BotSession)
    bot.configure_mock(
        deployment_id="deployment_1",
        status=BotSessionStatus.QUEUED,
        url="http://localhost:8000",
        internal_url="http://localhost:12345",
        returncode=None,
        log_id="test_42",
        process=MagicMock(spec=subprocess.Popen, returncode=0),
    )
    running_bots["deployment_1"] = bot

    async def advance_to_running(_delay: float) -> None:
        bot.status = BotSessionStatus.RUNNING

    monkeypatch.setattr(asyncio, "sleep", advance_to_running)

    _, response = await client.get("/bot/deployment_1?stream_response=true")

    assert response.status == 200
    events = [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert len(events) == 2
    assert events[0] == {
        "deployment_id": "deployment_1",
        "status": "queued",
        "url": "http://localhost:8000",
        "internal_url": "http://localhost:12345",
        "returncode": None,
    }
    assert events[1] == {
        "deployment_id": "deployment_1",
        "status": "running",
        "url": "http://localhost:8000",
        "internal_url": "http://localhost:12345",
        "returncode": None,
        "logs": None,
    }


async def test_stop_bot(client: SanicASGITestClient) -> None:
    running_bots["deployment_1"] = BotSession(
        deployment_id="deployment_1",
        status=BotSessionStatus.RUNNING,
        url="http://localhost:8000",
        internal_url="http://localhost:8000",
        port=8000,
        process=MagicMock(spec=subprocess.Popen, returncode=0),
        log_id="test_42",
    )
    _, response = await client.delete("/bot/deployment_1")
    assert response.status == 200
    assert response.json == {
        "deployment_id": "deployment_1",
        "status": "stopped",
        "url": "http://localhost:8000",
    }


async def test_stop_bot_not_found(client: SanicASGITestClient) -> None:
    _, response = await client.delete("/bot/non_existent_id")
    assert response.status == 404
    assert response.json == {"message": "Bot not found"}


async def test_get_training_with_logs(
    client: SanicASGITestClient, tmp_path: Path
) -> None:
    action_id = uuid.uuid4().hex
    log_id = "test_42"
    setup_logs_path(tmp_path, log_id)

    # Create a MagicMock with all required attributes to avoid serialization issues
    from rasa.model_manager.trainer_service import TrainingSessionStatus

    trainings[action_id] = MagicMock()
    trainings[action_id].training_id = action_id
    trainings[action_id].assistant_id = "assistant_1"
    trainings[action_id].client_id = "client_1"
    trainings[action_id].progress = 50
    trainings[action_id].status = TrainingSessionStatus.RUNNING
    trainings[action_id].log_id = log_id
    trainings[action_id].model_name = ""
    trainings[action_id].process = MagicMock()
    _, response = await client.get(f"/training/{action_id}")
    assert response.status == 200
    assert response.json == {
        "training_id": action_id,
        "assistant_id": "assistant_1",
        "client_id": "client_1",
        "progress": 50,
        "status": "running",  # This will be the string value from the enum
        "model_name": "",  # Updated to match the empty string we set
        "logs": f"test logs for {log_id}",
    }


async def test_get_bot_with_logs(
    client: SanicASGITestClient,
    tmp_path: Path,
) -> None:
    action_id = uuid.uuid4().hex
    log_id = "test_42"
    setup_logs_path(tmp_path, log_id)

    # Create a MagicMock with all required attributes to avoid serialization issues
    running_bots[action_id] = MagicMock()
    running_bots[action_id].deployment_id = action_id
    running_bots[action_id].status = BotSessionStatus.RUNNING
    running_bots[action_id].url = "http://localhost:8000"
    running_bots[action_id].log_id = log_id
    running_bots[action_id].returncode = None
    running_bots[action_id].process = MagicMock()  # Add the missing process attribute
    _, response = await client.get(f"/bot/{action_id}")
    assert response.status == 200
    assert response.json == {
        "deployment_id": action_id,
        "status": "running",  # This will be the string value from the enum
        "url": "http://localhost:8000",
        "returncode": None,
        "logs": f"test logs for {log_id}",
    }


async def test_get_model(
    client: SanicASGITestClient,
    trained_rasa_model_with_flows: str,
    test_public_key: str,
    monkeypatch: MonkeyPatch,
) -> None:
    os.makedirs(models_base_path(), exist_ok=True)
    model_name = os.path.basename(trained_rasa_model_with_flows).split(".")[0]
    shutil.copy(trained_rasa_model_with_flows, models_base_path())

    _, response = await client.get(
        f"/models/{model_name}",
    )
    assert response.status == 200


async def test_get_model_not_found(client: SanicASGITestClient) -> None:
    _, response = await client.get("/models/non_existent_model")
    assert response.status == 404
    assert response.json == {"message": "Model not found"}


@pytest.fixture()
def config_yaml() -> Text:
    calm_dir = Path(scaffold_path(ProjectTemplateName.DEFAULT))
    return (calm_dir / "config.yml").read_text(encoding="utf-8")


@pytest.fixture()
def endpoints_yaml() -> Text:
    calm_dir = Path(scaffold_path(ProjectTemplateName.DEFAULT))
    return (calm_dir / "endpoints.yml").read_text(encoding="utf-8")


async def test_defaults_happy_path(
    client: SanicASGITestClient, config_yaml: Text, endpoints_yaml: Text
):
    body = {"config": config_yaml, "endpoints": endpoints_yaml}
    _, response = await client.post("/defaults", json=body)

    assert response.status == HTTPStatus.OK

    payload = response.json
    for field in RasaDefaults.model_fields:
        assert field in payload
        assert payload[field]


async def test_defaults_missing_config(
    client: SanicASGITestClient, endpoints_yaml: Text
):
    body = {"endpoints": endpoints_yaml}
    _, response = await client.post("/defaults", json=body)

    assert response.status == HTTPStatus.BAD_REQUEST
    assert "Missing `config` key" in response.json["message"]


async def test_defaults_missing_endpoints(
    client: SanicASGITestClient, config_yaml: Text
):
    body = {"config": config_yaml}
    _, response = await client.post("/defaults", json=body)

    assert response.status == HTTPStatus.BAD_REQUEST
    assert "Missing `endpoints` key" in response.json["message"]


async def test_defaults_invalid_yaml(client: SanicASGITestClient):
    body = {"config": "::: this is not yaml :::", "endpoints": "{}"}
    _, response = await client.post("/defaults", json=body)

    assert response.status == HTTPStatus.INTERNAL_SERVER_ERROR
    assert "Failed to load defaults" in response.json["message"]


async def test_project_template_happy_path(client: SanicASGITestClient) -> None:
    _, response = await client.get("/project_template")  # default = calm

    assert response.status == HTTPStatus.OK

    payload = response.json
    expected = {
        "assistantName",
        "defaults",
        "version",
        *CALMUserData.model_fields.keys(),
    }
    assert expected == set(payload)
    assert payload["assistantName"] == ProjectTemplateName.DEFAULT.value
    assert payload["version"] == rasa.__version__


async def test_project_template_unknown_template(client: SanicASGITestClient) -> None:
    _, response = await client.get("/project_template?template=does_not_exist")
    assert response.status == HTTPStatus.BAD_REQUEST
    assert "Unknown template" in response.json["message"]

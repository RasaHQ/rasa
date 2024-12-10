import asyncio
import uuid
from http import HTTPStatus
from pathlib import Path
import subprocess
from unittest.mock import MagicMock

import pytest
from sanic import Sanic
from pytest import MonkeyPatch
import os
import shutil
from sanic_testing.testing import SanicASGITestClient

from rasa.model_manager.model_api import (
    external_blueprint,
    internal_blueprint,
    running_bots,
    trainings,
)
from rasa.model_manager.trainer_service import TrainingSession, TrainingSessionStatus
from rasa.model_manager.utils import models_base_path
from rasa.model_manager.runner_service import BotSession
from rasa.model_manager import config


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
    running_bots["deployment_1"] = MagicMock(
        deployment_id="deployment_1", status="running", url="http://localhost:8000"
    )
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


async def test_stop_bot(client: SanicASGITestClient) -> None:
    running_bots["deployment_1"] = BotSession(
        deployment_id="deployment_1",
        status="running",
        url="http://localhost:8000",
        internal_url="http://localhost:8000",
        port=8000,
        process=MagicMock(spec=subprocess.Popen, returncode=0),
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
    setup_logs_path(tmp_path, action_id)

    trainings[action_id] = MagicMock(
        training_id=action_id,
        assistant_id="assistant_1",
        client_id="client_1",
        progress=50,
        status="running",
    )
    _, response = await client.get(f"/training/{action_id}")
    assert response.status == 200
    assert response.json == {
        "training_id": action_id,
        "assistant_id": "assistant_1",
        "client_id": "client_1",
        "progress": 50,
        "status": "running",
        "model_name": None,
        "logs": f"test logs for {action_id}",
    }


async def test_get_bot_with_logs(
    client: SanicASGITestClient,
    tmp_path: Path,
) -> None:
    action_id = uuid.uuid4().hex
    setup_logs_path(tmp_path, action_id)

    running_bots[action_id] = MagicMock(
        deployment_id=action_id, status="running", url="http://localhost:8000"
    )
    _, response = await client.get(f"/bot/{action_id}")
    assert response.status == 200
    assert response.json == {
        "deployment_id": action_id,
        "status": "running",
        "url": "http://localhost:8000",
        "returncode": None,
        "logs": f"test logs for {action_id}",
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

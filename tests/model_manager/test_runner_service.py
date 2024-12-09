from pathlib import Path
from aioresponses import aioresponses
import pytest
import os
import shutil
import subprocess
from unittest import mock
from pytest import MonkeyPatch
import asyncio
from moto import mock_aws
from rasa.model_manager.utils import models_base_path
import rasa.constants
from rasa.core.persistor import AWSPersistor
import boto3

from rasa.exceptions import ModelNotFound
from rasa.model_manager import config
from rasa.model_manager.runner_service import (
    BotSession,
    bot_path,
    is_bot_startup_finished,
    set_bot_status_to_running,
    set_bot_status_to_stopped,
    get_open_port,
    prepare_bot_directory,
    run_bot,
    update_bot_status,
    terminate_bot,
)
from rasa.env import REMOTE_STORAGE_PATH_ENV


@pytest.fixture
def mock_bot_session() -> BotSession:
    return BotSession(
        deployment_id="test_deployment",
        status="queued",
        process=mock.Mock(spec=subprocess.Popen),
        url="http://example.com",
        internal_url="http://localhost:5005",
        port=5005,
    )


def test_bot_path() -> None:
    deployment_id = "test_deployment"
    expected_path = os.path.abspath("working-data/bots/test_deployment")
    assert bot_path(deployment_id) == expected_path


async def test_is_bot_startup_finished_success(mock_bot_session: BotSession) -> None:
    with aioresponses() as mocked:
        # create a mock health server
        mocked.get("http://localhost:5005/license", status=200, body="ok")

        assert await is_bot_startup_finished(mock_bot_session)


async def test_is_bot_startup_finished_failure(mock_bot_session: BotSession) -> None:
    with aioresponses() as mocked:
        # create a mock health server
        mocked.get("http://localhost:5005/license", status=500)

        assert not await is_bot_startup_finished(mock_bot_session)


def test_update_bot_to_stopped(mock_bot_session: BotSession) -> None:
    mock_bot_session.process.returncode = 0
    set_bot_status_to_stopped(mock_bot_session)
    assert mock_bot_session.status == "stopped"


def test_update_bot_to_running(mock_bot_session: BotSession) -> None:
    set_bot_status_to_running(mock_bot_session)
    assert mock_bot_session.status == "running"


def test_get_open_port() -> None:
    port = get_open_port()
    assert isinstance(port, int)
    assert 1024 <= port <= 65535


def test_prepare_bot_directory(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(config, "SERVER_BASE_WORKING_DIRECTORY", str(tmp_path))

    model_name = "mytestmodel"

    bot_base_path = tmp_path / "test_bot"
    training_base_path = tmp_path / "models"
    os.makedirs(training_base_path, exist_ok=True)
    # create empty file to simulate a trained model
    with open(training_base_path / f"{model_name}.tar.gz", "w") as f:
        f.write("")

    encoded_configs = {
        "endpoints": "",
        "credentials": "",
    }

    prepare_bot_directory(str(bot_base_path), model_name, encoded_configs)

    assert os.path.exists(bot_base_path)
    assert os.path.exists(bot_base_path / "models")
    assert os.path.exists(bot_base_path / "endpoints.yml")
    assert os.path.exists(bot_base_path / "credentials.yml")


@mock_aws
def test_prepare_remote_bot_directory(
    tmp_path: Path, monkeypatch: MonkeyPatch, trained_rasa_model_with_flows: str
) -> None:
    monkeypatch.setattr(config, "SERVER_BASE_WORKING_DIRECTORY", str(tmp_path))
    monkeypatch.setattr(config, "SERVER_MODEL_REMOTE_STORAGE", "aws")
    monkeypatch.setenv(REMOTE_STORAGE_PATH_ENV, "models")

    bot_base_path = tmp_path / "test_bot"

    # create models base path
    os.makedirs(models_base_path(), exist_ok=True)
    model_name = os.path.basename(trained_rasa_model_with_flows).split(".")[0]

    remote_model_path = os.path.join(
        "models", os.path.basename(trained_rasa_model_with_flows)
    )

    # --- test setup
    region_name = "us-east-1"
    bucket_name = rasa.constants.DEFAULT_BUCKET_NAME
    conn = boto3.resource("s3", region_name=region_name)
    # We need to create the bucket in Moto's 'virtual' AWS account
    # prior to AWSPersistor instantiation
    conn.create_bucket(Bucket=bucket_name)
    # upload model file to bucket
    with open(trained_rasa_model_with_flows, "rb") as f:
        conn.meta.client.upload_fileobj(f, bucket_name, remote_model_path)

    def mock_aws_persistor(name: str) -> AWSPersistor:
        aws_persistor = AWSPersistor(bucket_name, region_name=region_name)
        monkeypatch.setattr(aws_persistor, "s3", conn)
        monkeypatch.setattr(aws_persistor, "bucket", conn.Bucket(bucket_name))
        return aws_persistor

    # --- actual test
    monkeypatch.setattr("rasa.core.persistor.get_persistor", mock_aws_persistor)

    encoded_configs = {
        "endpoints": "",
        "credentials": "",
    }

    # check if

    prepare_bot_directory(str(bot_base_path), model_name, encoded_configs)

    assert os.path.exists(bot_base_path)
    assert os.path.exists(bot_base_path / "models")
    assert os.path.exists(bot_base_path / "models" / f"{model_name}.tar.gz")
    assert os.path.exists(bot_base_path / "endpoints.yml")
    assert os.path.exists(bot_base_path / "credentials.yml")

    # check if a non existing model raises an exception

    with pytest.raises(ModelNotFound):
        prepare_bot_directory(str(bot_base_path), "non_existing_model", encoded_configs)


async def test_run_bot(
    tmp_path: Path, trained_rasa_model_with_flows: str, monkeypatch: MonkeyPatch
) -> None:
    # we need a license for this test, otherwise the bot will not start
    assert os.getenv("RASA_PRO_LICENSE") is not None

    monkeypatch.setattr(config, "SERVER_BASE_WORKING_DIRECTORY", str(tmp_path))
    deployment_id = "test_deployment"
    base_url_path = "http://test_base_url"

    # create models base path
    os.makedirs(models_base_path(), exist_ok=True)
    model_name = os.path.basename(trained_rasa_model_with_flows).split(".")[0]
    shutil.copy(trained_rasa_model_with_flows, models_base_path())

    encoded_configs = {
        "endpoints": "",
        "credentials": "",
    }

    bot_session = run_bot(deployment_id, model_name, base_url_path, encoded_configs)
    assert bot_session.deployment_id == deployment_id
    assert bot_session.status == "queued"
    assert bot_session.process.pid is not None
    assert bot_session.url == f"{base_url_path}?deployment_id={deployment_id}"
    assert bot_session.internal_url == f"http://localhost:{bot_session.port}"

    while bot_session.status == "queued":
        await update_bot_status(bot_session)
        await asyncio.sleep(1)

    assert bot_session.status == "running"
    terminate_bot(bot_session)
    assert bot_session.status == "stopped"


async def test_update_bot_status_queued_to_stoped(mock_bot_session: BotSession) -> None:
    mock_bot_session.process.poll.return_value = 0  # type: ignore[attr-defined]
    mock_bot_session.process.returncode = 1
    await update_bot_status(mock_bot_session)
    assert mock_bot_session.status == "stopped"


async def test_update_bot_status_queued_to_running(
    mock_bot_session: BotSession,
) -> None:
    mock_bot_session.status = "queued"
    mock_bot_session.process.poll.return_value = None  # type: ignore[attr-defined]
    with aioresponses() as mocked:
        # create a mock health server
        mocked.get("http://localhost:5005/license", status=200)
        await update_bot_status(mock_bot_session)
        assert mock_bot_session.status == "running"
        assert mock_bot_session.returncode is None


async def test_update_bot_status_running_to_stopped(
    mock_bot_session: BotSession,
) -> None:
    # prepare mocks
    mock_bot_session.process.poll.return_value = 0  # type: ignore[attr-defined]
    mock_bot_session.process.returncode = 1

    # setup
    mock_bot_session.status = "running"

    await update_bot_status(mock_bot_session)
    assert mock_bot_session.status == "stopped"
    assert mock_bot_session.returncode == 1


def test_terminate_bot(mock_bot_session: BotSession) -> None:
    # set up mock return code
    mock_bot_session.process.returncode = 1

    terminate_bot(mock_bot_session)
    assert mock_bot_session.status == "stopped"
    assert mock_bot_session.returncode == 1
    mock_bot_session.process.terminate.assert_called_once()  # type: ignore[attr-defined]

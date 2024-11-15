import os
import sys
import time
from pathlib import Path
from typing import Text, Optional

import pytest
import asyncio
from _pytest.monkeypatch import MonkeyPatch

import rasa
import rasa.constants
import rasa.shared.utils.io
import rasa.model
from rasa.core.agent import load_agent
from rasa.exceptions import ModelNotFound


def test_get_latest_model(tmp_path: Path):
    path = tmp_path / "test_get_latest_model"
    path.mkdir()
    Path(path / "model_one.tar.gz").touch()

    # create second model later to be registered as distinct in Windows
    time.sleep(0.1)
    Path(path / "model_two.tar.gz").touch()

    path_of_latest = os.path.join(path, "model_two.tar.gz")
    assert rasa.model.get_latest_model(str(path)) == path_of_latest


def test_get_local_model(trained_rasa_model: str):
    assert rasa.model.get_local_model(trained_rasa_model) == trained_rasa_model


@pytest.mark.parametrize("model_path", ["foobar", "rasa", "README.md", None])
def test_get_local_model_exception(model_path: Optional[Text]):
    with pytest.raises(ModelNotFound):
        rasa.model.get_local_model(model_path)


def test_model_fingerprint():
    # As this tests in the root of the cloned `rasa` repo there is always a Git repo
    # and hence a fingerprint.
    assert rasa.model.project_fingerprint()


def test_model_fingerprint_with_no_git(monkeypatch: MonkeyPatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)

    assert rasa.model.project_fingerprint() is None


async def dummy_async_operation():
    await asyncio.sleep(0)
    return "fixture setup check"


@pytest.mark.asyncio
@pytest.mark.skipif(
    sys.platform != "win32", reason="The test needs to be executed only on Windows"
)
async def test_model_loading_windows(trained_rasa_model_windows: Text):
    windows_agent = await load_agent(model_path=trained_rasa_model_windows)
    assert windows_agent is not None

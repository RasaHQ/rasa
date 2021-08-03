from pathlib import Path

import pytest

from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.storage.storage import ModelStorage


@pytest.fixture()
def default_model_storage(tmp_path: Path) -> ModelStorage:
    return LocalModelStorage.create(tmp_path)

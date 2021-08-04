from pathlib import Path

from _pytest.monkeypatch import MonkeyPatch
import pytest

from rasa.engine.caching import CACHE_LOCATION_ENV, LocalTrainingCache, TrainingCache
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.training.hooks import TrainingHook


@pytest.fixture()
def default_model_storage(tmp_path: Path) -> ModelStorage:
    return LocalModelStorage.create(tmp_path)


@pytest.fixture()
def temp_cache(tmp_path: Path, monkeypatch: MonkeyPatch) -> LocalTrainingCache:
    monkeypatch.setenv(CACHE_LOCATION_ENV, str(tmp_path))
    return LocalTrainingCache()


@pytest.fixture()
def default_training_hook(
    temp_cache: TrainingCache, default_model_storage: ModelStorage
) -> TrainingHook:
    return TrainingHook(cache=temp_cache, model_storage=default_model_storage)

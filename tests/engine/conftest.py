from collections import Callable
from pathlib import Path

from _pytest.monkeypatch import MonkeyPatch
import pytest

from rasa.engine.caching import CACHE_LOCATION_ENV, LocalTrainingCache, TrainingCache
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.training.hooks import TrainingHook


@pytest.fixture()
def temp_cache(tmp_path: Path, local_cache_creator: Callable) -> LocalTrainingCache:
    return local_cache_creator(tmp_path)


@pytest.fixture()
def local_cache_creator(monkeypatch: MonkeyPatch) -> Callable:
    def create_local_cache(path: Path) -> LocalTrainingCache:
        monkeypatch.setenv(CACHE_LOCATION_ENV, str(path))
        return LocalTrainingCache()

    return create_local_cache


@pytest.fixture()
def default_training_hook(
    temp_cache: TrainingCache, default_model_storage: ModelStorage
) -> TrainingHook:
    return TrainingHook(cache=temp_cache, model_storage=default_model_storage)

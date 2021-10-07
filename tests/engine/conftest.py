from pathlib import Path
from typing import Callable

from _pytest.monkeypatch import MonkeyPatch
import pytest

from rasa.engine.caching import LocalTrainingCache, TrainingCache
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.training.hooks import TrainingHook


@pytest.fixture()
def temp_cache(tmp_path: Path, local_cache_creator: Callable) -> LocalTrainingCache:
    return local_cache_creator(tmp_path)


@pytest.fixture()
def local_cache_creator(monkeypatch: MonkeyPatch) -> Callable[..., LocalTrainingCache]:
    def create_local_cache(path: Path) -> LocalTrainingCache:
        monkeypatch.setattr(LocalTrainingCache, "_get_cache_location", lambda: path)
        return LocalTrainingCache()

    return create_local_cache


@pytest.fixture()
def default_training_hook(
    temp_cache: TrainingCache, default_model_storage: ModelStorage
) -> TrainingHook:
    return TrainingHook(cache=temp_cache, model_storage=default_model_storage)

from pathlib import Path
from typing import Callable

from _pytest.monkeypatch import MonkeyPatch
import pytest

from rasa.engine.caching import LocalTrainingCache


@pytest.fixture()
def temp_cache(tmp_path: Path, local_cache_creator: Callable) -> LocalTrainingCache:
    return local_cache_creator(tmp_path)


@pytest.fixture()
def local_cache_creator(monkeypatch: MonkeyPatch) -> Callable[..., LocalTrainingCache]:
    def create_local_cache(path: Path) -> LocalTrainingCache:
        monkeypatch.setattr(LocalTrainingCache, "_get_cache_location", lambda: path)
        return LocalTrainingCache()

    return create_local_cache

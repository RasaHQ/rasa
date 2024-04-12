from pathlib import Path
from typing import Callable, Dict, Any

from _pytest.monkeypatch import MonkeyPatch
import pytest

from rasa.dialogue_understanding.coexistence.intent_based_router import (
    IntentBasedRouter,
)
from rasa.dialogue_understanding.coexistence.llm_based_router import LLMBasedRouter
from rasa.engine.caching import LocalTrainingCache
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage


@DefaultV1Recipe.register(
    [
        DefaultV1Recipe.ComponentType.COEXISTENCE_ROUTER,
    ],
    is_trainable=False,
)
class SubclassedIntentBasedRouter(IntentBasedRouter):
    """Used for testing whether validation works with subclassed routers"""

    def __init__(
        self, config: Dict[str, Any], model_storage: ModelStorage, resource: Resource
    ):
        super().__init__(config, model_storage, resource)


@DefaultV1Recipe.register(
    [
        DefaultV1Recipe.ComponentType.COEXISTENCE_ROUTER,
    ],
    is_trainable=False,
)
class SubclassedLLMBasedRouter(LLMBasedRouter):
    """Used for testing whether validation works with subclassed routers"""

    def __init__(
        self, config: Dict[str, Any], model_storage: ModelStorage, resource: Resource
    ):
        super().__init__(config, model_storage, resource)


@pytest.fixture()
def temp_cache(tmp_path: Path, local_cache_creator: Callable) -> LocalTrainingCache:
    return local_cache_creator(tmp_path)


@pytest.fixture()
def local_cache_creator(monkeypatch: MonkeyPatch) -> Callable[..., LocalTrainingCache]:
    def create_local_cache(path: Path) -> LocalTrainingCache:
        monkeypatch.setattr(LocalTrainingCache, "_get_cache_location", lambda: path)
        return LocalTrainingCache()

    return create_local_cache

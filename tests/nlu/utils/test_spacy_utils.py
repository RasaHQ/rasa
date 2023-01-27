import pytest
from pytest import MonkeyPatch
from unittest.mock import MagicMock

from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.model import InvalidModelError
from rasa.nlu.utils.spacy_utils import SpacyNLP
import spacy


def test_spacy_runtime_model_version_compatibility(
    monkeypatch: MonkeyPatch,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
):
    def mock_info_return(mock_model_name):
        return {"spacy_version": ">=3.4.0,<3.5.0", "dummy_keys": "dummy_values"}

    def mock_spacy_obj_return(*args, **kwargs):
        return MagicMock()

    monkeypatch.setattr(spacy, "info", mock_info_return)
    monkeypatch.setattr(spacy, "load", mock_spacy_obj_return)

    # Test case that runs model on incompatible spacy runtime.
    spacy.about.__version__ = "3.10"
    with pytest.raises(InvalidModelError):
        _ = SpacyNLP.create(
            {"model": "some_model"},
            default_model_storage,
            Resource("spacy"),
            default_execution_context,
        )

    # Test case that runs model on compatible spacy runtime
    spacy.about.__version__ = "3.4.1"
    component = SpacyNLP.create(
        {"model": "some_model"},
        default_model_storage,
        Resource("spacy"),
        default_execution_context,
    )
    assert isinstance(component, SpacyNLP)

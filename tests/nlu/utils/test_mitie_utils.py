import shutil
from pathlib import Path

import pytest

from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.utils.mitie_utils import MitieNLP
import mitie

from rasa.shared.exceptions import RasaException


def test_provide(
    default_model_storage: ModelStorage, default_execution_context: ExecutionContext
):
    component = MitieNLP.create(
        MitieNLP.get_default_config(),
        default_model_storage,
        Resource("mitie"),
        default_execution_context,
    )

    model = component.provide()

    expected_path = Path("data", "total_word_feature_extractor.dat")
    assert model.model_path == expected_path
    assert isinstance(model.word_feature_extractor, mitie.total_word_feature_extractor)

    assert model.fingerprint() == str(expected_path)


def test_provide_different_path(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    tmp_path: Path,
):
    new_path = shutil.copy(Path("data", "total_word_feature_extractor.dat"), tmp_path)

    component = MitieNLP.create(
        {"model": new_path},
        default_model_storage,
        Resource("mitie"),
        default_execution_context,
    )

    model = component.provide()

    assert model.model_path == Path(new_path)
    assert isinstance(model.word_feature_extractor, mitie.total_word_feature_extractor)

    assert model.fingerprint() == str(new_path)


def test_invalid_path(
    default_model_storage: ModelStorage, default_execution_context: ExecutionContext
):
    with pytest.raises(RasaException):
        MitieNLP.create(
            {"model": "some-path"},
            default_model_storage,
            Resource("mitie"),
            default_execution_context,
        )

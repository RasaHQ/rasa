import pytest

from rasa.nlu import train
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.model import Interpreter, Trainer
from tests.nlu.conftest import DEFAULT_DATA_PATH
from rasa.shared.nlu.training_data.training_data import TrainingData
from tests.train import pipelines_for_tests, pipelines_for_non_windows_tests


@pytest.mark.timeout(600)
@pytest.mark.parametrize("language, pipeline", pipelines_for_tests())
@pytest.mark.trains_model
async def test_train_persist_load_parse(language, pipeline, component_builder, tmpdir):
    _config = RasaNLUModelConfig({"pipeline": pipeline, "language": language})

    (trained, _, persisted_path) = await train(
        _config,
        path=tmpdir.strpath,
        data=DEFAULT_DATA_PATH,
        component_builder=component_builder,
    )

    assert trained.pipeline

    loaded = Interpreter.load(persisted_path, component_builder)

    assert loaded.pipeline
    assert loaded.parse("Rasa is great!") is not None


@pytest.mark.timeout(600)
@pytest.mark.parametrize("language, pipeline", pipelines_for_non_windows_tests())
@pytest.mark.skip_on_windows
@pytest.mark.trains_model
async def test_train_persist_load_parse_non_windows(
    language, pipeline, component_builder, tmpdir
):
    await test_train_persist_load_parse(language, pipeline, component_builder, tmpdir)


@pytest.mark.parametrize("language, pipeline", pipelines_for_tests())
@pytest.mark.trains_model
def test_train_model_without_data(language, pipeline, component_builder, tmpdir):
    _config = RasaNLUModelConfig({"pipeline": pipeline, "language": language})

    trainer = Trainer(_config, component_builder)
    trainer.train(TrainingData())
    persisted_path = trainer.persist(tmpdir.strpath)

    loaded = Interpreter.load(persisted_path, component_builder)

    assert loaded.pipeline
    assert loaded.parse("Rasa is great!") is not None


@pytest.mark.timeout(600)
@pytest.mark.parametrize("language, pipeline", pipelines_for_non_windows_tests())
@pytest.mark.skip_on_windows
@pytest.mark.trains_model
def test_train_model_without_data_non_windows(
    language, pipeline, component_builder, tmpdir
):
    test_train_model_without_data(language, pipeline, component_builder, tmpdir)

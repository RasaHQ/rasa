from datetime import datetime
from pathlib import Path

from _pytest.tmpdir import TempPathFactory
import freezegun

import rasa
from rasa.engine.caching import TrainingCache
from rasa.engine.graph import GraphModelConfiguration, GraphSchema, SchemaNode
from rasa.engine import loader
from rasa.engine.runner.dask import DaskGraphRunner
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelMetadata, ModelStorage
from rasa.engine.training.graph_trainer import GraphTrainer
from rasa.shared.constants import ASSISTANT_ID_KEY
from rasa.shared.core.domain import Domain
from rasa.shared.data import TrainingType
from rasa.shared.importers.importer import TrainingDataImporter
from tests.engine.graph_components_test_classes import PersistableTestComponent


def test_loader_loads_graph_runner(
    default_model_storage: ModelStorage,
    temp_cache: TrainingCache,
    tmp_path: Path,
    tmp_path_factory: TempPathFactory,
    domain_path: Path,
):
    graph_trainer = GraphTrainer(
        model_storage=default_model_storage,
        cache=temp_cache,
        graph_runner_class=DaskGraphRunner,
    )

    test_value = "test_value"

    train_schema = GraphSchema(
        {
            "train": SchemaNode(
                needs={},
                uses=PersistableTestComponent,
                fn="train",
                constructor_name="create",
                config={"test_value": test_value},
                is_target=True,
            ),
            "load": SchemaNode(
                needs={"resource": "train"},
                uses=PersistableTestComponent,
                fn="run_inference",
                constructor_name="load",
                config={},
            ),
        }
    )
    predict_schema = GraphSchema(
        {
            "load": SchemaNode(
                needs={},
                uses=PersistableTestComponent,
                fn="run_inference",
                constructor_name="load",
                config={},
                is_target=True,
                resource=Resource("train"),
            )
        }
    )

    output_filename = tmp_path / "model.tar.gz"

    importer = TrainingDataImporter.load_from_dict(
        training_data_paths=[],
        domain_path=str(domain_path),
        config_path="data/test_config/config_unique_assistant_id.yml",
    )
    config = importer.get_config()

    trained_at = datetime.utcnow()
    with freezegun.freeze_time(trained_at):
        model_metadata = graph_trainer.train(
            GraphModelConfiguration(
                train_schema=train_schema,
                predict_schema=predict_schema,
                training_type=TrainingType.BOTH,
                assistant_id=config.get(ASSISTANT_ID_KEY),
                language=None,
                core_target=None,
                nlu_target=None,
            ),
            importer=importer,
            output_filename=output_filename,
        )

    assert isinstance(model_metadata, ModelMetadata)
    assert output_filename.is_file()

    loaded_model_storage_path = tmp_path_factory.mktemp("loaded model storage")

    model_metadata, loaded_predict_graph_runner = loader.load_predict_graph_runner(
        storage_path=loaded_model_storage_path,
        model_archive_path=output_filename,
        model_storage_class=LocalModelStorage,
        graph_runner_class=DaskGraphRunner,
    )

    assert loaded_predict_graph_runner.run() == {"load": test_value}

    assert model_metadata.predict_schema == predict_schema
    assert model_metadata.train_schema == train_schema
    assert model_metadata.model_id
    assert model_metadata.assistant_id == config.get(ASSISTANT_ID_KEY)
    assert model_metadata.domain.as_dict() == Domain.from_path(domain_path).as_dict()
    assert model_metadata.rasa_open_source_version == rasa.__version__
    assert model_metadata.trained_at == trained_at

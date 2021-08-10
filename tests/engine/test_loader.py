from datetime import datetime
from pathlib import Path

from _pytest.tmpdir import TempPathFactory

import rasa
from rasa.engine.caching import TrainingCache
from rasa.engine.graph import GraphSchema, SchemaNode
from rasa.engine import loader
from rasa.engine.runner.dask import DaskGraphRunner
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.training.graph_trainer import GraphTrainer
from rasa.shared.core.domain import Domain
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
                config={"test_value": test_value,},
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
            ),
        }
    )

    before_train_time = datetime.utcnow()

    output_filename = tmp_path / "model.tar.gz"
    predict_graph_runner = graph_trainer.train(
        train_schema=train_schema,
        predict_schema=predict_schema,
        domain_path=domain_path,
        output_filename=output_filename,
    )
    assert isinstance(predict_graph_runner, DaskGraphRunner)
    assert output_filename.is_file()
    assert predict_graph_runner.run() == {"load": test_value}

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
    assert model_metadata.domain.as_dict() == Domain.from_path(domain_path).as_dict()
    assert model_metadata.rasa_open_source_version == rasa.__version__
    assert before_train_time < model_metadata.trained_at < datetime.utcnow()

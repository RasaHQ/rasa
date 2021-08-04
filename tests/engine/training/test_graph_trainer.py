from pathlib import Path
from unittest.mock import Mock

from _pytest.monkeypatch import MonkeyPatch
from _pytest.tmpdir import TempPathFactory

from rasa.engine.caching import TrainingCache
from rasa.engine.graph import GraphSchema, SchemaNode
from rasa.engine.runner.dask import DaskGraphRunner
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.training.graph_trainer import GraphTrainer
from tests.engine.graph_components_test_classes import (
    AddInputs,
    AssertComponent, FileReader, PersistableTestComponent, SubtractByX,
)


def test_graph_trainer(
    default_model_storage: ModelStorage,
    temp_cache: TrainingCache,
    tmp_path: Path,
    domain_path: Path,
):
    graph_trainer = GraphTrainer(
        model_storage=default_model_storage,
        cache=temp_cache,
        graph_runner_class=DaskGraphRunner,
    )

    test_value_for_sub_directory = {"test": "test value sub dir"}
    test_value = {"test dir": "test value dir"}

    train_schema = GraphSchema(
        {
            "train": SchemaNode(
                needs={},
                uses=PersistableTestComponent,
                fn="train",
                constructor_name="create",
                config={
                    "test_value": test_value,
                    "test_value_for_sub_directory": test_value_for_sub_directory,
                },
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

    output_filename = tmp_path / "model.tar.gz"
    graph_runner = graph_trainer.train(
        train_schema=train_schema,
        predict_schema=predict_schema,
        domain_path=domain_path,
        output_filename=output_filename,
    )
    assert isinstance(graph_runner, DaskGraphRunner)
    assert output_filename.is_file()
    assert graph_runner.run() == {"load": test_value}

    # TODO: test unpacking from model package? (Or wait till graph_loader)


def test_graph_trainer_fingerprints_caches(
    default_model_storage: ModelStorage,
    temp_cache: TrainingCache,
    tmp_path: Path,
    tmp_path_factory: TempPathFactory,
    domain_path: Path,
    monkeypatch: MonkeyPatch,
):

    input_file = tmp_path / "input_file.txt"
    input_file.write_text("3")

    train_schema = GraphSchema(
        {
            "read_file": SchemaNode(
                needs={},
                uses=FileReader,
                fn="read",
                constructor_name="create",
                config={
                    "file_path": str(input_file),
                },
                is_input=True,
            ),
            "train": SchemaNode(
                needs={},
                uses=PersistableTestComponent,
                fn="train",
                constructor_name="create",
                config={
                    "test_value": "4",
                },
                is_target=True,
            ),
            "process": SchemaNode(
                needs={"resource": "train"},
                uses=PersistableTestComponent,
                fn="run_inference",
                constructor_name="load",
                config={},
            ),
            "add": SchemaNode(
                needs={"i1": "read_file", "i2": "process"},
                uses=AddInputs,
                fn="add",
                constructor_name="create",
                config={},
            ),
            "assert_node": SchemaNode(
                needs={"i": "add"},
                uses=AssertComponent,
                fn="run_assert",
                constructor_name="create",
                config={"value_to_assert": 7},
                is_target=True,
            ),
        }
    )

    mock = Mock()
    monkeypatch.setattr(AssertComponent, "mockable_method", mock)

    mock.assert_not_called()

    graph_trainer = GraphTrainer(
        model_storage=default_model_storage,
        cache=temp_cache,
        graph_runner_class=DaskGraphRunner,
    )

    output_filename = tmp_path / "model.tar.gz"
    graph_trainer.train(
        train_schema=train_schema,
        predict_schema=GraphSchema({}),
        domain_path=domain_path,
        output_filename=output_filename,
    )
    assert output_filename.is_file()

    mock.assert_called_once()

    new_tmp_path = tmp_path_factory.mktemp("new_model_storage_path")
    new_model_storage = LocalModelStorage.create(new_tmp_path)

    new_graph_trainer = GraphTrainer(
        model_storage=new_model_storage,
        cache=temp_cache,
        graph_runner_class=DaskGraphRunner,
    )

    print("******* SECOND RUN!!!!!")

    new_output_filename = new_tmp_path / "model.tar.gz"
    new_graph_trainer.train(
        train_schema=train_schema,
        predict_schema=GraphSchema({}),
        domain_path=domain_path,
        output_filename=new_output_filename,
        assert_fingerprint_status=True,
    )
    assert new_output_filename.is_file()

    mock.assert_called_once()


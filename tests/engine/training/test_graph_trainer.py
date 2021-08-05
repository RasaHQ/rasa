from pathlib import Path
from typing import Callable, Optional
from unittest.mock import Mock

from _pytest.monkeypatch import MonkeyPatch
from _pytest.tmpdir import TempPathFactory
import pytest

from rasa.engine.caching import LocalTrainingCache, TrainingCache
from rasa.engine.graph import GraphSchema, SchemaNode
from rasa.engine.runner.dask import DaskGraphRunner
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.training.graph_trainer import GraphTrainer
from tests.engine.graph_components_test_classes import (
    AddInputs,
    AssertComponent,
    FileReader,
    PersistableTestComponent,
    SubtractByX,
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
    temp_cache: TrainingCache,
    tmp_path: Path,
    schema_trainer_helper: Callable,
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
                config={"file_path": str(input_file),},
                is_input=True,
            ),
            "train": SchemaNode(
                needs={},
                uses=PersistableTestComponent,
                fn="train",
                constructor_name="create",
                config={"test_value": "4",},
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

    schema_trainer_helper(train_schema, temp_cache)

    # The first train should call the component
    mock.assert_called_once()

    second_mock = Mock()
    monkeypatch.setattr(AssertComponent, "mockable_method", second_mock)

    schema_trainer_helper(train_schema, temp_cache)

    # Nothing has changed so this time the component will be cached, not called.
    second_mock.assert_not_called()

    third_mock = Mock()
    monkeypatch.setattr(AssertComponent, "mockable_method", third_mock)

    train_schema.nodes["add"].config["something"] = "new"

    schema_trainer_helper(train_schema, temp_cache)

    # As we changed the config of "add", all its descendants will run.
    third_mock.assert_called_once()


@pytest.fixture
def schema_trainer_helper(
    default_model_storage: ModelStorage,
    temp_cache: TrainingCache,
    tmp_path: Path,
    tmp_path_factory: TempPathFactory,
    local_cache_creator: Callable,
    domain_path: Path,
):
    def train_with_schema(
        train_schema: GraphSchema,
        cache: Optional[TrainingCache] = None,
        model_storage: Optional[ModelStorage] = None,
        path: Optional[Path] = None,
    ) -> Path:
        if not path:
            path = tmp_path_factory.mktemp("model_storage_path")
        if not model_storage:
            model_storage = LocalModelStorage.create(path)
        if not cache:
            cache = local_cache_creator(path)

        graph_trainer = GraphTrainer(
            model_storage=model_storage,
            cache=cache,
            graph_runner_class=DaskGraphRunner,
        )

        output_filename = path / "model.tar.gz"
        graph_trainer.train(
            train_schema=train_schema,
            predict_schema=GraphSchema({}),
            domain_path=domain_path,
            output_filename=output_filename,
        )

        assert output_filename.is_file()
        return output_filename

    return train_with_schema

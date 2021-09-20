from pathlib import Path
from typing import Callable, Dict, Optional, Text, Type
from unittest.mock import Mock

from _pytest.monkeypatch import MonkeyPatch
from _pytest.tmpdir import TempPathFactory
import pytest

from rasa.engine.caching import TrainingCache
from rasa.engine.exceptions import GraphComponentException
from rasa.engine.graph import GraphComponent, GraphSchema, SchemaNode
from rasa.engine.runner.dask import DaskGraphRunner
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.training.graph_trainer import GraphTrainer
from rasa.shared.core.domain import Domain
from rasa.shared.importers.importer import TrainingDataImporter
from tests.engine.graph_components_test_classes import (
    AddInputs,
    AssertComponent,
    FileReader,
    PersistableTestComponent,
    ProvideX,
    SubtractByX,
)


def test_graph_trainer_returns_model_metadata(
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

    output_filename = tmp_path / "model.tar.gz"
    model_metadata = graph_trainer.train(
        train_schema=train_schema,
        predict_schema=predict_schema,
        importer=TrainingDataImporter.load_from_dict(domain_path=str(domain_path)),
        output_filename=output_filename,
    )
    assert model_metadata.model_id
    assert model_metadata.domain.as_dict() == Domain.from_path(domain_path).as_dict()
    assert model_metadata.train_schema == train_schema
    assert model_metadata.predict_schema == predict_schema


def test_graph_trainer_fingerprints_and_caches(
    temp_cache: TrainingCache,
    tmp_path: Path,
    train_with_schema: Callable,
    spy_on_all_components: Callable,
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
                config={"file_path": str(input_file)},
                is_input=True,
            ),
            "train": SchemaNode(
                needs={},
                uses=PersistableTestComponent,
                fn="train",
                constructor_name="create",
                config={"test_value": "4"},
                is_target=True,
            ),
            "process": SchemaNode(
                needs={"resource": "train"},
                uses=PersistableTestComponent,
                fn="run_inference",
                constructor_name="load",
                config={"wrap_output_in_cacheable": True},
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

    # The first train should call all the components and cache their outputs.
    mocks = spy_on_all_components(train_schema)
    train_with_schema(train_schema, temp_cache)
    assert node_call_counts(mocks) == {
        "read_file": 1,
        "train": 1,
        "process": 1,
        "add": 1,
        "assert_node": 1,
    }

    # Nothing has changed so this time so no components will run
    # (just input nodes during fingerprint run).
    mocks = spy_on_all_components(train_schema)
    train_with_schema(train_schema, temp_cache)
    assert node_call_counts(mocks) == {
        "read_file": 1,  # Inputs nodes are always called during the fingerprint run.
        "train": 0,
        "process": 0,
        "add": 0,
        "assert_node": 0,
    }

    # As we changed the config of "add", all its descendants will run.
    train_schema.nodes["add"].config["something"] = "new"
    mocks = spy_on_all_components(train_schema)
    train_with_schema(train_schema, temp_cache)
    assert node_call_counts(mocks) == {
        "read_file": 1,  # Inputs nodes are always called during the fingerprint run.
        "train": 0,
        "process": 0,
        "add": 1,
        "assert_node": 1,
    }

    # We always run everything when the `force_retraining` flag is set to `True`
    train_schema.nodes["add"].config["something"] = "new"
    mocks = spy_on_all_components(train_schema)
    train_with_schema(train_schema, temp_cache, force_retraining=True)
    assert node_call_counts(mocks) == {
        "read_file": 1,
        "train": 1,
        "process": 1,
        "add": 1,
        "assert_node": 1,
    }


def test_graph_trainer_always_reads_input(
    temp_cache: TrainingCache,
    tmp_path: Path,
    train_with_schema: Callable,
    spy_on_all_components: Callable,
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
                config={"file_path": str(input_file)},
                is_input=True,
            ),
            "subtract": SchemaNode(
                needs={"i": "read_file"},
                uses=SubtractByX,
                fn="subtract_x",
                constructor_name="create",
                config={"x": 1},
            ),
            "assert_node": SchemaNode(
                needs={"i": "subtract"},
                uses=AssertComponent,
                fn="run_assert",
                constructor_name="create",
                config={"value_to_assert": 2},
                is_target=True,
            ),
        }
    )

    # The first train should call all the components and cache their outputs.
    mocks = spy_on_all_components(train_schema)
    train_with_schema(train_schema, temp_cache)
    assert node_call_counts(mocks) == {
        "read_file": 1,
        "subtract": 1,
        "assert_node": 1,
    }

    # Nothing has changed so this time so no components will run
    # (just input nodes during fingerprint run).
    mocks = spy_on_all_components(train_schema)
    train_with_schema(train_schema, temp_cache)
    assert node_call_counts(mocks) == {
        "read_file": 1,
        "subtract": 0,
        "assert_node": 0,
    }

    # When we update the input file, all the nodes will run again and the assert_node
    # will fail.
    input_file.write_text("5")
    with pytest.raises(GraphComponentException):
        train_with_schema(train_schema, temp_cache)


def test_graph_trainer_with_non_cacheable_components(
    temp_cache: TrainingCache,
    tmp_path: Path,
    train_with_schema: Callable,
    spy_on_all_components: Callable,
):

    input_file = tmp_path / "input_file.txt"
    input_file.write_text("3")

    train_schema = GraphSchema(
        {
            "input": SchemaNode(
                needs={},
                uses=ProvideX,
                fn="provide",
                constructor_name="create",
                config={},
            ),
            "subtract": SchemaNode(
                needs={"i": "input"},
                uses=SubtractByX,
                fn="subtract_x",
                constructor_name="create",
                config={"x": 1},
                is_target=True,
            ),
        }
    )

    # The first train should call all the components.
    mocks = spy_on_all_components(train_schema)
    train_with_schema(train_schema, temp_cache)
    assert node_call_counts(mocks) == {
        "input": 1,
        "subtract": 1,
    }

    # Nothing has changed but none of the components can cache so all will have to
    # run again.
    mocks = spy_on_all_components(train_schema)
    train_with_schema(train_schema, temp_cache)
    assert node_call_counts(mocks) == {
        "input": 1,
        "subtract": 1,
    }


def node_call_counts(mocks: Dict[Text, Mock]) -> Dict[Text, int]:
    return {node_name: mocks[node_name].call_count for node_name, mock in mocks.items()}


@pytest.fixture
def train_with_schema(
    default_model_storage: ModelStorage,
    temp_cache: TrainingCache,
    tmp_path: Path,
    tmp_path_factory: TempPathFactory,
    local_cache_creator: Callable,
    domain_path: Path,
):
    def inner(
        train_schema: GraphSchema,
        cache: Optional[TrainingCache] = None,
        model_storage: Optional[ModelStorage] = None,
        path: Optional[Path] = None,
        force_retraining: bool = False,
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
            importer=TrainingDataImporter.load_from_dict(domain_path=str(domain_path)),
            output_filename=output_filename,
            force_retraining=force_retraining,
        )

        assert output_filename.is_file()
        return output_filename

    return inner


@pytest.fixture()
def spy_on_component(monkeypatch: MonkeyPatch) -> Callable:
    def inner(component_class: Type[GraphComponent], fn_name: Text) -> Mock:
        mock = Mock(wraps=getattr(component_class, fn_name))
        monkeypatch.setattr(component_class, fn_name, mock)
        return mock

    return inner


@pytest.fixture()
def spy_on_all_components(spy_on_component) -> Callable:
    def inner(schema: GraphSchema) -> Dict[Text, Mock]:
        return {
            node_name: spy_on_component(schema_node.uses, schema_node.fn)
            for node_name, schema_node in schema.nodes.items()
        }

    return inner

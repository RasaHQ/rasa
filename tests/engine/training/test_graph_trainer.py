import logging
from pathlib import Path
from typing import Callable, Dict, Optional, Text, Type, Any
from unittest.mock import Mock

from _pytest.logging import LogCaptureFixture
from _pytest.monkeypatch import MonkeyPatch
from _pytest.tmpdir import TempPathFactory
import pytest

from rasa.engine.caching import LocalTrainingCache, TrainingCache
from rasa.engine.exceptions import GraphComponentException
from rasa.engine.graph import (
    GraphComponent,
    GraphSchema,
    SchemaNode,
    GraphModelConfiguration,
    GraphNode,
    ExecutionContext,
    GraphNodeHook,
)
from rasa.engine.runner.dask import DaskGraphRunner
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.training.graph_trainer import GraphTrainer
from rasa.shared.core.domain import Domain
from rasa.shared.data import TrainingType
from rasa.shared.importers.importer import TrainingDataImporter
from tests.engine.graph_components_test_classes import (
    AddInputs,
    AssertComponent,
    FileReader,
    PersistableTestComponent,
    ProvideX,
    SubtractByX,
    CacheableComponent,
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
    model_metadata = graph_trainer.train(
        GraphModelConfiguration(
            train_schema=train_schema,
            predict_schema=predict_schema,
            assistant_id="test_assistant_id",
            language=None,
            core_target=None,
            nlu_target="nlu",
            training_type=TrainingType.BOTH,
        ),
        importer=TrainingDataImporter.load_from_dict(domain_path=str(domain_path)),
        output_filename=output_filename,
    )
    assert model_metadata.model_id
    assert model_metadata.assistant_id == "test_assistant_id"
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
    assert node_call_counts(mocks) == {"read_file": 1, "subtract": 1, "assert_node": 1}

    # Nothing has changed so this time so no components will run
    # (just input nodes during fingerprint run).
    mocks = spy_on_all_components(train_schema)
    train_with_schema(train_schema, temp_cache)
    assert node_call_counts(mocks) == {"read_file": 1, "subtract": 0, "assert_node": 0}

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
    assert node_call_counts(mocks) == {"input": 1, "subtract": 1}

    # Nothing has changed but none of the components can cache so all will have to
    # run again.
    mocks = spy_on_all_components(train_schema)
    train_with_schema(train_schema, temp_cache)
    assert node_call_counts(mocks) == {"input": 1, "subtract": 1}


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
            model_storage=model_storage, cache=cache, graph_runner_class=DaskGraphRunner
        )

        output_filename = path / "model.tar.gz"
        graph_trainer.train(
            GraphModelConfiguration(
                train_schema=train_schema,
                predict_schema=GraphSchema({}),
                assistant_id="test_assistant",
                language=None,
                core_target=None,
                nlu_target="nlu",
                training_type=TrainingType.BOTH,
            ),
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


def test_graph_trainer_train_logging(
    tmp_path: Path,
    temp_cache: TrainingCache,
    train_with_schema: Callable,
    caplog: LogCaptureFixture,
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
            "subtract 2": SchemaNode(
                needs={},
                uses=ProvideX,
                fn="provide",
                constructor_name="create",
                config={},
                is_target=True,
                is_input=True,
            ),
            "subtract": SchemaNode(
                needs={"i": "input"},
                uses=SubtractByX,
                fn="subtract_x",
                constructor_name="create",
                config={"x": 1},
                is_target=True,
                is_input=False,
            ),
        }
    )

    with caplog.at_level(logging.INFO, logger="rasa.engine.training.hooks"):
        train_with_schema(train_schema, temp_cache)

    caplog_info_records = list(
        filter(lambda x: x[1] == logging.INFO, caplog.record_tuples)
    )

    caplog_messages = list([record[2] for record in caplog_info_records])

    assert caplog_messages == [
        "Starting to train component 'SubtractByX'.",
        "Finished training component 'SubtractByX'.",
    ]


def test_graph_trainer_train_logging_with_cached_components(
    tmp_path: Path,
    temp_cache: TrainingCache,
    train_with_schema: Callable,
    caplog: LogCaptureFixture,
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
                is_input=False,
            ),
            "cache_able_node": SchemaNode(
                needs={"suffix": "input"},
                uses=CacheableComponent,
                fn="run",
                constructor_name="create",
                config={},
                is_target=True,
                is_input=False,
            ),
        }
    )

    # Train to cache
    train_with_schema(train_schema, temp_cache)

    # Train a second time
    with caplog.at_level(logging.INFO, logger="rasa.engine.training.hooks"):
        train_with_schema(train_schema, temp_cache)

        caplog_info_records = list(
            filter(lambda x: x[1] == logging.INFO, caplog.record_tuples)
        )
        caplog_messages_set = set([record[2] for record in caplog_info_records])

        assert caplog_messages_set == {
            "Starting to train component 'SubtractByX'.",
            "Finished training component 'SubtractByX'.",
            "Restored component 'CacheableComponent' from cache.",
        }


def test_resources_fingerprints_are_unique_when_cached(
    temp_cache: LocalTrainingCache, train_with_schema: Callable
):
    train_schema = GraphSchema(
        {
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
                config={},
            ),
            "assert_node": SchemaNode(
                needs={"i": "process"},
                uses=AssertComponent,
                fn="run_assert",
                constructor_name="create",
                config={"value_to_assert": "4"},
                is_target=True,
            ),
        }
    )

    # Train to cache
    train_with_schema(train_schema, temp_cache)

    train_schema.nodes["train"].config["test_value"] = "5"
    train_schema.nodes["assert_node"].config["value_to_assert"] = "5"
    train_with_schema(train_schema, temp_cache)

    # Add something to the config so only "assert_node" re-runs.
    train_schema.nodes["assert_node"].config["something"] = "something"
    # This breaks when `Resource`s use the node name as a fingerprint.
    # This is because the `Resource` for the first run is retrieved from the cache which
    # returns 4 whereas it should be the second resource which returns 5, and the schema
    # assert_node expects 5 now.
    train_with_schema(train_schema, temp_cache)


def test_resources_fingerprints_remain_after_being_cached(
    temp_cache: LocalTrainingCache, train_with_schema: Callable
):
    train_schema = GraphSchema(
        {
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
                config={},
                is_target=True,
            ),
        }
    )

    # Train and cache.
    train_with_schema(train_schema, temp_cache)

    # We can determine if a cached `Resource` has a static fingerprint by comparing two
    # subsequent cache entries of a child node.
    import sqlalchemy as sa

    with temp_cache._sessionmaker.begin() as session:
        # This will get the cache entry for the "process" node.
        query_for_most_recently_used_entry = sa.select(temp_cache.CacheEntry).order_by(
            temp_cache.CacheEntry.last_used.desc()
        )
        entry = session.execute(query_for_most_recently_used_entry).scalars().first()
        # The fingerprint key will incorporate the fingerprint of the `Resource`
        # provided by the "train" node. We save this key to compare after the next run.
        fingerprint_key = entry.fingerprint_key
        # Deleting the entry will force it to be recreated next train.
        delete_query = sa.delete(temp_cache.CacheEntry).where(
            temp_cache.CacheEntry.fingerprint_key == fingerprint_key
        )
        session.execute(delete_query)

    # In this second train, the Resource output of "train" will be retrieved from the
    # cache.
    train_with_schema(train_schema, temp_cache)

    with temp_cache._sessionmaker.begin() as session:
        # This will get the new cache entry for the "process" node.
        query_for_most_recently_used_entry = sa.select(temp_cache.CacheEntry).order_by(
            temp_cache.CacheEntry.last_used.desc()
        )
        entry = session.execute(query_for_most_recently_used_entry).scalars().first()
        # Assert the fingerprint key of the new entry is the same. This confirms that
        # the Resource from the cache has the same fingerprint.
        assert entry.fingerprint_key == fingerprint_key


@pytest.mark.parametrize(
    "on_before, on_after",
    [(lambda: True, lambda: 2 / 0), (lambda: 2 / 0, lambda: True)],
)
def test_exception_handling_for_on_before_hook(
    on_before: Callable,
    on_after: Callable,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
):
    schema_node = SchemaNode(
        needs={}, uses=ProvideX, fn="provide", constructor_name="create", config={}
    )

    class MyHook(GraphNodeHook):
        def on_after_node(
            self,
            node_name: Text,
            execution_context: ExecutionContext,
            config: Dict[Text, Any],
            output: Any,
            input_hook_data: Dict,
        ) -> None:
            on_before()

        def on_before_node(
            self,
            node_name: Text,
            execution_context: ExecutionContext,
            config: Dict[Text, Any],
            received_inputs: Dict[Text, Any],
        ) -> Dict:
            on_after()
            return {}

    node = GraphNode.from_schema_node(
        "some_node",
        schema_node,
        default_model_storage,
        default_execution_context,
        hooks=[MyHook()],
    )

    with pytest.raises(GraphComponentException):
        node()

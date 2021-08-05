import dataclasses
from typing import Text
import uuid

from rasa.engine.caching import TrainingCache
from rasa.engine.graph import ExecutionContext, GraphNode, GraphSchema, SchemaNode
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.training import fingerprinting
from rasa.engine.training.components import (
    CachedComponent,
    FingerprintComponent,
    FingerprintStatus,
)
from tests.engine.graph_components_test_classes import CacheableText


def test_cached_component_returns_value_from_cache(default_model_storage: ModelStorage):

    cached_output = CacheableText("Cache me!!")

    node = GraphNode(
        node_name="cached",
        component_class=CachedComponent,
        constructor_name="create",
        component_config={"output": cached_output},
        fn_name="get_cached_output",
        inputs={},
        eager=False,
        model_storage=default_model_storage,
        resource=None,
        execution_context=ExecutionContext(GraphSchema({}), "1"),
    )

    returned_output = node()["cached"]
    assert returned_output.text == "Cache me!!"


def test_cached_component_replace_schema_node():
    schema_node = SchemaNode(
        needs={"i1": "first_input", "i2": "second_input"},
        uses=FingerprintComponent,
        fn="add",
        constructor_name="load",
        config={"a": 1},
        eager=False,
        is_input=False,
        resource=Resource("hello")
    )

    CachedComponent.replace_schema_node(schema_node, 2)

    assert schema_node == SchemaNode(
        needs={"i1": "first_input", "i2": "second_input"},
        uses=CachedComponent,
        fn="get_cached_output",
        constructor_name="create",
        config={"output": 2},
        eager=False,
        is_input=False,
        resource=Resource("hello")
    )


def test_fingerprint_component_replace_schema_node(temp_cache: TrainingCache):
    schema_node = SchemaNode(
        needs={"i1": "first_input", "i2": "second_input"},
        uses=CachedComponent,
        fn="add",
        constructor_name="load",
        config={"a": 1},
        eager=False,
        is_input=False,
        resource=Resource("hello")
    )

    FingerprintComponent.replace_schema_node(schema_node, temp_cache, "some_node_name")

    assert schema_node == SchemaNode(
        needs={"i1": "first_input", "i2": "second_input"},
        uses=FingerprintComponent,
        fn="run",
        constructor_name="create",
        config={"a": 1, "cache": temp_cache, "node_name": "some_node_name"},
        eager=True,
        is_input=False,
        resource=Resource("hello")
    )


@dataclasses.dataclass
class FingerprintableText:
    text: Text

    def fingerprint(self):
        return self.text


def test_fingerprint_component_hit(
    default_model_storage: ModelStorage, temp_cache: TrainingCache
):

    cached_output = CacheableText("Cache me!!")
    output_fingerprint = uuid.uuid4().hex
    component_config = {"x": 1}
    fingerprint_key = fingerprinting.calculate_fingerprint_key(
        node_name="original_node",
        config=component_config,
        inputs={
            "param_1": FingerprintableText("input_1"),
            "param_2": FingerprintableText("input_2"),
        },
    )
    temp_cache.cache_output(
        fingerprint_key=fingerprint_key,
        output=cached_output,
        output_fingerprint=output_fingerprint,
        model_storage=default_model_storage,
    )

    node = GraphNode(
        node_name="fingerprint_node",
        component_class=FingerprintComponent,
        constructor_name="create",
        component_config={
            **component_config,
            "cache": temp_cache,
            "node_name": "original_node",
        },
        fn_name="run",
        inputs={"param_1": "parent_node_1", "param_2": "parent_node_2"},
        eager=False,
        model_storage=default_model_storage,
        resource=None,
        execution_context=ExecutionContext(GraphSchema({}), "1"),
    )

    returned_output = node(
        {
            "parent_node_1": FingerprintableText("input_1"),
            "parent_node_2": FingerprintStatus(
                is_hit=True, output_fingerprint="input_2"
            ),
        }
    )["fingerprint_node"]

    assert returned_output.is_hit is True
    assert returned_output.output_fingerprint == output_fingerprint
    assert returned_output.output_fingerprint == returned_output.fingerprint()


def test_fingerprint_component_miss(
    default_model_storage: ModelStorage, temp_cache: TrainingCache
):

    component_config = {"x": 1}

    node = GraphNode(
        node_name="fingerprint_node",
        component_class=FingerprintComponent,
        constructor_name="create",
        component_config={
            **component_config,
            "cache": temp_cache,
            "node_name": "original_node",
        },
        fn_name="run",
        inputs={"param_1": "parent_node_1", "param_2": "parent_node_2"},
        eager=False,
        model_storage=default_model_storage,
        resource=None,
        execution_context=ExecutionContext(GraphSchema({}), "1"),
    )

    returned_output = node(
        {
            "parent_node_1": FingerprintableText("input_1"),
            "parent_node_2": FingerprintStatus(
                is_hit=True, output_fingerprint="input_2"
            ),
        }
    )["fingerprint_node"]

    assert returned_output.is_hit is False
    assert returned_output.output_fingerprint is None
    assert returned_output.fingerprint() != returned_output.output_fingerprint
    assert returned_output.fingerprint() != returned_output.fingerprint()

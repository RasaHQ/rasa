import uuid

import pytest

from rasa.engine.caching import TrainingCache
from rasa.engine.exceptions import GraphComponentException
from rasa.engine.graph import ExecutionContext, GraphNode, GraphSchema
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.training.components import CachedComponent
from tests.engine.graph_components_test_classes import CacheableText


def test_cached_component_returns_value_from_cache(
    default_model_storage: ModelStorage, temp_cache: TrainingCache
):
    output_fingerprint = uuid.uuid4().hex

    cached_output = CacheableText("Cache me!!")

    temp_cache.cache_output(
        fingerprint_key="some_key",
        output=cached_output,
        output_fingerprint=output_fingerprint,
        model_storage=default_model_storage,
    )

    node = GraphNode(
        node_name="cached",
        component_class=CachedComponent,
        constructor_name="create",
        component_config={
            "cache": temp_cache,
            "cached_node_name": "the_original_node",
            "output_fingerprint": output_fingerprint,
        },
        fn_name="get_cached_output",
        inputs={},
        eager=False,
        model_storage=default_model_storage,
        resource=None,
        execution_context=ExecutionContext(GraphSchema({}), "1"),
    )

    returned_output = node()["cached"]
    assert returned_output.text == "Cache me!!"


def test_cached_component_returns_resource_from_cache(
    default_model_storage: ModelStorage, temp_cache: TrainingCache
):
    output_fingerprint = uuid.uuid4().hex

    resource = Resource("the_original_node")
    with default_model_storage.write_to(resource) as temporary_directory:
        file = temporary_directory / "file.txt"
        file.write_text("Cache me!!")

    temp_cache.cache_output(
        fingerprint_key="some_key",
        output=resource,
        output_fingerprint=output_fingerprint,
        model_storage=default_model_storage,
    )

    node = GraphNode(
        node_name="the_cached_node",
        component_class=CachedComponent,
        constructor_name="create",
        component_config={
            "cache": temp_cache,
            "cached_node_name": "the_original_node",
            "output_fingerprint": output_fingerprint,
        },
        fn_name="get_cached_output",
        inputs={},
        eager=False,
        model_storage=default_model_storage,
        resource=None,
        execution_context=ExecutionContext(GraphSchema({}), "1"),
    )

    returned_output = node()["the_cached_node"]
    assert isinstance(returned_output, Resource)

    with default_model_storage.read_from(returned_output) as resource_directory:
        file = resource_directory / "file.txt"
        assert file.read_text() == "Cache me!!"


def test_cached_component_with_no_cached_value(
    default_model_storage: ModelStorage, temp_cache: TrainingCache
):
    output_fingerprint = uuid.uuid4().hex

    non_cacheable_output = "Cache me!!"

    temp_cache.cache_output(
        fingerprint_key="some_key",
        output=non_cacheable_output,
        output_fingerprint=output_fingerprint,
        model_storage=default_model_storage,
    )

    node = GraphNode(
        node_name="cached",
        component_class=CachedComponent,
        constructor_name="create",
        component_config={
            "cache": temp_cache,
            "cached_node_name": "the_original_node",
            "output_fingerprint": output_fingerprint,
        },
        fn_name="get_cached_output",
        inputs={},
        eager=False,
        model_storage=default_model_storage,
        resource=None,
        execution_context=ExecutionContext(GraphSchema({}), "1"),
    )

    with pytest.raises(GraphComponentException):
        node()

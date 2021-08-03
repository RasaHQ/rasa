from rasa.engine.caching import TrainingCache
from rasa.engine.graph import ExecutionContext, GraphNode, GraphSchema
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.training.hooks import TrainingHook
import rasa.shared.utils.io
from tests.engine.graph_components_test_classes import (
    CacheableComponent,
    CacheableText,
)


def test_training_hook_saves_to_cache(
    default_model_storage: ModelStorage,
    temp_cache: TrainingCache,
    default_training_hook: TrainingHook,
):
    node = GraphNode(
        node_name="hello",
        component_class=CacheableComponent,
        constructor_name="create",
        component_config={},
        fn_name="run",
        inputs={"suffix": "input_node"},
        eager=False,
        model_storage=default_model_storage,
        resource=None,
        execution_context=ExecutionContext(GraphSchema({}), "1"),
        hooks=[default_training_hook],
    )

    node({"input_node": "Joe"})

    # This is the same key that the hook will generate
    fingerprint_key = rasa.shared.utils.io.deep_container_fingerprint(
        {
            "node_name": "hello",
            "config": {"prefix": "Hello "},
            "inputs": {"input_node": "Joe"},
        }
    )

    output_fingerprint_key = temp_cache.get_cached_output_fingerprint(fingerprint_key)
    assert output_fingerprint_key

    cached_result = temp_cache.get_cached_result(
        output_fingerprint_key=output_fingerprint_key,
        model_storage=default_model_storage,
        node_name="hello",
    )
    assert isinstance(cached_result, CacheableText)
    assert cached_result.text == "Hello Joe"

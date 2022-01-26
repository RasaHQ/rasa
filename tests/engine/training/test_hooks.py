from rasa.engine.caching import TrainingCache
from rasa.engine.graph import ExecutionContext, GraphNode, GraphSchema, SchemaNode
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.training import fingerprinting
from rasa.engine.training.components import PrecomputedValueProvider
from rasa.engine.training.hooks import TrainingHook
from tests.engine.graph_components_test_classes import CacheableComponent, CacheableText


def test_training_hook_saves_to_cache(
    default_model_storage: ModelStorage, temp_cache: TrainingCache
):
    # We need an execution context so the hook can determine the class of the graph
    # component
    execution_context = ExecutionContext(
        GraphSchema(
            {
                "hello": SchemaNode(
                    needs={},
                    constructor_name="create",
                    fn="run",
                    config={},
                    uses=CacheableComponent,
                )
            }
        ),
        "1",
    )
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
        execution_context=execution_context,
        hooks=[
            TrainingHook(
                cache=temp_cache,
                model_storage=default_model_storage,
                pruned_schema=execution_context.graph_schema,
            )
        ],
    )

    node(("input_node", "Joe"))

    # This is the same key that the hook will generate
    fingerprint_key = fingerprinting.calculate_fingerprint_key(
        graph_component_class=CacheableComponent,
        config={"prefix": "Hello "},
        inputs={"suffix": "Joe"},
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


def test_training_hook_does_not_cache_cached_component(
    default_model_storage: ModelStorage, temp_cache: TrainingCache
):
    # We need an execution context so the hook can determine the class of the graph
    # component
    execution_context = ExecutionContext(
        GraphSchema(
            {
                "hello": SchemaNode(
                    needs={},
                    constructor_name="create",
                    fn="run",
                    config={},
                    uses=PrecomputedValueProvider,
                )
            }
        ),
        "1",
    )
    node = GraphNode(
        node_name="hello",
        component_class=PrecomputedValueProvider,
        constructor_name="create",
        component_config={"output": CacheableText("hi")},
        fn_name="get_value",
        inputs={},
        eager=False,
        model_storage=default_model_storage,
        resource=None,
        execution_context=execution_context,
        hooks=[
            TrainingHook(
                cache=temp_cache,
                model_storage=default_model_storage,
                pruned_schema=execution_context.graph_schema,
            )
        ],
    )

    node(("input_node", "Joe"))

    # This is the same key that the hook will generate
    fingerprint_key = fingerprinting.calculate_fingerprint_key(
        graph_component_class=PrecomputedValueProvider,
        config={"output": CacheableText("hi")},
        inputs={},
    )

    # The hook should not cache the output of a PrecomputedValueProvider
    assert not temp_cache.get_cached_output_fingerprint(fingerprint_key)

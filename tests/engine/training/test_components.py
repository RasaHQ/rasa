from rasa.engine.graph import ExecutionContext, GraphNode, GraphSchema
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.training.components import CachedComponent
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

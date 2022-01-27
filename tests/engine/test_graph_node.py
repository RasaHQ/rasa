from __future__ import annotations
from typing import Any, Dict, Optional, Text
from unittest.mock import Mock

import pytest

import rasa.shared
import rasa.shared.utils
import rasa.shared.utils.io
from rasa.engine.exceptions import GraphComponentException
from rasa.engine.graph import ExecutionContext, GraphComponent, GraphNode, GraphSchema
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.storage.resource import Resource
from tests.engine.graph_components_test_classes import (
    AddInputs,
    ExecutionContextAware,
    ProvideX,
    SubtractByX,
    PersistableTestComponent,
)


def test_calling_component(default_model_storage: ModelStorage):
    node = GraphNode(
        node_name="add_node",
        component_class=AddInputs,
        constructor_name="create",
        component_config={},
        fn_name="add",
        inputs={"i1": "input_node1", "i2": "input_node2"},
        eager=False,
        model_storage=default_model_storage,
        resource=None,
        execution_context=ExecutionContext(GraphSchema({}), "1"),
    )

    result = node(("input_node1", 3), ("input_node2", 4))

    assert result == ("add_node", 7)


@pytest.mark.parametrize("x, output", [(None, 5), (0, 5), (1, 4), (2, 3)])
def test_component_config(
    x: Optional[int], output: int, default_model_storage: ModelStorage
):
    node = GraphNode(
        node_name="subtract",
        component_class=SubtractByX,
        constructor_name="create",
        component_config={"x": x} if x else {},
        fn_name="subtract_x",
        inputs={"i": "input_node"},
        eager=False,
        model_storage=default_model_storage,
        resource=None,
        execution_context=ExecutionContext(GraphSchema({}), "1"),
    )

    result = node(("input_node", 5))

    assert result == ("subtract", output)


def test_can_use_alternate_constructor(default_model_storage: ModelStorage):
    node = GraphNode(
        node_name="provide",
        component_class=ProvideX,
        constructor_name="create_with_2",
        component_config={},
        fn_name="provide",
        inputs={},
        eager=False,
        model_storage=default_model_storage,
        resource=None,
        execution_context=ExecutionContext(GraphSchema({}), "1"),
    )

    result = node()
    assert result == ("provide", 2)


@pytest.mark.parametrize("eager", [True, False])
def test_eager_and_not_eager(eager: bool, default_model_storage: ModelStorage):
    run_mock = Mock()
    create_mock = Mock()

    class SpyComponent(GraphComponent):
        @classmethod
        def create(
            cls,
            config: Dict,
            model_storage: ModelStorage,
            resource: Resource,
            execution_context: ExecutionContext,
        ) -> SpyComponent:
            create_mock()
            return cls()

        def run(self):
            return run_mock()

    node = GraphNode(
        node_name="spy_node",
        component_class=SpyComponent,
        constructor_name="create",
        component_config={},
        fn_name="run",
        inputs={},
        eager=eager,
        model_storage=default_model_storage,
        resource=None,
        execution_context=ExecutionContext(GraphSchema({}), "1"),
    )

    if eager:
        assert create_mock.called
    else:
        assert not create_mock.called

    assert not run_mock.called

    node()

    assert create_mock.call_count == 1
    assert run_mock.called


def test_non_eager_can_use_inputs_for_constructor(default_model_storage: ModelStorage):
    node = GraphNode(
        node_name="provide",
        component_class=ProvideX,
        constructor_name="create",
        component_config={},
        fn_name="provide",
        inputs={"x": "input_node"},
        eager=False,
        model_storage=default_model_storage,
        resource=None,
        execution_context=ExecutionContext(GraphSchema({}), "1"),
    )

    result = node(("input_node", 5))

    assert result == ("provide", 5)


def test_execution_context(default_model_storage: ModelStorage):
    context = ExecutionContext(GraphSchema({}), "some_id")
    node = GraphNode(
        node_name="execution_context_aware",
        component_class=ExecutionContextAware,
        constructor_name="create",
        component_config={},
        fn_name="get_execution_context",
        inputs={},
        eager=False,
        model_storage=default_model_storage,
        resource=None,
        execution_context=context,
    )

    context.model_id = "a_new_id"

    result = node()[1]
    assert result.model_id == "some_id"
    assert result.node_name == "execution_context_aware"


def test_constructor_exception(default_model_storage: ModelStorage):
    class BadConstructor(GraphComponent):
        @classmethod
        def create(
            cls,
            config: Dict[Text, Any],
            model_storage: ModelStorage,
            resource: Resource,
            execution_context: ExecutionContext,
        ) -> BadConstructor:
            raise ValueError("oh no!")

        def run(self) -> None:
            ...

    with pytest.raises(GraphComponentException):
        GraphNode(
            node_name="bad_constructor",
            component_class=BadConstructor,
            constructor_name="create",
            component_config={},
            fn_name="run",
            inputs={},
            eager=True,
            model_storage=default_model_storage,
            resource=None,
            execution_context=ExecutionContext(GraphSchema({}), "some_id"),
        )


def test_fn_exception(default_model_storage: ModelStorage):
    class BadFn(GraphComponent):
        @classmethod
        def create(
            cls,
            config: Dict[Text, Any],
            model_storage: ModelStorage,
            resource: Resource,
            execution_context: ExecutionContext,
        ) -> BadFn:
            return cls()

        def run(self) -> None:
            raise ValueError("Oh no!")

    node = GraphNode(
        node_name="bad_fn",
        component_class=BadFn,
        constructor_name="create",
        component_config={},
        fn_name="run",
        inputs={},
        eager=True,
        model_storage=default_model_storage,
        resource=None,
        execution_context=ExecutionContext(GraphSchema({}), "some_id"),
    )

    with pytest.raises(GraphComponentException):
        node()


def test_writing_to_resource_during_training(default_model_storage: ModelStorage):
    node_name = "some_name"

    test_value_for_sub_directory = {"test": "test value sub dir"}
    test_value = {"test dir": "test value dir"}

    node = GraphNode(
        node_name=node_name,
        component_class=PersistableTestComponent,
        constructor_name="create",
        component_config={
            "test_value": test_value,
            "test_value_for_sub_directory": test_value_for_sub_directory,
        },
        fn_name="train",
        inputs={},
        eager=False,
        model_storage=default_model_storage,
        resource=None,
        execution_context=ExecutionContext(GraphSchema({}), "123"),
    )

    _, resource = node()

    assert resource == Resource(node_name)

    with default_model_storage.read_from(resource) as directory:
        assert (
            rasa.shared.utils.io.read_json_file(directory / "test.json") == test_value
        )
        assert (
            rasa.shared.utils.io.read_json_file(directory / "sub_dir" / "test.json")
            == test_value_for_sub_directory
        )


def test_loading_from_resource_not_eager(default_model_storage: ModelStorage):
    previous_resource = Resource("previous resource")
    parent_node_name = "parent"
    test_value = {"test": "test value"}

    # Pretend resource persisted itself before
    with default_model_storage.write_to(previous_resource) as directory:
        rasa.shared.utils.io.dump_obj_as_json_to_file(
            directory / "test.json", test_value
        )

    node_name = "some_name"
    node = GraphNode(
        node_name=node_name,
        component_class=PersistableTestComponent,
        constructor_name="load",
        component_config={},
        fn_name="run_train_process",
        inputs={"resource": parent_node_name},
        eager=False,
        model_storage=default_model_storage,
        # The `GraphComponent` should load from this resource
        resource=None,
        execution_context=ExecutionContext(GraphSchema({}), "123"),
    )

    _, value = node((parent_node_name, previous_resource))

    assert value == test_value


def test_loading_from_resource_eager(default_model_storage: ModelStorage):
    previous_resource = Resource("previous resource")
    test_value = {"test": "test value"}

    # Pretend resource persisted itself before
    with default_model_storage.write_to(previous_resource) as directory:
        rasa.shared.utils.io.dump_obj_as_json_to_file(
            directory / "test.json", test_value
        )

    node_name = "some_name"
    node = GraphNode(
        node_name=node_name,
        component_class=PersistableTestComponent,
        constructor_name="load",
        component_config={},
        fn_name="run_inference",
        inputs={},
        eager=True,
        model_storage=default_model_storage,
        # The `GraphComponent` should load from this resource
        resource=previous_resource,
        execution_context=ExecutionContext(GraphSchema({}), "123"),
    )

    actual_node_name, value = node()

    assert actual_node_name == node_name
    assert value == test_value


def test_config_with_nested_dict_override(default_model_storage: ModelStorage):
    class ComponentWithNestedDictConfig(GraphComponent):
        @staticmethod
        def get_default_config() -> Dict[Text, Any]:
            return {"nested-dict": {"key1": "value1", "key2": "value2"}}

        @classmethod
        def create(
            cls,
            config: Dict,
            model_storage: ModelStorage,
            resource: Resource,
            execution_context: ExecutionContext,
            **kwargs: Any,
        ) -> ComponentWithNestedDictConfig:
            return cls()

        def run(self) -> None:
            return None

    node = GraphNode(
        node_name="nested_dict_config",
        component_class=ComponentWithNestedDictConfig,
        constructor_name="create",
        component_config={"nested-dict": {"key2": "override-value2"}},
        fn_name="run",
        inputs={},
        eager=True,
        model_storage=default_model_storage,
        resource=None,
        execution_context=ExecutionContext(GraphSchema({}), "123"),
    )

    expected_config = {"nested-dict": {"key1": "value1", "key2": "override-value2"}}

    for key, value in expected_config.items():
        assert key in node._component_config
        if isinstance(value, dict):
            for nested_key, nested_value in expected_config[key].items():
                assert nested_key in node._component_config[key]
                assert node._component_config[key][nested_key] == nested_value

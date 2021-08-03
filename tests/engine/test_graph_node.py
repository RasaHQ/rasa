from __future__ import annotations
from typing import Any, Dict, Optional, Text
from unittest.mock import Mock

import pytest

from rasa.engine.exceptions import GraphComponentException
from rasa.engine.graph import (
    ExecutionContext,
    GraphComponent,
    GraphNode,
)
from tests.engine.graph_components_test_classes import (
    AddInputs,
    ExecutionContextAware,
    ProvideX,
    SubtractByX,
)


def test_calling_component():
    node = GraphNode(
        node_name="add_node",
        component_class=AddInputs,
        constructor_name="create",
        component_config={},
        fn_name="add",
        inputs={"i1": "input_node1", "i2": "input_node2"},
        eager=False,
        execution_context=ExecutionContext({}, "1"),
    )

    result = node({"input_node1": 3}, {"input_node2": 4})
    assert result == {"add_node": 7}


@pytest.mark.parametrize("x, output", [(None, 5), (0, 5), (1, 4), (2, 3)])
def test_component_config(x: Optional[int], output: int):
    node = GraphNode(
        node_name="subtract",
        component_class=SubtractByX,
        constructor_name="create",
        component_config={"x": x} if x else {},
        fn_name="subtract_x",
        inputs={"i": "input_node"},
        eager=False,
        execution_context=ExecutionContext({}, "1"),
    )

    result = node({"input_node": 5})
    assert result == {"subtract": output}


def test_can_use_alternate_constructor():
    node = GraphNode(
        node_name="provide",
        component_class=ProvideX,
        constructor_name="create_with_2",
        component_config={},
        fn_name="provide",
        inputs={},
        eager=False,
        execution_context=ExecutionContext({}, "1"),
    )

    result = node()
    assert result == {"provide": 2}


@pytest.mark.parametrize("eager", [True, False])
def test_eager_and_not_eager(eager: bool):
    run_mock = Mock()
    create_mock = Mock()

    class SpyComponent(GraphComponent):
        default_config = {}

        @classmethod
        def create(
            cls, config: Dict, execution_context: ExecutionContext
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
        execution_context=ExecutionContext({}, "1"),
    )

    if eager:
        assert create_mock.called
    else:
        assert not create_mock.called

    assert not run_mock.called

    node()

    assert create_mock.call_count == 1
    assert run_mock.called


def test_non_eager_can_use_inputs_for_constructor():
    node = GraphNode(
        node_name="provide",
        component_class=ProvideX,
        constructor_name="create",
        component_config={},
        fn_name="provide",
        inputs={"x": "input_node"},
        eager=False,
        execution_context=ExecutionContext({}, "1"),
    )

    result = node({"input_node": 5})
    assert result == {"provide": 5}


def test_execution_context():
    node = GraphNode(
        node_name="model_id",
        component_class=ExecutionContextAware,
        constructor_name="create",
        component_config={},
        fn_name="get_model_id",
        inputs={},
        eager=False,
        execution_context=ExecutionContext({}, "some_id"),
    )

    result = node()
    assert result == {"model_id": "some_id"}


def test_constructor_exception():
    class BadConstructor(GraphComponent):
        default_config = {}

        @classmethod
        def create(
            cls, config: Dict[Text, Any], execution_context: ExecutionContext
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
            execution_context=ExecutionContext({}, "some_id"),
        )


def test_fn_exception():
    class BadFn(GraphComponent):
        default_config = {}

        @classmethod
        def create(
            cls, config: Dict[Text, Any], execution_context: ExecutionContext
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
        execution_context=ExecutionContext({}, "some_id"),
    )

    with pytest.raises(GraphComponentException):
        node()

from __future__ import annotations
from typing import Dict, List, Optional, Text

import pytest

from rasa.engine.graph import (
    ExecutionContext,
    GraphComponent,
    GraphSchema,
    SchemaNode,
)
from rasa.engine.exceptions import GraphRunError
from rasa.engine.runner.dask import DaskGraphRunner


class AddInputs(GraphComponent):
    default_config = {}

    @classmethod
    def create(cls, config: Dict, execution_context: ExecutionContext) -> AddInputs:
        return cls()

    def supported_languages(self) -> List[Text]:
        return ["en"]

    def required_packages(self) -> List[Text]:
        return []

    def add(self, i1: int, i2: int) -> int:
        return i1 + i2


class SubtractByX(GraphComponent):
    default_config = {"x": 0}

    def __init__(self, x: int) -> None:
        self._x = x

    @classmethod
    def create(cls, config: Dict, execution_context: ExecutionContext) -> SubtractByX:
        return cls(config["x"])

    def supported_languages(self) -> List[Text]:
        return ["en"]

    def required_packages(self) -> List[Text]:
        return []

    def subtract_x(self, i: int) -> int:
        return i - self._x


class ProvideX(GraphComponent):
    default_config = {}

    def __init__(self) -> None:
        self.x = 1

    @classmethod
    def create(
        cls, config: Dict, execution_context: ExecutionContext, x: Optional[int] = None
    ) -> ProvideX:
        instance = cls()
        if x:
            instance.x = x
        return instance

    @classmethod
    def create_with_2(
        cls, config: Dict, execution_context: ExecutionContext
    ) -> ProvideX:
        return cls.create(config, execution_context, 2)

    def supported_languages(self) -> List[Text]:
        return ["en"]

    def required_packages(self) -> List[Text]:
        return []

    def provide(self) -> int:
        return self.x


class ExecutionContextAware(GraphComponent):
    default_config = {}

    def __init__(self, model_id: Text) -> None:
        self.model_id = model_id

    @classmethod
    def create(
        cls, config: Dict, execution_context: ExecutionContext
    ) -> ExecutionContextAware:
        return cls(execution_context.model_id)

    def supported_languages(self) -> List[Text]:
        return ["en"]

    def required_packages(self) -> List[Text]:
        return []

    def get_model_id(self) -> Text:
        return self.model_id


@pytest.mark.parametrize("eager", [True, False])
def test_multi_node_graph_run(eager: bool):
    graph_schema: GraphSchema = {
        "add": SchemaNode(
            needs={"i1": "first_input", "i2": "second_input"},
            uses=AddInputs,
            fn="add",
            constructor_name="create",
            config={},
            eager=eager,
        ),
        "subtract_2": SchemaNode(
            needs={"i": "add"},
            uses=SubtractByX,
            fn="subtract_x",
            constructor_name="create",
            config={"x": 2},
            eager=eager,
            is_target=True,
        ),
    }

    execution_context = ExecutionContext(graph_schema=graph_schema, model_id="1")

    runner = DaskGraphRunner(
        graph_schema=graph_schema, execution_context=execution_context
    )
    results = runner.run(inputs={"first_input": 3, "second_input": 4})
    assert results["subtract_2"] == 5


@pytest.mark.parametrize("eager", [True, False])
def test_target_override(eager: bool):
    graph_schema: GraphSchema = {
        "add": SchemaNode(
            needs={"i1": "first_input", "i2": "second_input"},
            uses=AddInputs,
            fn="add",
            constructor_name="create",
            config={},
            eager=eager,
        ),
        "subtract_2": SchemaNode(
            needs={"i": "add"},
            uses=SubtractByX,
            fn="subtract_x",
            constructor_name="create",
            config={"x": 3},
            eager=eager,
            is_target=True,
        ),
    }

    execution_context = ExecutionContext(graph_schema=graph_schema, model_id="1")

    runner = DaskGraphRunner(
        graph_schema=graph_schema, execution_context=execution_context
    )
    results = runner.run(inputs={"first_input": 3, "second_input": 4}, targets=["add"])
    assert results == {"add": 7}


@pytest.mark.parametrize("x, output", [(None, 5), (0, 5), (1, 4), (2, 3)])
def test_default_config(x: Optional[int], output: int):
    graph_schema: GraphSchema = {
        "subtract": SchemaNode(
            needs={"i": "input"},
            uses=SubtractByX,
            fn="subtract_x",
            constructor_name="create",
            config={"x": x} if x else {},
            is_target=True,
        ),
    }

    runner = DaskGraphRunner(
        graph_schema=graph_schema,
        execution_context=ExecutionContext(graph_schema=graph_schema, model_id="1"),
    )
    results = runner.run(inputs={"input": 5})
    assert results["subtract"] == output


def test_empty_schema():
    runner = DaskGraphRunner(
        graph_schema={},
        execution_context=ExecutionContext(graph_schema={}, model_id="1"),
    )
    results = runner.run()
    assert not results


def test_no_inputs():
    graph_schema: GraphSchema = {
        "provide": SchemaNode(
            needs={},
            uses=ProvideX,
            fn="provide",
            constructor_name="create",
            config={},
            is_target=True,
        ),
    }
    runner = DaskGraphRunner(
        graph_schema=graph_schema,
        execution_context=ExecutionContext(graph_schema=graph_schema, model_id="1"),
    )
    results = runner.run()
    assert results["provide"] == 1


def test_no_target():
    graph_schema: GraphSchema = {
        "provide": SchemaNode(
            needs={}, uses=ProvideX, fn="provide", constructor_name="create", config={},
        ),
    }
    runner = DaskGraphRunner(
        graph_schema=graph_schema,
        execution_context=ExecutionContext(graph_schema=graph_schema, model_id="1"),
    )
    results = runner.run()
    assert not results


def test_non_eager_can_use_inputs_for_constructor():
    graph_schema: GraphSchema = {
        "provide": SchemaNode(
            needs={"x": "input"},
            uses=ProvideX,
            fn="provide",
            constructor_name="create",
            config={},
            eager=False,
            is_target=True,
        ),
    }
    runner = DaskGraphRunner(
        graph_schema=graph_schema,
        execution_context=ExecutionContext(graph_schema=graph_schema, model_id="1"),
    )
    results = runner.run(inputs={"input": 5})
    assert results["provide"] == 5


def test_can_use_alternate_constructor():
    graph_schema: GraphSchema = {
        "provide": SchemaNode(
            needs={},
            uses=ProvideX,
            fn="provide",
            constructor_name="create_with_2",
            config={},
            is_target=True,
        ),
    }
    runner = DaskGraphRunner(
        graph_schema=graph_schema,
        execution_context=ExecutionContext(graph_schema=graph_schema, model_id="1"),
    )
    results = runner.run()
    assert results["provide"] == 2


def test_execution_context():
    graph_schema: GraphSchema = {
        "model_id": SchemaNode(
            needs={},
            uses=ExecutionContextAware,
            fn="get_model_id",
            constructor_name="create",
            config={},
            is_target=True,
        ),
    }
    runner = DaskGraphRunner(
        graph_schema=graph_schema,
        execution_context=ExecutionContext(
            graph_schema=graph_schema, model_id="some_id"
        ),
    )
    results = runner.run()
    assert results["model_id"] == "some_id"


def test_loop():
    graph_schema: GraphSchema = {
        "subtract_a": SchemaNode(
            needs={"i": "subtract_b"},
            uses=SubtractByX,
            fn="subtract_x",
            constructor_name="create",
            config={},
            is_target=False,
        ),
        "subtract_b": SchemaNode(
            needs={"i": "subtract_a"},
            uses=SubtractByX,
            fn="subtract_x",
            constructor_name="create",
            config={},
            is_target=True,
        ),
    }
    runner = DaskGraphRunner(
        graph_schema=graph_schema,
        execution_context=ExecutionContext(graph_schema=graph_schema, model_id="1"),
    )
    with pytest.raises(GraphRunError):
        runner.run()


def test_input_value_is_node_name():
    graph_schema: GraphSchema = {
        "provide": SchemaNode(
            needs={},
            uses=ProvideX,
            fn="provide",
            constructor_name="create",
            config={},
            is_target=True,
        ),
    }
    runner = DaskGraphRunner(
        graph_schema=graph_schema,
        execution_context=ExecutionContext(graph_schema=graph_schema, model_id="1"),
    )
    with pytest.raises(GraphRunError):
        runner.run(inputs={"input": "provide"})

from __future__ import annotations
from typing import Optional

import pytest

from rasa.engine.graph import ExecutionContext, GraphSchema, SchemaNode
from rasa.engine.exceptions import GraphRunError
from rasa.engine.runner.dask import DaskGraphRunner
from rasa.engine.storage.storage import ModelStorage
from tests.engine.graph_components_test_classes import (
    AddInputs,
    AssertComponent,
    ExecutionContextAware,
    ProvideX,
    SubtractByX,
    PersistableTestComponent,
)


@pytest.mark.parametrize("eager", [True, False])
def test_multi_node_graph_run(eager: bool, default_model_storage: ModelStorage):
    graph_schema = GraphSchema(
        {
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
    )

    execution_context = ExecutionContext(graph_schema=graph_schema, model_id="1")

    runner = DaskGraphRunner(
        graph_schema=graph_schema,
        model_storage=default_model_storage,
        execution_context=execution_context,
    )
    results = runner.run(inputs={"first_input": 3, "second_input": 4})
    assert results["subtract_2"] == 5


@pytest.mark.parametrize("eager", [True, False])
def test_target_override(eager: bool, default_model_storage: ModelStorage):
    graph_schema = GraphSchema(
        {
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
    )

    execution_context = ExecutionContext(graph_schema=graph_schema, model_id="1")

    runner = DaskGraphRunner(
        graph_schema=graph_schema,
        model_storage=default_model_storage,
        execution_context=execution_context,
    )
    results = runner.run(inputs={"first_input": 3, "second_input": 4}, targets=["add"])
    assert results == {"add": 7}


@pytest.mark.parametrize("x, output", [(None, 5), (0, 5), (1, 4), (2, 3)])
def test_default_config(
    x: Optional[int], output: int, default_model_storage: ModelStorage
):
    graph_schema = GraphSchema(
        {
            "subtract": SchemaNode(
                needs={"i": "input"},
                uses=SubtractByX,
                fn="subtract_x",
                constructor_name="create",
                config={"x": x} if x else {},
                is_target=True,
            )
        }
    )

    runner = DaskGraphRunner(
        graph_schema=graph_schema,
        model_storage=default_model_storage,
        execution_context=ExecutionContext(graph_schema=graph_schema, model_id="1"),
    )
    results = runner.run(inputs={"input": 5})
    assert results["subtract"] == output


def test_empty_schema(default_model_storage: ModelStorage):
    empty_schema = GraphSchema({})
    runner = DaskGraphRunner(
        graph_schema=empty_schema,
        model_storage=default_model_storage,
        execution_context=ExecutionContext(graph_schema=empty_schema, model_id="1"),
    )
    results = runner.run()
    assert not results


def test_no_inputs(default_model_storage: ModelStorage):
    graph_schema = GraphSchema(
        {
            "provide": SchemaNode(
                needs={},
                uses=ProvideX,
                fn="provide",
                constructor_name="create",
                config={},
                is_target=True,
            )
        }
    )
    runner = DaskGraphRunner(
        graph_schema=graph_schema,
        model_storage=default_model_storage,
        execution_context=ExecutionContext(graph_schema=graph_schema, model_id="1"),
    )
    results = runner.run()
    assert results["provide"] == 1


def test_no_target(default_model_storage: ModelStorage):
    graph_schema = GraphSchema(
        {
            "provide": SchemaNode(
                needs={},
                uses=ProvideX,
                fn="provide",
                constructor_name="create",
                config={},
            )
        }
    )
    runner = DaskGraphRunner(
        graph_schema=graph_schema,
        model_storage=default_model_storage,
        execution_context=ExecutionContext(graph_schema=graph_schema, model_id="1"),
    )
    results = runner.run()
    assert not results


def test_unused_node(default_model_storage: ModelStorage):
    graph_schema = GraphSchema(
        {
            "provide": SchemaNode(
                needs={},
                uses=ProvideX,
                fn="provide",
                constructor_name="create",
                config={},
                is_target=True,
            ),
            # This node will not fail as it will be pruned because it is not a target
            # or a target's ancestor.
            "assert_false": SchemaNode(
                needs={"i": "input"},
                uses=AssertComponent,
                fn="run_assert",
                constructor_name="create",
                config={"value_to_assert": "some_value"},
            ),
        }
    )
    runner = DaskGraphRunner(
        graph_schema=graph_schema,
        model_storage=default_model_storage,
        execution_context=ExecutionContext(graph_schema=graph_schema, model_id="1"),
    )
    results = runner.run(inputs={"input": "some_other_value"})
    assert results == {"provide": 1}


def test_non_eager_can_use_inputs_for_constructor(default_model_storage: ModelStorage):
    graph_schema = GraphSchema(
        {
            "provide": SchemaNode(
                needs={"x": "input"},
                uses=ProvideX,
                fn="provide",
                constructor_name="create",
                config={},
                eager=False,
                is_target=True,
            )
        }
    )
    runner = DaskGraphRunner(
        graph_schema=graph_schema,
        model_storage=default_model_storage,
        execution_context=ExecutionContext(graph_schema=graph_schema, model_id="1"),
    )
    results = runner.run(inputs={"input": 5})
    assert results["provide"] == 5


def test_can_use_alternate_constructor(default_model_storage: ModelStorage):
    graph_schema = GraphSchema(
        {
            "provide": SchemaNode(
                needs={},
                uses=ProvideX,
                fn="provide",
                constructor_name="create_with_2",
                config={},
                is_target=True,
            )
        }
    )
    runner = DaskGraphRunner(
        graph_schema=graph_schema,
        model_storage=default_model_storage,
        execution_context=ExecutionContext(graph_schema=graph_schema, model_id="1"),
    )
    results = runner.run()
    assert results["provide"] == 2


def test_execution_context(default_model_storage: ModelStorage):
    graph_schema = GraphSchema(
        {
            "execution_context_aware": SchemaNode(
                needs={},
                uses=ExecutionContextAware,
                fn="get_execution_context",
                constructor_name="create",
                config={},
                is_target=True,
            )
        }
    )
    context = ExecutionContext(graph_schema=graph_schema, model_id="some_id")
    runner = DaskGraphRunner(
        graph_schema=graph_schema,
        model_storage=default_model_storage,
        execution_context=context,
    )
    context.model_id = "a_new_id"
    result = runner.run()["execution_context_aware"]
    assert result.model_id == "some_id"
    assert result.node_name == "execution_context_aware"


def test_input_value_is_node_name(default_model_storage: ModelStorage):
    graph_schema = GraphSchema(
        {
            "provide": SchemaNode(
                needs={},
                uses=ProvideX,
                fn="provide",
                constructor_name="create",
                config={},
                is_target=True,
            )
        }
    )
    runner = DaskGraphRunner(
        graph_schema=graph_schema,
        model_storage=default_model_storage,
        execution_context=ExecutionContext(graph_schema=graph_schema, model_id="1"),
    )
    with pytest.raises(GraphRunError):
        runner.run(inputs={"input": "provide"})


def test_loading_from_previous_node(default_model_storage: ModelStorage):
    test_value_for_sub_directory = {"test": "test value sub dir"}
    test_value = {"test dir": "test value dir"}

    graph_schema = GraphSchema(
        {
            "train": SchemaNode(
                needs={},
                uses=PersistableTestComponent,
                fn="train",
                constructor_name="create",
                config={
                    "test_value": test_value,
                    "test_value_for_sub_directory": test_value_for_sub_directory,
                },
            ),
            "load": SchemaNode(
                needs={"resource": "train"},
                uses=PersistableTestComponent,
                fn="run_inference",
                constructor_name="load",
                config={},
                is_target=True,
            ),
        }
    )

    runner = DaskGraphRunner(
        graph_schema=graph_schema,
        model_storage=default_model_storage,
        execution_context=ExecutionContext(graph_schema=graph_schema, model_id="1"),
    )

    results = runner.run()

    assert results["load"] == test_value

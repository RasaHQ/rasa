from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Text

import dask.local
import dask.core
import asyncio

from rasa.engine.exceptions import GraphRunError
from rasa.engine.graph import ExecutionContext, GraphNode, GraphNodeHook, GraphSchema
from rasa.engine.runner.interface import GraphRunner
from rasa.engine.storage.storage import ModelStorage

logger = logging.getLogger(__name__)


class DaskGraphRunner(GraphRunner):
    """Dask implementation of a `GraphRunner`."""

    def __init__(
        self,
        graph_schema: GraphSchema,
        model_storage: ModelStorage,
        execution_context: ExecutionContext,
        hooks: Optional[List[GraphNodeHook]] = None,
    ) -> None:
        """Initializes a `DaskGraphRunner`.

        Args:
            graph_schema: The graph schema that will be run.
            model_storage: Storage which graph components can use to persist and load
                themselves.
            execution_context: Information about the current graph run to be passed to
                each node.
            hooks: These are called before and after the execution of each node.
        """
        self._graph_schema = graph_schema
        self._instantiated_nodes: Dict[Text, GraphNode] = self._instantiate_nodes(
            graph_schema, model_storage, execution_context, hooks
        )
        self._execution_context: ExecutionContext = execution_context

    @classmethod
    def create(
        cls,
        graph_schema: GraphSchema,
        model_storage: ModelStorage,
        execution_context: ExecutionContext,
        hooks: Optional[List[GraphNodeHook]] = None,
    ) -> DaskGraphRunner:
        """Creates the runner (see parent class for full docstring)."""
        return cls(graph_schema, model_storage, execution_context, hooks)

    @staticmethod
    def _instantiate_nodes(
        graph_schema: GraphSchema,
        model_storage: ModelStorage,
        execution_context: ExecutionContext,
        hooks: Optional[List[GraphNodeHook]] = None,
    ) -> Dict[Text, GraphNode]:
        return {
            node_name: GraphNode.from_schema_node(
                node_name, schema_node, model_storage, execution_context, hooks
            )
            for node_name, schema_node in graph_schema.nodes.items()
        }

    def _build_dask_graph(self, schema: GraphSchema) -> Dict[Text, Any]:
        """Builds a dask graph from the instantiated graph.

        For more information about dask graphs
        see: https://docs.dask.org/en/latest/spec.html
        """
        run_graph = {
            node_name: (
                self._instantiated_nodes[node_name],
                *schema_node.needs.values(),
            )
            for node_name, schema_node in schema.nodes.items()
        }
        return run_graph

    async def run(
        self,
        inputs: Optional[Dict[Text, Any]] = None,
        targets: Optional[List[Text]] = None,
    ) -> Dict[Text, Any]:
        """Runs the graph (see parent class for full docstring)."""
        run_targets = targets if targets else self._graph_schema.target_names
        minimal_schema = self._graph_schema.minimal_graph_schema(run_targets)
        run_graph = self._build_dask_graph(minimal_schema)

        if inputs:
            self._add_inputs_to_graph(inputs, run_graph)

        logger.debug(
            f"Running graph with inputs: {inputs}, targets: {targets} "
            f"and {self._execution_context}."
        )

        try:
            dask_result = await execute_dask_graph(run_graph, run_targets)
            return dict(dask_result)
        except KeyError as e:
            raise GraphRunError(
                f"Could not find key {e} in the graph. Error running runner. "
                f"Please check that you are running bot developed with CALM instead "
                f"of bot developed with previous version of dialog management (DM1)."
            ) from e
        except RuntimeError as e:
            raise GraphRunError("Error running runner.") from e

    @staticmethod
    def _add_inputs_to_graph(inputs: Optional[Dict[Text, Any]], graph: Any) -> None:
        if inputs is None:
            return

        for input_name, input_value in inputs.items():
            if isinstance(input_value, str) and input_value in graph.keys():
                raise GraphRunError(
                    f"Input value '{input_value}' clashes with a node name. Make sure "
                    f"that none of the input names passed to the `run` method are the "
                    f"same as node names in the graph schema."
                )
            graph[input_name] = (input_name, input_value)


async def _execute_task(arg, cache, dsk=None):
    """Do the actual work of collecting data and executing a function.

    Examples:
    --------
    >>> inc = lambda x: x + 1
    >>> add = lambda x, y: x + y
    >>> cache = {'x': 1, 'y': 2}

    Compute tasks against a cache
    >>> _execute_task((add, 'x', 1), cache)  # Compute task in naive manner
    2
    >>> _execute_task((add, (inc, 'x'), 1), cache)  # Support nested computation
    3

    Also grab data from cache
    >>> _execute_task('x', cache)
    1

    Support nested lists
    >>> list(_execute_task(['x', 'y'], cache))
    [1, 2]

    >>> list(map(list, _execute_task([['x', 'y'], ['y', 'x']], cache)))
    [[1, 2], [2, 1]]

    >>> _execute_task('foo', cache)  # Passes through on non-keys
    'foo'
    """
    if isinstance(arg, list):
        return [await _execute_task(a, cache) for a in arg]
    elif dask.core.istask(arg):
        func, args = arg[0], arg[1:]
        # Note: Don't assign the subtask results to a variable. numpy detects
        # temporaries by their reference count and can execute certain
        # operations in-place.
        awaited_args = await asyncio.gather(*(_execute_task(a, cache) for a in args))
        if asyncio.iscoroutinefunction(func):
            return await func(*awaited_args)
        else:
            return func(*awaited_args)
    elif not dask.core.ishashable(arg):
        return arg
    elif arg in cache:
        return cache[arg]
    else:
        return arg


async def execute_dask_graph(dsk: Dict[str, Any], result: List[str]):
    """Asynchronous get function.

    This is a general version of various asynchronous schedulers for dask.  It
    takes a ``concurrent.futures.Executor.submit`` function to form a more
    specific ``get`` method that walks through the dask array with parallel
    workers, avoiding repeat computation and minimizing memory use.

    Build on the version of `get_async` in `dask.local`. This version is
    asynchronous and uses `asyncio` to manage the event loop. It is designed to
    be used in an `async` function.

    The default dask implementation uses an internal event loop, which means
    blocking the outer event loop (e.g. sanic or webhooks).

    Parameters
    ----------
    dsk : dict
        A dask dictionary specifying a workflow
    result : key or list of keys
        Keys corresponding to desired data

    See Also:
    --------
    threaded.get
    """
    cache = None

    results = set(result)

    # if start_state_from_dask fails, we will have something
    # to pass to the final block.
    state = {}
    keyorder = dask.local.order(dsk)

    state = dask.local.start_state_from_dask(dsk, cache=cache, sortkey=keyorder.get)

    if state["waiting"] and not state["ready"]:
        raise ValueError("Found no accessible jobs in dask")

    async def fire_task():
        """Fire off a task to the thread pool."""
        # start a new job
        # Get the next task to compute (most recently added)
        key = state["ready"].pop()
        # Notify task is running
        state["running"].add(key)

        # Prep args to send
        data = {
            dep: state["cache"][dep] for dep in dask.local.get_dependencies(dsk, key)
        }

        result = await _execute_task(dsk[key], data)
        state["cache"][key] = result
        dask.local.finish_task(dsk, key, state, results, keyorder.get)

    # Main loop, wait on tasks to finish, insert new ones
    while state["ready"]:
        await fire_task()

    return dask.local.nested_get(result, state["cache"])

from __future__ import annotations
from collections import ChainMap
from typing import (
    Any,
    Text,
    Dict,
    List,
    Union,
)

import dask
import dask.threaded

from rasa.architecture_prototype.interfaces import DaskGraph, GraphSchema


def minimal_dask_graph(dask_graph: DaskGraph, targets: List[Text]) -> DaskGraph:
    dependencies = _all_dependencies(dask_graph, targets)

    return {
        step_name: step
        for step_name, step in dask_graph.items()
        if step_name in dependencies
    }


def _all_dependencies(dask_graph: DaskGraph, targets: List[Text]) -> List[Text]:
    required = []
    for target in targets:
        required.append(target)
        target_dependencies = dask_graph[target][1:]
        for dependency in target_dependencies:
            required += _all_dependencies(dask_graph, [dependency])

    return required


def minimal_graph_schema(graph_schema: GraphSchema, targets: List[Text]) -> GraphSchema:
    dependencies = _all_dependencies_schema(graph_schema, targets)

    return {
        step_name: step
        for step_name, step in graph_schema.items()
        if step_name in dependencies
    }


def _all_dependencies_schema(
    graph_schema: GraphSchema, targets: List[Text]
) -> List[Text]:
    required = []
    for target in targets:
        required.append(target)
        target_dependencies = graph_schema[target]["needs"].values()
        for dependency in target_dependencies:
            required += _all_dependencies_schema(graph_schema, [dependency])

    return required


def run_dask_graph(
    dask_graph: DaskGraph, target_names: Union[Text, List[Text]],
) -> Dict[Text, Any]:
    return dict(ChainMap(*dask.get(dask_graph, target_names)))  # FIXME: threaded


def visualise_dask_graph(dask_graph: DaskGraph, filename: Text,) -> None:
    dask.visualize(dask_graph, filename=filename)

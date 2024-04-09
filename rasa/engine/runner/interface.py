from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Text

from rasa.engine.graph import ExecutionContext, GraphNodeHook, GraphSchema
from rasa.engine.storage.storage import ModelStorage


class GraphRunner(ABC):
    """A `GraphRunner` is responsible for running a `GraphSchema`."""

    @classmethod
    @abstractmethod
    def create(
        cls,
        graph_schema: GraphSchema,
        model_storage: ModelStorage,
        execution_context: ExecutionContext,
        hooks: Optional[List[GraphNodeHook]] = None,
    ) -> GraphRunner:
        """Creates a new instance of a `GraphRunner`.

        Args:
            graph_schema: The graph schema that will be instantiated and run.
            model_storage: Storage which graph components can use to persist and load
                themselves.
            execution_context: Context that will be passed to every `GraphComponent`.
            hooks: These are called before and after the execution of each node.

        Returns: Instantiated `GraphRunner`
        """
        ...

    @abstractmethod
    async def run(
        self,
        inputs: Optional[Dict[Text, Any]] = None,
        targets: Optional[List[Text]] = None,
    ) -> Dict[Text, Any]:
        """Runs the instantiated graph with the given inputs and targets.

        Args:
            inputs: Input nodes to be added to the graph. These can be referenced by
                name in the "needs" key of a node in the schema.
            targets: Nodes whose output is needed and must always run.

        Returns: A mapping of target node name to output value.
        """
        ...

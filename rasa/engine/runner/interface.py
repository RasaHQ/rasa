from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Text

from rasa.engine.graph import ExecutionContext, GraphSchema


class GraphRunner(ABC):
    """A `GraphRunner` is responsible for running a `GraphSchema`."""

    @classmethod
    @abstractmethod
    def create(
        cls, graph_schema: GraphSchema, execution_context: ExecutionContext
    ) -> GraphRunner:
        """Creates a new instance of a `GraphRunner`.

        Args:
            graph_schema: The graph schema that will be instantiated and run.
            execution_context: Context that will be passed to every `GraphComponent`.

        Returns: Instantiated `GraphRunner`
        """
        ...

    @abstractmethod
    def run(
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

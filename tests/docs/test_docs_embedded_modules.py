import inspect

from rasa.engine.graph import GraphComponent
from data.test_classes.graph_component_interface import (
    GraphComponent as GraphComponentDocs,
)


def test_graph_copy_does_not_diverge():
    """Tests that the module embedded in the docs doesn't diverge from the actual one.

    It is currently not possible to embed a class from within a module in the docs
    without embedding the rest of the module file. To avoid this we have copied the
    `GraphComponent` to a separate file which is embeded in the docs.
    """
    # If this fails then copy the latest implementation of
    # `rasa.engine.graph.GraphComponent` to `data.test_classes.graph_component_interface
    assert inspect.getsource(GraphComponent) == inspect.getsource(GraphComponentDocs)

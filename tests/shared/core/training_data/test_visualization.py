from pathlib import Path
from typing import Text

import rasa.shared.utils.io
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import ActionExecuted, SlotSet, UserUttered
from rasa.shared.core.training_data import visualization
import rasa.utils.io
from rasa.shared.nlu.constants import TEXT, INTENT
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData

import pytest


def test_style_transfer():
    r = visualization._transfer_style({"class": "dashed great"}, {"class": "myclass"})
    assert r["class"] == "myclass dashed"


def test_style_transfer_empty():
    r = visualization._transfer_style({"class": "dashed great"}, {"something": "else"})
    assert r["class"] == "dashed"


def test_common_action_prefix():
    this = [
        ActionExecuted("action_listen"),
        ActionExecuted("greet"),
        UserUttered("hey"),
        ActionExecuted("amazing"),
        # until this point they are the same
        SlotSet("my_slot", "a"),
        ActionExecuted("a"),
        ActionExecuted("after_a"),
    ]
    other = [
        ActionExecuted("action_listen"),
        ActionExecuted("greet"),
        UserUttered("hey"),
        ActionExecuted("amazing"),
        # until this point they are the same
        SlotSet("my_slot", "b"),
        ActionExecuted("b"),
        ActionExecuted("after_b"),
    ]
    num_common = visualization._length_of_common_action_prefix(this, other)

    assert num_common == 3


def test_common_action_prefix_equal():
    this = [
        ActionExecuted("action_listen"),
        ActionExecuted("greet"),
        UserUttered("hey"),
        ActionExecuted("amazing"),
    ]
    other = [
        ActionExecuted("action_listen"),
        ActionExecuted("greet"),
        UserUttered("hey"),
        ActionExecuted("amazing"),
    ]
    num_common = visualization._length_of_common_action_prefix(this, other)

    assert num_common == 3


def test_common_action_prefix_unequal():
    this = [
        ActionExecuted("action_listen"),
        ActionExecuted("greet"),
        UserUttered("hey"),
    ]
    other = [
        ActionExecuted("greet"),
        ActionExecuted("action_listen"),
        UserUttered("hey"),
    ]
    num_common = visualization._length_of_common_action_prefix(this, other)

    assert num_common == 0


def test_graph_persistence(domain: Domain, tmp_path: Path):
    from os.path import isfile
    from networkx.drawing import nx_pydot
    import rasa.shared.core.training_data.loading as core_loading

    story_steps = core_loading.load_data_from_resource(
        "data/test_yaml_stories/stories.yml", domain
    )
    out_file = str(tmp_path / "graph.html")
    generated_graph = visualization.visualize_stories(
        story_steps,
        domain,
        output_file=out_file,
        max_history=3,
        should_merge_nodes=False,
    )

    generated_graph = nx_pydot.to_pydot(generated_graph)

    assert isfile(out_file)

    content = rasa.shared.utils.io.read_file(out_file)

    assert "isClient = true" in content
    assert "graph = `{}`".format(generated_graph.to_string()) in content


def test_merge_nodes(domain: Domain, tmp_path: Path):
    from os.path import isfile
    import rasa.shared.core.training_data.loading as core_loading

    story_steps = core_loading.load_data_from_resource(
        "data/test_yaml_stories/stories.yml", domain
    )
    out_file = str(tmp_path / "graph.html")
    visualization.visualize_stories(
        story_steps,
        domain,
        output_file=out_file,
        max_history=3,
        should_merge_nodes=True,
    )
    assert isfile(out_file)


def test_story_visualization(domain: Domain, tmp_path: Path):
    import rasa.shared.core.training_data.loading as core_loading

    story_steps = core_loading.load_data_from_resource(
        "data/test_yaml_stories/stories.yml", domain
    )
    out_file = tmp_path / "graph.html"
    generated_graph = visualization.visualize_stories(
        story_steps,
        domain,
        output_file=str(out_file),
        max_history=3,
        should_merge_nodes=False,
    )

    assert str(None) not in out_file.read_text()
    assert "/affirm" in out_file.read_text()
    assert len(generated_graph.nodes()) == 51
    assert len(generated_graph.edges()) == 56


def test_story_visualization_with_training_data(
    domain: Domain, tmp_path: Path, nlu_data_path: Text
):
    import rasa.shared.core.training_data.loading as core_loading

    story_steps = core_loading.load_data_from_resource(
        "data/test_yaml_stories/stories.yml", domain
    )
    out_file = tmp_path / "graph.html"
    test_text = "test text"
    test_intent = "affirm"
    generated_graph = visualization.visualize_stories(
        story_steps,
        domain,
        output_file=str(out_file),
        max_history=3,
        should_merge_nodes=False,
        nlu_training_data=TrainingData(
            [Message({TEXT: test_text, INTENT: test_intent})]
        ),
    )

    assert test_text in out_file.read_text()
    assert test_intent not in out_file.read_text()

    assert len(generated_graph.nodes()) == 51
    assert len(generated_graph.edges()) == 56


def test_story_visualization_with_merging(domain: Domain):
    import rasa.shared.core.training_data.loading as core_loading

    story_steps = core_loading.load_data_from_resource(
        "data/test_yaml_stories/stories.yml", domain
    )
    generated_graph = visualization.visualize_stories(
        story_steps, domain, output_file=None, max_history=3, should_merge_nodes=True
    )
    assert 15 < len(generated_graph.nodes()) < 33

    assert 20 < len(generated_graph.edges()) < 33


@pytest.mark.parametrize(
    "input_nodes, input_edges, remove_count, expected_nodes, expected_edges",
    [
        (
            [-2, -1, 0, 1, 2, 3, 4, 5],
            [(-2, 0), (-1, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
            3,
            set([0, 1, 2, 3, 4, 5, -1]),
            [(-1, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
        ),
        (
            [-3, -2, -1, 0, 1, 2, 3, 4, 5],
            [(-3, -2), (-2, -1), (-1, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
            4,
            set([-3, -1, 0, 1, 2, 3, 4, 5]),
            [(-1, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
        ),
    ],
)
def test_remove_auxiliary_nodes(
    input_nodes, input_edges, remove_count, expected_nodes, expected_edges
):
    import networkx as nx

    # Create a sample graph
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(input_nodes)
    graph.add_edges_from(input_edges)

    # Call the method to remove auxiliary nodes
    visualization._remove_auxiliary_nodes(graph, remove_count)

    # Check if the expected nodes are removed
    assert set(graph.nodes()) == expected_nodes, "Nodes mismatch"

    # Check if the edges are updated correctly
    assert list(graph.edges()) == expected_edges, "Edges mismatch"

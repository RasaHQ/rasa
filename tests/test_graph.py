from rasa_core.training import StoryGraph


def test_node_ordering():
    example_graph = {
        "a": ["b", "c", "d"],
        "b": [],
        "c": ["d"],
        "d": [],
        "e": ["f"],
        "f": []}
    sorted_nodes, removed_edges = StoryGraph.topological_sort(example_graph)
    assert list(sorted_nodes) == ['e', 'f', 'a', 'c', 'd', 'b']
    assert removed_edges == set()


def test_node_ordering_with_cycle():
    example_graph = {
        "a": ["b", "c", "d"],
        "b": [],
        "c": ["d"],
        "d": ["a"],
        "e": ["f"],
        "f": ["e"]}
    sorted_nodes, removed_edges = StoryGraph.topological_sort(example_graph)
    assert list(sorted_nodes) == ['e', 'f', 'a', 'c', 'd', 'b']
    assert len(removed_edges) == 2
    assert ("f", "e") in removed_edges
    assert ("d", "a") in removed_edges

from rasa.core.training.structures import StoryGraph


def check_graph_is_sorted(g, sorted_nodes, removed_edges):
    incoming_edges = {k: [s for s, vs in g.items() if k in vs] for k in g.keys()}

    visited = set()
    for n in sorted_nodes:
        deps = incoming_edges.get(n, [])
        # checks that all incoming edges are from nodes we have already visited
        assert all(
            [d in visited or (d, n) in removed_edges for d in deps]
        ), "Found an incoming edge from a node that wasn't visited yet!"
        visited.add(n)


def test_node_ordering():
    example_graph = {
        "a": ["b", "c", "d"],
        "b": [],
        "c": ["d"],
        "d": [],
        "e": ["f"],
        "f": [],
    }
    sorted_nodes, removed_edges = StoryGraph.topological_sort(example_graph)

    assert removed_edges == set()
    check_graph_is_sorted(example_graph, sorted_nodes, removed_edges)


def test_node_ordering_with_cycle():
    example_graph = {
        "a": ["b", "c", "d"],
        "b": [],
        "c": ["d"],
        "d": ["a"],
        "e": ["f"],
        "f": ["e"],
    }
    sorted_nodes, removed_edges = StoryGraph.topological_sort(example_graph)

    check_graph_is_sorted(example_graph, sorted_nodes, removed_edges)

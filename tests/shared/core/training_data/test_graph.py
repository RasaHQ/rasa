from rasa.shared.core.training_data.structures import StoryGraph
import rasa.shared.core.training_data.loading
from rasa.shared.core.domain import Domain


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
    # sorting removed_edges converting set converting it to list
    assert removed_edges == list()
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


def test_is_empty():
    assert StoryGraph([]).is_empty()


def test_consistent_fingerprints():
    stories_path = "data/test_yaml_stories/stories.yml"
    domain_path = "data/test_domains/default_with_slots.yml"
    domain = Domain.load(domain_path)
    story_steps = rasa.shared.core.training_data.loading.load_data_from_resource(
        stories_path, domain
    )
    story_graph = StoryGraph(story_steps)

    # read again
    story_steps_2 = rasa.shared.core.training_data.loading.load_data_from_resource(
        stories_path, domain
    )
    story_graph_2 = StoryGraph(story_steps_2)

    fingerprint = story_graph.fingerprint()
    fingerprint_2 = story_graph_2.fingerprint()

    assert fingerprint == fingerprint_2


def test_unique_checkpoint_names():
    stories_path = "data/test_yaml_stories/story_with_two_equal_or_statements.yml"
    domain_path = "data/test_domains/default_with_slots.yml"
    domain = Domain.load(domain_path)
    story_steps = rasa.shared.core.training_data.loading.load_data_from_resource(
        stories_path, domain
    )
    start_checkpoint_names = {
        chk.name for s in story_steps for chk in s.start_checkpoints
    }

    # first story:
    # START_CHECKPOINT, GENR_OR_XXXXX for first OR, GENR_OR_YYYYY for second OR

    # additional in second story:
    # GENR_OR_ZZZZZ as entities are different from first OR in first story
    assert len(start_checkpoint_names) == 4

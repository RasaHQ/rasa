from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random
from collections import defaultdict, deque

from typing import Any, Text, List, Dict

from rasa_core.events import UserUttered, ActionExecuted
from rasa_core.interpreter import RegexInterpreter
from rasa_core.training_utils import STORY_START
from rasa_core.training_utils.story_graph import StoryGraph
from rasa_nlu.training_data import TrainingData

EDGE_NONE_LABEL = "NONE"


class UserMessageGenerator(object):
    def __init__(self, training_data):
        self.training_data = training_data
        self.mapping = self._create_reverse_mapping(self.training_data)

    def _create_reverse_mapping(self, data):
        # type: (TrainingData) -> Dict[Text, List[Dict[Text, Any]]]
        """Create a mapping from intent to messages

        This allows a faster intent lookup."""

        d = defaultdict(list)
        for example in data.training_examples:
            if example.get("intent", {}) is not None:
                d[example.get("intent", {})].append(example)
        return d

    @staticmethod
    def _contains_same_entity(entities, e):
        return (entities.get(e.get("entity")) is None or
                entities.get(e.get("entity")) != e.get("value"))

    def message_for_data(self, structured_info):
        """Find a data sample with the same intent and entities.

        Given the parsed data from a message (intent and entities) finds a
        message in the data that has the same intent and entities."""

        if structured_info.get("intent") is not None:
            intent_name = structured_info.get("intent", {}).get("name")
            usable_examples = self.mapping.get(intent_name, [])[:]
            random.shuffle(usable_examples)
            for example in usable_examples:
                entities = {e.get("entity"): e.get("value")
                            for e in example.get("entities", [])}
                for e in structured_info.get("entities", []):
                    if self._contains_same_entity(entities, e):
                        break
                else:
                    return example.text
        return structured_info.get("text")


def _fingerprint_node(G, node, max_history):
    """Fingerprint a node in a graph.

    Can be used to identify nodes that are similar and can be merged within the
    graph.
    Generates all paths starting at `node` following the directed graph up to
    the length of `max_history`, and returns a set of strings describing the
    found paths. If the fingerprint creation for two nodes results in the same
    sets these nodes are indistinguishable if we walk along the path and only
    remember max history number of nodes we have visited. Hence, if we randomly
    walk on our directed graph, always only remembering the last `max_history`
    nodes we have visited, we can never remember if we have visited node A or
    node B if both have the same fingerprint. """

    # the candidate list contains all node paths that haven't been
    # extended till `max_history` length yet.
    candidates = deque()
    candidates.append([node])
    continuations = []
    while len(candidates) > 0:
        candidate = candidates.pop()
        last = candidate[-1]
        empty = True
        for _, succ_node in G.out_edges(last):
            next_candidate = candidate[:]
            next_candidate.append(succ_node)
            # if the path is already long enough, we add it to the results,
            # otherwise we add it to the candidates
            # that we still need to visit
            if len(next_candidate) == max_history:
                continuations.append(next_candidate)
            else:
                candidates.append(next_candidate)
            empty = False
        if empty:
            continuations.append(candidate)
    return {" - ".join([G.node[node]["label"]
                        for node in continuation])
            for continuation in continuations}


def _incoming_edges(G, node):
    return {(prev_node, k) for prev_node, _, k in G.in_edges(node, keys=True)}


def _outgoing_edges(G, node):
    return {(succ_node, k) for _, succ_node, k in G.out_edges(node, keys=True)}


def _outgoing_edges_are_similar(G, node_a, node_b):
    """If the outgoing edges from the two nodes are similar enough,
    it doesn't matter if you are in a or b.

    As your path will be the same because the outgoing edges will lead you to
    the same nodes anyways."""

    ignored = {node_b, node_a}
    a_edges = {(target, k) for target, k in _outgoing_edges(G, node_a) if
               target not in ignored}
    b_edges = {(target, k) for target, k in _outgoing_edges(G, node_b) if
               target not in ignored}
    return a_edges == b_edges or not a_edges or not b_edges


def _nodes_are_equivalent(G, node_a, node_b, max_history):
    """Decides if two nodes are equivalent based on their fingerprints."""
    return (
        G.node[node_a]["label"] == G.node[node_b]["label"] and (
            _outgoing_edges_are_similar(G, node_a, node_b) or
            _incoming_edges(G, node_a) == _incoming_edges(G, node_b) or
            _fingerprint_node(G, node_a, max_history) ==
            _fingerprint_node(G, node_b, max_history)))


def _add_edge(G, u, v, key, label=None):
    """Adds an edge to the graph if the edge is not already present. Uses the
    label as the key."""

    if key is None or key == EDGE_NONE_LABEL:
        # Can't use `None` as a label
        if not G.has_edge(u, v, key=EDGE_NONE_LABEL):
            G.add_edge(u, v, key=EDGE_NONE_LABEL, label="")
    elif not G.has_edge(u, v, key):
        G.add_edge(u, v, key=key, label=label)


def _merge_equivalent_nodes(G, max_history):
    """Searches for equivalent nodes in the graph and merges them."""

    changed = True
    # every node merge changes the graph and can trigger previously
    # impossible node merges - we need to repeat until
    # the graph doesn't change anymore
    while changed:
        changed = False
        remaining_node_ids = [n for n in G.nodes() if n > 0]
        for idx, i in enumerate(remaining_node_ids):
            if G.has_node(i):
                for j in remaining_node_ids[
                         idx + 1:]:  # assumes node equivalence is cumulative
                    if G.has_node(j) and \
                            _nodes_are_equivalent(G, i, j, max_history):
                        changed = True
                        # moves all outgoing edges to the other node
                        j_outgoing_edges = G.out_edges(j, keys=True, data=True)
                        for _, succ_node, k, d in j_outgoing_edges:
                            _add_edge(G, i, succ_node, k, d.get("label"))
                            G.remove_edge(j, succ_node)
                        # moves all incoming edges to the other node
                        j_incoming_edges = G.in_edges(j, keys=True, data=True)
                        for prev_node, _, k, d in j_incoming_edges:
                            _add_edge(G, prev_node, i, k, d.get("label"))
                            G.remove_edge(prev_node, j)
                        G.remove_node(j)


def _replace_edge_labels_with_nodes(G, next_id, interpreter, training_data,
                                    fontsize):
    """User messages are created as edge labels. This removes the labels and
    creates nodes instead.

    The algorithms (e.g. merging) are simpler if the user messages are labels
    on the edges. But it sometimes
    looks better if in the final graphs the user messages are nodes instead
    of edge labels."""

    if training_data:
        message_generator = UserMessageGenerator(training_data)
    else:
        message_generator = None

    for s, e, k, d in G.edges(keys=True, data=True):
        if k != EDGE_NONE_LABEL:
            if message_generator and d.get("label", k) is not None:
                parsed_info = interpreter.parse(d.get("label", k))
                label = message_generator.message_for_data(parsed_info)
            else:
                label = d.get("label", k)
            next_id += 1
            G.remove_edge(s, e, k)
            G.add_node(next_id, label=label, style="filled",
                       fillcolor="lightblue", shape="box", fontsize=fontsize)
            G.add_edge(s, next_id)
            G.add_edge(next_id, e)


def _persist_graph(G, output_file):
    """Plots the graph and persists it into a file. Uses graphviz (needs to
    be installed!)."""
    import networkx as nx

    A = nx.nx_agraph.to_agraph(G)  # convert to a graphviz graph
    A.layout("dot", args="-Goverlap=false -Gsplines=true -Gconcentrate=true "
                         "-Gfontname=typewriter")
    A.draw(output_file)


def visualize_stories(story_steps,
                      output_file=None,
                      max_history=2,
                      interpreter=RegexInterpreter(),
                      training_data=None,
                      fontsize=12):
    """Given a set of stories, generates a graph visualizing the flows in the
    stories.

    Visualization is always a trade off between making the graph as small as
    possible while
    at the same time making sure the meaning doesn't change to "much". The
    algorithm will
    compress the graph generated from the stories to merge nodes that are
    similar. Hence,
    the algorithm might create paths through the graph that aren't actually
    specified in the
    stories, but we try to minimize that.

    Output file defines if and where a file containing the plotted graph
    should be stored.

    The history defines how much 'memory' the graph has. This influences in
    which situations the
    algorithm will merge nodes. Nodes will only be merged if they are equal
    within the history, this
    means the larger the history is we take into account the less likely it
    is we merge any nodes.

    The training data parameter can be used to pass in a Rasa NLU training
    data instance. It will
    be used to replace the user messages from the story file with actual
    messages from the training data."""
    import networkx as nx

    story_graph = StoryGraph(story_steps)
    G = nx.MultiDiGraph()
    next_node_idx = 0
    G.add_node(0, label="START", fillcolor="green", style="filled",
               fontsize=fontsize)
    G.add_node(-1, label="END", fillcolor="red", style="filled",
               fontsize=fontsize)

    checkpoint_indices = defaultdict(list)
    checkpoint_indices[STORY_START] = [(0, None)]

    for step in story_graph.ordered_steps():
        current_nodes = checkpoint_indices[step.start_checkpoint_name()]
        message = None
        for el in step.events:
            if isinstance(el, UserUttered):
                message = interpreter.parse(el.text)
            elif isinstance(el, ActionExecuted):
                next_node_idx += 1
                G.add_node(next_node_idx, label=el.action_name,
                           fontsize=fontsize)

                for current_node, tailing_message in current_nodes:
                    msg = message if message else tailing_message
                    if msg:
                        message_key = msg.get("intent", {}).get("name", None)
                        message_label = msg.get("text", None)
                    else:
                        message_key = None
                        message_label = None

                    _add_edge(G, current_node, next_node_idx, message_key,
                              message_label)

                current_nodes = [(next_node_idx, None)]
                message = None
        if not step.end_checkpoint_name():
            for current_node, _ in current_nodes:
                if message:
                    G.add_edge(current_node, -1,
                               key=EDGE_NONE_LABEL, label=message)
                else:
                    G.add_edge(current_node, -1, key=EDGE_NONE_LABEL)
        else:
            updated_nodes = [(c, message) for c, _ in current_nodes]
            checkpoint_indices[step.end_checkpoint_name()].extend(updated_nodes)

    _merge_equivalent_nodes(G, max_history)
    _replace_edge_labels_with_nodes(G, next_node_idx, interpreter,
                                    training_data, fontsize)

    if output_file:
        _persist_graph(G, output_file)
    return G

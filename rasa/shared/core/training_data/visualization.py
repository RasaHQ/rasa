from collections import defaultdict, deque

import random
from typing import (
    Any,
    Text,
    List,
    Deque,
    Dict,
    Optional,
    Set,
    TYPE_CHECKING,
    Union,
    cast,
)

import rasa.shared.utils.io
from rasa.shared.constants import INTENT_MESSAGE_PREFIX
from rasa.shared.core.constants import ACTION_LISTEN_NAME
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import UserUttered, ActionExecuted, Event
from rasa.shared.core.generator import TrainingDataGenerator
from rasa.shared.core.training_data.structures import StoryGraph, StoryStep
from rasa.shared.nlu.constants import (
    ENTITY_ATTRIBUTE_VALUE,
    INTENT,
    TEXT,
    ENTITY_ATTRIBUTE_TYPE,
    INTENT_NAME_KEY,
)

if TYPE_CHECKING:
    from rasa.shared.nlu.training_data.training_data import TrainingData
    from rasa.shared.nlu.training_data.message import Message
    import networkx

EDGE_NONE_LABEL = "NONE"

START_NODE_ID = 0
END_NODE_ID = -1
TMP_NODE_ID = -2

VISUALIZATION_TEMPLATE_PATH = "visualization.html"


class UserMessageGenerator:
    def __init__(self, nlu_training_data: "TrainingData") -> None:
        self.nlu_training_data = nlu_training_data
        self.mapping = self._create_reverse_mapping(self.nlu_training_data)

    @staticmethod
    def _create_reverse_mapping(
        data: "TrainingData",
    ) -> Dict[Dict[Text, Any], List["Message"]]:
        """Create a mapping from intent to messages.

        This allows a faster intent lookup.
        """
        d = defaultdict(list)
        for example in data.training_examples:
            if example.get(INTENT, {}) is not None:
                d[example.get(INTENT, {})].append(example)
        return d

    @staticmethod
    def _contains_same_entity(entities: Dict[Text, Any], e: Dict[Text, Any]) -> bool:
        return entities.get(e.get(ENTITY_ATTRIBUTE_TYPE)) is None or entities.get(
            e.get(ENTITY_ATTRIBUTE_TYPE)
        ) != e.get(ENTITY_ATTRIBUTE_VALUE)

    def message_for_data(self, structured_info: Dict[Text, Any]) -> Any:
        """Find a data sample with the same intent."""
        if structured_info.get(INTENT) is not None:
            intent_name = structured_info.get(INTENT, {}).get(INTENT_NAME_KEY)
            usable_examples = self.mapping.get(intent_name, [])[:]
            random.shuffle(usable_examples)

            if usable_examples:
                return usable_examples[0].get(TEXT)

        return structured_info.get(TEXT)


def _fingerprint_node(
    graph: "networkx.MultiDiGraph", node: int, max_history: int
) -> Set[Text]:
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
    node B if both have the same fingerprint.
    """
    # the candidate list contains all node paths that haven't been
    # extended till `max_history` length yet.
    candidates: Deque = deque()
    candidates.append([node])
    continuations = []
    while len(candidates) > 0:
        candidate = candidates.pop()
        last = candidate[-1]
        empty = True
        for _, succ_node in graph.out_edges(last):
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
    return {
        " - ".join([graph.nodes[node]["label"] for node in continuation])
        for continuation in continuations
    }


def _incoming_edges(graph: "networkx.MultiDiGraph", node: int) -> set:
    return {(prev_node, k) for prev_node, _, k in graph.in_edges(node, keys=True)}


def _outgoing_edges(graph: "networkx.MultiDiGraph", node: int) -> set:
    return {(succ_node, k) for _, succ_node, k in graph.out_edges(node, keys=True)}


def _outgoing_edges_are_similar(
    graph: "networkx.MultiDiGraph", node_a: int, node_b: int
) -> bool:
    """If the outgoing edges from the two nodes are similar enough,
    it doesn't matter if you are in a or b.

    As your path will be the same because the outgoing edges will lead you to
    the same nodes anyways.
    """
    ignored = {node_b, node_a}
    a_edges = {
        (target, k)
        for target, k in _outgoing_edges(graph, node_a)
        if target not in ignored
    }
    b_edges = {
        (target, k)
        for target, k in _outgoing_edges(graph, node_b)
        if target not in ignored
    }
    return a_edges == b_edges or not a_edges or not b_edges


def _nodes_are_equivalent(
    graph: "networkx.MultiDiGraph", node_a: int, node_b: int, max_history: int
) -> bool:
    """Decides if two nodes are equivalent based on their fingerprints."""
    return graph.nodes[node_a]["label"] == graph.nodes[node_b]["label"] and (
        _outgoing_edges_are_similar(graph, node_a, node_b)
        or _incoming_edges(graph, node_a) == _incoming_edges(graph, node_b)
        or _fingerprint_node(graph, node_a, max_history)
        == _fingerprint_node(graph, node_b, max_history)
    )


def _add_edge(
    graph: "networkx.MultiDiGraph",
    u: int,
    v: int,
    key: Optional[Text],
    label: Optional[Text] = None,
    **kwargs: Any,
) -> None:
    """Adds an edge to the graph if the edge is not already present. Uses the
    label as the key.
    """
    if key is None:
        key = EDGE_NONE_LABEL

    if key == EDGE_NONE_LABEL:
        label = ""

    if not graph.has_edge(u, v, key=EDGE_NONE_LABEL):
        graph.add_edge(u, v, key=key, label=label, **kwargs)
    else:
        d = graph.get_edge_data(u, v, key=EDGE_NONE_LABEL)
        _transfer_style(kwargs, d)


def _transfer_style(
    source: Dict[Text, Any], target: Dict[Text, Any]
) -> Dict[Text, Any]:
    """Copy over class names from source to target for all special classes.

    Used if a node is highlighted and merged with another node.
    """
    clazzes = source.get("class", "")

    special_classes = {"dashed", "active"}

    if "class" not in target:
        target["class"] = ""

    for c in special_classes:
        if c in clazzes and c not in target["class"]:
            target["class"] += " " + c

    target["class"] = target["class"].strip()
    return target


def _merge_equivalent_nodes(graph: "networkx.MultiDiGraph", max_history: int) -> None:
    """Searches for equivalent nodes in the graph and merges them."""
    changed = True
    # every node merge changes the graph and can trigger previously
    # impossible node merges - we need to repeat until
    # the graph doesn't change anymore
    while changed:
        changed = False
        remaining_node_ids = [n for n in graph.nodes() if n > 0]
        for idx, i in enumerate(remaining_node_ids):
            if graph.has_node(i):
                # assumes node equivalence is cumulative
                for j in remaining_node_ids[idx + 1 :]:
                    if graph.has_node(j) and _nodes_are_equivalent(
                        graph, i, j, max_history
                    ):
                        # make sure we keep special styles
                        _transfer_style(
                            graph.nodes(data=True)[j], graph.nodes(data=True)[i]
                        )

                        changed = True
                        # moves all outgoing edges to the other node
                        j_outgoing_edges = list(
                            graph.out_edges(j, keys=True, data=True)
                        )
                        for _, succ_node, k, d in j_outgoing_edges:
                            _add_edge(
                                graph,
                                i,
                                succ_node,
                                k,
                                d.get("label"),
                                **{"class": d.get("class", "")},
                            )
                            graph.remove_edge(j, succ_node)
                        # moves all incoming edges to the other node
                        j_incoming_edges = list(graph.in_edges(j, keys=True, data=True))
                        for prev_node, _, k, d in j_incoming_edges:
                            _add_edge(
                                graph,
                                prev_node,
                                i,
                                k,
                                d.get("label"),
                                **{"class": d.get("class", "")},
                            )
                            graph.remove_edge(prev_node, j)
                        graph.remove_node(j)


def _replace_edge_labels_with_nodes(
    graph: "networkx.MultiDiGraph", next_id: int, nlu_training_data: "TrainingData"
) -> None:
    """Replaces edge labels with nodes.

    User messages are created as edge labels. This removes the labels and
    creates nodes instead.

    The algorithms (e.g. merging) are simpler if the user messages are labels
    on the edges. But it sometimes
    looks better if in the final graphs the user messages are nodes instead
    of edge labels.
    """
    if nlu_training_data:
        message_generator = UserMessageGenerator(nlu_training_data)
    else:
        message_generator = None

    edges = list(graph.edges(keys=True, data=True))
    for s, e, k, d in edges:
        if k != EDGE_NONE_LABEL:
            label = d.get("label", k)

            if message_generator:
                parsed_info = {TEXT: label}
                if label.startswith(INTENT_MESSAGE_PREFIX):
                    parsed_info[INTENT] = {INTENT_NAME_KEY: label[1:]}

                label = message_generator.message_for_data(parsed_info)
            next_id += 1
            graph.remove_edge(s, e, k)
            graph.add_node(
                next_id,
                label=label,
                shape="rect",
                style="filled",
                fillcolor="lightblue",
                **_transfer_style(d, {"class": "intent"}),
            )
            graph.add_edge(s, next_id, **{"class": d.get("class", "")})
            graph.add_edge(next_id, e, **{"class": d.get("class", "")})


def visualization_html_path() -> Text:
    import importlib

    with importlib.resources.path(__package__, VISUALIZATION_TEMPLATE_PATH) as r:
        return str(r)


def persist_graph(graph: "networkx.Graph", output_file: Text) -> None:
    """Plots the graph and persists it into a html file."""
    import networkx as nx

    expg = nx.nx_pydot.to_pydot(graph)

    template = rasa.shared.utils.io.read_file(visualization_html_path())

    # Insert graph into template
    template = template.replace("// { is-client }", "isClient = true", 1)
    graph_as_text = expg.to_string()
    # escape backslashes
    graph_as_text = graph_as_text.replace("\\", "\\\\")
    template = template.replace("// { graph-content }", f"graph = `{graph_as_text}`", 1)

    rasa.shared.utils.io.write_text_file(template, output_file)


def _length_of_common_action_prefix(this: List[Event], other: List[Event]) -> int:
    """Calculate number of actions that two conversations have in common."""
    num_common_actions = 0
    t_cleaned = cast(
        List[Union[ActionExecuted, UserUttered]],
        [e for e in this if e.type_name in {"user", "action"}],
    )
    o_cleaned = cast(
        List[Union[ActionExecuted, UserUttered]],
        [e for e in other if e.type_name in {"user", "action"}],
    )

    for i, e in enumerate(t_cleaned):
        o = o_cleaned[i]
        if i == len(o_cleaned):
            break
        elif isinstance(e, UserUttered) and isinstance(o, UserUttered):
            continue
        elif (
            isinstance(e, ActionExecuted)
            and isinstance(o, ActionExecuted)
            and o.action_name == e.action_name
        ):
            num_common_actions += 1
        else:
            break
    return num_common_actions


def _add_default_nodes(graph: "networkx.MultiDiGraph", fontsize: int = 12) -> None:
    """Add the standard nodes we need."""
    graph.add_node(
        START_NODE_ID,
        label="START",
        fillcolor="green",
        style="filled",
        fontsize=fontsize,
        **{"class": "start active"},
    )
    graph.add_node(
        END_NODE_ID,
        label="END",
        fillcolor="red",
        style="filled",
        fontsize=fontsize,
        **{"class": "end"},
    )
    graph.add_node(TMP_NODE_ID, label="TMP", style="invis", **{"class": "invisible"})


def _create_graph(fontsize: int = 12) -> "networkx.MultiDiGraph":
    """Create a graph and adds the default nodes."""
    import networkx as nx

    graph = nx.MultiDiGraph()
    _add_default_nodes(graph, fontsize)
    return graph


def _add_message_edge(
    graph: "networkx.MultiDiGraph",
    message: Optional[Dict[Text, Any]],
    current_node: int,
    next_node_idx: int,
    is_current: bool,
) -> None:
    """Create an edge based on the user message."""
    if message:
        message_key = message.get("intent", {}).get("name", None)
        message_label = message.get("text", None)
    else:
        message_key = None
        message_label = None

    _add_edge(
        graph,
        current_node,
        next_node_idx,
        message_key,
        message_label,
        **{"class": "active" if is_current else ""},
    )


def visualize_neighborhood(
    current: Optional[List[Event]],
    event_sequences: List[List[Event]],
    output_file: Optional[Text] = None,
    max_history: int = 2,
    nlu_training_data: Optional["TrainingData"] = None,
    should_merge_nodes: bool = True,
    max_distance: int = 1,
    fontsize: int = 12,
) -> "networkx.MultiDiGraph":
    """Given a set of event lists, visualizing the flows."""
    graph = _create_graph(fontsize)
    _add_default_nodes(graph)

    next_node_idx = START_NODE_ID
    special_node_idx = -3
    path_ellipsis_ends = set()

    for events in event_sequences:
        if current and max_distance:
            prefix = _length_of_common_action_prefix(current, events)
        else:
            prefix = len(events)

        message = None
        current_node = START_NODE_ID
        idx = 0
        is_current = events == current

        for idx, el in enumerate(events):
            if not prefix:
                idx -= 1
                break
            if isinstance(el, UserUttered):
                message = el.parse_data
                message[TEXT] = f"{INTENT_MESSAGE_PREFIX}{el.intent_name}"  # type: ignore[literal-required]
            elif (
                isinstance(el, ActionExecuted) and el.action_name != ACTION_LISTEN_NAME
            ):
                next_node_idx += 1
                graph.add_node(
                    next_node_idx,
                    label=el.action_name,
                    fontsize=fontsize,
                    **{"class": "active" if is_current else ""},
                )

                _add_message_edge(
                    graph, message, current_node, next_node_idx, is_current
                )
                current_node = next_node_idx

                message = None
                prefix -= 1

        # determine what the end node of the conversation is going to be
        # this can either be an ellipsis "...", the conversation end node
        # "END" or a "TMP" node if this is the active conversation
        if is_current:
            event_idx = events[idx]
            if (
                isinstance(event_idx, ActionExecuted)
                and event_idx.action_name == ACTION_LISTEN_NAME
            ):
                next_node_idx += 1
                if message is None:
                    label = "  ?  "
                else:
                    intent = cast(dict, message).get("intent", {})
                    label = intent.get("name", "  ?  ")
                graph.add_node(
                    next_node_idx,
                    label=label,
                    shape="rect",
                    **{"class": "intent dashed active"},
                )
                target = next_node_idx
            elif current_node:
                d = graph.nodes(data=True)[current_node]
                d["class"] = "dashed active"
                target = TMP_NODE_ID
            else:
                target = TMP_NODE_ID
        elif idx == len(events) - 1:
            target = END_NODE_ID
        elif current_node and current_node not in path_ellipsis_ends:
            graph.add_node(special_node_idx, label="...", **{"class": "ellipsis"})
            target = special_node_idx
            path_ellipsis_ends.add(current_node)
            special_node_idx -= 1
        else:
            target = END_NODE_ID

        _add_message_edge(graph, message, current_node, target, is_current)

    if should_merge_nodes:
        _merge_equivalent_nodes(graph, max_history)
    _replace_edge_labels_with_nodes(graph, next_node_idx, nlu_training_data)

    _remove_auxiliary_nodes(graph, special_node_idx)

    if output_file:
        persist_graph(graph, output_file)
    return graph


def _remove_auxiliary_nodes(
    graph: "networkx.MultiDiGraph", special_node_idx: int
) -> None:
    """Remove any temporary or unused nodes."""
    graph.remove_node(TMP_NODE_ID)

    if not graph.predecessors(END_NODE_ID):
        graph.remove_node(END_NODE_ID)

    # remove duplicated "..." nodes after merging
    predecessors_seen = set()
    for i in range(special_node_idx + 1, TMP_NODE_ID):
        predecessors = graph.predecessors(i)
        for pred in predecessors:
            if pred in predecessors_seen:
                graph.remove_node(i)
        predecessors_seen.update(predecessors)


def visualize_stories(
    story_steps: List[StoryStep],
    domain: Domain,
    output_file: Optional[Text],
    max_history: int,
    nlu_training_data: Optional["TrainingData"] = None,
    should_merge_nodes: bool = True,
    fontsize: int = 12,
) -> "networkx.MultiDiGraph":
    """Given a set of stories, generates a graph visualizing the flows in the stories.

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
    messages from the training data.
    """
    story_graph = StoryGraph(story_steps)

    g = TrainingDataGenerator(
        story_graph,
        domain,
        use_story_concatenation=False,
        tracker_limit=100,
        augmentation_factor=0,
    )
    completed_trackers = g.generate()
    event_sequences = [t.events for t in completed_trackers]

    graph = visualize_neighborhood(
        None,
        event_sequences,
        output_file,
        max_history,
        nlu_training_data,
        should_merge_nodes,
        max_distance=1,
        fontsize=fontsize,
    )
    return graph

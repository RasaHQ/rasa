import json
import logging
from collections import deque, defaultdict

import uuid
import typing
from typing import (
    List,
    Text,
    Deque,
    Dict,
    Optional,
    Tuple,
    Any,
    Set,
    ValuesView,
    Union,
    Sequence,
)

import rasa.shared.utils.io
from rasa.shared.core.constants import (
    ACTION_LISTEN_NAME,
    ACTION_SESSION_START_NAME,
    ACTION_UNLIKELY_INTENT_NAME,
)
from rasa.shared.core.conversation import Dialogue
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import (
    UserUttered,
    ActionExecuted,
    Event,
    SessionStarted,
    SlotSet,
)
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import RasaCoreException


if typing.TYPE_CHECKING:
    import networkx as nx

logger = logging.getLogger(__name__)

# Checkpoint id used to identify story starting blocks
STORY_START = "STORY_START"

# Checkpoint id used to identify story end blocks
STORY_END = None

# need abbreviations otherwise they are not visualized well
GENERATED_CHECKPOINT_PREFIX = "GENR_"
CHECKPOINT_CYCLE_PREFIX = "CYCL_"

GENERATED_HASH_LENGTH = 5

FORM_PREFIX = "form: "
# prefix for storystep ID to get reproducible sorting results
# will get increased with each new instance
STEP_COUNT = 1


class EventTypeError(RasaCoreException, ValueError):
    """Represents an error caused by a Rasa Core event not being of the expected
    type.

    """


class Checkpoint:
    """Represents places where trackers split.

    This currently happens if
    - users place manual checkpoints in their stories
    - have `or` statements for intents in their stories.
    """

    def __init__(
        self, name: Text, conditions: Optional[Dict[Text, Any]] = None
    ) -> None:
        """Creates `Checkpoint`.

        Args:
            name: Name of the checkpoint.
            conditions: Slot conditions for this checkpoint.
        """
        self.name = name
        self.conditions = conditions if conditions else {}

    def as_story_string(self) -> Text:
        dumped_conds = json.dumps(self.conditions) if self.conditions else ""
        return f"{self.name}{dumped_conds}"

    def filter_trackers(
        self, trackers: List[DialogueStateTracker]
    ) -> List[DialogueStateTracker]:
        """Filters out all trackers that do not satisfy the conditions."""
        if not self.conditions:
            return trackers

        for slot_name, slot_value in self.conditions.items():
            trackers = [t for t in trackers if t.get_slot(slot_name) == slot_value]
        return trackers

    def __repr__(self) -> Text:
        return "Checkpoint(name={!r}, conditions={})".format(
            self.name, json.dumps(self.conditions)
        )


class StoryStep:
    """A StoryStep is a section of a story block between two checkpoints.

    NOTE: Checkpoints are not only limited to those manually written
    in the story file, but are also implicitly created at points where
    multiple intents are separated in one line by chaining them with "OR"s.
    """

    def __init__(
        self,
        block_name: Text,
        start_checkpoints: Optional[List[Checkpoint]] = None,
        end_checkpoints: Optional[List[Checkpoint]] = None,
        events: Optional[List[Union[Event, List[Event]]]] = None,
        source_name: Optional[Text] = None,
    ) -> None:
        """Initialise `StoryStep` default attributes."""
        self.end_checkpoints = end_checkpoints if end_checkpoints else []
        self.start_checkpoints = start_checkpoints if start_checkpoints else []
        self.events = events if events else []
        self.block_name = block_name
        self.source_name = source_name
        # put a counter prefix to uuid to get reproducible sorting results
        global STEP_COUNT
        self.id = "{}_{}".format(STEP_COUNT, uuid.uuid4().hex)
        STEP_COUNT += 1

    def create_copy(self, use_new_id: bool) -> "StoryStep":
        copied = StoryStep(
            self.block_name,
            self.start_checkpoints,
            self.end_checkpoints,
            self.events[:],
            self.source_name,
        )
        if not use_new_id:
            copied.id = self.id
        return copied

    def add_user_message(self, user_message: UserUttered) -> None:
        self.add_event(user_message)

    def add_event(self, event: Event) -> None:
        self.events.append(event)

    def add_events(self, events: List[Event]) -> None:
        self.events.append(events)

    @staticmethod
    def _checkpoint_string(story_step_element: Checkpoint) -> Text:
        return f"> {story_step_element.as_story_string()}\n"

    @staticmethod
    def _user_string(story_step_element: UserUttered, e2e: bool) -> Text:
        return f"* {story_step_element.as_story_string(e2e)}\n"

    @staticmethod
    def _bot_string(story_step_element: Event) -> Text:
        return f"    - {story_step_element.as_story_string()}\n"

    @staticmethod
    def _event_to_story_string(event: Event, e2e: bool) -> Optional[Text]:
        if isinstance(event, UserUttered):
            return event.as_story_string(e2e=e2e)
        return event.as_story_string()

    @staticmethod
    def _or_string(story_step_element: Sequence[Event], e2e: bool) -> Optional[Text]:
        for event in story_step_element:
            # OR statement can also contain `slot_was_set`, and
            # we're going to ignore this events when representing
            # the story as a string
            if not isinstance(event, UserUttered) and not isinstance(event, SlotSet):
                raise EventTypeError(
                    "OR statement events must be of type `UserUttered` or `SlotSet`."
                )

        event_as_strings = [
            StoryStep._event_to_story_string(element, e2e)
            for element in story_step_element
        ]
        result = " OR ".join([event for event in event_as_strings if event is not None])

        return f"* {result}\n"

    def as_story_string(self, flat: bool = False, e2e: bool = False) -> Text:
        """Returns a story as a string."""
        # if the result should be flattened, we
        # will exclude the caption and any checkpoints.
        if flat:
            result = ""
        else:
            result = f"\n## {self.block_name}\n"
            for checkpoint in self.start_checkpoints:
                if checkpoint.name != STORY_START:
                    result += self._checkpoint_string(checkpoint)

        for event in self.events:
            if (
                self.is_action_listen(event)
                or self.is_action_session_start(event)
                or self.is_action_unlikely_intent(event)
                or isinstance(event, SessionStarted)
            ):
                continue

            if isinstance(event, UserUttered):
                result += self._user_string(event, e2e)
            elif isinstance(event, Event):
                converted = event.as_story_string()
                if converted:
                    result += self._bot_string(event)
            elif isinstance(event, list):
                # The story reader classes support reading stories in
                # conversion mode.  When this mode is enabled, OR statements
                # are represented as lists of events.
                or_string = self._or_string(event, e2e)
                if or_string:
                    result += or_string
            else:
                raise Exception(f"Unexpected element in story step: {event}")

        if not flat:
            for checkpoint in self.end_checkpoints:
                result += self._checkpoint_string(checkpoint)
        return result

    @staticmethod
    def is_action_listen(event: Event) -> bool:
        # this is not an `isinstance` because
        # we don't want to allow subclasses here
        return type(event) == ActionExecuted and event.action_name == ACTION_LISTEN_NAME

    @staticmethod
    def is_action_unlikely_intent(event: Event) -> bool:
        """Checks if the executed action is a `action_unlikely_intent`."""
        return (
            type(event) == ActionExecuted
            and event.action_name == ACTION_UNLIKELY_INTENT_NAME
        )

    @staticmethod
    def is_action_session_start(event: Event) -> bool:
        """Checks if the executed action is a `action_session_start`."""
        # this is not an `isinstance` because
        # we don't want to allow subclasses here
        return (
            type(event) == ActionExecuted
            and event.action_name == ACTION_SESSION_START_NAME
        )

    def _add_action_listen(self, events: List[Event]) -> None:
        if not events or not self.is_action_listen(events[-1]):
            # do not add second action_listen
            events.append(ActionExecuted(ACTION_LISTEN_NAME))

    def explicit_events(
        self, domain: Domain, should_append_final_listen: bool = True
    ) -> List[Event]:
        """Returns events contained in the story step including implicit events.

        Not all events are always listed in the story dsl. This
        includes listen actions as well as implicitly
        set slots. This functions makes these events explicit and
        returns them with the rest of the steps events.
        """
        events: List[Event] = []

        for e in self.events:
            if isinstance(e, UserUttered):
                self._add_action_listen(events)
                events.append(e)
                events.extend(domain.slots_for_entities(e.entities))
            else:
                events.append(e)

        if not self.end_checkpoints and should_append_final_listen:
            self._add_action_listen(events)

        return events

    def __repr__(self) -> Text:
        return (
            "StoryStep("
            "block_name={!r}, "
            "start_checkpoints={!r}, "
            "end_checkpoints={!r}, "
            "events={!r})".format(
                self.block_name,
                self.start_checkpoints,
                self.end_checkpoints,
                self.events,
            )
        )


class RuleStep(StoryStep):
    """A Special type of StoryStep representing a Rule."""

    def __init__(
        self,
        block_name: Optional[Text] = None,
        start_checkpoints: Optional[List[Checkpoint]] = None,
        end_checkpoints: Optional[List[Checkpoint]] = None,
        events: Optional[List[Union[Event, List[Event]]]] = None,
        source_name: Optional[Text] = None,
        condition_events_indices: Optional[Set[int]] = None,
    ) -> None:
        super().__init__(
            block_name, start_checkpoints, end_checkpoints, events, source_name
        )
        self.condition_events_indices = (
            condition_events_indices if condition_events_indices else set()
        )

    def create_copy(self, use_new_id: bool) -> "StoryStep":
        copied = RuleStep(
            self.block_name,
            self.start_checkpoints,
            self.end_checkpoints,
            self.events[:],
            self.source_name,
            self.condition_events_indices,
        )
        if not use_new_id:
            copied.id = self.id
        return copied

    def __repr__(self) -> Text:
        return (
            "RuleStep("
            "block_name={!r}, "
            "start_checkpoints={!r}, "
            "end_checkpoints={!r}, "
            "events={!r})".format(
                self.block_name,
                self.start_checkpoints,
                self.end_checkpoints,
                self.events,
            )
        )

    def get_rules_condition(self) -> List[Union[Event, List[Event]]]:
        """Returns a list of events forming a condition of the Rule."""
        return [
            event
            for event_id, event in enumerate(self.events)
            if event_id in self.condition_events_indices
        ]

    def get_rules_events(self) -> List[Union[Event, List[Event]]]:
        """Returns a list of events forming the Rule, that are not conditions."""
        return [
            event
            for event_id, event in enumerate(self.events)
            if event_id not in self.condition_events_indices
        ]

    def add_event_as_condition(self, event: Event) -> None:
        """Adds event to the Rule as part of its condition.

        Args:
            event: The event to be added.
        """
        self.condition_events_indices.add(len(self.events))
        self.events.append(event)


class Story:
    def __init__(
        self,
        story_steps: Optional[List[StoryStep]] = None,
        story_name: Optional[Text] = None,
    ) -> None:
        self.story_steps = story_steps if story_steps else []
        self.story_name = story_name

    @staticmethod
    def from_events(events: List[Event], story_name: Optional[Text] = None) -> "Story":
        """Create a story from a list of events."""
        story_step = StoryStep(story_name)
        for event in events:
            story_step.add_event(event)
        return Story([story_step], story_name)

    def as_dialogue(self, sender_id: Text, domain: Domain) -> Dialogue:
        events = []
        for step in self.story_steps:
            events.extend(
                step.explicit_events(domain, should_append_final_listen=False)
            )

        events.append(ActionExecuted(ACTION_LISTEN_NAME))
        return Dialogue(sender_id, events)

    def as_story_string(self, flat: bool = False, e2e: bool = False) -> Text:
        story_content = ""
        for step in self.story_steps:
            story_content += step.as_story_string(flat, e2e)

        if flat:
            if self.story_name:
                name = self.story_name
            else:
                name = "Generated Story {}".format(hash(story_content))
            return f"## {name}\n{story_content}"
        else:
            return story_content


class StoryGraph:
    """Graph of the story-steps pooled from all stories in the training data."""

    def __init__(
        self,
        story_steps: List[StoryStep],
        story_end_checkpoints: Optional[Dict[Text, Text]] = None,
    ) -> None:
        self.story_steps = story_steps
        self.step_lookup = {s.id: s for s in self.story_steps}
        ordered_ids, cyclic_edges = StoryGraph.order_steps(story_steps)
        self.ordered_ids = ordered_ids
        self.cyclic_edge_ids = cyclic_edges
        if story_end_checkpoints:
            self.story_end_checkpoints = story_end_checkpoints
        else:
            self.story_end_checkpoints = {}

    def __hash__(self) -> int:
        """Return hash for the story step.

        Returns:
            Hash of the story step.
        """
        return int(self.fingerprint(), 16)

    def fingerprint(self) -> Text:
        """Returns a unique hash for the stories which is stable across python runs.

        Returns:
            fingerprint of the stories
        """
        from rasa.shared.core.training_data.story_writer.yaml_story_writer import (
            YAMLStoryWriter,
        )

        stories_as_yaml = YAMLStoryWriter().stories_to_yaml(self.story_steps)
        return rasa.shared.utils.io.deep_container_fingerprint(stories_as_yaml)

    def ordered_steps(self) -> List[StoryStep]:
        """Returns the story steps ordered by topological order of the DAG."""
        return [self._get_step(step_id) for step_id in self.ordered_ids]

    def cyclic_edges(self) -> List[Tuple[Optional[StoryStep], Optional[StoryStep]]]:
        """Returns the story steps ordered by topological order of the DAG."""
        return [
            (self._get_step(source), self._get_step(target))
            for source, target in self.cyclic_edge_ids
        ]

    def merge(self, other: Optional["StoryGraph"]) -> "StoryGraph":
        """Merge two StoryGraph together."""
        if not other:
            return self

        steps = self.story_steps.copy() + other.story_steps
        story_end_checkpoints = self.story_end_checkpoints.copy().update(
            other.story_end_checkpoints
        )
        return StoryGraph(steps, story_end_checkpoints)

    @staticmethod
    def overlapping_checkpoint_names(
        cps: List[Checkpoint], other_cps: List[Checkpoint]
    ) -> Set[Text]:
        """Find overlapping checkpoints names."""
        return {cp.name for cp in cps} & {cp.name for cp in other_cps}

    def with_cycles_removed(self) -> "StoryGraph":
        """Create a graph with the cyclic edges removed from this graph."""
        story_end_checkpoints = self.story_end_checkpoints.copy()
        cyclic_edge_ids = self.cyclic_edge_ids
        # we need to remove the start steps and replace them with steps ending
        # in a special end checkpoint

        story_steps = {s.id: s for s in self.story_steps}

        # collect all overlapping checkpoints
        # we will remove unused start ones
        all_overlapping_cps = set()

        if self.cyclic_edge_ids:
            # we are going to do this in a recursive way. we are going to
            # remove one cycle and then we are going to
            # let the cycle detection run again
            # this is not inherently necessary so if this becomes a performance
            # issue, we can change it. It is actually enough to run the cycle
            # detection only once and then remove one cycle after another, but
            # since removing the cycle is done by adding / removing edges and
            #  nodes
            # the logic is a lot easier if we only need to make sure the
            # change is consistent if we only change one compared to
            # changing all of them.

            for s, e in cyclic_edge_ids:
                cid = generate_id(max_chars=GENERATED_HASH_LENGTH)
                prefix = GENERATED_CHECKPOINT_PREFIX + CHECKPOINT_CYCLE_PREFIX
                # need abbreviations otherwise they are not visualized well
                sink_cp_name = prefix + "SINK_" + cid
                connector_cp_name = prefix + "CONN_" + cid
                source_cp_name = prefix + "SRC_" + cid
                story_end_checkpoints[sink_cp_name] = source_cp_name

                overlapping_cps = self.overlapping_checkpoint_names(
                    story_steps[s].end_checkpoints, story_steps[e].start_checkpoints
                )

                all_overlapping_cps.update(overlapping_cps)

                # change end checkpoints of starts
                start = story_steps[s].create_copy(use_new_id=False)
                start.end_checkpoints = [
                    cp for cp in start.end_checkpoints if cp.name not in overlapping_cps
                ]
                start.end_checkpoints.append(Checkpoint(sink_cp_name))
                story_steps[s] = start

                needs_connector = False

                for k, step in list(story_steps.items()):
                    additional_ends = []
                    for original_cp in overlapping_cps:
                        for cp in step.start_checkpoints:
                            if cp.name == original_cp:
                                if k == e:
                                    cp_name = source_cp_name
                                else:
                                    cp_name = connector_cp_name
                                    needs_connector = True

                                if not self._is_checkpoint_in_list(
                                    cp_name, cp.conditions, step.start_checkpoints
                                ):
                                    # add checkpoint only if it was not added
                                    additional_ends.append(
                                        Checkpoint(cp_name, cp.conditions)
                                    )

                    if additional_ends:
                        updated = step.create_copy(use_new_id=False)
                        updated.start_checkpoints.extend(additional_ends)
                        story_steps[k] = updated

                if needs_connector:
                    start.end_checkpoints.append(Checkpoint(connector_cp_name))

        # the process above may generate unused checkpoints
        # we need to find them and remove them
        self._remove_unused_generated_cps(
            story_steps, all_overlapping_cps, story_end_checkpoints
        )

        return StoryGraph(list(story_steps.values()), story_end_checkpoints)

    @staticmethod
    def _checkpoint_difference(
        cps: List[Checkpoint], cp_name_to_ignore: Set[Text]
    ) -> List[Checkpoint]:
        """Finds checkpoints which names are
        different form names of checkpoints to ignore.
        """
        return [cp for cp in cps if cp.name not in cp_name_to_ignore]

    def _remove_unused_generated_cps(
        self,
        story_steps: Dict[Text, StoryStep],
        overlapping_cps: Set[Text],
        story_end_checkpoints: Dict[Text, Text],
    ) -> None:
        """Finds unused generated checkpoints
        and remove them from story steps.
        """
        unused_cps = self._find_unused_checkpoints(
            story_steps.values(), story_end_checkpoints
        )

        unused_overlapping_cps = unused_cps.intersection(overlapping_cps)

        unused_genr_cps = {
            cp_name
            for cp_name in unused_cps
            if cp_name is not None and cp_name.startswith(GENERATED_CHECKPOINT_PREFIX)
        }

        k_to_remove = set()
        for k, step in story_steps.items():
            # changed all ends
            updated = step.create_copy(use_new_id=False)
            updated.start_checkpoints = self._checkpoint_difference(
                updated.start_checkpoints, unused_overlapping_cps
            )

            # remove generated unused end checkpoints
            updated.end_checkpoints = self._checkpoint_difference(
                updated.end_checkpoints, unused_genr_cps
            )

            if (
                step.start_checkpoints
                and not updated.start_checkpoints
                or step.end_checkpoints
                and not updated.end_checkpoints
            ):
                # remove story step if the generated checkpoints
                # were the only ones
                k_to_remove.add(k)

            story_steps[k] = updated

        # remove unwanted story steps
        for k in k_to_remove:
            del story_steps[k]

    @staticmethod
    def _is_checkpoint_in_list(
        checkpoint_name: Text, conditions: Dict[Text, Any], cps: List[Checkpoint]
    ) -> bool:
        """Checks if checkpoint with name and conditions is
        already in the list of checkpoints.
        """
        for cp in cps:
            if checkpoint_name == cp.name and conditions == cp.conditions:
                return True
        return False

    @staticmethod
    def _find_unused_checkpoints(
        story_steps: ValuesView[StoryStep], story_end_checkpoints: Dict[Text, Text]
    ) -> Set[Optional[Text]]:
        """Finds all unused checkpoints."""
        collected_start = {STORY_END, STORY_START}
        collected_end = {STORY_END, STORY_START}

        for step in story_steps:
            for start in step.start_checkpoints:
                collected_start.add(start.name)
            for end in step.end_checkpoints:
                start_name = story_end_checkpoints.get(end.name, end.name)
                collected_end.add(start_name)

        return collected_end.symmetric_difference(collected_start)

    def _get_step(self, step_id: Text) -> StoryStep:
        """Looks a story step up by its id."""
        return self.step_lookup[step_id]

    @staticmethod
    def order_steps(
        story_steps: List[StoryStep],
    ) -> Tuple[deque, List[Tuple[Text, Text]]]:
        """Topological sort of the steps returning the ids of the steps."""
        checkpoints = StoryGraph._group_by_start_checkpoint(story_steps)
        graph = {
            s.id: {
                other.id for end in s.end_checkpoints for other in checkpoints[end.name]
            }
            for s in story_steps
        }
        return StoryGraph.topological_sort(graph)

    @staticmethod
    def _group_by_start_checkpoint(
        story_steps: List[StoryStep],
    ) -> Dict[Text, List[StoryStep]]:
        """Returns all the start checkpoint of the steps."""
        checkpoints = defaultdict(list)
        for step in story_steps:
            for start in step.start_checkpoints:
                checkpoints[start.name].append(step)
        return checkpoints

    @staticmethod
    def topological_sort(
        graph: Dict[Text, Set[Text]],
    ) -> Tuple[deque, List[Tuple[Text, Text]]]:
        """Creates a top sort of a directed graph. This is an unstable sorting!

        The function returns the sorted nodes as well as the edges that need
        to be removed from the graph to make it acyclic (and hence, sortable).

        The graph should be represented as a dictionary, e.g.:

        >>> example_graph = {
        ...         "a": set("b", "c", "d"),
        ...         "b": set(),
        ...         "c": set("d"),
        ...         "d": set(),
        ...         "e": set("f"),
        ...         "f": set()}
        >>> StoryGraph.topological_sort(example_graph)
        (deque([u'e', u'f', u'a', u'c', u'd', u'b']), [])
        """
        # noinspection PyPep8Naming
        GRAY, BLACK = 0, 1

        ordered: Deque = deque()
        unprocessed = sorted(set(graph))
        visited_nodes = {}

        removed_edges = set()

        def dfs(node: Text) -> None:
            visited_nodes[node] = GRAY
            for k in sorted(graph.get(node, set())):
                sk = visited_nodes.get(k, None)
                if sk == GRAY:
                    removed_edges.add((node, k))
                    continue
                if sk == BLACK:
                    continue
                unprocessed.remove(k)
                dfs(k)
            ordered.appendleft(node)
            visited_nodes[node] = BLACK

        while unprocessed:
            dfs(unprocessed.pop())

        return ordered, sorted(removed_edges)

    def visualize(self, output_file: Optional[Text] = None) -> "nx.MultiDiGraph":
        import networkx as nx
        from rasa.shared.core.training_data import visualization
        from colorhash import ColorHash

        graph = nx.MultiDiGraph()
        next_node_idx = [0]
        nodes = {"STORY_START": 0, "STORY_END": -1}

        def ensure_checkpoint_is_drawn(cp: Checkpoint) -> None:
            if cp.name not in nodes:
                next_node_idx[0] += 1
                nodes[cp.name] = next_node_idx[0]

                if cp.name.startswith(GENERATED_CHECKPOINT_PREFIX):
                    # colors generated checkpoints based on their hash
                    color = ColorHash(cp.name[-GENERATED_HASH_LENGTH:]).hex
                    graph.add_node(
                        next_node_idx[0],
                        label=_cap_length(cp.name),
                        style="filled",
                        fillcolor=color,
                    )
                else:
                    graph.add_node(next_node_idx[0], label=_cap_length(cp.name))

        graph.add_node(
            nodes["STORY_START"], label="START", fillcolor="green", style="filled"
        )
        graph.add_node(nodes["STORY_END"], label="END", fillcolor="red", style="filled")

        for step in self.story_steps:
            next_node_idx[0] += 1
            step_idx = next_node_idx[0]

            graph.add_node(
                next_node_idx[0],
                label=_cap_length(step.block_name),
                style="filled",
                fillcolor="lightblue",
                shape="rect",
            )

            for c in step.start_checkpoints:
                ensure_checkpoint_is_drawn(c)
                graph.add_edge(nodes[c.name], step_idx)
            for c in step.end_checkpoints:
                ensure_checkpoint_is_drawn(c)
                graph.add_edge(step_idx, nodes[c.name])

            if not step.end_checkpoints:
                graph.add_edge(step_idx, nodes["STORY_END"])

        if output_file:
            visualization.persist_graph(graph, output_file)

        return graph

    def is_empty(self) -> bool:
        """Checks if `StoryGraph` is empty."""
        return not self.story_steps

    def __repr__(self) -> Text:
        """Returns text representation of object."""
        return f"{self.__class__.__name__}: {len(self.story_steps)} story steps"

    def has_e2e_stories(self) -> bool:
        """
        Checks if there are end-to-end (E2E) stories present in the story steps.

        An E2E story is determined by checking if any `UserUttered` event has
        associated text within the story steps.

        Returns:
            bool: True if any E2E story (i.e., a `UserUttered` event with text)
            is found, False otherwise.
        """
        if not self.story_steps:
            return False
        for story_step in self.story_steps:
            for event in story_step.events:
                if isinstance(event, UserUttered):
                    if event.text:
                        return True
        return False


def generate_id(prefix: Text = "", max_chars: Optional[int] = None) -> Text:
    """Generate a random UUID.

    Args:
        prefix: String to prefix the ID with.
        max_chars: Maximum number of characters.

    Returns:
        Generated random UUID.
    """
    import uuid

    gid = uuid.uuid4().hex
    if max_chars:
        gid = gid[:max_chars]

    return f"{prefix}{gid}"


def _cap_length(s: Text, char_limit: int = 20, append_ellipsis: bool = True) -> Text:
    """Makes sure the string doesn't exceed the passed char limit.

    Appends an ellipsis if the string is too long.
    """
    if len(s) > char_limit:
        if append_ellipsis:
            return s[: char_limit - 3] + "..."
        else:
            return s[:char_limit]
    else:
        return s

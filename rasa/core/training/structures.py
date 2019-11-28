import json
import logging
import sys
import uuid
from collections import deque, defaultdict
from typing import List, Text, Dict, Optional, Tuple, Any, Set, ValuesView

from rasa.core import utils
from rasa.core.actions.action import ACTION_LISTEN_NAME
from rasa.core.conversation import Dialogue
from rasa.core.domain import Domain
from rasa.core.events import (
    UserUttered,
    ActionExecuted,
    Form,
    FormValidation,
    SlotSet,
    Event,
    ActionExecutionRejected,
)

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


class StoryStringHelper:
    """A helper class to mark story steps that are inside a form with `form: `
    """

    def __init__(
        self,
        active_form=None,
        form_validation=True,
        form_rejected=False,
        form_prefix_string="",
        no_form_prefix_string="",
    ):
        # track active form
        self.active_form = active_form
        # track whether a from should be validated
        self.form_validation = form_validation
        # track whether a from was rejected
        self.form_rejected = form_rejected
        # save story strings with form prefix for later
        self.form_prefix_string = form_prefix_string
        # save story strings without form prefix for later
        self.no_form_prefix_string = no_form_prefix_string


class Checkpoint:
    def __init__(
        self, name: Optional[Text], conditions: Optional[Dict[Text, Any]] = None
    ) -> None:

        self.name = name
        self.conditions = conditions if conditions else {}

    def as_story_string(self):
        dumped_conds = json.dumps(self.conditions) if self.conditions else ""
        return f"{self.name}{dumped_conds}"

    def filter_trackers(self, trackers):
        """Filters out all trackers that do not satisfy the conditions."""

        if not self.conditions:
            return trackers

        for slot_name, slot_value in self.conditions.items():
            trackers = [t for t in trackers if t.get_slot(slot_name) == slot_value]
        return trackers

    def __repr__(self):
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
        block_name: Optional[Text] = None,
        start_checkpoints: Optional[List[Checkpoint]] = None,
        end_checkpoints: Optional[List[Checkpoint]] = None,
        events: Optional[List[Event]] = None,
    ) -> None:

        self.end_checkpoints = end_checkpoints if end_checkpoints else []
        self.start_checkpoints = start_checkpoints if start_checkpoints else []
        self.events = events if events else []
        self.block_name = block_name
        # put a counter prefix to uuid to get reproducible sorting results
        global STEP_COUNT
        self.id = "{}_{}".format(STEP_COUNT, uuid.uuid4().hex)
        STEP_COUNT += 1

        self.story_string_helper = StoryStringHelper()

    def create_copy(self, use_new_id):
        copied = StoryStep(
            self.block_name,
            self.start_checkpoints,
            self.end_checkpoints,
            self.events[:],
        )
        if not use_new_id:
            copied.id = self.id
        return copied

    def add_user_message(self, user_message):
        self.add_event(user_message)

    def add_event(self, event):
        self.events.append(event)

    @staticmethod
    def _checkpoint_string(story_step_element: UserUttered) -> Text:
        return "> {}\n".format(story_step_element.as_story_string())

    @staticmethod
    def _user_string(
        story_step_element: UserUttered, e2e: bool, prefix: Text = ""
    ) -> Text:
        return "* {}{}\n".format(prefix, story_step_element.as_story_string(e2e))

    def _store_user_strings(
        self, story_step_element: UserUttered, e2e: bool, prefix: Text = ""
    ) -> None:
        self.story_string_helper.no_form_prefix_string += self._user_string(
            story_step_element, e2e
        )
        self.story_string_helper.form_prefix_string += self._user_string(
            story_step_element, e2e, prefix
        )

    @staticmethod
    def _bot_string(story_step_element, prefix=""):
        return "    - {}{}\n".format(prefix, story_step_element.as_story_string())

    def _store_bot_strings(self, story_step_element, prefix=""):
        self.story_string_helper.no_form_prefix_string += self._bot_string(
            story_step_element
        )
        self.story_string_helper.form_prefix_string += self._bot_string(
            story_step_element, prefix
        )

    def _reset_stored_strings(self):
        self.story_string_helper.form_prefix_string = ""
        self.story_string_helper.no_form_prefix_string = ""

    def as_story_string(self, flat: bool = False, e2e: bool = False) -> Text:
        # if the result should be flattened, we
        # will exclude the caption and any checkpoints.

        for s in self.start_checkpoints:
            if s.name == STORY_START:
                # first story step in the story, so reset helper
                self.story_string_helper = StoryStringHelper()

        if flat:
            result = ""
        else:
            result = f"\n## {self.block_name}\n"
            for s in self.start_checkpoints:
                if s.name != STORY_START:
                    result += self._checkpoint_string(s)

        for s in self.events:
            if isinstance(s, UserUttered):
                if self.story_string_helper.active_form is None:
                    result += self._user_string(s, e2e)
                else:
                    # form is active
                    # it is not known whether the form will be
                    # successfully executed, so store this
                    # story string for later
                    self._store_user_strings(s, e2e, FORM_PREFIX)

            elif isinstance(s, Form):
                # form got either activated or deactivated
                self.story_string_helper.active_form = s.name

                if self.story_string_helper.active_form is None:
                    # form deactivated, so form succeeded,
                    # so add story string with form prefix
                    result += self.story_string_helper.form_prefix_string
                    # remove all stored story strings
                    self._reset_stored_strings()

                result += self._bot_string(s)

            elif isinstance(s, FormValidation):
                self.story_string_helper.form_validation = s.validate

            elif isinstance(s, ActionExecutionRejected):
                if s.action_name == self.story_string_helper.active_form:
                    # form rejected
                    self.story_string_helper.form_rejected = True

            elif isinstance(s, ActionExecuted):
                if self._is_action_listen(s):
                    pass
                elif self.story_string_helper.active_form is None:
                    result += self._bot_string(s)
                else:
                    # form is active
                    if self.story_string_helper.form_rejected:
                        if (
                            self.story_string_helper.form_prefix_string
                            and self.story_string_helper.form_validation
                            and s.action_name == self.story_string_helper.active_form
                        ):
                            # if there is something in `form_prefix_string`,
                            # add action_listen before it,
                            # because this form user input will be ignored by core
                            # and therefore action_listen will not be automatically
                            # added during reading the stories
                            result += self._bot_string(
                                ActionExecuted(ACTION_LISTEN_NAME)
                            )
                            result += self.story_string_helper.form_prefix_string
                        elif self.story_string_helper.no_form_prefix_string:
                            result += self.story_string_helper.no_form_prefix_string
                        # form rejected, add story string without form prefix
                        result += self._bot_string(s)
                    else:
                        # form succeeded, so add story string with form prefix
                        result += self.story_string_helper.form_prefix_string
                        result += self._bot_string(s, FORM_PREFIX)

                    # remove all stored story strings
                    self._reset_stored_strings()

                    if s.action_name == self.story_string_helper.active_form:
                        # form was successfully executed
                        self.story_string_helper.form_rejected = False

                self.story_string_helper.form_validation = True

            elif isinstance(s, SlotSet):
                if self.story_string_helper.active_form is None:
                    result += self._bot_string(s)
                else:
                    # form is active
                    # it is not known whether the form will be
                    # successfully executed, so store this
                    # story string for later
                    # slots should be always printed without prefix
                    self._store_bot_strings(s)

            elif isinstance(s, Event):
                converted = s.as_story_string()
                if converted:
                    if self.story_string_helper.active_form is None:
                        result += self._bot_string(s)
                    else:
                        # form is active
                        # it is not known whether the form will be
                        # successfully executed, so store this
                        # story string for later
                        self._store_bot_strings(s, FORM_PREFIX)

            else:
                raise Exception(f"Unexpected element in story step: {s}")

        if (
            not self.end_checkpoints
            and self.story_string_helper.active_form is not None
        ):
            # there are no end checkpoints
            # form is active
            # add story string with form prefix
            result += self.story_string_helper.form_prefix_string
            # remove all stored story strings
            self._reset_stored_strings()

        if not flat:
            for e in self.end_checkpoints:
                result += "> {}\n".format(e.as_story_string())
        return result

    @staticmethod
    def _is_action_listen(event):
        # this is not an `isinstance` because
        # we don't want to allow subclasses here
        return type(event) == ActionExecuted and event.action_name == ACTION_LISTEN_NAME

    def _add_action_listen(self, events):
        if not events or not self._is_action_listen(events[-1]):
            # do not add second action_listen
            events.append(ActionExecuted(ACTION_LISTEN_NAME))

    def explicit_events(
        self, domain: Domain, should_append_final_listen: bool = True
    ) -> List[Event]:
        """Returns events contained in the story step
            including implicit events.

        Not all events are always listed in the story dsl. This
        includes listen actions as well as implicitly
        set slots. This functions makes these events explicit and
        returns them with the rest of the steps events."""

        events = []

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

    def __repr__(self):
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


class Story:
    def __init__(
        self, story_steps: List[StoryStep] = None, story_name: Optional[Text] = None
    ) -> None:
        self.story_steps = story_steps if story_steps else []
        self.story_name = story_name

    @staticmethod
    def from_events(events, story_name=None):
        """Create a story from a list of events."""

        story_step = StoryStep()
        for event in events:
            story_step.add_event(event)
        return Story([story_step], story_name)

    def as_dialogue(self, sender_id, domain):
        events = []
        for step in self.story_steps:
            events.extend(
                step.explicit_events(domain, should_append_final_listen=False)
            )

        events.append(ActionExecuted(ACTION_LISTEN_NAME))
        return Dialogue(sender_id, events)

    def as_story_string(self, flat: bool = False, e2e: bool = False) -> Text:
        story_content = ""

        # initialize helper for first story step
        story_string_helper = StoryStringHelper()

        for step in self.story_steps:
            # use helper from previous story step
            step.story_string_helper = story_string_helper
            # create string for current story step
            story_content += step.as_story_string(flat, e2e)
            # override helper for next story step
            story_string_helper = step.story_string_helper

        if flat:
            if self.story_name:
                name = self.story_name
            else:
                name = "Generated Story {}".format(hash(story_content))
            return f"## {name}\n{story_content}"
        else:
            return story_content

    def dump_to_file(
        self, filename: Text, flat: bool = False, e2e: bool = False
    ) -> None:
        from rasa.utils import io

        io.write_text_file(self.as_story_string(flat, e2e), filename, append=True)


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
        self_as_string = self.as_story_string()
        text_hash = utils.get_text_hash(self_as_string)

        return int(text_hash, 16)

    def ordered_steps(self) -> List[StoryStep]:
        """Returns the story steps ordered by topological order of the DAG."""

        return [self.get(step_id) for step_id in self.ordered_ids]

    def cyclic_edges(self) -> List[Tuple[Optional[StoryStep], Optional[StoryStep]]]:
        """Returns the story steps ordered by topological order of the DAG."""

        return [
            (self.get(source), self.get(target))
            for source, target in self.cyclic_edge_ids
        ]

    def merge(self, other: Optional["StoryGraph"]) -> "StoryGraph":
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
        """Find overlapping checkpoints names"""

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
                cid = utils.generate_id(max_chars=GENERATED_HASH_LENGTH)
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
            different form names of checkpoints to ignore"""

        return [cp for cp in cps if cp.name not in cp_name_to_ignore]

    def _remove_unused_generated_cps(
        self,
        story_steps: Dict[Text, StoryStep],
        overlapping_cps: Set[Text],
        story_end_checkpoints: Dict[Text, Text],
    ) -> None:
        """Finds unused generated checkpoints
            and remove them from story steps."""

        unused_cps = self._find_unused_checkpoints(
            story_steps.values(), story_end_checkpoints
        )

        unused_overlapping_cps = unused_cps.intersection(overlapping_cps)

        unused_genr_cps = {
            cp_name
            for cp_name in unused_cps
            if cp_name.startswith(GENERATED_CHECKPOINT_PREFIX)
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
            already in the list of checkpoints."""

        for cp in cps:
            if checkpoint_name == cp.name and conditions == cp.conditions:
                return True
        return False

    @staticmethod
    def _find_unused_checkpoints(
        story_steps: ValuesView[StoryStep], story_end_checkpoints: Dict[Text, Text]
    ) -> Set[Text]:
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

    def get(self, step_id: Text) -> Optional[StoryStep]:
        """Looks a story step up by its id."""

        return self.step_lookup.get(step_id)

    def as_story_string(self) -> Text:
        """Convert the graph into the story file format."""

        story_content = ""
        for step in self.story_steps:
            story_content += step.as_story_string(flat=False)
        return story_content

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
        """Returns all the start checkpoint of the steps"""

        checkpoints = defaultdict(list)
        for step in story_steps:
            for start in step.start_checkpoints:
                checkpoints[start.name].append(step)
        return checkpoints

    @staticmethod
    def topological_sort(
        graph: Dict[Text, Set[Text]]
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

        ordered = deque()
        unprocessed = sorted(set(graph))
        visited_nodes = {}

        removed_edges = set()

        def dfs(node):
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

    def visualize(self, output_file=None):
        import networkx as nx
        from rasa.core.training import visualization  # pytype: disable=pyi-error
        from colorhash import ColorHash

        graph = nx.MultiDiGraph()
        next_node_idx = [0]
        nodes = {"STORY_START": 0, "STORY_END": -1}

        def ensure_checkpoint_is_drawn(cp):
            if cp.name not in nodes:
                next_node_idx[0] += 1
                nodes[cp.name] = next_node_idx[0]

                if cp.name.startswith(GENERATED_CHECKPOINT_PREFIX):
                    # colors generated checkpoints based on their hash
                    color = ColorHash(cp.name[-GENERATED_HASH_LENGTH:]).hex
                    graph.add_node(
                        next_node_idx[0],
                        label=utils.cap_length(cp.name),
                        style="filled",
                        fillcolor=color,
                    )
                else:
                    graph.add_node(next_node_idx[0], label=utils.cap_length(cp.name))

        graph.add_node(
            nodes["STORY_START"], label="START", fillcolor="green", style="filled"
        )
        graph.add_node(nodes["STORY_END"], label="END", fillcolor="red", style="filled")

        for step in self.story_steps:
            next_node_idx[0] += 1
            step_idx = next_node_idx[0]

            graph.add_node(
                next_node_idx[0],
                label=utils.cap_length(step.block_name),
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

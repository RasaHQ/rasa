from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
import json
import logging
import uuid
from collections import deque, defaultdict

import typing
from typing import List, Text, Dict, Optional, Tuple, Any, Deque, Set

from rasa_core import utils
from rasa_core.actions.action import ACTION_LISTEN_NAME
from rasa_core.conversation import Dialogue
from rasa_core.events import UserUttered, ActionExecuted, Event

if typing.TYPE_CHECKING:
    from rasa_core.domain import Domain

logger = logging.getLogger(__name__)

# Checkpoint id used to identify story starting blocks
STORY_START = "STORY_START"

# Checkpoint id used to identify story end blocks
STORY_END = None


class Checkpoint(object):
    def __init__(self, name, conditions=None):
        # type: (Optional[Text], Optional[Dict[Text, Any]]) -> None

        self.name = name
        self.conditions = conditions if conditions else {}

    def as_story_string(self):
        dumped_conds = json.dumps(self.conditions) if self.conditions else ""
        return "{}{}".format(self.name, dumped_conds)

    def filter_trackers(self, trackers):
        """Filters out all trackers that do not satisfy the conditions."""

        if not self.conditions:
            return trackers

        for slot_name, slot_value in self.conditions.items():
            trackers = [t
                        for t in trackers
                        if t.tracker.get_slot(slot_name) == slot_value]
        return trackers

    def __str__(self):
        return "Checkpoint({})".format(self.as_story_string())


class StoryStep(object):
    def __init__(self,
                 block_name=None,  # type: Optional[Text]
                 start_checkpoint=None,  # type: Optional[Checkpoint]
                 end_checkpoint=None,  # type: Optional[Checkpoint]
                 events=None  # type: Optional[List[Event]]
                 ):
        # type: (...) -> None

        self.end_checkpoint = end_checkpoint
        self.start_checkpoint = start_checkpoint
        self.events = events if events else []
        self.block_name = block_name
        self.id = uuid.uuid4().hex  # type: Text

    def start_checkpoint_name(self):
        return self.start_checkpoint.name if self.start_checkpoint else None

    def end_checkpoint_name(self):
        return self.end_checkpoint.name if self.end_checkpoint else None

    def create_copy(self, use_new_id):
        copied = StoryStep(self.block_name, self.start_checkpoint,
                           self.end_checkpoint,
                           self.events[:])
        if not use_new_id:
            copied.id = self.id
        return copied

    def add_user_message(self, user_message):
        self.add_event(user_message)

    @staticmethod
    def _is_action_listen(event):
        return (isinstance(event, ActionExecuted) and
                event.action_name == ACTION_LISTEN_NAME)

    def add_event(self, event):
        # stories never contain the action listen events they are implicit
        # and added after a story is read and converted to a dialogue
        if not self._is_action_listen(event):
            self.events.append(event)

    def as_story_string(self, flat=False):
        # if the result should be flattened, we
        # will exclude the caption and any checkpoints.
        if flat:
            result = ""
        else:
            result = "\n## {}\n".format(self.block_name)
            if self.start_checkpoint_name() != STORY_START:
                cp = self.start_checkpoint.as_story_string()
                result += "> {}\n".format(cp)
        for s in self.events:
            if isinstance(s, UserUttered):
                result += "* {}\n".format(s.as_story_string())
            elif isinstance(s, Event):
                converted = s.as_story_string()
                if converted:
                    result += "    - {}\n".format(s.as_story_string())
            else:
                raise Exception("Unexpected element in story step: "
                                "{}".format(s))

        if not flat:
            if self.end_checkpoint != STORY_END:
                cp = self.end_checkpoint.as_story_string()
                result += "> {}\n".format(cp)
        return result

    def explicit_events(self, domain, should_append_final_listen=True):
        # type: (Domain, bool) -> List[Event]
        """Returns events contained in the story step including implicit events.

        Not all events are always listed in the story dsl. This
        includes listen actions as well as implicitly
        set slots. This functions makes these events explicit and
        returns them with the rest of the steps events."""

        events = []

        for e in self.events:
            if isinstance(e, UserUttered):
                events.append(ActionExecuted(ACTION_LISTEN_NAME))
                events.append(e)
                events.extend(domain.slots_for_entities(e.entities))
            else:
                events.append(e)

        if self.end_checkpoint == STORY_END and should_append_final_listen:
            events.append(ActionExecuted(ACTION_LISTEN_NAME))
        return events


class Story(object):
    def __init__(self, story_steps=None):
        # type: (List[StoryStep]) -> None
        self.story_steps = story_steps if story_steps else []

    def as_dialogue(self, sender_id, domain):
        events = []
        for step in self.story_steps:
            events.extend(
                    step.explicit_events(domain,
                                         should_append_final_listen=False))

        events.append(ActionExecuted(ACTION_LISTEN_NAME))
        return Dialogue(sender_id, events)

    def as_story_string(self, flat=False):
        story_content = ""
        for step in self.story_steps:
            story_content += step.as_story_string(flat)

        if flat:
            return "## Generated Story {}\n{}".format(
                    hash(story_content), story_content)
        else:
            return story_content

    def dump_to_file(self, filename, flat=False):
        with io.open(filename, "a") as f:
            f.write(self.as_story_string(flat))


class StoryGraph(object):
    def __init__(self, story_steps, story_end_checkpoints=None):
        # type: (List[StoryStep]) -> None
        self.story_steps = story_steps
        self.step_lookup = {s.id: s for s in self.story_steps}
        ordered_ids, cyclic_edges = StoryGraph.order_steps(story_steps)
        self.ordered_ids = ordered_ids
        self.cyclic_edge_ids = cyclic_edges
        if story_end_checkpoints:
            self.story_end_checkpoints = story_end_checkpoints
        else:
            self.story_end_checkpoints = {}

    def ordered_steps(self):
        # type: () -> List[StoryStep]
        """Returns the story steps ordered by topological order of the DAG."""

        return [self.get(step_id) for step_id in self.ordered_ids]

    def cyclic_edges(self):
        # type: () -> List[Tuple[Optional[StoryStep], Optional[StoryStep]]]
        """Returns the story steps ordered by topological order of the DAG."""

        return [(self.get(source), self.get(target))
                for source, target in self.cyclic_edge_ids]

    def with_cycles_removed(self):
        # type: () -> StoryGraph
        """Create a graph with the cyclic edges removed from this graph."""

        if not self.cyclic_edge_ids:
            return self

        story_end_checkpoints = self.story_end_checkpoints.copy()
        cyclic_edge_ids = self.cyclic_edge_ids
        # we need to remove the start steps and replace them with steps ending
        # in a special end checkpoint
        story_steps = {s.id: s.create_copy(use_new_id=False)
                       for s in self.story_steps}

        # we are going to do this in a recursive way. we are going to remove
        # one cycle and then we are going to let the cycle detection run again
        # this is not inherently necessary so if this becomes a performance
        # issue, we can change it. It is actually enough to run the cycle
        # detection only once and then remove one cycle after another, but
        # since removing the cycle is done by adding / removing edges and nodes
        # the logic is a lot easier if we only need to make sure the change is
        # consistent if we only change one compared to changing all of them.

        s, e = cyclic_edge_ids.pop()

        cid = utils.generate_id()
        start_cid = "CYCLE_S_" + cid
        connector_cid = "CYCLE_C_" + cid
        end_cid = "CYCLE_E_" + cid
        story_end_checkpoints[start_cid] = end_cid

        # changed all starts
        start = story_steps[s]
        original_end = start.end_checkpoint
        start.end_checkpoint = Checkpoint(start_cid)

        needs_connector = False

        for k, step in story_steps.items()[:]:
            if (step.start_checkpoint
                    and step.start_checkpoint.name == original_end.name):

                if k == e:
                    cid = end_cid
                else:
                    cid = connector_cid
                    needs_connector = True

                modified = step.create_copy(use_new_id=True)
                modified.start_checkpoint = Checkpoint(
                        cid,
                        step.start_checkpoint.conditions)
                story_steps[modified.id] = modified

        if needs_connector:
            modified = start.create_copy(use_new_id=True)
            modified.end_checkpoint = Checkpoint(connector_cid)
            story_steps[modified.id] = modified

        # remove next cycles in another call (will create a new graph!)
        return StoryGraph(story_steps.values(),
                          story_end_checkpoints).with_cycles_removed()

    def get(self, step_id):
        # type: (Text) -> Optional[StoryStep]
        """Looks a story step up by its id."""

        return self.step_lookup.get(step_id)

    def as_story_string(self):
        # type: () -> Text
        """Convert the graph into the story file format."""

        story_content = ""
        for step in self.story_steps:
            story_content += step.as_story_string(flat=False)
        return story_content

    @staticmethod
    def order_steps(story_steps):
        # type: (List[StoryStep]) -> Deque[Text]
        """Topological sort of the steps returning the ids of the steps."""

        checkpoints = StoryGraph._group_by_start_checkpoint(story_steps)
        graph = {s.id: [other.id
                        for other in checkpoints[s.end_checkpoint_name()]]
                 for s in story_steps}
        return StoryGraph.topological_sort(graph)

    @staticmethod
    def _group_by_start_checkpoint(story_steps):
        # type: (List[StoryStep]) -> Dict[Text, List[StoryStep]]
        """Returns all the start checkpoint of the steps"""

        checkpoints = defaultdict(list)
        for step in story_steps:
            checkpoints[step.start_checkpoint_name()].append(step)
        return checkpoints

    @staticmethod
    def topological_sort(
            graph  # type: Dict[Text, List[Text]]
    ):
        # type: (...) -> Tuple[Deque[Text], Set[Tuple[Text, Text]]]
        """Creates a top sort of a directed graph. This is an unstable sorting!

        The function returns the sorted nodes as well as the edges that need
        to be removed from the graph to make it acyclic (and hence, sortable).

        The graph should be represented as a dictionary, e.g.:

        >>> example_graph = {
        ...         "a": ["b", "c", "d"],
        ...         "b": [],
        ...         "c": ["d"],
        ...         "d": [],
        ...         "e": ["f"],
        ...         "f": []}
        >>> StoryGraph.topological_sort(example_graph)
        (deque([u'e', u'f', u'a', u'c', u'd', u'b']), [])
        """

        GRAY, BLACK = 0, 1
        ordered = deque()
        unprocessed = set(graph)
        visited_nodes = {}

        removed_edges = set()

        def dfs(node):
            visited_nodes[node] = GRAY
            for k in graph.get(node, ()):
                sk = visited_nodes.get(k, None)
                if sk == GRAY:
                    removed_edges.add((node, k))
                    continue
                if sk == BLACK:
                    continue
                unprocessed.discard(k)
                dfs(k)
            ordered.appendleft(node)
            visited_nodes[node] = BLACK

        while unprocessed:
            dfs(unprocessed.pop())
        return ordered, removed_edges

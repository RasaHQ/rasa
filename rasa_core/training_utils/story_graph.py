from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random
from collections import deque, defaultdict
import typing
from rasa_core.domain import Domain
from typing import List, Text, Dict, Optional

from rasa_core.interpreter import RegexInterpreter, NaturalLanguageInterpreter
from rasa_core import utils

if typing.TYPE_CHECKING:
    from rasa_core.training_utils.dsl import StoryStep, Story, \
    TrainingsDataExtractor


class StoryGraph(object):
    def __init__(self, story_steps):
        # type: (List[StoryStep]) -> None
        self.story_steps = story_steps
        self.step_lookup = {s.id: s for s in self.story_steps}
        self.ordered_ids = StoryGraph.order_steps(story_steps)

    def ordered_steps(self):
        # type: () -> List[StoryStep]
        """Returns the story steps ordered by topological order of the DAG."""

        return [self.get(step_id) for step_id in self.ordered_ids]

    def get(self, step_id):
        # type: (Text) -> Optional[StoryStep]
        """Looks a story step up by its id."""

        return self.step_lookup.get(step_id)

    def build_stories(self,
                      domain,
                      max_number_of_trackers=2000):
        # type: (Domain, NaturalLanguageInterpreter, bool, int) -> List[Story]
        """Build the stories of a graph."""
        from rasa_core.training_utils.dsl import STORY_START, Story

        active_trackers = {STORY_START: [Story()]}
        rand = random.Random(42)

        for step in self.ordered_steps():
            if step.start_checkpoint_name() in active_trackers:
                # these are the trackers that reached this story step
                # and that need to handle all events of the step
                incoming_trackers = active_trackers[step.start_checkpoint_name()]

                # TODO: we can't use tracker filter here to filter for
                #       checkpoint conditions since we don't have trackers.
                #       this code should rather use the code from the dsl.

                if max_number_of_trackers is not None:
                    incoming_trackers = utils.subsample_array(
                            incoming_trackers, max_number_of_trackers, rand)

                events = step.explicit_events(domain)
                # need to copy the tracker as multiple story steps might
                # start with the same checkpoint and all of them
                # will use the same set of incoming trackers
                if events:
                    trackers = [Story(tracker.story_steps + [step])
                                for tracker in incoming_trackers]
                else:
                    trackers = []  # small optimization

                # update our tracker dictionary with the trackers that handled
                # the events of the step and that can now be used for further
                # story steps that start with the checkpoint this step ended on
                if step.end_checkpoint_name() not in active_trackers:
                    active_trackers[step.end_checkpoint_name()] = []
                active_trackers[step.end_checkpoint_name()].extend(trackers)

        return active_trackers[None]

    def as_story_string(self):
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
    def topological_sort(graph):
        """Creates a topsort of a directed graph. This is an unstable sorting!

        The graph should be represented as a dictionary, e.g.:

        >>> example_graph = {
        ...         "a": ["b", "c", "d"],
        ...         "b": [],
        ...         "c": ["d"],
        ...         "d": [],
        ...         "e": ["f"],
        ...         "f": []}
        >>> StoryGraph.topological_sort(example_graph)
        deque([u'e', u'f', u'a', u'c', u'd', u'b'])
        """
        GRAY, BLACK = 0, 1
        ordered = deque()
        unprocessed = set(graph)
        visited_nodes = {}

        def dfs(node):
            visited_nodes[node] = GRAY
            for k in graph.get(node, ()):
                sk = visited_nodes.get(k, None)
                if sk == GRAY:
                    raise ValueError("Cycle found at node: {}".format(sk))
                if sk == BLACK:
                    continue
                unprocessed.discard(k)
                dfs(k)
            ordered.appendleft(node)
            visited_nodes[node] = BLACK

        while unprocessed:
            dfs(unprocessed.pop())
        return ordered

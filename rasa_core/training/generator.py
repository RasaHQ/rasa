# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import json
import logging
import random
from collections import defaultdict, namedtuple

import typing
from tqdm import tqdm
from typing import Optional, List, Text, Set, Dict

from rasa_core import utils
from rasa_core.channels import UserMessage
from rasa_core.events import (
    ActionExecuted, UserUttered,
    ActionReverted, UserUtteranceReverted)
from rasa_core.trackers import DialogueStateTracker
from rasa_core.training.structures import (
    StoryGraph, STORY_END, STORY_START, StoryStep,
    GENERATED_CHECKPOINT_PREFIX)

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from rasa_core.domain import Domain

ExtractorConfig = namedtuple("ExtractorConfig", "remove_duplicates "
                                                "augmentation_factor "
                                                "max_number_of_trackers "
                                                "tracker_limit "
                                                "use_story_concatenation "
                                                "rand")

# define types
TrackerLookupDict = Dict[Optional[Text], List[DialogueStateTracker]]


class TrainingDataGenerator(object):
    def __init__(
            self,
            story_graph,  # type: StoryGraph
            domain,  # type: Domain
            remove_duplicates=True,  # type: bool
            augmentation_factor=20,  # type: int
            max_number_of_trackers=2000,  # type: int
            tracker_limit=None,  # type: Optional[int]
            use_story_concatenation=True  # type: bool
    ):
        """Given a set of story parts, generates all stories that are possible.

        The different story parts can end and start with checkpoints
        and this generator will match start and end checkpoints to
        connect complete stories. Afterwards, duplicate stories will be
        removed and the data is augmented (if augmentation is enabled)."""

        self.hashed_featurizations = set()
        self.story_graph = story_graph.with_cycles_removed()
        self.domain = domain
        self.config = ExtractorConfig(
                remove_duplicates=remove_duplicates,
                augmentation_factor=augmentation_factor,
                max_number_of_trackers=max_number_of_trackers,
                tracker_limit=tracker_limit,
                use_story_concatenation=use_story_concatenation,
                rand=random.Random(42))

    def generate(self):
        # type: () -> List[DialogueStateTracker]

        self._mark_first_action_in_story_steps_as_unpredictable()

        unused_checkpoints = set()  # type: Set[Text]
        previous_unused = set()  # type: Set[Text]

        everything_reachable_is_reached = False

        used_checkpoints = set()  # type: Set[Text]
        active_trackers = defaultdict(list)  # type: TrackerLookupDict

        init_tracker = DialogueStateTracker(
                UserMessage.DEFAULT_SENDER_ID,
                self.domain.slots,
                self.domain.topics,
                self.domain.default_topic,
                max_event_history=self.config.tracker_limit
        )
        active_trackers[STORY_START].append(init_tracker)

        finished_trackers = []

        phase = 0
        min_num_phases = 3 if self.config.augmentation_factor > 0 else 0

        # we will continue generating data until we have reached all
        # checkpoints that seem to be reachable. This is a heuristic,
        # if we did not reach any new checkpoints in an iteration, we
        # assume we have reached all and stop.
        while not everything_reachable_is_reached or phase < min_num_phases:
            phase_name = "data generation round {}".format(phase)
            num_trackers = self._count_trackers(active_trackers)
            logger.debug("Starting {} ... (using {} trackers)"
                         "".format(phase_name, num_trackers))

            pbar = tqdm(self.story_graph.ordered_steps(),
                        desc="Processed Story Blocks")

            for step in pbar:
                incoming_trackers = []
                for start in step.start_checkpoints:
                    if not active_trackers[start.name]:
                        # need to skip - there was no previous step that
                        # had this start checkpoint as an end checkpoint
                        unused_checkpoints.add(start.name)
                    else:
                        ts = start.filter_trackers(active_trackers[start.name])
                        incoming_trackers.extend(ts)
                        used_checkpoints.add(start.name)

                if incoming_trackers:
                    # these are the trackers that reached this story
                    # step and that need to handle all events of the step
                    incoming_trackers = self._subsample_trackers(
                            incoming_trackers)

                    trackers = self._process_step(step, incoming_trackers)

                    # update progress bar
                    pbar.set_postfix({
                        "# trackers": len(incoming_trackers)})

                    # update our tracker dictionary with the trackers
                    # that handled the events of the step and
                    # that can now be used for further story steps
                    # that start with the checkpoint this step ended with
                    for end in step.end_checkpoints:
                        active_trackers[end.name].extend(trackers)

                    if not step.end_checkpoints:
                        active_trackers[STORY_END].extend(trackers)

            # trackers that reached the end of a story
            completed = [t for t in active_trackers[STORY_END]]
            finished_trackers.extend(completed)
            active_trackers = self._create_start_trackers(active_trackers)
            logger.debug("Finished phase. ({} training samples found)"
                         "".format(len(finished_trackers)))

            # check if we reached all nodes that can be reached
            # if we reached at least one more node this round than last one,
            # we assume there is still something left to reach and we continue
            unused = unused_checkpoints - used_checkpoints
            everything_reachable_is_reached = unused == previous_unused

            # prepare next round
            previous_unused = unused
            phase += 1

        unused_checkpoints -= used_checkpoints
        self._issue_unused_checkpoint_notification(unused_checkpoints)
        logger.debug("Found {} training examples."
                     "".format(len(finished_trackers)))

        return finished_trackers

    @staticmethod
    def _count_trackers(active_trackers):
        # type: (TrackerLookupDict) -> int
        """Count the number of trackers in the tracker dictionary."""
        return sum(len(ts) for ts in active_trackers.values())

    def _subsample_trackers(self, incoming_trackers):
        # type: (List[DialogueStateTracker]) -> List[DialogueStateTracker]
        """Subsample the list of trackers to retrieve a random subset."""

        # if flows get very long and have a lot of forks we
        # get into trouble by collecting to many trackers
        # hence the sub sampling
        if self.config.max_number_of_trackers is not None:
            return utils.subsample_array(incoming_trackers,
                                         self.config.max_number_of_trackers,
                                         rand=self.config.rand)
        else:
            return incoming_trackers

    def _create_start_trackers(self, active_trackers):
        # type: (TrackerLookupDict) -> TrackerLookupDict
        """One phase is one traversal of all story steps.

        We need to do some cleanup before processing them again."""

        glue_mapping = self.story_graph.story_end_checkpoints
        if self.config.use_story_concatenation:
            glue_mapping[STORY_END] = STORY_START

        next_active_trackers = defaultdict(list)
        for end, start in glue_mapping.items():
            ending_trackers = active_trackers.get(end, [])
            if start == STORY_START:
                ending_trackers = utils.subsample_array(
                        ending_trackers,
                        self.config.augmentation_factor,
                        rand=self.config.rand)

            # This is where the augmentation magic happens. We
            # will reuse all the trackers that reached the
            # end checkpoint `None` (which is the end of a
            # story) and start processing all steps again. So instead
            # of starting with a fresh tracker, the second and
            # all following phases will reuse a couple of the trackers
            # that made their way to a story end.
            for t in ending_trackers:
                # this is a nasty thing - all stories end and
                # start with action listen - so after logging the first
                # actions in the next phase the trackers would
                # contain action listen followed by action listen.
                # to fix this we are going to "undo" the last action listen
                if start == STORY_START:
                    t.update(ActionReverted())
                next_active_trackers[start].append(t)
        return next_active_trackers

    def _process_step(
            self,
            step,  # type: StoryStep
            incoming_trackers  # type: List[DialogueStateTracker]
    ):
        # type: (...) -> List[DialogueStateTracker]
        """Processes a steps events with all trackers.

        The trackers that reached the steps starting checkpoint will
        be used to process the events. Collects and returns training
        data while processing the story step."""

        events = step.explicit_events(self.domain)
        # need to copy the tracker as multiple story steps
        # might start with the same checkpoint and all of them
        # will use the same set of incoming trackers
        trackers = [tracker.copy() for tracker in
                    incoming_trackers] if events else []  # small optimization
        new_trackers = []
        for event in events:
            for tracker in trackers:
                if isinstance(event, (ActionReverted, UserUtteranceReverted)):
                    new_trackers.append(tracker.copy())

                tracker.update(event)

        trackers.extend(new_trackers)
        if self.config.remove_duplicates:
            trackers = self._remove_duplicate_trackers(trackers)

        return trackers

    def _remove_duplicate_trackers(self, trackers):
        # type: (List[DialogueStateTracker]) -> List[DialogueStateTracker]
        """Removes trackers that create equal featurizations.

        From multiple trackers that create equal featurizations
        we only need to keep one. Because as we continue processing
        events and story steps, all trackers that created the
        same featurization once will do so in the future (as we
        feed the same events to all trackers)."""

        # collected trackers that created different featurizations
        unique_trackers = []

        for tracker in trackers:
            states = self.domain.states_for_tracker_history(tracker)
            hashed = hash(tuple((frozenset(s) for s in states)))

            # only continue with trackers that created a
            # hashed_featurization we haven't observed
            if hashed not in self.hashed_featurizations:
                self.hashed_featurizations.add(hashed)
                unique_trackers.append(tracker)

        return unique_trackers

    def _mark_first_action_in_story_steps_as_unpredictable(self):
        # type: () -> None
        """Mark actions which shouldn't be used during ML training.

        If a story starts with an action, we can not use
        that first action as a training example, as there is no
        history. There is one exception though, we do want to
        predict action listen. But because stories never
        contain action listen events (they are added when a
        story gets converted to a dialogue) we need to apply a
        small trick to avoid marking actions occurring after
        an action listen as unpredictable."""

        for step in self.story_graph.story_steps:
            # TODO: this does not work if a step is the conversational start
            #       as well as an intermediary part of a conversation.
            #       This means a checkpoint can either have multiple
            #       checkpoints OR be the start of a conversation
            #       but not both.
            if STORY_START in {s.name for s in step.start_checkpoints}:
                for i, e in enumerate(step.events):
                    if isinstance(e, UserUttered):
                        # if there is a user utterance, that means before the
                        # user uttered something there has to be
                        # an action listen. therefore, any action that comes
                        # after this user utterance isn't the first
                        # action anymore and the tracker used for prediction
                        # is not empty anymore. Hence, it is fine
                        # to predict anything that occurs after an utterance.
                        break
                    if isinstance(e, ActionExecuted):
                        e.unpredictable = True
                        break

    def _issue_unused_checkpoint_notification(self, unused_checkpoints):
        # type: (Set[Text]) -> None
        """Warns about unused story blocks.

        Unused steps are ones having a start checkpoint
        that no one provided)."""

        # running through the steps first will result in only one warning
        # per block (as one block might have multiple steps)
        collected = set()
        for step in self.story_graph.story_steps:
            for start in step.start_checkpoints:
                if start.name in unused_checkpoints:
                    # After processing, there shouldn't be a story part left.
                    # This indicates a start checkpoint that doesn't exist
                    collected.add((start.name, step.block_name))
        for cp, block_name in collected:
            if not cp.startswith(GENERATED_CHECKPOINT_PREFIX):
                logger.warning("Unsatisfied start checkpoint '{}' "
                               "in block '{}'".format(cp, block_name))

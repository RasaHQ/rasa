# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import logging
import random
from collections import defaultdict, namedtuple, deque

import io
import numpy as np
import typing
from numpy import ndarray
from tqdm import tqdm
from typing import Optional, List, Text, Tuple, Set, Dict, Any

from rasa_core import utils
from rasa_core.channels import UserMessage
from rasa_core.events import ActionExecuted, UserUttered, Event, ActionReverted
from rasa_core.trackers import DialogueStateTracker
from rasa_core.training.data import DialogueTrainingData
from rasa_core.training.structures import (
    StoryGraph, STORY_END, STORY_START, StoryStep, GENERATED_CHECKPOINT_PREFIX)

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from rasa_core.domain import Domain
    from rasa_core.featurizers import Featurizer

ExtractorConfig = namedtuple("ExtractorConfig", "remove_duplicates "
                                                "augmentation_factor "
                                                "max_history "
                                                "max_number_of_trackers "
                                                "tracker_limit "
                                                "use_story_concatenation "
                                                "rand")

TrackerResult = namedtuple("TrackerResult", "features "
                                            "labels "
                                            "unique_trackers")


class FeaturizedTracker(object):
    """A tracker wrapper that caches the featurization of the tracker."""

    def __init__(self, tracker, max_history, featurization=None):
        # type: (DialogueStateTracker, int) -> None

        self.tracker = tracker
        self.max_history = max_history

        if featurization is None:
            self.featurization = deque([], max_history + 1)
        else:
            self.featurization = featurization

    def create_copy(self):
        """Creates a deep copy of this featurized tracker. """

        features = deque(self.featurization, self.max_history + 1)
        tracker_copy = copy.deepcopy(self.tracker)
        return FeaturizedTracker(tracker_copy, self.max_history, features)

    def undo_last_action(self):
        # type: () -> None
        """Reverts the last action of the tracker (usually action listen)."""

        self.tracker.update(ActionReverted())
        self.featurization.pop()

    def feauturize_current_state(self, domain):
        # type: (Domain) -> List[Dict[Text, Any]]
        """Featurizes the tracker and caches the featurization."""

        self.featurization.append(domain.get_active_features(self.tracker))
        return list(self.featurization)

    def update(self, event):
        # type: (Event) -> None
        """Logs an event on the tracker"""

        self.tracker.update(event)

    def previously_executed_action(self):
        """Returns the previously logged action."""

        for e in reversed(self.tracker.events):
            if isinstance(e, ActionExecuted):
                return e.action_name
        return None

    @classmethod
    def from_domain(cls, domain, max_history, tracker_limit=None):
        # type: (Domain, int) -> FeaturizedTracker
        """Creates a featurized tracker from a domain."""

        tracker_limit = max_history * 2 if not tracker_limit else tracker_limit
        tracker = DialogueStateTracker(UserMessage.DEFAULT_SENDER_ID,
                                       domain.slots,
                                       domain.topics,
                                       domain.default_topic,
                                       max_event_history=tracker_limit)
        return cls(tracker, max_history)


TrackerLookupDict = Dict[Optional[Text], List[FeaturizedTracker]]


class TrainingsDataGenerator(object):
    def __init__(
            self,
            story_graph,  # type: StoryGraph
            domain,  # type: Domain
            featurizer,  # type: Featurizer
            remove_duplicates=True,  # type: bool
            augmentation_factor=20,  # type: int
            max_history=1,  # type: int
            max_number_of_trackers=2000,  # type: int
            tracker_limit=None,  # type: Optional[int]
            use_story_concatenation=True  # type: bool
    ):
        # type: (...) -> None
        """Given a set of story parts, generates all stories that are possible.

        The different story parts can end and start with checkpoints
        and this generator will match start and end checkpoints to
        connect complete stories. Afterwards, duplicate stories will be
        removed and the data is augmented (if augmentation is enabled)."""

        self.events_metadata = defaultdict(set)
        self.story_graph = story_graph.with_cycles_removed()
        self.domain = domain
        self.featurizer = featurizer
        self.config = ExtractorConfig(
                remove_duplicates=remove_duplicates,
                augmentation_factor=augmentation_factor,
                max_history=max_history,
                max_number_of_trackers=max_number_of_trackers,
                tracker_limit=tracker_limit,
                use_story_concatenation=use_story_concatenation,
                rand=random.Random(42))

    def generate(self):
        # type: () -> DialogueTrainingData

        self._mark_first_action_in_story_steps_as_unpredictable()

        all_features = []  # type: List[ndarray]
        all_actions = []  # type: List[int]
        unused_checkpoints = set()  # type: Set[Text]
        used_checkpoints = set()  # type: Set[Text]

        init_tracker = FeaturizedTracker.from_domain(
                self.domain, self.config.max_history,
                self.config.tracker_limit)
        active_trackers = defaultdict(list)
        active_trackers[STORY_START].append(init_tracker)
        finished_trackers = []

        phases = self._phase_names()

        for i, phase_name in enumerate(phases):
            num_trackers = self._count_trackers(active_trackers)

            logger.debug("Starting {} (phase {} of {})... (using {} trackers)"
                         "".format(phase_name, i + 1, len(phases),
                                   num_trackers))

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
                            incoming_trackers, phase_idx=i)

                    features, labels, trackers = self._process_step(
                            step, incoming_trackers)

                    # collect all the training samples created while
                    # processing the steps events with the trackers
                    all_features.extend(features)
                    all_actions.extend(labels)

                    # update progress bar
                    pbar.set_postfix({
                        "# trackers": len(incoming_trackers),
                        "samples": len(all_actions)})

                    # update our tracker dictionary with the trackers
                    # that handled the events of the step and
                    # that can now be used for further story steps
                    # that start with the checkpoint this step ended with
                    for end in step.end_checkpoints:
                        active_trackers[end.name].extend(trackers)

                    if not step.end_checkpoints:
                        active_trackers[STORY_END].extend(trackers)

            # trackers that reached the end of a story
            completed = [t.tracker for t in active_trackers[STORY_END]]
            finished_trackers.extend(completed)
            active_trackers = self._create_start_trackers(active_trackers)
            logger.debug("Finished phase. ({} training samples found)".format(
                    len(all_actions)))

        unused_checkpoints -= used_checkpoints
        self._issue_unused_checkpoint_notification(unused_checkpoints)
        logger.debug("Found {} action examples.".format(len(all_actions)))

        X = np.array(all_features)
        y = np.array(all_actions)

        metadata = {"events": self.events_metadata,
                    "trackers": finished_trackers}

        if self.config.remove_duplicates:
            X_unique, y_unique = self._deduplicate_training_data(X, y)
            logger.debug("Deduplicated to {} unique action examples.".format(
                    y_unique.shape[0]))
            return DialogueTrainingData(X_unique, y_unique, metadata)
        else:
            return DialogueTrainingData(X, y, metadata)

    def _phase_names(self):
        # type: () -> List[Text]
        """Create names for the different data generation phases"""

        phases = ["normal generation"]
        for i in range(1, self.config.max_history + 1):
            phases.append("augmentation round {})".format(i))
        return phases

    @staticmethod
    def _count_trackers(active_trackers):
        # type: (TrackerLookupDict) -> int
        """Count the number of trackers in the tracker dictionary."""
        return sum(len(ts) for ts in active_trackers.values())

    def _subsample_trackers(self, incoming_trackers, phase_idx):
        # type: (List[FeaturizedTracker], int) -> List[FeaturizedTracker]
        """Subsample the list of trackers to retrieve a random subset."""

        # if flows get very long and have a lot of forks we
        # get into trouble by collecting to many trackers
        # hence the sub sampling
        if self.config.max_number_of_trackers is not None:
            return utils.subsample_array(incoming_trackers,
                                         self.config.max_number_of_trackers,
                                         self.config.rand)
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
                        self.config.rand)

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
                    t.undo_last_action()
                next_active_trackers[start].append(t)
        return next_active_trackers

    def _process_step(self, step, incoming_trackers):
        # type: (StoryStep, List[FeaturizedTracker]) -> TrackerResult
        """Processes a steps events with all trackers.

        The trackers that reached the steps starting checkpoint will
        be used to process the events. Collects and returns training
        data while processing the story step."""

        events = step.explicit_events(self.domain)
        # need to copy the tracker as multiple story steps
        # might start with the same checkpoint and all of them
        # will use the same set of incoming trackers
        trackers = [tracker.create_copy() for tracker in
                    incoming_trackers] if events else []  # small optimization

        training_features = []
        training_labels = []

        for event in events:
            result = self._process_event_with_trackers(
                    event, trackers)
            training_features.extend(result.features)
            training_labels.extend(result.labels)
            trackers = result.unique_trackers
        return TrackerResult(training_features, training_labels, trackers)

    def _process_event_with_trackers(self, event, trackers):
        # type: (Event, List[FeaturizedTracker]) -> TrackerResult
        """Logs an event to all trackers.

        Removes trackers that create equal featurizations.

        From multiple trackers that create equal featurizations
        we only need to keep one. Because as we continue processing
        events and story steps, all trackers that created the
        same featurization once will do so in the future (as we
        feed the same events to all trackers)."""

        # collected trackers that created different featurizations
        unique_trackers = []
        featurizations = set()

        # collected training data
        features = []
        labels = []

        for tracker in trackers:
            if isinstance(event, ActionExecuted):
                state_features = tracker.feauturize_current_state(self.domain)
                feature_vector = self.domain.slice_feature_history(
                        self.featurizer, state_features,
                        self.config.max_history)
                hashed = utils.HashableNDArray(feature_vector)

                # only continue with trackers that created a
                # featurization we haven't observed at this event
                if (hashed not in featurizations
                        or not self.config.remove_duplicates):
                    featurizations.add(hashed)
                    if not event.unpredictable:
                        # only actions which can be predicted at a stories start
                        a_idx = self.domain.index_for_action(event.action_name)

                        features.append(feature_vector)
                        labels.append(a_idx)
                    unique_trackers.append(tracker)
            else:
                unique_trackers.append(tracker)
            tracker.update(event)
            if not isinstance(event, ActionExecuted):
                action_name = tracker.previously_executed_action()
                self.events_metadata[action_name].add(event)

        return TrackerResult(features, labels, unique_trackers)

    @staticmethod
    def _deduplicate_training_data(X, y):
        # type: (ndarray, ndarray) -> Tuple[ndarray, ndarray]
        """Make sure every training example in X occurs exactly once."""

        # we need to concat X and y to make sure that
        # we do NOT throw out contradicting examples
        # (same featurization but different labels).
        # appends y to X so it appears to be just another feature
        if not utils.is_training_data_empty(X):
            casted_y = np.broadcast_to(
                    np.reshape(y, (y.shape[0], 1, 1)),
                    (y.shape[0], X.shape[1], 1))
            concatenated = np.concatenate((X, casted_y), axis=2)
            t_data = np.unique(concatenated, axis=0)
            X_unique = t_data[:, :, :-1]
            y_unique = np.array(t_data[:, 0, -1], dtype=casted_y.dtype)
            return X_unique, y_unique
        else:
            return X, y

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
                logger.warn("Unsatisfied start checkpoint '{}' "
                            "in block '{}'".format(cp, block_name))

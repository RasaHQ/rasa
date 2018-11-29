# -*- coding: utf-8 -*-
from collections import defaultdict, namedtuple, deque

import copy
import logging
import random
from tqdm import tqdm
from typing import Optional, List, Text, Set, Dict, Tuple

from rasa_core import utils
from rasa_core.domain import Domain
from rasa_core.events import (
    ActionExecuted, UserUttered,
    ActionReverted, UserUtteranceReverted, Restarted, Event)
from rasa_core.trackers import DialogueStateTracker
from rasa_core.training.structures import (
    StoryGraph, STORY_START, StoryStep,
    GENERATED_CHECKPOINT_PREFIX)

logger = logging.getLogger(__name__)

ExtractorConfig = namedtuple("ExtractorConfig", "remove_duplicates "
                                                "unique_last_num_states "
                                                "augmentation_factor "
                                                "max_number_of_trackers "
                                                "tracker_limit "
                                                "use_story_concatenation "
                                                "rand")


class TrackerWithCachedStates(DialogueStateTracker):
    """A tracker wrapper that caches the state creation of the tracker."""

    def __init__(self, sender_id, slots,
                 max_event_history=None,
                 domain=None
                 ):
        super(TrackerWithCachedStates, self).__init__(
            sender_id, slots, max_event_history)
        self._states = None
        self.domain = domain

    def past_states(self, domain: Domain) -> deque:
        """Return the states of the tracker based on the logged events."""

        # we need to make sure this is the same domain, otherwise things will
        # go south. but really, the same tracker shouldn't be used across
        # domains
        assert domain == self.domain

        # if don't have it cached, we use the domain to calculate the states
        # from the events
        if self._states is None:
            self._states = super(TrackerWithCachedStates, self).past_states(
                domain)

        return self._states

    def clear_states(self) -> None:
        """Reset the states."""
        self._states = None

    def init_copy(self)-> 'TrackerWithCachedStates':
        """Create a new state tracker with the same initial values."""
        return type(self)("",
                          self.slots.values(),
                          self._max_event_history,
                          self.domain)

    def copy(self, sender_id: Text = "") -> 'TrackerWithCachedStates':
        """Creates a duplicate of this tracker.

        A new tracker will be created and all events
        will be replayed."""

        # This is an optimization, we could use the original copy, but
        # the states would be lost and we would need to recalculate them

        tracker = self.init_copy()
        tracker.sender_id = sender_id

        for event in self.events:
            tracker.update(event, skip_states=True)

        tracker._states = copy.copy(self._states)

        return tracker  # yields the final state

    def _append_current_state(self) -> None:
        if self._states is None:
            self._states = self.past_states(self.domain)
        else:
            state = self.domain.get_active_states(self)
            self._states.append(frozenset(state.items()))

    def update(self, event: Event, skip_states: bool = False) -> None:
        """Modify the state of the tracker according to an ``Event``. """

        # if `skip_states` is `True`, this function behaves exactly like the
        # normal update of the `DialogueStateTracker`

        if self._states is None and not skip_states:
            # rest of this function assumes we have the previous state
            # cached. let's make sure it is there.
            self._states = self.past_states(self.domain)

        super(TrackerWithCachedStates, self).update(event)

        if not skip_states:
            if isinstance(event, ActionExecuted):
                pass
            elif isinstance(event, ActionReverted):
                self._states.pop()  # removes the state after the action
                self._states.pop()  # removes the state used for the action
            elif isinstance(event, UserUtteranceReverted):
                self.clear_states()
            elif isinstance(event, Restarted):
                self.clear_states()
            else:
                self._states.pop()

            self._append_current_state()


# define types
TrackerLookupDict = Dict[Optional[Text],
                         List[TrackerWithCachedStates]]

TrackersTuple = Tuple[List[TrackerWithCachedStates],
                      List[TrackerWithCachedStates]]


class TrainingDataGenerator(object):
    def __init__(
        self,
        story_graph: StoryGraph,
        domain: Domain,
        remove_duplicates: bool = True,
        unique_last_num_states: Optional[int] = None,
        augmentation_factor: int = 20,
        tracker_limit: Optional[int] = None,
        use_story_concatenation: bool = True,
        debug_plots: bool = False
    ):
        """Given a set of story parts, generates all stories that are possible.

        The different story parts can end and start with checkpoints
        and this generator will match start and end checkpoints to
        connect complete stories. Afterwards, duplicate stories will be
        removed and the data is augmented (if augmentation is enabled)."""

        self.story_graph = story_graph.with_cycles_removed()
        if debug_plots:
            self.story_graph.visualize('story_blocks_connections.html')

        self.domain = domain

        # 10x factor is a heuristic for augmentation rounds
        max_number_of_trackers = augmentation_factor * 10

        self.config = ExtractorConfig(
            remove_duplicates=remove_duplicates,
            unique_last_num_states=unique_last_num_states,
            augmentation_factor=augmentation_factor,
            max_number_of_trackers=max_number_of_trackers,
            tracker_limit=tracker_limit,
            use_story_concatenation=use_story_concatenation,
            rand=random.Random(42)
        )
        # hashed featurization of all finished trackers
        self.hashed_featurizations = set()

    @staticmethod
    def _phase_name(everything_reachable_is_reached, phase):
        if everything_reachable_is_reached:
            return "augmentation round {}".format(phase)
        else:
            return "data generation round {}".format(phase)

    def generate(self, silent: bool = False) -> List[TrackerWithCachedStates]:
        if (self.config.remove_duplicates and
                self.config.unique_last_num_states):
            logger.debug("Generated trackers will be deduplicated "
                         "based on their unique last {} states."
                         "".format(self.config.unique_last_num_states))

        self._mark_first_action_in_story_steps_as_unpredictable()

        active_trackers = defaultdict(list)

        init_tracker = TrackerWithCachedStates(
            "",
            self.domain.slots,
            max_event_history=self.config.tracker_limit,
            domain=self.domain
        )
        active_trackers[STORY_START].append(init_tracker)

        # trackers that are sent to a featurizer
        finished_trackers = []
        # keep story end trackers separately for augmentation
        story_end_trackers = []

        phase = 0  # one phase is one traversal of all story steps.
        min_num_aug_phases = 3 if self.config.augmentation_factor > 0 else 0
        logger.debug("Number of augmentation rounds is {}"
                     "".format(min_num_aug_phases))

        # placeholder to track gluing process of checkpoints
        used_checkpoints = set()
        previous_unused = set()
        everything_reachable_is_reached = False

        # we will continue generating data until we have reached all
        # checkpoints that seem to be reachable. This is a heuristic,
        # if we did not reach any new checkpoints in an iteration, we
        # assume we have reached all and stop.
        while (not everything_reachable_is_reached or
               phase < min_num_aug_phases):
            phase_name = self._phase_name(everything_reachable_is_reached,
                                          phase)

            num_active_trackers = self._count_trackers(active_trackers)

            if num_active_trackers:
                logger.debug("Starting {} ... (with {} trackers)"
                             "".format(phase_name, num_active_trackers))
            else:
                logger.debug("There are no trackers for {}"
                             "".format(phase_name))
                break

            # track unused checkpoints for this phase
            unused_checkpoints = set()  # type: Set[Text]

            pbar = tqdm(self.story_graph.ordered_steps(),
                        desc="Processed Story Blocks",
                        disable=silent)
            for step in pbar:
                incoming_trackers = []  # type: List[TrackerWithCachedStates]
                for start in step.start_checkpoints:
                    if active_trackers[start.name]:
                        ts = start.filter_trackers(active_trackers[start.name])
                        incoming_trackers.extend(ts)
                        used_checkpoints.add(start.name)
                    elif start.name not in used_checkpoints:
                        # need to skip - there was no previous step that
                        # had this start checkpoint as an end checkpoint
                        # it will be processed in next phases
                        unused_checkpoints.add(start.name)

                if not incoming_trackers:
                    # if there are no trackers,
                    # we can skip the rest of the loop
                    continue

                # these are the trackers that reached this story
                # step and that need to handle all events of the step

                if self.config.remove_duplicates:
                    incoming_trackers, end_trackers = \
                        self._remove_duplicate_trackers(incoming_trackers)
                    # append end trackers to finished trackers
                    finished_trackers.extend(end_trackers)

                if everything_reachable_is_reached:
                    # augmentation round
                    incoming_trackers = self._subsample_trackers(
                        incoming_trackers)

                # update progress bar
                pbar.set_postfix({"# trackers": "{:d}".format(
                    len(incoming_trackers))})

                trackers, end_trackers = self._process_step(step,
                                                            incoming_trackers)
                # add end trackers to finished trackers
                finished_trackers.extend(end_trackers)

                # update our tracker dictionary with the trackers
                # that handled the events of the step and
                # that can now be used for further story steps
                # that start with the checkpoint this step ended with

                for end in step.end_checkpoints:

                    start_name = self._find_start_checkpoint_name(end.name)

                    active_trackers[start_name].extend(trackers)

                    if start_name in used_checkpoints:
                        # add end checkpoint as unused
                        # if this checkpoint was processed as
                        # start one before
                        unused_checkpoints.add(start_name)

                if not step.end_checkpoints:
                    unique_ends = self._remove_duplicate_story_end_trackers(
                        trackers)
                    story_end_trackers.extend(unique_ends)

            num_finished = len(finished_trackers) + len(story_end_trackers)
            logger.debug("Finished phase ({} training samples found)."
                         "".format(num_finished))

            # prepare next round
            phase += 1

            if not everything_reachable_is_reached:
                # check if we reached all nodes that can be reached
                # if we reached at least one more node this round
                # than last one, we assume there is still
                # something left to reach and we continue

                unused_checkpoints = self._add_unused_end_checkpoints(
                    set(active_trackers.keys()),
                    unused_checkpoints,
                    used_checkpoints
                )
                active_trackers = self._filter_active_trackers(
                    active_trackers,
                    unused_checkpoints
                )
                num_active_trackers = self._count_trackers(active_trackers)

                everything_reachable_is_reached = (
                    unused_checkpoints == previous_unused or
                    num_active_trackers == 0)
                previous_unused = unused_checkpoints

                if everything_reachable_is_reached:
                    # should happen only once

                    previous_unused -= used_checkpoints
                    # add trackers with unused checkpoints
                    # to finished trackers
                    for start_name in previous_unused:
                        finished_trackers.extend(active_trackers[start_name])

                    logger.debug("Data generation rounds finished.")
                    logger.debug("Found {} unused checkpoints"
                                 "".format(len(previous_unused)))
                    phase = 0
                else:
                    logger.debug("Found {} unused checkpoints "
                                 "in current phase."
                                 "".format(len(unused_checkpoints)))
                    logger.debug("Found {} active trackers "
                                 "for these checkpoints."
                                 "".format(num_active_trackers))

            if everything_reachable_is_reached:
                # augmentation round, so we process only
                # story end checkpoints
                # reset used checkpoints
                used_checkpoints = set()  # type: Set[Text]

                # generate active trackers for augmentation
                active_trackers = \
                    self._create_start_trackers_for_augmentation(
                        story_end_trackers)

        finished_trackers.extend(story_end_trackers)
        self._issue_unused_checkpoint_notification(previous_unused)
        logger.debug("Found {} training trackers."
                     "".format(len(finished_trackers)))

        return finished_trackers

    @staticmethod
    def _count_trackers(active_trackers: TrackerLookupDict) -> int:
        """Count the number of trackers in the tracker dictionary."""
        return sum(len(ts) for ts in active_trackers.values())

    def _subsample_trackers(
        self,
        incoming_trackers: List[TrackerWithCachedStates]
    ) -> List[TrackerWithCachedStates]:
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

    def _find_start_checkpoint_name(self, end_name: Text) -> Text:
        """Find start checkpoint name given end checkpoint name of a cycle"""
        return self.story_graph.story_end_checkpoints.get(end_name, end_name)

    @staticmethod
    def _add_unused_end_checkpoints(start_checkpoints: Set[Text],
                                    unused_checkpoints: Set[Text],
                                    used_checkpoints: Set[Text]
                                    ) -> Set[Text]:
        """Add unused end checkpoints
            if they were never encountered as start checkpoints
        """

        return unused_checkpoints.union(
            set(start_name
                for start_name in start_checkpoints
                if start_name not in used_checkpoints)
        )

    @staticmethod
    def _filter_active_trackers(active_trackers: TrackerLookupDict,
                                unused_checkpoints: Set[Text]
                                ) -> TrackerLookupDict:
        """Filter active trackers that ended with unused checkpoint
            or are parts of loops."""
        next_active_trackers = defaultdict(list)

        for start_name in unused_checkpoints:
            # process trackers ended with unused checkpoints further
            if start_name != STORY_START:
                # there is no point to process STORY_START checkpoint again
                next_active_trackers[start_name] = \
                    active_trackers.get(start_name, [])

        return next_active_trackers

    def _create_start_trackers_for_augmentation(
        self,
        story_end_trackers: List[TrackerWithCachedStates]
    ) -> TrackerLookupDict:
        """This is where the augmentation magic happens.

            We will reuse all the trackers that reached the
            end checkpoint `None` (which is the end of a
            story) and start processing all steps again. So instead
            of starting with a fresh tracker, the second and
            all following phases will reuse a couple of the trackers
            that made their way to a story end.

            We need to do some cleanup before processing them again.
        """
        next_active_trackers = defaultdict(list)

        if self.config.use_story_concatenation:
            ending_trackers = utils.subsample_array(
                story_end_trackers,
                self.config.augmentation_factor,
                rand=self.config.rand
            )
            for t in ending_trackers:
                # this is a nasty thing - all stories end and
                # start with action listen - so after logging the first
                # actions in the next phase the trackers would
                # contain action listen followed by action listen.
                # to fix this we are going to "undo" the last action listen

                # tracker should be copied,
                # otherwise original tracker is updated
                aug_t = t.copy()
                aug_t.update(ActionReverted())
                next_active_trackers[STORY_START].append(aug_t)

        return next_active_trackers

    def _process_step(
        self,
        step: StoryStep,
        incoming_trackers: List[TrackerWithCachedStates]
    ) -> TrackersTuple:
        """Processes a steps events with all trackers.

        The trackers that reached the steps starting checkpoint will
        be used to process the events. Collects and returns training
        data while processing the story step."""

        events = step.explicit_events(self.domain)

        trackers = []
        if events:  # small optimization

            # need to copy the tracker as multiple story steps
            # might start with the same checkpoint and all of them
            # will use the same set of incoming trackers

            for tracker in incoming_trackers:
                # sender id is used to be able for a human to see where the
                # messages and events for this tracker came from - to do this
                # we concatenate the story block names of the blocks that
                # contribute to the trackers events
                if tracker.sender_id:
                    if step.block_name not in tracker.sender_id.split(" > "):
                        new_sender = (tracker.sender_id +
                                      " > " + step.block_name)
                    else:
                        new_sender = tracker.sender_id
                else:
                    new_sender = step.block_name
                trackers.append(tracker.copy(new_sender))

        end_trackers = []
        for event in events:
            for tracker in trackers:
                if isinstance(event, (ActionReverted,
                                      UserUtteranceReverted,
                                      Restarted)):
                    end_trackers.append(tracker.copy(tracker.sender_id))
                tracker.update(event)

        # end trackers should be returned separately
        # to avoid using them for augmentation
        return trackers, end_trackers

    def _remove_duplicate_trackers(self,
                                   trackers: List[TrackerWithCachedStates]
                                   ) -> TrackersTuple:
        """Removes trackers that create equal featurizations
            for current story step.

        From multiple trackers that create equal featurizations
        we only need to keep one. Because as we continue processing
        events and story steps, all trackers that created the
        same featurization once will do so in the future (as we
        feed the same events to all trackers)."""

        step_hashed_featurizations = set()

        # collected trackers that created different featurizations
        unique_trackers = []  # for current step
        end_trackers = []  # for all steps

        for tracker in trackers:
            states = tuple(tracker.past_states(self.domain))
            hashed = hash(states)

            # only continue with trackers that created a
            # hashed_featurization we haven't observed
            if hashed not in step_hashed_featurizations:
                if self.config.unique_last_num_states:
                    last_states = states[-self.config.unique_last_num_states:]
                    last_hashed = hash(last_states)

                    if last_hashed not in step_hashed_featurizations:
                        step_hashed_featurizations.add(last_hashed)
                        unique_trackers.append(tracker)
                    elif (len(states) > len(last_states) and
                          hashed not in self.hashed_featurizations):
                        self.hashed_featurizations.add(hashed)
                        end_trackers.append(tracker)
                else:
                    unique_trackers.append(tracker)

                step_hashed_featurizations.add(hashed)

        return unique_trackers, end_trackers

    def _remove_duplicate_story_end_trackers(
        self,
        trackers: List[TrackerWithCachedStates]
    ) -> List[TrackerWithCachedStates]:
        """Removes trackers that reached story end and
            created equal featurizations."""

        # collected trackers that created different featurizations
        unique_trackers = []  # for all steps

        # deduplication of finished trackers is needed,
        # otherwise featurization does a lot of unnecessary work

        for tracker in trackers:
            states = tuple(tracker.past_states(self.domain))
            hashed = hash(states)

            # only continue with trackers that created a
            # hashed_featurization we haven't observed

            if hashed not in self.hashed_featurizations:
                self.hashed_featurizations.add(hashed)
                unique_trackers.append(tracker)

        return unique_trackers

    def _mark_first_action_in_story_steps_as_unpredictable(self) -> None:
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

    def _issue_unused_checkpoint_notification(
        self,
        unused_checkpoints: Set[Text]
    ) -> None:
        """Warns about unused story blocks.

        Unused steps are ones having a start or end checkpoint
        that no one provided."""

        if STORY_START in unused_checkpoints:
            logger.warning("There is no starting story block "
                           "in the training data. "
                           "All your story blocks start with some checkpoint. "
                           "There should be at least one story block "
                           "that starts without any checkpoint.")

        # running through the steps first will result in only one warning
        # per block (as one block might have multiple steps)
        collected_start = set()
        collected_end = set()
        for step in self.story_graph.story_steps:
            for start in step.start_checkpoints:
                if start.name in unused_checkpoints:
                    # After processing, there shouldn't be a story part left.
                    # This indicates a start checkpoint that doesn't exist
                    collected_start.add((start.name, step.block_name))

            for end in step.end_checkpoints:
                if end.name in unused_checkpoints:
                    # After processing, there shouldn't be a story part left.
                    # This indicates an end checkpoint that doesn't exist
                    collected_end.add((end.name, step.block_name))

        for cp, block_name in collected_start:
            if not cp.startswith(GENERATED_CHECKPOINT_PREFIX):
                logger.warning("Unsatisfied start checkpoint '{}' "
                               "in block '{}'. "
                               "Remove this checkpoint or add "
                               "story blocks that end "
                               "with this checkpoint.".format(cp, block_name))

        for cp, block_name in collected_end:
            if not cp.startswith(GENERATED_CHECKPOINT_PREFIX):
                logger.warning("Unsatisfied end checkpoint '{}' "
                               "in block '{}'. "
                               "Remove this checkpoint or add "
                               "story blocks that start "
                               "with this checkpoint.".format(cp, block_name))

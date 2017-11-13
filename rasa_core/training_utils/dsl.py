# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import io
import json
import logging
import os
import random
import re
import uuid
from collections import deque
from random import Random

import numpy as np
from numpy import ndarray
from tqdm import tqdm
from typing import Optional, List, Text, Tuple, Set, Any, Dict

from rasa_core.actions.action import ActionListen, ACTION_LISTEN_NAME
from rasa_core.channels import UserMessage
from rasa_core.conversation import Dialogue
from rasa_core.domain import Domain
from rasa_core.events import ActionExecuted, UserUttered, Event, \
    ActionReverted
from rasa_core.featurizers import Featurizer
from rasa_core.interpreter import RegexInterpreter, NaturalLanguageInterpreter
from rasa_core.trackers import DialogueStateTracker
from rasa_core.training_utils.story_graph import StoryGraph
from rasa_core import utils

logger = logging.getLogger(__name__)

# Checkpoint used to identify story starting blocks
STORY_START = "STORY_START"


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


class StoryParseError(Exception):
    """Raised if there is an error while parsing the story file."""

    def __init__(self, message):
        self.message = message


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
                result += "    - {}\n".format(s.as_story_string())
            else:
                raise Exception("Unexpected element in story step: " + s)

        if not flat:
            if self.end_checkpoint is not None:
                cp = self.end_checkpoint.as_story_string()
                result += "> {}\n".format(cp)
        return result

    def explicit_events(self, domain, should_append_final_listen=True):
        # type: (Domain, NaturalLanguageInterpreter) -> List[Event]
        """Returns events contained in the story step including implicit events.

        Not all events are always listed in the story dsl. This
        includes listen actions as well as implicitly
        set slots. This functions makes these events explicit and
        returns them with the rest of the steps events."""

        events = []

        for e in self.events:
            if isinstance(e, UserUttered):
                events.append(ActionExecuted(ActionListen().name()))
                events.append(e)
                events.extend(domain.slots_for_entities(e.entities))
            else:
                events.append(e)

        if self.end_checkpoint is None and should_append_final_listen:
            events.append(ActionExecuted(ActionListen().name()))
        return events


class StoryStepBuilder(object):
    def __init__(self, name):
        self.name = name
        self.story_steps = []
        self.current_steps = []
        self.start_checkpoints = []

    def add_checkpoint(self, name, conditions):
        # Depending on the state of the story part this
        # is either a start or an end check point
        if not self.current_steps:
            self.start_checkpoints.append(Checkpoint(name, conditions))
        else:
            if conditions:
                logger.warn("End or intermediate checkpoints "
                            "do not support conditions! "
                            "(checkpoint: {})".format(name))
            additional_steps = []
            for t in self.current_steps:
                if t.end_checkpoint is not None:
                    tcp = t.create_copy(use_new_id=True)
                    tcp.end_checkpoint = Checkpoint(name)
                    additional_steps.append(tcp)
                else:
                    t.end_checkpoint = Checkpoint(name)
            self.current_steps.extend(additional_steps)

    def _prev_end_checkpoints(self):
        if not self.current_steps:
            return self.start_checkpoints
        else:
            end_names = {s.end_checkpoint_name() for s in self.current_steps}
            return [Checkpoint(name) for name in end_names]

    def add_user_messages(self, messages):
        self.ensure_current_steps()

        if len(messages) == 1:
            # If there is only one possible intent, we'll keep things simple
            for t in self.current_steps:
                t.add_user_message(messages[0])
        else:
            # If there are multiple different intents the
            # user can use the express the same thing
            # we need to copy the blocks and create one
            # copy for each possible message
            generated_checkpoint = "GENERATED_M_{}".format(uuid.uuid4().hex)
            updated_steps = []
            for t in self.current_steps:
                for m in messages:
                    copied = t.create_copy(use_new_id=True)
                    copied.add_user_message(m)
                    copied.end_checkpoint = Checkpoint(generated_checkpoint)
                    updated_steps.append(copied)
            self.current_steps = updated_steps

    def add_event(self, event):
        self.ensure_current_steps()
        for t in self.current_steps:
            t.add_event(event)

    def ensure_current_steps(self):
        completed = [step
                     for step in self.current_steps
                     if step.end_checkpoint is not None]
        unfinished = [step
                      for step in self.current_steps
                      if step.end_checkpoint is None]
        self.story_steps.extend(completed)
        if unfinished:
            self.current_steps = unfinished
        else:
            self.current_steps = self._next_story_steps()

    def flush(self):
        if self.current_steps:
            self.story_steps.extend(self.current_steps)
            self.current_steps = []

    def _next_story_steps(self):
        start_checkpoints = self._prev_end_checkpoints() or [Checkpoint(STORY_START)]
        current_turns = [StoryStep(block_name=self.name, start_checkpoint=s)
                         for s in start_checkpoints]
        return current_turns


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

        events.append(ActionExecuted(ActionListen().name()))
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

    def dump_to_file(self, file_name, flat=False):
        with io.open(file_name, "a") as f:
            f.write(self.as_story_string(flat))


class StoryFileReader(object):
    """Helper class to read a story file."""

    def __init__(self, domain, interpreter, template_vars=None):
        self.story_steps = []
        self.current_step_builder = None  # type: Optional[StoryStepBuilder]
        self.domain = domain
        self.interpreter = interpreter
        self.template_variables = template_vars if template_vars else {}

    @staticmethod
    def read_from_file(file_name, domain, interpreter=RegexInterpreter(),
                       template_variables=None):
        """Given a json file reads the contained stories."""

        try:
            with io.open(file_name, "r") as f:
                lines = f.readlines()
            reader = StoryFileReader(domain, interpreter, template_variables)
            return reader.process_lines(lines)
        except Exception as e:
            raise Exception("Failed to parse '{}'. {}".format(
                    os.path.abspath(file_name), e))

    @staticmethod
    def _parse_event_line(line, parameter_default_value=""):
        """Tries to parse a single line as an event with arguments."""

        # the regex matches "slot{"a": 1}" as well as "slot["a"]"
        m = re.search('^([^\[{]+)([\[{].+)?', line)
        if m is not None:
            event_name = m.group(1).strip()
            slots_str = m.group(2)
            parameters = {}
            if slots_str is not None and slots_str.strip():
                parsed_slots = json.loads(slots_str)
                if isinstance(parsed_slots, list):
                    for slot in parsed_slots:
                        parameters[slot] = parameter_default_value
                elif isinstance(parsed_slots, dict):
                    parameters = parsed_slots
                else:
                    raise Exception(
                            "Invalid slot string in line '{}'.".format(line))
            return event_name, parameters
        else:
            logger.debug("Failed to parse action line '{}'. ".format(line))
            return "", {}

    def process_lines(self, lines):
        # type: (List[Text]) -> List[StoryStep]

        for idx, line in enumerate(lines):
            line_num = idx + 1
            try:
                line = self._replace_template_variables(
                        self._clean_up_line(line))
                if line.strip() == "":
                    continue
                elif line.startswith("#"):  # reached a new story block
                    name = line[1:].strip("# ")
                    self.new_story_part(name)
                elif line.startswith(">"):  # reached a checkpoint
                    name, conditions = self._parse_event_line(line[1:].strip())
                    self.add_checkpoint(name, conditions)
                elif line.startswith(
                        "-"):  # reached a slot, event, or executed action
                    event_name, parameters = self._parse_event_line(line[1:])
                    self.add_event(event_name, parameters)
                elif line.startswith("*"):  # reached a user message
                    user_messages = [el.strip() for el in
                                     line[1:].split(" OR ")]
                    self.add_user_messages(user_messages)
                else:  # reached an unknown type of line
                    logger.warn("Skipping line {}. No valid command found. "
                                "Line Content: '{}'".format(line_num, line))
            except Exception as e:
                msg = "Error in line {}: {}".format(line_num, e.message)
                logger.error(msg, exc_info=1)
                raise Exception(msg)
        self._add_current_stories_to_result()
        return self.story_steps

    def _replace_template_variables(self, line):
        def process_match(matchobject):
            varname = matchobject.group(1)
            if varname in self.template_variables:
                return self.template_variables[varname]
            else:
                raise ValueError("Unknown variable `{var}` "
                                 "in template line '{line}'".format(var=varname,
                                                                    line=line))

        template_rx = re.compile(r"`([^`]+)`")
        return template_rx.sub(process_match, line)

    @staticmethod
    def _clean_up_line(line):
        # type: (Text) -> Text
        """Removes comments and trailing spaces"""

        return re.sub(r'<!--.*?-->', '', line).strip()

    def _add_current_stories_to_result(self):
        if self.current_step_builder:
            self.current_step_builder.flush()
            self.story_steps.extend(self.current_step_builder.story_steps)

    def new_story_part(self, name):
        self._add_current_stories_to_result()
        self.current_step_builder = StoryStepBuilder(name)

    def add_checkpoint(self, name, conditions):
        # type: (Text) -> None

        # Ensure story part already has a name
        if not self.current_step_builder:
            raise StoryParseError("Checkpoint '{}' is at an invalid location. "
                                  "Expected a story start.".format(name))

        self.current_step_builder.add_checkpoint(name, conditions)

    def add_user_messages(self, messages):
        if not self.current_step_builder:
            raise StoryParseError("User message '{}' at invalid location. "
                                  "Expected story start.".format(messages))
        parsed_messages = []
        for m in messages:
            parse_data = self.interpreter.parse(m)
            utterance = UserUttered.from_parse_data(m, parse_data)
            if utterance.intent.get("name") not in self.domain.intents:
                logger.warn("Found unknown intent '{}'. Please, make sure "
                            "that all intents are listed in your domain "
                            "yaml.".format(utterance.intent.get("name")))
            parsed_messages.append(utterance)
        self.current_step_builder.add_user_messages(parsed_messages)

    def add_event(self, event_name, parameters):
        parsed = Event.from_story_string(event_name, parameters, self.domain,
                                         default=ActionExecuted)
        if parsed is None:
            raise StoryParseError("Unknown event '{}'. It is Neither an event "
                                  "nor an action).".format(event_name))
        self.current_step_builder.add_event(parsed)


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

    @classmethod
    def from_domain(cls, domain, max_history):
        # type: (Domain, int) -> FeaturizedTracker
        """Creates a featurized tracker from a domain."""

        tracker = DialogueStateTracker(UserMessage.DEFAULT_SENDER_ID,
                                       domain.slots,
                                       domain.topics,
                                       domain.default_topic,
                                       max_event_history=max_history * 2)
        return cls(tracker, max_history)


TrackerLookupDict = Dict[Optional[Text], List[FeaturizedTracker]]


class TrainingsDataExtractor(object):
    def __init__(self,
                 story_graph,  # type: StoryGraph
                 domain,  # type: Domain
                 featurizer  # type: Featurizer
                 ):
        # type: (...) -> None

        self.story_graph = story_graph
        self.domain = domain
        self.featurizer = featurizer

    def extract_trainings_data(self,
                               remove_duplicates=True,
                               augmentation_factor=20,
                               max_history=1,
                               max_number_of_trackers=2000):
        # type: (bool, int, int) -> Tuple[ndarray, ndarray]
        """Given a set of story parts, generates all stories that are possible.

        The different story parts can end and start with checkpoints
        and this generator will match start and end checkpoints to
        connect complete stories. Afterwards, duplicate stories will be
        removed and the data is augmented (if augmentation is enabled)."""

        self._mark_first_action_in_story_steps_as_unpredictable()
        rand = random.Random(42)

        all_features = []  # type: List[ndarray]
        all_actions = []  # type: List[int]
        unused_checkpoints = set()  # type: Set[Text]
        init_tracker = FeaturizedTracker.from_domain(self.domain, max_history)
        active_trackers = {STORY_START: [init_tracker]}

        phases = (["normal generation"] +
                  ["augmentation round {})".format(i)
                   for i in range(1, max_history + 1)])

        for i, phase in enumerate(phases):
            num_trackers = len(active_trackers[STORY_START])
            logger.debug("Starting {} (phase {} of {})... "
                         "(using {} trackers)".format(phase,
                                                      i + 1,
                                                      len(phases),
                                                      num_trackers))

            pbar = tqdm(self.story_graph.ordered_steps(),
                        desc="Processed Story Blocks")
            for step in pbar:
                if step.start_checkpoint:
                    start = step.start_checkpoint
                else:
                    start = Checkpoint(None)
                if start.name not in active_trackers:
                    # need to skip - there was no previous step that
                    # had this start checkpoint as an end checkpoint
                    unused_checkpoints.add(start.name)
                else:
                    # these are the trackers that reached this story
                    # step and that need to handle all events of the step
                    incoming_trackers = active_trackers[start.name]
                    incoming_trackers = start.filter_trackers(incoming_trackers)

                    incoming_trackers = self._subsample_trackers(
                            incoming_trackers,
                            max_number_of_trackers,
                            augmentation_factor, phase_idx=i,
                            rand=rand)

                    features, labels, trackers = self._process_step(
                            step,
                            incoming_trackers,
                            max_history)

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
                    if step.end_checkpoint_name() not in active_trackers:
                        active_trackers[step.end_checkpoint_name()] = []
                    active_trackers[step.end_checkpoint_name()].extend(trackers)

            logger.debug("Finished phase. ({} training samples found)".format(
                    len(all_actions)))
            active_trackers = self._prepare_next_phase(active_trackers,
                                                       augmentation_factor,
                                                       rand)

        self._issue_unused_checkpoint_notification(unused_checkpoints)
        logger.debug("Found {} action examples.".format(len(all_actions)))

        X = np.array(all_features)
        y = np.array(all_actions)

        if remove_duplicates:
            X_unique, y_unique = self._deduplicate_training_data(X, y)
            logger.debug("Deduplicated to {} unique action examples.".format(
                    y_unique.shape[0]))
            return X_unique, y_unique
        else:
            return X, y

    @staticmethod
    def _subsample_trackers(incoming_trackers, max_number_of_trackers,
                            augmentation_factor,
                            phase_idx, rand):
        # if flows get very long and have a lot of forks we
        # get into trouble by collecting to many trackers
        # hence the sub sampling
        if phase_idx == 0:
            if max_number_of_trackers is not None:
                return utils.subsample_array(incoming_trackers,
                                             max_number_of_trackers, rand)
            else:
                return incoming_trackers
        else:
            # after the first phase we always sample max
            # `augmentation_factor` samples
            return utils.subsample_array(incoming_trackers,
                                         augmentation_factor, rand)

    @staticmethod
    def _prepare_next_phase(active_trackers,  # type: TrackerLookupDict
                            augmentation_factor,  # type: int
                            rand  # type: Random
                            ):
        # type: (...) -> Dict[Optional[Text], List[FeaturizedTracker]]
        """One phase is one traversal of all story steps.

        We need to do some cleanup before processing them again."""

        ending_trackers = active_trackers.get(None, [])
        subsampled_trackers = utils.subsample_array(ending_trackers,
                                                    augmentation_factor, rand)
        active_trackers = {STORY_START: []}

        # This is where the augmentation magic happens. We
        # will reuse all the trackers that reached the
        # end checkpoint `None` (which is the end of a
        # story) and start processing all steps again. So instead
        # of starting with a fresh tracker, the second and
        # all following phases will reuse a couple of the trackers
        # that made their way to a story end.
        for t in subsampled_trackers:
            # this is a nasty thing - all stories end and
            # start with action listen - so after logging the first
            # actions in the next phase the trackers would
            # contain action listen followed by action listen.
            # to fix this we are going to "undo" the last action listen
            t.undo_last_action()
            active_trackers[STORY_START].append(t)
        return active_trackers

    def _process_step(self,
                      step,  # type: StoryStep
                      incoming_trackers,  # type: List[FeaturizedTracker]
                      max_history  # type: int
                      ):
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
            features, labels, trackers = self._process_event_with_trackers(
                    event, trackers,
                    max_history)
            training_features.extend(features)
            training_labels.extend(labels)
        return training_features, training_labels, trackers

    def _process_event_with_trackers(self,
                                     event,  # type: Event
                                     trackers,  # type: List[FeaturizedTracker]
                                     max_history  # type: int
                                     ):
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
        training_features = []
        training_labels = []

        for tracker in trackers:
            if isinstance(event, ActionExecuted):
                state_features = tracker.feauturize_current_state(self.domain)
                feature_vector = self.domain.slice_feature_history(
                        self.featurizer, state_features, max_history)
                hashed = utils.HashableNDArray(feature_vector)

                # only continue with trackers that created a
                # featurization we haven't observed at this event
                if hashed not in featurizations:
                    featurizations.add(hashed)
                    if not event.unpredictable:
                        # only actions which can be predicted at a stories start
                        training_features.append(feature_vector)
                        training_labels.append(
                                self.domain.index_for_action(event.action_name))
                    unique_trackers.append(tracker)
            else:
                unique_trackers.append(tracker)
            tracker.update(event)

        return training_features, training_labels, unique_trackers

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
                    np.reshape(y, (y.shape[0], 1, 1)), (y.shape[0], X.shape[1], 1))
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
            if step.start_checkpoint_name() == STORY_START:
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
            if step.start_checkpoint in unused_checkpoints:
                # After processing, there shouldn't be a story part left.
                # This indicates a start checkpoint that doesn't exist
                collected.add((step.start_checkpoint, step.block_name))
        for block_name, cp in collected:
            logger.warn("Unsatisfied start checkpoint '{}' "
                        "in block '{}'".format(cp, block_name))

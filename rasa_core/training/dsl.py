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
import re
from collections import deque

from typing import Optional, List, Text, Any, Dict

from rasa_core import utils
from rasa_core.channels import UserMessage
from rasa_core.domain import Domain
from rasa_core.events import (
    ActionExecuted, UserUttered, Event, ActionReverted)
from rasa_core.interpreter import RegexInterpreter
from rasa_core.trackers import DialogueStateTracker
from rasa_core.training.structures import (
    Checkpoint, STORY_END, STORY_START, StoryStep)

logger = logging.getLogger(__name__)


class StoryParseError(Exception):
    """Raised if there is an error while parsing the story file."""

    def __init__(self, message):
        self.message = message


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
                if t.end_checkpoint != STORY_END:
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
            generated_checkpoint = utils.generate_id("GENERATED_M_")
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
                     if step.end_checkpoint != STORY_END]
        unfinished = [step
                      for step in self.current_steps
                      if step.end_checkpoint == STORY_END]
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
        start_checkpoints = self._prev_end_checkpoints() or [
            Checkpoint(STORY_START)]
        current_turns = [StoryStep(block_name=self.name, start_checkpoint=s)
                         for s in start_checkpoints]
        return current_turns


class StoryFileReader(object):
    """Helper class to read a story file."""

    def __init__(self, domain, interpreter, template_vars=None):
        self.story_steps = []
        self.current_step_builder = None  # type: Optional[StoryStepBuilder]
        self.domain = domain
        self.interpreter = interpreter
        self.template_variables = template_vars if template_vars else {}

    @staticmethod
    def read_from_file(filename, domain, interpreter=RegexInterpreter(),
                       template_variables=None):
        """Given a json file reads the contained stories."""

        try:
            with io.open(filename, "r") as f:
                lines = f.readlines()
            reader = StoryFileReader(domain, interpreter, template_variables)
            return reader.process_lines(lines)
        except Exception as e:
            raise Exception("Failed to parse '{}'. {}".format(
                    os.path.abspath(filename), e))

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

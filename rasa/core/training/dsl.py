# -*- coding: utf-8 -*-
import asyncio
import io
import json
import logging
import os
import re
import warnings
from typing import Optional, List, Text, Any, Dict, AnyStr, TYPE_CHECKING

from rasa.core import utils
from rasa.core.constants import INTENT_MESSAGE_PREFIX
from rasa.core.events import ActionExecuted, UserUttered, Event, SlotSet
from rasa.core.exceptions import StoryParseError
from rasa.core.interpreter import RegexInterpreter
from rasa.core.training.structures import (
    Checkpoint,
    STORY_START,
    StoryStep,
    GENERATED_CHECKPOINT_PREFIX,
    GENERATED_HASH_LENGTH,
    FORM_PREFIX,
)
from rasa.nlu.training_data.formats import MarkdownReader


if TYPE_CHECKING:
    from rasa.nlu.training_data import Message


logger = logging.getLogger(__name__)


class EndToEndReader(MarkdownReader):
    def _parse_item(self, line: Text) -> Optional["Message"]:
        """Parses an md list item line based on the current section type.

        Matches expressions of the form `<intent>:<example>. For the
        syntax of <example> see the Rasa NLU docs on training data:
        https://rasa.com/docs/nlu/dataformat/#markdown-format"""

        item_regex = re.compile(r"\s*(.+?):\s*(.*)")
        match = re.match(item_regex, line)
        if match:
            intent = match.group(1)
            self.current_title = intent
            message = match.group(2)
            example = self._parse_training_example(message)
            example.data["true_intent"] = intent
            return example

        raise ValueError(
            "Encountered invalid end-to-end format for message "
            "`{}`. Please visit the documentation page on "
            "end-to-end evaluation at https://rasa.com/docs/core/"
            "evaluation#end-to-end-evaluation-of-rasa-nlu-and-"
            "core".format(line)
        )


class StoryStepBuilder(object):
    def __init__(self, name):
        self.name = name
        self.story_steps = []
        self.current_steps = []
        self.start_checkpoints = []

    def add_checkpoint(self, name: Text, conditions: Optional[Dict[Text, Any]]) -> None:

        # Depending on the state of the story part this
        # is either a start or an end check point
        if not self.current_steps:
            self.start_checkpoints.append(Checkpoint(name, conditions))
        else:
            if conditions:
                logger.warning(
                    "End or intermediate checkpoints "
                    "do not support conditions! "
                    "(checkpoint: {})".format(name)
                )
            additional_steps = []
            for t in self.current_steps:
                if t.end_checkpoints:
                    tcp = t.create_copy(use_new_id=True)
                    tcp.end_checkpoints = [Checkpoint(name)]
                    additional_steps.append(tcp)
                else:
                    t.end_checkpoints = [Checkpoint(name)]
            self.current_steps.extend(additional_steps)

    def _prev_end_checkpoints(self):
        if not self.current_steps:
            return self.start_checkpoints
        else:
            # makes sure we got each end name only once
            end_names = {e.name for s in self.current_steps for e in s.end_checkpoints}
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
            prefix = GENERATED_CHECKPOINT_PREFIX + "OR_"
            generated_checkpoint = utils.generate_id(prefix, GENERATED_HASH_LENGTH)
            updated_steps = []
            for t in self.current_steps:
                for m in messages:
                    copied = t.create_copy(use_new_id=True)
                    copied.add_user_message(m)
                    copied.end_checkpoints = [Checkpoint(generated_checkpoint)]
                    updated_steps.append(copied)
            self.current_steps = updated_steps

    def add_event(self, event):
        self.ensure_current_steps()
        for t in self.current_steps:
            t.add_event(event)

    def ensure_current_steps(self):
        completed = [step for step in self.current_steps if step.end_checkpoints]
        unfinished = [step for step in self.current_steps if not step.end_checkpoints]
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
        start_checkpoints = self._prev_end_checkpoints()
        if not start_checkpoints:
            start_checkpoints = [Checkpoint(STORY_START)]
        current_turns = [
            StoryStep(block_name=self.name, start_checkpoints=start_checkpoints)
        ]
        return current_turns


class StoryFileReader(object):
    """Helper class to read a story file."""

    def __init__(self, domain, interpreter, template_vars=None, use_e2e=False):
        self.story_steps = []
        self.current_step_builder = None  # type: Optional[StoryStepBuilder]
        self.domain = domain
        self.interpreter = interpreter
        self.template_variables = template_vars if template_vars else {}
        self.use_e2e = use_e2e

    @staticmethod
    async def read_from_folder(
        resource_name,
        domain,
        interpreter=RegexInterpreter(),
        template_variables=None,
        use_e2e=False,
        exclusion_percentage=None,
    ):
        """Given a path reads all contained story files."""
        import rasa.nlu.utils as nlu_utils

        if not os.path.exists(resource_name):
            raise ValueError(
                "Story file or folder could not be found. Make "
                "sure '{}' exists and points to a story folder "
                "or file.".format(os.path.abspath(resource_name))
            )

        story_steps = []
        for f in nlu_utils.list_files(resource_name):
            steps = await StoryFileReader.read_from_file(
                f, domain, interpreter, template_variables, use_e2e
            )
            story_steps.extend(steps)

        # if exclusion percentage is not 100
        if exclusion_percentage and exclusion_percentage is not 100:
            import random

            idx = int(round(exclusion_percentage / 100.0 * len(story_steps)))
            random.shuffle(story_steps)
            story_steps = story_steps[:-idx]

        return story_steps

    @staticmethod
    async def read_from_file(
        filename,
        domain,
        interpreter=RegexInterpreter(),
        template_variables=None,
        use_e2e=False,
    ):
        """Given a md file reads the contained stories."""

        try:
            with open(filename, "r", encoding="utf-8") as f:
                lines = f.readlines()
            reader = StoryFileReader(domain, interpreter, template_variables, use_e2e)
            return await reader.process_lines(lines)
        except ValueError as err:
            file_info = "Invalid story file format. Failed to parse '{}'".format(
                os.path.abspath(filename)
            )
            logger.exception(file_info)
            if not err.args:
                err.args = ("",)
            err.args = err.args + (file_info,)
            raise

    @staticmethod
    def _parameters_from_json_string(s: Text, line: Text) -> Dict[Text, Any]:
        """Parse the passed string as json and create a parameter dict."""

        if s is None or not s.strip():
            # if there is no strings there are not going to be any parameters
            return {}

        try:
            parsed_slots = json.loads(s)
            if isinstance(parsed_slots, dict):
                return parsed_slots
            else:
                raise Exception(
                    "Parsed value isn't a json object "
                    "(instead parser found '{}')"
                    ".".format(type(parsed_slots))
                )
        except Exception as e:
            raise ValueError(
                "Invalid to parse arguments in line "
                "'{}'. Failed to decode parameters"
                "as a json object. Make sure the event"
                "name is followed by a proper json "
                "object. Error: {}".format(line, e)
            )

    @staticmethod
    def _parse_event_line(line):
        """Tries to parse a single line as an event with arguments."""

        # the regex matches "slot{"a": 1}"
        m = re.search("^([^{]+)([{].+)?", line)
        if m is not None:
            event_name = m.group(1).strip()
            slots_str = m.group(2)
            parameters = StoryFileReader._parameters_from_json_string(slots_str, line)
            return event_name, parameters
        else:
            warnings.warn(
                "Failed to parse action line '{}'. Ignoring this line.".format(line)
            )
            return "", {}

    async def process_lines(self, lines: List[AnyStr]) -> List[StoryStep]:

        for idx, line in enumerate(lines):
            line_num = idx + 1
            try:
                line = self._replace_template_variables(self._clean_up_line(line))
                if line.strip() == "":
                    continue
                elif line.startswith("#"):
                    # reached a new story block
                    name = line[1:].strip("# ")
                    self.new_story_part(name)
                elif line.startswith(">"):
                    # reached a checkpoint
                    name, conditions = self._parse_event_line(line[1:].strip())
                    self.add_checkpoint(name, conditions)
                elif re.match(r"^[*\-]\s+{}".format(FORM_PREFIX), line):
                    logger.debug(
                        "Skipping line {}, "
                        "because it was generated by "
                        "form action".format(line)
                    )
                elif line.startswith("-"):
                    # reached a slot, event, or executed action
                    event_name, parameters = self._parse_event_line(line[1:])
                    self.add_event(event_name, parameters)
                elif line.startswith("*"):
                    # reached a user message
                    user_messages = [el.strip() for el in line[1:].split(" OR ")]
                    if self.use_e2e:
                        await self.add_e2e_messages(user_messages, line_num)
                    else:
                        await self.add_user_messages(user_messages, line_num)
                else:
                    # reached an unknown type of line
                    logger.warning(
                        "Skipping line {}. "
                        "No valid command found. "
                        "Line Content: '{}'"
                        "".format(line_num, line)
                    )
            except Exception as e:
                msg = "Error in line {}: {}".format(line_num, e)
                logger.error(msg, exc_info=1)
                raise ValueError(msg)
        self._add_current_stories_to_result()
        return self.story_steps

    def _replace_template_variables(self, line):
        def process_match(matchobject):
            varname = matchobject.group(1)
            if varname in self.template_variables:
                return self.template_variables[varname]
            else:
                raise ValueError(
                    "Unknown variable `{var}` "
                    "in template line '{line}'"
                    "".format(var=varname, line=line)
                )

        template_rx = re.compile(r"`([^`]+)`")
        return template_rx.sub(process_match, line)

    @staticmethod
    def _clean_up_line(line: Text) -> Text:
        """Removes comments and trailing spaces"""

        return re.sub(r"<!--.*?-->", "", line).strip()

    def _add_current_stories_to_result(self):
        if self.current_step_builder:
            self.current_step_builder.flush()
            self.story_steps.extend(self.current_step_builder.story_steps)

    def new_story_part(self, name):
        self._add_current_stories_to_result()
        self.current_step_builder = StoryStepBuilder(name)

    def add_checkpoint(self, name: Text, conditions: Optional[Dict[Text, Any]]) -> None:

        # Ensure story part already has a name
        if not self.current_step_builder:
            raise StoryParseError(
                "Checkpoint '{}' is at an invalid location. "
                "Expected a story start.".format(name)
            )

        self.current_step_builder.add_checkpoint(name, conditions)

    async def _parse_message(self, message, line_num):
        if message.startswith(INTENT_MESSAGE_PREFIX):
            parse_data = await RegexInterpreter().parse(message)
        else:
            parse_data = await self.interpreter.parse(message)
        utterance = UserUttered(
            message, parse_data.get("intent"), parse_data.get("entities"), parse_data
        )
        intent_name = utterance.intent.get("name")
        if intent_name not in self.domain.intents:
            logger.warning(
                "Found unknown intent '{}' on line {}. "
                "Please, make sure that all intents are "
                "listed in your domain yaml."
                "".format(intent_name, line_num)
            )
        return utterance

    async def add_user_messages(self, messages, line_num):
        if not self.current_step_builder:
            raise StoryParseError(
                "User message '{}' at invalid location. "
                "Expected story start.".format(messages)
            )
        parsed_messages = await asyncio.gather(
            *[self._parse_message(m, line_num) for m in messages]
        )
        self.current_step_builder.add_user_messages(parsed_messages)

    async def add_e2e_messages(self, e2e_messages, line_num):
        if not self.current_step_builder:
            raise StoryParseError(
                "End-to-end message '{}' at invalid "
                "location. Expected story start."
                "".format(e2e_messages)
            )
        e2e_reader = EndToEndReader()
        parsed_messages = []
        for m in e2e_messages:
            message = e2e_reader._parse_item(m)
            parsed = await self._parse_message(message.text, line_num)

            parsed.parse_data["true_intent"] = message.data["true_intent"]
            parsed.parse_data["true_entities"] = message.data.get("entities") or []
            parsed_messages.append(parsed)
        self.current_step_builder.add_user_messages(parsed_messages)

    def add_event(self, event_name, parameters):

        # add 'name' only if event is not a SlotSet,
        # because there might be a slot with slot_key='name'
        if "name" not in parameters and event_name != SlotSet.type_name:
            parameters["name"] = event_name

        parsed_events = Event.from_story_string(
            event_name, parameters, default=ActionExecuted
        )
        if parsed_events is None:
            raise StoryParseError(
                "Unknown event '{}'. It is Neither an event "
                "nor an action).".format(event_name)
            )
        if self.current_step_builder is None:
            raise StoryParseError(
                "Failed to handle event '{}'. There is no "
                "started story block available. "
                "".format(event_name)
            )

        for p in parsed_events:
            self.current_step_builder.add_event(p)

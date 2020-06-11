import asyncio
import json
import logging
import os
import re
from typing import Optional, List, Text, Any, Dict, TYPE_CHECKING, Iterable

import rasa.utils.io as io_utils
from rasa.constants import DOCS_BASE_URL, DOCS_URL_STORIES, DOCS_URL_DOMAINS
from rasa.core import utils
from rasa.core.constants import INTENT_MESSAGE_PREFIX
from rasa.core.events import ActionExecuted, UserUttered, Event, SlotSet
from rasa.core.exceptions import StoryParseError
from rasa.core.interpreter import RegexInterpreter, NaturalLanguageInterpreter
from rasa.core.training.structures import (
    Checkpoint,
    STORY_START,
    StoryStep,
    GENERATED_CHECKPOINT_PREFIX,
    GENERATED_HASH_LENGTH,
    FORM_PREFIX,
)
from rasa.nlu.training_data.formats import MarkdownReader
from rasa.core.domain import Domain
from rasa.utils.common import raise_warning

if TYPE_CHECKING:
    from rasa.nlu.training_data import Message

logger = logging.getLogger(__name__)


class EndToEndReader(MarkdownReader):
    def __init__(self) -> None:
        super().__init__()
        self._regex_interpreter = RegexInterpreter()

    def _parse_item(self, line: Text) -> Optional["Message"]:
        f"""Parses an md list item line based on the current section type.

        Matches expressions of the form `<intent>:<example>. For the
        syntax of <example> see the Rasa docs on NLU training data:
        {DOCS_BASE_URL}/nlu/training-data-format/#markdown-format"""

        # Match three groups:
        # 1) Potential "form" annotation
        # 2) The correct intent
        # 3) Optional entities
        # 4) The message text
        form_group = fr"({FORM_PREFIX}\s*)*"
        item_regex = re.compile(r"\s*" + form_group + r"([^{}]+?)({.*})*:\s*(.*)")
        match = re.match(item_regex, line)

        if not match:
            raise ValueError(
                "Encountered invalid end-to-end format for message "
                "`{}`. Please visit the documentation page on "
                "end-to-end testing at {}/user-guide/testing-your-assistant/"
                "#end-to-end-testing/".format(line, DOCS_BASE_URL)
            )

        intent = match.group(2)
        self.current_title = intent
        message = match.group(4)
        example = self.parse_training_example(message)

        # If the message starts with the `INTENT_MESSAGE_PREFIX` potential entities
        # are annotated in the json format (e.g. `/greet{"name": "Rasa"})
        if message.startswith(INTENT_MESSAGE_PREFIX):
            parsed = self._regex_interpreter.synchronous_parse(message)
            example.data["entities"] = parsed["entities"]

        example.data["true_intent"] = intent
        return example

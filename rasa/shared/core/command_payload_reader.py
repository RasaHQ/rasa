import copy
import re
from typing import List, Optional

import structlog

from rasa.dialogue_understanding.commands.set_slot_command import SetSlotExtractor
from rasa.shared.core.domain import Domain
from rasa.shared.nlu.constants import COMMANDS, SET_SLOT_COMMAND, TEXT
from rasa.shared.nlu.training_data.message import Message

structlogger = structlog.get_logger()

SET_SLOTS_PATTERN = r"(?P<slot_name>[^(),=]+)=(?P<slot_value>[^(),]+)"
MAX_NUMBER_OF_SLOTS = 10


class CommandPayloadReader:
    @staticmethod
    def unpack_regex_message(
        message: Message,
        domain: Optional[Domain] = None,
        entity_extractor_name: Optional[
            str
        ] = SetSlotExtractor.COMMAND_PAYLOAD_READER.value,
    ) -> Message:
        """Extracts commands from user text and adds them to the message."""
        user_text = message.get(TEXT).strip()

        # prevent ReDos attacks by placing a limit on the number of slots
        if CommandPayloadReader.is_above_slot_limit(user_text):
            return message

        matches = CommandPayloadReader.find_matches(user_text)
        if not matches:
            structlogger.warning(
                "message.parsing.failed", user_text=copy.deepcopy(user_text)
            )
            return message

        return CommandPayloadReader.extract_commands_from_pattern_matches(
            message, matches, domain, entity_extractor_name
        )

    @staticmethod
    def find_matches(user_text: str) -> List[re.Match]:
        return re.findall(SET_SLOTS_PATTERN, user_text)

    @staticmethod
    def extract_commands_from_pattern_matches(
        message: Message,
        matches: List[re.Match],
        domain: Optional[Domain] = None,
        entity_extractor_name: Optional[
            str
        ] = SetSlotExtractor.COMMAND_PAYLOAD_READER.value,
    ) -> Message:
        """Extract attributes from the matches and validate them via the domain."""
        commands = []

        for match in matches:
            slot_name = match[0].strip()
            slot_value = match[1].strip()

            domain_slot_names = [slot.name for slot in domain.slots] if domain else []

            if domain and slot_name not in domain_slot_names:
                structlogger.warning(
                    "slot.not.found",
                    slot_name=slot_name,
                )
                return message

            slot_value = (
                slot_value
                if slot_value.lower() not in {"none", "null", "undefined"}
                else None
            )

            # Create new SetSlot commands from the extracted attributes.
            commands.append(
                {
                    "command": SET_SLOT_COMMAND,
                    "name": slot_name,
                    "value": slot_value,
                    "extractor": entity_extractor_name,
                }
            )

            structlogger.debug(
                "slot.set.command.added",
                slot_name=slot_name,
            )

        # set the command(s) on the Message object
        message.set(COMMANDS, commands, add_to_output=True)
        return message

    @staticmethod
    def is_above_slot_limit(user_text: str) -> bool:
        """Prevent ReDoS attacks by limiting the number of slots."""
        if user_text.count("=") > MAX_NUMBER_OF_SLOTS:
            structlogger.warning(
                "too.many.slots",
                user_text=copy.deepcopy(user_text),
                slot_limit=10,
            )
            return True
        return False

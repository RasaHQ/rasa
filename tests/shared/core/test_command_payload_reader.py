from typing import List

import pytest
from pytest import CaptureFixture

from rasa.dialogue_understanding.commands.set_slot_command import SetSlotExtractor
from rasa.shared.core.command_payload_reader import (
    CommandPayloadReader,
)
from rasa.shared.core.domain import Domain
from rasa.shared.nlu.constants import COMMANDS, SET_SLOT_COMMAND, TEXT
from rasa.shared.nlu.training_data.message import Message


@pytest.mark.parametrize(
    "user_text, expected_output",
    [
        (
            "/SetSlots(merchant=visa, amount=1000, date=2022-01-01)",
            [("merchant", "visa"), (" amount", "1000"), (" date", "2022-01-01")],
        )
    ],
)
def test_command_payload_reader_find_match_successfully(
    user_text: str, expected_output: List
) -> None:
    matches = CommandPayloadReader.find_matches(user_text)
    assert matches == expected_output


@pytest.mark.parametrize("user_text", ["random message", ""])
def test_command_payload_reader_find_match_no_match(user_text: str) -> None:
    matches = CommandPayloadReader.find_matches(user_text)
    assert matches == []


@pytest.mark.parametrize(
    "slots, expected_values",
    [(["merchant", "amount", "date"], ["visa", "1000", "2022-01-01"])],
)
def test_command_payload_reader_unpack_regex_message(
    slots: List[str], expected_values: List[str]
) -> None:
    user_text = (
        f"/SetSlots({slots[0]}={expected_values[0]}, {slots[1]}={expected_values[1]}, "
        f"{slots[2]}={expected_values[2]})"
    )
    message = Message({TEXT: user_text})
    domain = Domain.from_yaml(
        f"""
        slots:
          {slots[0]}:
            type: text
          {slots[1]}:
            type: float
          {slots[2]}:
            type: text
        """
    )

    processed_message = CommandPayloadReader.unpack_regex_message(message, domain)
    assert processed_message.get(COMMANDS) == [
        {
            "command": SET_SLOT_COMMAND,
            "name": slots[0],
            "value": expected_values[0],
            "extractor": SetSlotExtractor.COMMAND_PAYLOAD_READER.value,
        },
        {
            "command": SET_SLOT_COMMAND,
            "name": slots[1],
            "value": expected_values[1],
            "extractor": SetSlotExtractor.COMMAND_PAYLOAD_READER.value,
        },
        {
            "command": SET_SLOT_COMMAND,
            "name": slots[2],
            "value": expected_values[2],
            "extractor": SetSlotExtractor.COMMAND_PAYLOAD_READER.value,
        },
    ]


@pytest.mark.parametrize(
    "user_text, expected_warning",
    [
        ("random message", "message.parsing.failed"),
        ("", "message.parsing.failed"),
        ("/SetSlots(amount=100)", "slot.not.found"),
    ],
)
def test_command_payload_reader_unpack_regex_message_logs_warning(
    user_text: str,
    capsys: CaptureFixture,
    expected_warning: str,
) -> None:
    message = Message({TEXT: user_text})
    domain = Domain.from_yaml(
        """
        slots:
          merchant:
            type: text
        """
    )

    processed_message = CommandPayloadReader.unpack_regex_message(message, domain)
    assert message.get(COMMANDS) is None
    assert processed_message == message

    log_output = capsys.readouterr().out
    assert expected_warning in log_output


def test_command_payload_reader_unpack_regex_message_prevent_ReDoS(
    capsys: CaptureFixture,
) -> None:
    user_text = "/SetSlots(" + "a=" * 11 + "1)"
    message = Message({TEXT: user_text})
    domain = Domain.from_yaml(
        """
        slots:
          a:
            type: text
        """
    )

    processed_message = CommandPayloadReader.unpack_regex_message(message, domain)
    assert message.get(COMMANDS) is None
    assert processed_message == message

    log_output = capsys.readouterr().out
    assert "too.many.slots" in log_output


@pytest.mark.parametrize(
    "user_text",
    [
        "/SetSlots(address=null)",
        "/SetSlots(address=None)",
        "/SetSlots(address=undefined)",
        "/SetSlots(address=none)",
        "/SetSlots(address=Null)",
    ],
)
def test_command_payload_reader_unpack_regex_message_null_value(user_text: str) -> None:
    message = Message({TEXT: user_text})
    domain = Domain.from_yaml(
        """
        slots:
          address:
            type: text
        """
    )

    processed_message = CommandPayloadReader.unpack_regex_message(message, domain)
    assert processed_message.get(COMMANDS) == [
        {
            "command": SET_SLOT_COMMAND,
            "name": "address",
            "value": None,
            "extractor": SetSlotExtractor.COMMAND_PAYLOAD_READER.value,
        }
    ]

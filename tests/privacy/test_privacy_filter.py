import datetime
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock

import freezegun
import pytest
from pytest import MonkeyPatch
from structlog.testing import capture_logs

from rasa.privacy.constants import GLINER_LABELS
from rasa.privacy.privacy_config import AnonymizationMethod
from rasa.privacy.privacy_filter import PrivacyFilter
from rasa.shared.core.events import (
    ActionExecuted,
    BotUttered,
    Event,
    SessionStarted,
    SlotSet,
    UserUttered,
)
from tests.utilities import filter_logs

freezegun.config.configure(extend_ignore_list=["transformers"])


@pytest.fixture(scope="session")
def anonymization_rules() -> Dict[str, AnonymizationMethod]:
    return {
        "email": AnonymizationMethod.from_dict(
            {"type": "redact", "keep_left": 2, "keep_right": 2}
        ),
        "phone_number": AnonymizationMethod.from_dict(
            {"type": "redact", "keep_right": 3}
        ),
        "address": AnonymizationMethod.from_dict({"type": "mask"}),
        "name": AnonymizationMethod.from_dict({"type": "mask"}),
    }


@pytest.fixture(scope="session")
def privacy_filter_with_loaded_gliner(
    anonymization_rules: Dict[str, AnonymizationMethod],
) -> PrivacyFilter:
    return PrivacyFilter(anonymization_rules)


@pytest.fixture
def privacy_filter_with_unloaded_gliner(
    anonymization_rules: Dict[str, AnonymizationMethod],
    monkeypatch: MonkeyPatch,
) -> PrivacyFilter:
    """Fixture for PrivacyFilter with GLiNER not loaded."""
    monkeypatch.setattr(
        "gliner.GLiNER.from_pretrained", MagicMock(side_effect=ImportError)
    )
    return PrivacyFilter(anonymization_rules)


@pytest.fixture(scope="session")
def tracker_events() -> List[Event]:
    return [
        SlotSet("session_started_metadata", {"user_id": "12345"}),
        ActionExecuted("action_session_start"),
        UserUttered(
            "Hi, my name is John Doe, I need help with my order.",
            parse_data={
                "entities": [{"entity": "name", "value": "John Doe"}],
            },
        ),
        SlotSet("name", "John Doe"),
        BotUttered(
            "Hi John Doe, please provide your email, phone number, and "
            "address to proceed."
        ),
        ActionExecuted("action_listen"),
        UserUttered(
            "Sure, my email is j.doe@test.com, my phone number is +1234567890, and "
            "my address is 123 Main St. Passport number is 123456789.",
        ),
        SlotSet("email", "j.doe@test.com"),
        SlotSet("phone_number", "+1234567890"),
        SlotSet("address", "123 Main St"),
        BotUttered(
            "The following details have been recorded: "
            "Name: John Doe. "
            "Email Address: j.doe@test.com. "
            "Phone Number: +1234567890. "
            "Address: 123 Main St. "
            "Passport Number: 123456789. "
            "Thank you for providing your details."
        ),
        ActionExecuted("action_listen"),
    ]


@freezegun.freeze_time("2023-10-01T12:00:00+00:00")
def test_privacy_filter_anonymize(
    privacy_filter_with_loaded_gliner: PrivacyFilter,
    tracker_events: List[Event],
) -> None:
    """Test anonymization of tracker events."""
    # When
    anonymized_events = privacy_filter_with_loaded_gliner.anonymize(tracker_events, [])

    # Then
    assert len(anonymized_events) == len(tracker_events)

    assert (
        anonymized_events[0] == tracker_events[0]
    )  # SlotSet for session_started_metadata unchanged
    assert (
        anonymized_events[1] == tracker_events[1]
    )  # ActionExecuted for action_session_start unchanged

    first_user_event = anonymized_events[2]
    assert isinstance(first_user_event, UserUttered)
    assert first_user_event.text == "Hi, my name is [NAME], I need help with my order."
    assert first_user_event.parse_data == {
        "entities": [{"entity": "name", "value": "[NAME]"}],
        "text": "Hi, my name is [NAME], I need help with my order.",
    }
    assert first_user_event.anonymized_at == datetime.datetime.fromisoformat(
        "2023-10-01T12:00:00+00:00"
    )

    name_slot_event = anonymized_events[3]
    assert isinstance(name_slot_event, SlotSet)
    assert name_slot_event.key == "name"
    assert name_slot_event.value == "[NAME]"
    assert name_slot_event.anonymized_at == datetime.datetime.fromisoformat(
        "2023-10-01T12:00:00+00:00"
    )

    first_bot_event = anonymized_events[4]
    assert isinstance(first_bot_event, BotUttered)
    assert "John Doe" not in first_bot_event.text
    assert "[NAME]" in first_bot_event.text
    assert first_bot_event.anonymized_at == datetime.datetime.fromisoformat(
        "2023-10-01T12:00:00+00:00"
    )

    assert anonymized_events[5] == tracker_events[5]

    second_user_event = anonymized_events[6]
    assert isinstance(second_user_event, UserUttered)
    assert second_user_event.text == (
        "Sure, my email is j.**********om, my phone number is ********890, and "
        "my address is [ADDRESS]. Passport number is [PASSPORT NUMBER]."
    )

    email_slot_event = anonymized_events[7]
    assert isinstance(email_slot_event, SlotSet)
    assert email_slot_event.key == "email"
    assert email_slot_event.value == "j.**********om"
    assert email_slot_event.anonymized_at == datetime.datetime.fromisoformat(
        "2023-10-01T12:00:00+00:00"
    )

    phone_slot_event = anonymized_events[8]
    assert isinstance(phone_slot_event, SlotSet)
    assert phone_slot_event.key == "phone_number"
    assert phone_slot_event.value == "********890"
    assert phone_slot_event.anonymized_at == datetime.datetime.fromisoformat(
        "2023-10-01T12:00:00+00:00"
    )

    address_slot_event = anonymized_events[9]
    assert isinstance(address_slot_event, SlotSet)
    assert address_slot_event.key == "address"
    assert address_slot_event.value == "[ADDRESS]"
    assert address_slot_event.anonymized_at == datetime.datetime.fromisoformat(
        "2023-10-01T12:00:00+00:00"
    )

    second_bot_event = anonymized_events[10]
    assert isinstance(second_bot_event, BotUttered)
    assert second_bot_event.text == (
        "The following details have been recorded: "
        "Name: [NAME]. "
        "Email Address: j.**********om. "
        "Phone Number: ********890. "
        "Address: [ADDRESS]. "
        "Passport Number: [PASSPORT NUMBER]. "
        "Thank you for providing your details."
    )
    assert second_bot_event.anonymized_at == datetime.datetime.fromisoformat(
        "2023-10-01T12:00:00+00:00"
    )
    assert (
        anonymized_events[11] == tracker_events[11]
    )  # ActionExecuted for action_listen unchanged


def test_privacy_filter_loads_model(
    privacy_filter_with_loaded_gliner: PrivacyFilter,
    anonymization_rules: Dict[str, AnonymizationMethod],
) -> None:
    """Test that the PrivacyFilter loads the anonymization model correctly."""
    assert privacy_filter_with_loaded_gliner.model is not None
    assert privacy_filter_with_loaded_gliner.anonymization_rules == anonymization_rules
    assert privacy_filter_with_loaded_gliner.labels == GLINER_LABELS


def test_privacy_filter_no_model_loaded(
    monkeypatch: MonkeyPatch, anonymization_rules: Dict[str, AnonymizationMethod]
) -> None:
    monkeypatch.setattr(
        "gliner.GLiNER.from_pretrained", MagicMock(side_effect=ImportError)
    )

    with capture_logs() as caplog:
        privacy_filter = PrivacyFilter(anonymization_rules)
        log = filter_logs(
            caplog,
            "rasa.privacy.privacy_filter.gliner_import_error",
            "warning",
            [
                "Optional GLiNER library is not installed. Please install it "
                "if you wish to use additional PII detection to the slot "
                "based approach."
            ],
            True,
        )
        assert len(log) == 1

    assert privacy_filter.model is None
    assert privacy_filter.anonymization_rules == anonymization_rules


def test_privacy_filter_find_sensitive_slots(
    privacy_filter_with_loaded_gliner: PrivacyFilter,
    tracker_events: List[Event],
) -> None:
    """Test finding sensitive slots in tracker events."""
    sensitive_slots = privacy_filter_with_loaded_gliner._find_sensitive_slots(
        tracker_events
    )

    assert len(sensitive_slots) == 4
    assert sensitive_slots[0].key == "name"
    assert sensitive_slots[1].key == "email"
    assert sensitive_slots[2].key == "phone_number"
    assert sensitive_slots[3].key == "address"


def test_privacy_filter_find_sensitive_slots_no_sensitive_data(
    privacy_filter_with_unloaded_gliner: PrivacyFilter,
) -> None:
    sensitive_slots = privacy_filter_with_unloaded_gliner._find_sensitive_slots(
        [
            SessionStarted(),
            SlotSet("session_started_metadata", {"user_id": "12345"}),
            ActionExecuted("action_session_start"),
        ]
    )

    assert len(sensitive_slots) == 0


def test_privacy_filter_anonymize_sensitive_slot_event(
    privacy_filter_with_unloaded_gliner: PrivacyFilter,
    tracker_events: List[Event],
) -> None:
    """Test anonymizing a sensitive slot event."""
    anonymized_event = (
        privacy_filter_with_unloaded_gliner._anonymize_sensitive_slot_event(
            SlotSet("name", "John Doe")
        )
    )
    assert anonymized_event.key == "name"
    assert anonymized_event.value == "[NAME]"


@pytest.mark.parametrize(
    "slot_event, expected_value",
    [
        (SlotSet("email", "john.doe@example.com"), "jo****************om"),
        (SlotSet("address", "123 Main St, Springfield"), "[ADDRESS]"),
    ],
)
def test_privacy_filter_anonymize_value(
    privacy_filter_with_unloaded_gliner: PrivacyFilter,
    slot_event: SlotSet,
    expected_value: str,
) -> None:
    """Test anonymizing a value."""
    anonymized_value = privacy_filter_with_unloaded_gliner._anonymize_value(slot_event)
    assert anonymized_value == expected_value


@pytest.mark.parametrize(
    "input_text, expected_anonymization",
    [
        (
            "I'd like to pay with my credit card 3782-8224-6310-0051",
            "[CREDIT CARD NUMBER]",
        ),
        ("My social security number is 123-45-6789.", "[SOCIAL SECURITY NUMBER]"),
        ("I've rented the car with license plate ABC1234.", "[LICENSE PLATE NUMBER]"),
    ],
)
def test_privacy_filter_anonymize_edge_cases_new_entities(
    privacy_filter_with_loaded_gliner: PrivacyFilter,
    input_text: str,
    expected_anonymization: str,
) -> None:
    anonymized_slots = privacy_filter_with_loaded_gliner._anonymize_sensitive_slots(
        events=[]
    )
    output_text = privacy_filter_with_loaded_gliner._anonymize_edge_cases(
        input_text, anonymized_slots
    )
    assert expected_anonymization in output_text
    assert input_text != output_text


def test_privacy_filter_anonymize_edge_cases_no_changes(
    privacy_filter_with_loaded_gliner: PrivacyFilter,
) -> None:
    input_text = "This text does not contain any sensitive information."
    anonymized_slots = privacy_filter_with_loaded_gliner._anonymize_sensitive_slots(
        events=[]
    )
    output_text = privacy_filter_with_loaded_gliner._anonymize_edge_cases(
        input_text, anonymized_slots
    )
    assert output_text == input_text


def test_privacy_filter_anonymize_edge_cases_no_double_anonymization(
    privacy_filter_with_loaded_gliner: PrivacyFilter,
) -> None:
    input_text = "[NAME]'s email is dr**************om."
    anonymized_slots = privacy_filter_with_loaded_gliner._anonymize_sensitive_slots(
        events=[
            SlotSet("email", "dr.watson@test.com"),
            SlotSet("name", "Watson"),
        ]
    )
    output_text = privacy_filter_with_loaded_gliner._anonymize_edge_cases(
        input_text, anonymized_slots
    )

    assert (
        output_text == input_text
    ), "Anonymization should not change already anonymized text."


def test_privacy_filter_anonymize_edge_cases_no_model_loaded(
    privacy_filter_with_unloaded_gliner: PrivacyFilter,
) -> None:
    input_text = "This text does not contain any sensitive information."
    with capture_logs() as caplog:
        output = privacy_filter_with_unloaded_gliner._anonymize_edge_cases(
            input_text, anonymized_slots={}
        )
        log = filter_logs(
            caplog,
            "rasa.privacy.privacy_filter.gliner_model_not_loaded",
            "debug",
            ["GLiNER model is not loaded, skipping PII detection."],
            True,
        )
        assert len(log) == 1

    assert output == input_text, "Output should be unchanged when model is not loaded."


@freezegun.freeze_time("2023-10-01 12:00:00")
@pytest.mark.parametrize(
    "event, expected_event",
    [
        (
            UserUttered(
                "Please deliver my package to 123 Main St, Springfield.",
                parse_data={
                    "entities": [
                        {"entity": "address", "value": "123 Main St, Springfield"}
                    ]
                },
            ),
            UserUttered(
                "Please deliver my package to [ADDRESS].",
                parse_data={"entities": [{"entity": "address", "value": "[ADDRESS]"}]},
            ),
        ),
        (
            UserUttered(
                "My phone number is +1234567890.",
            ),
            UserUttered(
                "My phone number is ********890.",
            ),
        ),
        (
            SlotSet("address", "123 Main St, Springfield"),
            SlotSet("address", "[ADDRESS]"),
        ),
        (
            SlotSet("phone_number", "+1234567890"),
            SlotSet("phone_number", "********890"),
        ),
        (
            BotUttered(
                "Confirming your order will be sent to 123 Main St, Springfield."
            ),
            BotUttered("Confirming your order will be sent to [ADDRESS]."),
        ),
        (
            BotUttered("Your phone number +1234567890 has been recorded."),
            BotUttered("Your phone number ********890 has been recorded."),
        ),
    ],
)
def test_privacy_filter_anonymize_event_supported_events(
    event: Event,
    expected_event: Event,
    privacy_filter_with_unloaded_gliner: PrivacyFilter,
) -> None:
    """Test anonymization of various supported events."""
    anonymized_slots = {
        "address:123 Main St, Springfield": SlotSet("address", "[ADDRESS]"),
        "phone_number:+1234567890": SlotSet("phone_number", "********890"),
    }
    anonymized_event = privacy_filter_with_unloaded_gliner._anonymize_event(
        event, anonymized_slots=anonymized_slots
    )
    assert anonymized_event == expected_event


def test_privacy_filter_anonymize_event_unsupported_events(
    privacy_filter_with_unloaded_gliner: PrivacyFilter,
    tracker_events: List[Event],
) -> None:
    """Test anonymization of unsupported events."""
    unsupported_event = ActionExecuted("action_listen")
    anonymized_event = privacy_filter_with_unloaded_gliner._anonymize_event(
        unsupported_event, anonymized_slots={}
    )
    assert anonymized_event == unsupported_event


@pytest.mark.parametrize(
    "no_text_event, event_type",
    [
        (UserUttered(text=""), "user"),
        (BotUttered(text=""), "bot"),
    ],
)
def test_privacy_filter_anonymize_event_no_text(
    privacy_filter_with_unloaded_gliner: PrivacyFilter,
    no_text_event: Event,
    event_type: str,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test anonymization of events with no text."""
    mock_anonymize_edge_cases = Mock()
    monkeypatch.setattr(
        privacy_filter_with_unloaded_gliner,
        "_anonymize_edge_cases",
        mock_anonymize_edge_cases,
    )

    with capture_logs() as caplog:
        anonymized_event = privacy_filter_with_unloaded_gliner._anonymize_event(
            no_text_event, anonymized_slots={}
        )
        log = filter_logs(
            caplog,
            f"rasa.privacy.privacy_filter.{event_type}_event_no_text",
            "debug",
            None,
            True,
        )
        assert len(log) == 1
    assert anonymized_event == no_text_event
    mock_anonymize_edge_cases.assert_not_called()


@pytest.mark.parametrize(
    "slot_value",
    [
        None,
        "",
        False,
        0,
    ],
)
def test_privacy_filter_anonymize_sensitive_slot_event_empty_value(
    privacy_filter_with_unloaded_gliner: PrivacyFilter,
    slot_value: Any,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test anonymizing a sensitive slot event with no value."""
    mock_anonymize_value = Mock()
    monkeypatch.setattr(
        privacy_filter_with_unloaded_gliner, "_anonymize_value", mock_anonymize_value
    )

    anonymized_event = (
        privacy_filter_with_unloaded_gliner._anonymize_sensitive_slot_event(
            SlotSet("name", slot_value)
        )
    )
    assert anonymized_event.key == "name"
    assert (
        anonymized_event.value == slot_value
    ), "Value should remain unchanged when empty."
    mock_anonymize_value.assert_not_called()

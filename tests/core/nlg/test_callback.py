import logging

from pytest import LogCaptureFixture
from rasa.core.nlg.callback import CallbackNaturalLanguageGenerator, nlg_request_format
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import UserUttered
from rasa.shared.core.slots import TextSlot
from rasa.shared.core.trackers import DialogueStateTracker, EventVerbosity


def test_nlg_request_format(
    default_tracker: DialogueStateTracker,
) -> None:
    assert nlg_request_format(
        utter_action="utter_one_id",
        tracker=default_tracker,
        output_channel="test",
    ) == {
        "response": "utter_one_id",
        "id": None,
        "arguments": {},
        "channel": {"name": "test"},
        "tracker": default_tracker.current_state(EventVerbosity.ALL),
    }


def test_nlg_request_format_with_arguments(
    default_tracker: DialogueStateTracker,
) -> None:
    assert nlg_request_format(
        utter_action="utter_one_id",
        tracker=default_tracker,
        output_channel="test",
        some_arg="some_value",
    ) == {
        "response": "utter_one_id",
        "id": None,
        "arguments": {"some_arg": "some_value"},
        "channel": {"name": "test"},
        "tracker": default_tracker.current_state(EventVerbosity.ALL),
    }


def test_nlg_request_format_with_response_ids(
    default_tracker: DialogueStateTracker,
) -> None:
    assert nlg_request_format(
        utter_action="utter_one_id",
        tracker=default_tracker,
        output_channel="test",
        response_id="1",
    ) == {
        "response": "utter_one_id",
        "id": "1",
        "arguments": {},
        "channel": {"name": "test"},
        "tracker": default_tracker.current_state(EventVerbosity.ALL),
    }


def test_callback_nlg_fetch_response_id() -> None:
    # Arrange
    utter_action = "utter_greet"

    name_slot = TextSlot(
        name="name", mappings=[{}], initial_value="Bob", influence_conversation=False
    )
    logged_in_slot = TextSlot(
        name="logged_in",
        mappings=[{}],
        initial_value=True,
        influence_conversation=False,
    )
    tracker = DialogueStateTracker.from_events(
        sender_id="interpolated_crv",
        evts=[UserUttered("Hello")],
        slots=[name_slot, logged_in_slot],
    )
    output_channel = "default"
    domain = Domain.load("data/test_nlg/domain_with_response_ids.yml")

    # Act
    response_id = CallbackNaturalLanguageGenerator.fetch_response_id(
        utter_action=utter_action,
        tracker=tracker,
        output_channel=output_channel,
        domain_responses=domain.responses,
    )

    # Assert
    assert response_id == "ID_2"


def test_callback_nlg_fetch_response_id_with_no_domain_responses(
    caplog: LogCaptureFixture,
) -> None:
    # Arrange
    utter_action = "utter_greet"

    tracker = DialogueStateTracker.from_events(
        sender_id="no_domain_responses",
        evts=[UserUttered("Hello")],
    )
    output_channel = "default"

    # Act
    with caplog.at_level(logging.DEBUG):
        response_id = CallbackNaturalLanguageGenerator.fetch_response_id(
            utter_action=utter_action,
            tracker=tracker,
            output_channel=output_channel,
            domain_responses=None,
        )

    # Assert
    assert response_id is None
    assert "Failed to fetch response id. Responses not provided." in caplog.text


def test_callback_nlg_fetch_response_id_with_no_response_id(
    caplog: LogCaptureFixture,
) -> None:
    # Arrange
    utter_action = "utter_goodbye"

    tracker = DialogueStateTracker.from_events(
        sender_id="no_response",
        evts=[UserUttered("Goodbye")],
    )
    output_channel = "default"
    domain = Domain.load("data/test_nlg/domain_with_response_ids.yml")

    # Act
    with caplog.at_level(logging.DEBUG):
        response_id = CallbackNaturalLanguageGenerator.fetch_response_id(
            utter_action=utter_action,
            tracker=tracker,
            output_channel=output_channel,
            domain_responses=domain.responses,
        )

    # Assert
    assert response_id is None
    assert f"Failed to fetch response id for action '{utter_action}'." in caplog.text

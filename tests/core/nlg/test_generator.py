import textwrap
from typing import Text

import pytest

from rasa.core.nlg.generator import ResponseVariationFilter
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import UserUttered
from rasa.shared.core.slots import TextSlot
from rasa.shared.core.trackers import DialogueStateTracker


def test_response_variation_filter_get_response_variation_id_interpolated_crv() -> None:
    """Test that the correct response variation id is retrieved.

    The test uses a conditional response variation with an interpolated slot value.
    """
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
    response_variation_filter = ResponseVariationFilter(domain.responses)

    # Act
    response_variation_id = response_variation_filter.get_response_variation_id(
        utter_action, tracker, output_channel
    )

    # Assert
    assert response_variation_id == "ID_2"


def test_response_variation_filter_get_response_id_default_variation() -> None:
    # Arrange
    utter_action = "utter_greet"

    tracker = DialogueStateTracker.from_events(
        sender_id="default_variation", evts=[UserUttered("Hello")], slots=[]
    )
    output_channel = "default"
    domain = Domain.load("data/test_nlg/domain_with_response_ids.yml")
    response_variation_filter = ResponseVariationFilter(domain.responses)

    # Act
    response_variation_id = response_variation_filter.get_response_variation_id(
        utter_action, tracker, output_channel
    )

    # Assert
    assert response_variation_id == "ID_1"


def test_response_variation_filter_get_response_id_multiple_conditions() -> None:
    """Test that the first response variation id is retrieved if multiple conditions are met."""  # noqa: E501
    # Arrange
    utter_action = "utter_greet"

    membership_slot = TextSlot(
        name="membership",
        mappings=[{}],
        initial_value="gold",
        influence_conversation=False,
    )
    logged_in_slot = TextSlot(
        name="logged_in",
        mappings=[{}],
        initial_value=True,
        influence_conversation=False,
    )
    tracker = DialogueStateTracker.from_events(
        sender_id="multiple_conditions",
        evts=[UserUttered("Hello")],
        slots=[membership_slot, logged_in_slot],
    )
    output_channel = "default"

    domain_yaml = textwrap.dedent(
        """
        version: "3.1"

        intents:
        - greet

        slots:
          membership:
             type: text
             mappings:
             - type: custom
          logged_in:
             type: bool
             influence_conversation: false
             mappings:
             - type: custom

        responses:
          utter_greet:
            - text: "Hello"
              id: "ID_0"

            - text: "Hello valued customer!"
              id: "ID_1"
              condition:
              - type: slot
                name: membership
                value: gold

            - text: "Welcome back!"
              id: "ID_2"
              condition:
              - type: slot
                name: logged_in
                value: true
        """
    )

    domain = Domain.from_yaml(domain_yaml)
    response_variation_filter = ResponseVariationFilter(domain.responses)

    # Act
    response_variation_id = response_variation_filter.get_response_variation_id(
        utter_action, tracker, output_channel
    )

    # Assert
    assert response_variation_id == "ID_1"


def test_response_variation_filter_get_response_id_chained_conditions() -> None:
    # Arrange
    utter_action = "utter_greet"

    membership_slot = TextSlot(
        name="membership",
        mappings=[{}],
        initial_value="gold",
        influence_conversation=False,
    )
    logged_in_slot = TextSlot(
        name="logged_in",
        mappings=[{}],
        initial_value=True,
        influence_conversation=False,
    )
    tracker = DialogueStateTracker.from_events(
        sender_id="chained_conditions",
        evts=[UserUttered("Hello")],
        slots=[membership_slot, logged_in_slot],
    )
    output_channel = "default"

    domain_yaml = textwrap.dedent(
        """
        version: "3.1"

        intents:
        - greet

        slots:
          membership:
             type: text
             mappings:
             - type: custom
          logged_in:
             type: bool
             influence_conversation: false
             mappings:
             - type: custom

        responses:
          utter_greet:
            - text: "Hello"
              id: "ID_0"

            - text: "Welcome back!"
              id: "ID_1"
              condition:
              - type: slot
                name: logged_in
                value: true
              - type: slot
                name: membership
                value: gold
        """
    )

    domain = Domain.from_yaml(domain_yaml)
    response_variation_filter = ResponseVariationFilter(domain.responses)

    # Act
    response_variation_id = response_variation_filter.get_response_variation_id(
        utter_action, tracker, output_channel
    )

    # Assert
    assert response_variation_id == "ID_1"


def test_response_variation_filter_get_response_id_with_channels() -> None:
    # Arrange
    utter_action = "utter_greet"

    membership_slot = TextSlot(
        name="membership",
        mappings=[{}],
        initial_value="gold",
        influence_conversation=False,
    )
    tracker = DialogueStateTracker.from_events(
        sender_id="crv_channel", evts=[UserUttered("Hello")], slots=[membership_slot]
    )

    domain_yaml = textwrap.dedent(
        """
        version: "3.1"

        intents:
        - greet

        slots:
          membership:
             type: text
             mappings:
             - type: custom
          logged_in:
             type: bool
             influence_conversation: false
             mappings:
             - type: custom

        responses:
          utter_greet:
            - text: "How can I help you today?"
              id: "ID_0"

            - text: ""
              id: "ID_1"
              condition:
              - type: slot
                name: membership
                value: gold
              channel: app

            - text: ""
              id: "ID_2"
              condition:
              - type: slot
                name: membership
                value: gold
              channel: web
        """
    )

    domain = Domain.from_yaml(domain_yaml)
    response_variation_filter = ResponseVariationFilter(domain.responses)

    # Act
    response_variation_id = response_variation_filter.get_response_variation_id(
        utter_action, tracker, output_channel="app"
    )

    # Assert
    assert response_variation_id == "ID_1"

    # Act
    response_variation_id = response_variation_filter.get_response_variation_id(
        utter_action, tracker, output_channel="web"
    )

    # Assert
    assert response_variation_id == "ID_2"

    # Act
    response_variation_id = response_variation_filter.get_response_variation_id(
        utter_action, tracker, output_channel="default"
    )

    # Assert
    assert response_variation_id == "ID_0"


@pytest.mark.parametrize(
    "initial_value, output_channel, expected_id",
    [
        ("gold", "app", "ID_1"),
        ("silver", "app", "ID_2"),
        ("bronze", "default", "ID_3"),
        ("gold", "default", "ID_4"),
    ],
)
def test_response_variation_filter_get_response_id_edge_cases(
    initial_value: Text, output_channel: Text, expected_id: Text
) -> None:
    # Arrange
    utter_action = "utter_greet"

    membership_slot = TextSlot(
        name="membership",
        mappings=[{}],
        initial_value=initial_value,
        influence_conversation=False,
    )
    tracker = DialogueStateTracker.from_events(
        sender_id="edge_cases", evts=[UserUttered("Hello")], slots=[membership_slot]
    )

    domain_yaml = textwrap.dedent(
        """
        version: "3.1"

        intents:
        - greet

        slots:
          membership:
             type: text
             mappings:
             - type: custom
          logged_in:
             type: bool
             influence_conversation: false
             mappings:
             - type: custom

        responses:
          utter_greet:
            - text: "How can I help you today?"
              id: "ID_0"

            - text: ""
              id: "ID_1"
              condition:
              - type: slot
                name: membership
                value: gold
              channel: app

            - text: ""
              id: "ID_2"
              condition:
              - type: slot
                name: membership
                value: silver
              channel: app

            - text: ""
              id: "ID_3"
              condition:
              - type: slot
                name: membership
                value: bronze

            - text: ""
              id: "ID_4"
              condition:
              - type: slot
                name: membership
                value: gold
        """
    )

    domain = Domain.from_yaml(domain_yaml)
    response_variation_filter = ResponseVariationFilter(domain.responses)

    # Act
    response_variation_id = response_variation_filter.get_response_variation_id(
        utter_action, tracker, output_channel=output_channel
    )

    # Assert
    assert response_variation_id == expected_id


def test_response_variation_filter_raises_exception_duplicate_ids() -> None:
    # Arrange
    utter_action = "utter_greet"

    membership_slot = TextSlot(
        name="membership",
        mappings=[{}],
        initial_value="gold",
        influence_conversation=False,
    )
    tracker = DialogueStateTracker.from_events(
        sender_id="duplicate_ids", evts=[UserUttered("Hello")], slots=[membership_slot]
    )

    domain_yaml = textwrap.dedent(
        """
        version: "3.1"

        intents:
        - greet

        slots:
          membership:
             type: text
             mappings:
             - type: custom
          logged_in:
             type: bool
             influence_conversation: false
             mappings:
             - type: custom

        responses:
          utter_greet:
            - text: "How can I help you today?"
              id: "ID_0"

            - text: ""
              id: "ID_1"

            - text: ""
              id: "ID_1"
        """
    )

    domain = Domain.from_yaml(domain_yaml)
    response_variation_filter = ResponseVariationFilter(domain.responses)

    expected_message = "Duplicate response id 'ID_1' defined in the domain."

    with pytest.warns(UserWarning, match=expected_message):
        result = response_variation_filter.get_response_variation_id(
            utter_action, tracker, output_channel="app"
        )

    assert result is None

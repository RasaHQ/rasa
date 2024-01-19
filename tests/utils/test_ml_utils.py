from rasa.shared.core.domain import Domain

from rasa.utils.ml_utils import (
    extract_ai_response_examples,
    extract_participant_messages_from_transcript,
    form_utterances_to_action,
)


def test_participant_messages_from_transcript_handles_empty_string() -> None:
    assert extract_participant_messages_from_transcript("") == []


def test_participant_messages_from_transcript_handles_no_participant() -> None:
    assert extract_participant_messages_from_transcript("Hello") == []


def test_participant_messages_from_transcript_handles_ai_participant() -> None:
    assert extract_participant_messages_from_transcript("AI: Hello") == ["Hello"]


def test_participant_messages_from_transcript_handles_human_participant() -> None:
    assert extract_participant_messages_from_transcript("USER: Hello\nAI: Hi!") == [
        "Hi!"
    ]


def test_participant_messages_from_transcript_handles_multiple() -> None:
    assert extract_participant_messages_from_transcript(
        "USER: Hello\nAI: Hi!", "USER"
    ) == ["Hello"]


def test_participant_messages_from_transcript_handles_multiple_messages() -> None:
    assert extract_participant_messages_from_transcript(
        "USER: Hello\nAI: Hi!\nAI: How are you?"
    ) == ["Hi!", "How are you?"]


def test_extract_ai_response_examples_handles_empty_responses() -> None:
    assert extract_ai_response_examples({}) == []


def test_extract_ai_response_examples_handles_empty_text() -> None:
    assert extract_ai_response_examples({"utter_greet": [{"text": ""}]}) == []


def test_extract_ai_response_examples_handles_no_text() -> None:
    assert extract_ai_response_examples({"utter_greet": []}) == []


def test_extract_ai_response_examples_handles_single_text() -> None:
    assert extract_ai_response_examples({"utter_greet": [{"text": "hello"}]}) == [
        "hello"
    ]


def test_extract_ai_response_examples_handles_multiple_texts() -> None:
    assert extract_ai_response_examples(
        {"utter_greet": [{"text": "hello"}], "utter_goodbye": [{"text": "goodbye"}]}
    ) == ["hello", "goodbye"]


def test_extract_ai_response_examples_handles_multiple_variations() -> None:
    assert extract_ai_response_examples(
        {"utter_greet": [{"text": "hello"}, {"text": "hi"}]}
    ) == ["hello", "hi"]


def test_form_utterances_to_action_handles_empty_domain() -> None:
    assert form_utterances_to_action(Domain.empty()) == {}


def test_form_utterances_to_action_maps_slots_to_forms() -> None:
    domain = Domain.from_yaml(
        """
        forms:
            form1:
                required_slots:
                    - slot1
                    - slot2"""
    )
    assert form_utterances_to_action(domain) == {
        "utter_ask_slot1": "form1",
        "utter_ask_slot2": "form1",
    }

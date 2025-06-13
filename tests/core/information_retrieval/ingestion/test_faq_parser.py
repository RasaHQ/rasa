from typing import List, Tuple

import pytest
import structlog
from langchain.schema import Document

from rasa.core.information_retrieval.ingestion.faq_parser import _format_faq_documents
from tests.utilities import filter_logs


@pytest.mark.parametrize(
    "input_text, expected_pairs, expected_warning_event",
    [
        # Simple QA
        (
            "Q: What is Finley?\nA: Finley is your assistant.",
            [("What is Finley?", "Finley is your assistant.")],
            None,
        ),
        # Extra spacing
        (
            "  Q: Who made Finley?  \n   A: The FinX team.  ",
            [("Who made Finley?", "The FinX team.")],
            None,
        ),
        (
            "Q:    What is Finley?\nA:      Finley is your assistant.",
            [("What is Finley?", "Finley is your assistant.")],
            None,
        ),
        (
            "    Q:    What is Finley?\n     A:      Finley is your assistant.",
            [("What is Finley?", "Finley is your assistant.")],
            None,
        ),
        (
            "\n\n\tQ: What is Finley?\n\t\t     A:      Finley is your assistant.",
            [("What is Finley?", "Finley is your assistant.")],
            None,
        ),
        # Multi-line answer
        (
            "Q: What does Finley do?\nA: Finley helps with:\n - booking\n - support",
            [("What does Finley do?", "Finley helps with:\n - booking\n - support")],
            None,
        ),
        # Multi-line question
        (
            "Q: What does\n Finley do?\nA: Finley helps with booking and support",
            [("What does\n Finley do?", "Finley helps with booking and support")],
            None,
        ),
        # Multiline quest and answer
        (
            "Q: What does\n Finley do?\nA: Finley helps with:\n - booking\n - support",
            [("What does\n Finley do?", "Finley helps with:\n - booking\n - support")],
            None,
        ),
        # Multiple QA pairs
        (
            "Q: Who?\nA: Finley.\n\nQ: What?\nA: Assistant.",
            [("Who?", "Finley."), ("What?", "Assistant.")],
            None,
        ),
        (
            "Q: What is Finley?\n"
            "A: Finley is your assistant.\n\n"
            "Q: How do I use Finley\n"
            "   in my daily routine?\n"
            "A: You can talk to Finley\n"
            "   via your favorite messaging app.\n\n"
            "Q: Can Finley help with scheduling?\n"
            "A: Yes, Finley can:\n"
            " - Schedule meetings\n"
            " - Send reminders\n\n"
            "Q: What languages does Finley support?\n"
            "A: English, Deutsch, Français, Español, and more.\n\n"
            "Q: Is my data safe with Finley?\n"
            "A: Absolutely! Finley uses end-to-end encryption & complies with GDPR.",
            [
                ("What is Finley?", "Finley is your assistant."),
                (
                    "How do I use Finley\n   in my daily routine?",
                    "You can talk to Finley\n   via your favorite messaging app.",
                ),
                (
                    "Can Finley help with scheduling?",
                    "Yes, Finley can:\n - Schedule meetings\n - Send reminders",
                ),
                (
                    "What languages does Finley support?",
                    "English, Deutsch, Français, Español, and more.",
                ),
                (
                    "Is my data safe with Finley?",
                    "Absolutely! Finley uses end-to-end encryption "
                    "& complies with GDPR.",
                ),
            ],
            None,
        ),
        # Special characters
        (
            "Q: Wie heißt Finley?\nA: Finley ist Ihr Assistent für die Fin@X~App!",
            [("Wie heißt Finley?", "Finley ist Ihr Assistent für die Fin@X~App!")],
            None,
        ),
        # Invalid formats
        (
            "Q: Where is Finley?",
            [],
            "faq_parser.format_faq_documents.invalid_chunk_skipped",
        ),
        (
            "A: Finley helps with booking and support",
            [],
            "faq_parser.format_faq_documents.invalid_chunk_skipped",
        ),
        (
            "Question: What is Finley?\nAnswer: Finley is your assistant.",
            [],
            "faq_parser.format_faq_documents.invalid_chunk_skipped",
        ),
        (
            "q: What is Finley?\na: Finley is your assistant.",
            [],
            "faq_parser.format_faq_documents.invalid_chunk_skipped",
        ),
        (
            "Q: What is Finley? A: Finley is your assistant.",
            [],
            "faq_parser.format_faq_documents.invalid_chunk_skipped",
        ),
    ],
)
def test_format_faq_documents_regex(
    input_text: str, expected_pairs: List[Tuple], expected_warning_event: str
):
    # Given
    input_docs = [Document(page_content=input_text, metadata={})]

    # When
    with structlog.testing.capture_logs() as caplog:
        parsed = _format_faq_documents(input_docs)

        if expected_warning_event is not None:
            logs = filter_logs(
                caplog,
                expected_warning_event,
                "warning",
            )

    parsed_faq_pairs = [(doc.page_content, doc.metadata["answer"]) for doc in parsed]

    # Then
    assert parsed_faq_pairs == expected_pairs
    if expected_warning_event is not None:
        assert len(logs) == 1


@pytest.mark.parametrize(
    "input_text, expected_warnings",
    [
        (
            # No duplicates
            "Q: Who is Finley?\nA: Finley is your smart assistant for the FinX App.\n\n"
            "Q: What is FinX?\nA: FinX is a secure productivity platform.",
            [],
        ),
        (
            # Exact duplicate QA pair
            "Q: Who is Finley?\nA: Finley is your smart assistant for the FinX App.\n\n"
            "Q: Who is Finley?\nA: Finley is your smart assistant for the FinX App.",
            ["faq_parser.duplicate_qa_pair_found"],
        ),
        (
            # Same question, different answers
            "Q: Who is Finley?\nA: Finley is your smart assistant for the FinX App.\n\n"
            "Q: Who is Finley?\nA: A digital assistant built by FinX.",
            ["faq_parser.inconsistent_answer"],
        ),
        (
            # Both: one duplicate and one inconsistent answer
            "Q: Who is Finley?\nA: Finley is your smart assistant for the FinX App.\n\n"
            "Q: Who is Finley?\nA: Finley is your smart assistant for the FinX App.\n\n"
            "Q: Who is Finley?\nA: A digital assistant built by FinX.",
            ["faq_parser.duplicate_qa_pair_found", "faq_parser.inconsistent_answer"],
        ),
    ],
)
def test_format_faq_documents_duplicate_warnings(
    input_text: str, expected_warnings: List[str]
):
    # Given
    input_docs = [Document(page_content=input_text, metadata={})]

    # When
    with structlog.testing.capture_logs() as caplog:
        _format_faq_documents(input_docs)
        logs = []
        for expected_warning in expected_warnings:
            logs.extend(
                filter_logs(
                    caplog,
                    expected_warning,
                    "warning",
                )
            )

    # Then
    for expected_warning in expected_warnings:
        assert any(log["event"] == expected_warning for log in logs)

    if not expected_warnings:
        unexpected = [log["event"] for log in logs if log["level"] == "warning"]
        assert not unexpected

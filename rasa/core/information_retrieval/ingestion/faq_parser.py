"""Utilities for parsing FAQ-style documents (Q/A pairs) used in extractive search."""

import re
from collections import defaultdict
from typing import TYPE_CHECKING, List

import structlog

from rasa.shared.constants import (
    DOCUMENT_TYPE_FAQ,
    FAQ_DOCUMENT_ENTRY_SEPARATOR,
    FAQ_DOCUMENT_LINE_SEPARATOR,
    FAQ_DOCUMENT_METADATA_ANSWER,
    FAQ_DOCUMENT_METADATA_TITLE,
    FAQ_DOCUMENT_METADATA_TYPE,
    FAQ_INPUT_DATA_ANSWER_LINE_PREFIX,
    FAQ_INPUT_DATA_QUESTION_LINE_PREFIX,
)

if TYPE_CHECKING:
    from langchain.schema import Document

_FAQ_PAIR_PATTERN = re.compile(
    rf"{re.escape(FAQ_INPUT_DATA_QUESTION_LINE_PREFIX)}\s*"
    rf"(?P<question>.*?)\s*{FAQ_DOCUMENT_LINE_SEPARATOR}\s*"
    rf"{re.escape(FAQ_INPUT_DATA_ANSWER_LINE_PREFIX)}\s*"
    rf"(?P<answer>.*)",
    re.DOTALL,
)


structlogger = structlog.get_logger()


def _format_faq_documents(documents: List["Document"]) -> List["Document"]:
    """Splits each loaded file into individual FAQs.

    Args:
        documents: Documents representing whole files containing FAQs.

    Returns:
        List of Document objects, each containing a separate FAQ.

    Examples:
        An example of a file containing FAQs:

        Q: Who is Finley?
        A: Finley is your smart assistant for the FinX App. You can add him to your
           favorite messenger and tell him what you need help with.

        Q: How does Finley work?
        A: Finley is powered by the latest chatbot technology leveraging a unique
           interplay of large language models and secure logic.

    More details in documentation: https://rasa.com/docs/reference/config/policies/extractive-search/
    """
    structured_faqs = []
    from langchain.schema import Document

    for document in documents:
        chunks = document.page_content.strip().split(FAQ_DOCUMENT_ENTRY_SEPARATOR)

        for chunk in chunks:
            match = _FAQ_PAIR_PATTERN.match(chunk.strip())

            if not match:
                structlogger.warning(
                    "faq_parser.format_faq_documents.invalid_chunk_skipped",
                    event_info=(
                        "Chunk does not match expected QA format. "
                        "Please refer to the documentation: "
                        "https://rasa.com/docs/reference/config/"
                        "policies/extractive-search/"
                    ),
                    chunk_preview=chunk[:100],
                )
                continue

            question = match.group("question").strip()
            answer = match.group("answer").strip()
            title = _sanitize_title(question)

            formatted_document = Document(
                page_content=question,
                metadata={
                    FAQ_DOCUMENT_METADATA_TITLE: title,
                    FAQ_DOCUMENT_METADATA_TYPE: DOCUMENT_TYPE_FAQ,
                    FAQ_DOCUMENT_METADATA_ANSWER: answer,
                },
            )

            structured_faqs.append(formatted_document)

            structlogger.debug(
                "faq_parser.format_faq_documents.parsed_chunk",
                event_info="Parsed chunk.",
                title=title,
                question=question,
                answer=answer,
                parsed_chunk_preview=chunk[:100],
            )

    structlogger.debug(
        "faq_parser.format_faq_documents.parsed_chunks",
        event_info=(
            f"Retrieved {len(structured_faqs)} FAQ pair(s)"
            f"from {len(documents)} document(s)."
        ),
        num_structured_faqs=len(structured_faqs),
        num_documents=len(documents),
    )
    _check_and_parsed_faq_documents_for_duplicates(structured_faqs)
    return structured_faqs


def _sanitize_title(title: str) -> str:
    title = title.lower()
    # Remove all whitespaces with "_"
    title = re.sub(r"\s+", "_", title)
    # Remove all non alpha-numeric characters
    title = re.sub(r"[^\w]", "", title)
    # Collapse multiple "_"
    title = re.sub(r"_+", "_", title)
    # Clean up edges
    return title.strip("_")


def _check_and_parsed_faq_documents_for_duplicates(documents: List["Document"]) -> None:
    seen_qa_pairs = set()
    seen_questions: defaultdict = defaultdict(list)

    for doc in documents:
        question = doc.page_content.strip()
        answer = doc.metadata.get(FAQ_DOCUMENT_METADATA_ANSWER, "").strip()

        if not question or not answer:
            continue

        if (question, answer) in seen_qa_pairs:
            structlogger.warning(
                "faq_parser.duplicate_qa_pair_found",
                event_info="Duplicate QA pair found.",
                question=question,
                answer_preview=answer,
            )
            continue

        if question in seen_questions and seen_questions[question] != answer:
            structlogger.warning(
                "faq_parser.inconsistent_answer",
                event_info="Duplicate question with different answer found.",
                question=question,
                previous_answers=seen_questions[question],
                new_answer=answer,
            )

        seen_qa_pairs.add((question, answer))
        seen_questions[question].append(answer)

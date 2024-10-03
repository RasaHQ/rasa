from pathlib import Path
from typing import Any, Dict, List, Optional, Text

import numpy as np
import structlog
from langchain.schema.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from rasa.shared.constants import REQUIRED_SLOTS_KEY
from rasa.shared.core.domain import KEY_RESPONSES_TEXT, Domain
from rasa.shared.utils.llm import AI

structlogger = structlog.get_logger()


def persist_faiss_vector_store(path: Path, vecor_store: Optional[FAISS]) -> None:
    """Persists the given vector store to the given path.

    Args:
        path: path to persist the vector store to
        vecor_store: vector store to persist
    """
    if vecor_store:
        vecor_store.save_local(str(path))


def load_faiss_vector_store(path: Path, embedder: Embeddings) -> Optional[FAISS]:
    """Loads the vector store from the given path.

    Args:
        path: path to load the vector store from
        embedder: embedder to use for the vector store

    Returns:
        The loaded vector store or None if the path does not exist.
    """
    if path.exists():
        return FAISS.load_local(
            str(path), embedder, allow_dangerous_deserialization=True
        )
    else:
        return None


def extract_participant_messages_from_transcript(
    transcript: Text, participant: Text = AI
) -> List[Text]:
    r"""Extracts repsonses from the transcript.

    Args:
        transcript: transcript text to process
        participant: participant prefix (AI or Human, AI by default)

    Example:
        >>> extract_participant_messages_from_transcript("USER: Hello\\nAI: Hi there!")
        ["Hi there!"]

    Returns:
        List of extracted responses.
    """
    prefix = participant + ": "
    return [
        line[len(prefix) :]
        for line in transcript.split("\n")
        if line.startswith(prefix)
    ]


def extract_ai_response_examples(
    responses: Dict[Text, List[Dict[Text, Any]]],
) -> List[str]:
    """Extracts the responses from the domain.

    Only responses that have a text field are extracted.

    Example:
        >>> responses = {
        ...     "utter_greet": [
        ...         {"text": "hello"},
        ...     ],
        ...     "utter_goodbye": [
        ...         {"text": "goodbye"},
        ...         {"text": ""},
        ...     ],
        ... }
        >>> extract_ai_response_examples(responses)
        ["hello", "goodbye"]

    Args:
        responses: the responses from the domain

    Returns:
    A list of responses.
    """
    example_ai_responses = []

    for variations in responses.values():
        if not variations:
            continue
        for variation in variations:
            if response_text := variation.get(KEY_RESPONSES_TEXT):
                example_ai_responses.append(response_text)

    return example_ai_responses


def response_for_template(
    template_name: str, responses: Dict[Text, List[Dict[Text, Any]]]
) -> Optional[str]:
    """Returns an interaction for the given template name from the responses.

    Args:
        template_name: the name of the template
        responses: the responses from the domain

    Returns:
    The response text or None if no response was found.
    """
    rsps = responses.get(template_name, [])
    usuable_responses = [r for r in rsps if r.get(KEY_RESPONSES_TEXT)]
    if usuable_responses:
        selected_response = np.random.choice(usuable_responses)  # type: ignore
        return selected_response.get(KEY_RESPONSES_TEXT)
    else:
        structlogger.warning(
            "response_template.not_found",
            template_name=template_name,
        )
        return None


def form_utterances_to_action(domain: Domain) -> Dict[str, str]:
    """Returns a dictionary of form utterances to their form name.

    Args:
        domain: the domain of the assistant


    Returns:
    A dictionary of form utterances to their form name.
    """
    form_slots = [
        (slot_name, form_name)
        for form_name, props in domain.forms.items()
        for slot_name in props.get(REQUIRED_SLOTS_KEY, [])
    ]

    return {f"utter_ask_{slot_name}": form_name for slot_name, form_name in form_slots}

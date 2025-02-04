import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import jsonschema
import numpy as np
import structlog
from pydantic import BaseModel, ConfigDict

from rasa.core.constants import (
    UTTER_SOURCE_METADATA_KEY,
)
from rasa.e2e_test.constants import (
    KEY_JUSTIFICATION,
    KEY_SCORE,
)
from rasa.e2e_test.e2e_config import LLMJudgeConfig
from rasa.shared.constants import MODEL_CONFIG_KEY, OPENAI_PROVIDER, PROVIDER_CONFIG_KEY
from rasa.shared.core.events import BotUttered
from rasa.shared.exceptions import RasaException
from rasa.shared.utils.llm import DEFAULT_OPENAI_EMBEDDING_MODEL_NAME, embedder_factory

if TYPE_CHECKING:
    from rasa.shared.core.events import Event


structlogger = structlog.get_logger()


DEFAULT_EMBEDDINGS_CONFIG = {
    PROVIDER_CONFIG_KEY: OPENAI_PROVIDER,
    MODEL_CONFIG_KEY: DEFAULT_OPENAI_EMBEDDING_MODEL_NAME,
}

ELIGIBLE_UTTER_SOURCE_METADATA = [
    "EnterpriseSearchPolicy",
    "ContextualResponseRephraser",
    "IntentlessPolicy",
]

GROUNDEDNESS_JSON_SUB_SCHEMA = {
    "properties": {
        "statements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "statement": {"type": "string"},
                    "score": {"type": "number"},
                    "justification": {"type": "string"},
                },
                "required": ["statement", "score", "justification"],
            },
        }
    },
    "required": ["statements"],
}

ANSWER_RELEVANCE_JSON_SUB_SCHEMA = {
    "properties": {
        "question_variations": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["question_variations"],
}

LLM_JUDGE_OUTPUT_JSON_SCHEMA = {
    "type": "object",
    "oneOf": [
        GROUNDEDNESS_JSON_SUB_SCHEMA,
        ANSWER_RELEVANCE_JSON_SUB_SCHEMA,
    ],
}


class ScoreInputs(BaseModel):
    """Input data for the score calculation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    threshold: float
    matching_event: BotUttered
    user_question: str
    llm_judge_config: LLMJudgeConfig


def _calculate_similarity(
    user_question: str,
    generated_questions: List[str],
    llm_judge_config: LLMJudgeConfig,
) -> np.ndarray:
    """Calculate the cosine similarity between the user question and the generated questions."""  # noqa: E501
    embedding_client = embedder_factory(
        llm_judge_config.embeddings_config_as_dict, DEFAULT_EMBEDDINGS_CONFIG
    )

    user_question_embedding_response = embedding_client.embed([user_question])
    question_vector = np.asarray(user_question_embedding_response.data[0]).reshape(
        1, -1
    )

    gen_questions_embedding_response = embedding_client.embed(generated_questions)
    generated_questions_vectors = np.asarray(
        gen_questions_embedding_response.data
    ).reshape(len(generated_questions), -1)

    # calculate norm
    question_vector_norm = np.linalg.norm(question_vector, axis=1)
    generated_questions_vectors_norm = np.linalg.norm(
        generated_questions_vectors, axis=1
    )
    norm = generated_questions_vectors_norm * question_vector_norm
    norm = np.maximum(norm, 1e-10)

    # calculate the dot product
    dot_product = np.dot(generated_questions_vectors, question_vector.T).reshape(-1)

    # calculate and return cosine similarity
    return dot_product / norm


def calculate_relevance_score(
    processed_output: List[Union[str, Dict[str, Any]]],
    score_inputs: ScoreInputs,
) -> Tuple[float, str]:
    """Calculate the score based on the LLM response."""
    user_question = score_inputs.user_question
    llm_judge_config = score_inputs.llm_judge_config

    generated_questions = [output for output in processed_output]
    if all(not question for question in generated_questions):
        score = 0.0
        error_justification = "No relevant questions were generated"
        return score, error_justification

    cosine_sim = _calculate_similarity(
        user_question, generated_questions, llm_judge_config
    )

    score = cosine_sim.mean()

    if score < score_inputs.threshold:
        error_justifications = [
            f"Question '{generated_questions[i]}' "
            f"has a cosine similarity score of '{round(cosine_sim[i], 2)}' "
            f"with the user question '{user_question}'"
            for i in range(len(generated_questions))
        ]
        error_justification = ", ".join(error_justifications)
        return score, error_justification

    return score, ""


def calculate_groundedness_score(
    processed_output: List[Any],
    score_inputs: ScoreInputs,
) -> Tuple[float, str]:
    """Calculate the score based on the LLM response."""
    matching_event = score_inputs.matching_event

    total_statements = len(processed_output)
    correct_statements = sum([output.get(KEY_SCORE, 0) for output in processed_output])
    score = correct_statements / total_statements

    structlogger.debug(
        "generative_response_is_grounded_assertion.run_results",
        matching_event=repr(matching_event),
        score=score,
        justification=f"There were {correct_statements} correct statements "
        f"out of {total_statements} total extracted statements.",
    )

    if score < score_inputs.threshold:
        justifications = [
            output.get(KEY_JUSTIFICATION, "")
            for output in processed_output
            if output.get(KEY_SCORE, 0) == 0
        ]
        justification = ", ".join(justifications).replace(".", "")

        error_justification = (
            f"There were {total_statements - correct_statements} "
            f"incorrect statements out of {total_statements} total "
            f"extracted statements. The justifications for "
            f"these statements include: {justification}"
        )

        return score, error_justification

    return score, ""


def _find_matching_generative_events(
    turn_events: List["Event"], utter_source: Optional[str]
) -> List[BotUttered]:
    """Find the matching events for the generative response assertions."""
    if utter_source is None:
        return [
            event
            for event in turn_events
            if isinstance(event, BotUttered)
            and event.metadata.get(UTTER_SOURCE_METADATA_KEY)
            in ELIGIBLE_UTTER_SOURCE_METADATA
        ]

    return [
        event
        for event in turn_events
        if isinstance(event, BotUttered)
        and event.metadata.get(UTTER_SOURCE_METADATA_KEY) == utter_source
    ]


def _parse_llm_output(llm_response: str, bot_message: str) -> Dict[str, Any]:
    """Parse the LLM output."""
    llm_output = (
        llm_response.replace("```json\n", "").replace("```", "").replace("\n", "")
    )
    try:
        parsed_llm_output = json.loads(llm_output)
    except json.JSONDecodeError as exc:
        raise RasaException(
            f"Failed to parse the LLM Judge response '{llm_output}' for "
            f"the generative bot message '{bot_message}': {exc}"
        )

    return parsed_llm_output


def _validate_parsed_llm_output(
    parsed_llm_output: Dict[str, Any], bot_message: str
) -> None:
    """Validate the parsed LLM output."""
    try:
        jsonschema.validate(parsed_llm_output, LLM_JUDGE_OUTPUT_JSON_SCHEMA)
    except jsonschema.ValidationError as exc:
        raise RasaException(
            f"Failed to validate the LLM Judge json response "
            f"'{parsed_llm_output}' for the generative bot message "
            f"'{bot_message}'. Error: {exc}"
        )

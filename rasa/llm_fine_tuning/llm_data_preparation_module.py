from dataclasses import dataclass
from typing import List

from rasa.llm_fine_tuning.conversations import Conversation


@dataclass
class LLMDataExample:
    prompt: str
    output: str
    original_test_name: str
    original_user_utterance: str
    rephrased_user_utterance: str


def convert_to_fine_tuning_data(
    conversations: List[Conversation], output_dir: str
) -> List[LLMDataExample]:
    # TODO placeholder

    return []

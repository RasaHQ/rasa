from typing import List, Dict, Any, Tuple

from rasa.llm_fine_tuning.conversations import Conversation


def create_paraphrased_conversations(
    conversations: List[Conversation],
    rephrase_config_path: str,
    num_rephrases: int,
    output_dir: str,
) -> Tuple[List[Conversation], Dict[str, Any]]:
    # TODO placeholder

    # TODO validate config, use proper default values
    config = {"llm": {"model_name": "gpt-4"}}

    return [], config

from dataclasses import dataclass
from typing import List, Tuple

from rasa.llm_fine_tuning.llm_data_preparation_module import LLMDataExample


@dataclass
class AlpacaDataExample:
    instruction: str
    output: str


def split_llm_fine_tuning_data(
    fine_tuning_data: List[LLMDataExample],
    train_frac: float,
    output_format: str,
    output_dir: str,
) -> Tuple[List[AlpacaDataExample], List[AlpacaDataExample]]:
    # TODO placeholder

    return [], []

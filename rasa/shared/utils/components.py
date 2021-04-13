from typing import Optional, List

import rasa.shared
from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.model import Trainer
from rasa.nlu.tokenizers.tokenizer import Tokenizer


def get_tokenizer_from_nlu_config(
    nlu_config: Optional[RasaNLUModelConfig] = None,
) -> Optional[Tokenizer]:
    """Extracts the first Tokenizer in the NLU pipeline.

    Args:
        nlu_config: NLU Config.

    Returns:
        The first Tokenizer in the NLU pipeline, if any.
    """
    if not nlu_config:
        return None

    pipeline: List[Component] = Trainer(nlu_config, skip_validation=True).pipeline
    tokenizer: Optional[Tokenizer] = None
    for component in pipeline:
        if isinstance(component, Tokenizer):
            if tokenizer:
                rasa.shared.utils.io.raise_warning(
                    "The pipeline contains more than one tokenizer. "
                    "Only the first tokenizer will be used.",
                )
            tokenizer = component

    return tokenizer

from typing import Dict, Text, Any

import rasa.shared.utils.io
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.shared.constants import DOCS_URL_MIGRATION_GUIDE


class LanguageModelTokenizer(WhitespaceTokenizer):
    """
    This tokenizer is deprecated and will be removed in the future.

    The HFTransformersNLP component now sets the tokens
    for dense featurizable attributes of each message object.
    """

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:
        super().__init__(component_config)
        rasa.shared.utils.io.raise_warning(
            f"'{self.__class__.__name__}' is deprecated and "
            f"will be removed in the future. "
            f"It is recommended to use the '{WhitespaceTokenizer.__name__}' instead.",
            category=DeprecationWarning,
            docs=DOCS_URL_MIGRATION_GUIDE,
        )

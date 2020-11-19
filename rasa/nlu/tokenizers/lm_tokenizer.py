from typing import Dict, Text, Any, Optional

import rasa.shared.utils.io
from rasa.nlu.tokenizers.tokenizer import Tokenizer
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.shared.core.domain import Domain


class LanguageModelTokenizer(WhitespaceTokenizer):
    """This tokenizer is deprecated and will be removed in the future.

    Use the LanguageModelFeaturizer with any other Tokenizer instead.
    """

    def __init__(
        self, component_config: Dict[Text, Any] = None, domain: Optional[Domain] = None
    ) -> None:
        """Initializes LanguageModelTokenizer for tokenization.

        Args:
            component_config: Configuration for the component.
        """
        super().__init__(component_config, domain)
        rasa.shared.utils.io.raise_warning(
            f"'{self.__class__.__name__}' is deprecated and "
            f"will be removed in the future. "
            f"It is recommended to use the '{WhitespaceTokenizer.__name__}' or "
            f"another {Tokenizer.__name__} instead.",
            category=DeprecationWarning,
        )

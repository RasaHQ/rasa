from typing import Dict, Text, Any

import rasa.shared.utils.io
from rasa.nlu.tokenizers.tokenizer import Tokenizer
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer


class ConveRTTokenizer(WhitespaceTokenizer):
    """This tokenizer is deprecated and will be removed in the future.

    The ConveRTFeaturizer component now sets the sub-token information
    for dense featurizable attributes of each message object.
    """

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:
        """Initializes ConveRTTokenizer with the ConveRT model.

        Args:
            component_config: Configuration for the component.
        """
        super().__init__(component_config)
        rasa.shared.utils.io.raise_warning(
            f"'{self.__class__.__name__}' is deprecated and "
            f"will be removed in the future. "
            f"It is recommended to use the '{WhitespaceTokenizer.__name__}' or "
            f"another {Tokenizer.__name__} instead.",
            category=DeprecationWarning,
        )

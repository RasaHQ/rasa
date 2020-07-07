from typing import Any, Dict, List, Optional, Text

from rasa.core.utils import get_dict_hash
from rasa.nlu.constants import NUMBER_OF_SUB_TOKENS
from rasa.nlu.model import Metadata
from rasa.nlu.tokenizers.tokenizer import Token
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.nlu.training_data import Message
from rasa.utils import common
import rasa.utils.train_utils as train_utils
import tensorflow as tf


TF_HUB_MODULE_URL = "http://models.poly-ai.com/convert/v1/model.tar.gz"


class ConveRTTokenizer(WhitespaceTokenizer):
    """Tokenizer using ConveRT model.
    Loads the ConveRT(https://github.com/PolyAI-LDN/polyai-models#convert)
    model from TFHub and computes sub-word tokens for dense
    featurizable attributes of each message object.
    """

    defaults = {
        # Flag to check whether to split intents
        "intent_tokenization_flag": False,
        # Symbol on which intent should be split
        "intent_split_symbol": "_",
        # Regular expression to detect tokens
        "token_pattern": None,
    }

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:
        """Construct a new tokenizer using the WhitespaceTokenizer framework."""

        super().__init__(component_config)

        self.module = train_utils.load_tf_hub_model(TF_HUB_MODULE_URL)

        self.tokenize_signature = self.module.signatures["tokenize"]

    @classmethod
    def cache_key(
        cls, component_meta: Dict[Text, Any], model_metadata: Metadata
    ) -> Optional[Text]:
        _config = common.update_existing_keys(cls.defaults, component_meta)
        return f"{cls.name}-{get_dict_hash(_config)}"

    def provide_context(self) -> Dict[Text, Any]:
        return {"tf_hub_module": self.module}

    def _tokenize(self, sentence: Text) -> Any:

        return self.tokenize_signature(tf.convert_to_tensor([sentence]))[
            "default"
        ].numpy()

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        """Tokenize the text using the ConveRT model.
        ConveRT adds a special char in front of (some) words and splits words into
        sub-words. To ensure the entity start and end values matches the token values,
        tokenize the text first using the whitespace tokenizer. If individual tokens
        are split up into multiple tokens, add this information to the
        respected tokens.
        """

        # perform whitespace tokenization
        tokens_in = super().tokenize(message, attribute)

        tokens_out = []

        for token in tokens_in:
            # use ConveRT model to tokenize the text
            split_token_strings = self._tokenize(token.text)[0]

            # clean tokens (remove special chars and empty tokens)
            split_token_strings = self._clean_tokens(split_token_strings)

            token.set(NUMBER_OF_SUB_TOKENS, len(split_token_strings))

            tokens_out.append(token)

        return tokens_out

    @staticmethod
    def _clean_tokens(tokens: List[bytes]) -> List[Text]:
        """Encode tokens and remove special char added by ConveRT."""

        tokens = [string.decode("utf-8").replace("﹏", "") for string in tokens]
        return [string for string in tokens if string]

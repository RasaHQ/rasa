from typing import Any, Dict, List, Text

from rasa.nlu.tokenizers.tokenizer import Token
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.nlu.training_data import Message
from rasa.nlu.constants import MESSAGE_ATTRIBUTES, TOKENS_NAMES
import tensorflow as tf


class ConveRTTokenizer(WhitespaceTokenizer):

    provides = [TOKENS_NAMES[attribute] for attribute in MESSAGE_ATTRIBUTES]

    defaults = {
        # Flag to check whether to split intents
        "intent_tokenization_flag": False,
        # Symbol on which intent should be split
        "intent_split_symbol": "_",
        # Text will be tokenized with case sensitive as default
        "case_sensitive": True,
    }

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:
        """Construct a new tokenizer using the WhitespaceTokenizer framework."""

        super().__init__(component_config)

        self._load_tokenizer_params()

    def _load_tokenizer_params(self):

        # needed to load the ConveRT model
        import tensorflow_text
        import tensorflow_hub as tfhub

        model_url = "http://models.poly-ai.com/convert/v1/model.tar.gz"

        self.module = tfhub.load(model_url)

        self.tokenize_signature = self.module.signatures["tokenize"]

    def _tokenize(self, sentence: Text) -> Any:

        return self.tokenize_signature(tf.convert_to_tensor([sentence]))[
            "default"
        ].numpy()

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        """Tokenize the text using the ConveRT model.

        ConveRT adds a special char in front of (some) words and splits words into
        sub-words. To ensure the entity start and end values matches the token values,
        tokenize the text first using the whitespace tokenizer. If individual tokens
        are split up into multiple tokens, we make sure that the start end end value
        of the first and last respective tokens stay the same.
        """

        # perform whitespace tokenization
        tokens_in = super().tokenize(message, attribute)

        tokens_out = []

        for token in tokens_in:
            token_start, token_end, token_text = token.start, token.end, token.text

            # use ConveRT model to tokenize the text
            split_token_strings = self._tokenize(token_text)[0]

            # clean tokens (remove special chars and empty tokens)
            split_token_strings = self._clean_tokens(split_token_strings)

            _aligned_tokens = self._align_tokens(
                split_token_strings, token_end, token_start
            )
            tokens_out += _aligned_tokens

        return tokens_out

    def _clean_tokens(self, tokens: List[bytes]):
        """Encode tokens and remove special char added by ConveRT."""

        tokens = [string.decode("utf-8").replace("﹏", "") for string in tokens]
        return [string for string in tokens if string]

    def _align_tokens(self, tokens_in: List[Text], token_end: int, token_start: int):
        """Align sub-tokens of ConveRT with tokens return by the WhitespaceTokenizer.

        As ConveRT might split a single word into multiple tokens, we need to make
        sure that the start and end value of first and last sub-token matches the
        start and end value of the token return by the WhitespaceTokenizer as the
        entities are using those start and end values.
        """

        tokens_out = []

        current_token_offset = token_start

        for index, string in enumerate(tokens_in):
            if index == 0:
                if index == len(tokens_in) - 1:
                    s_token_end = token_end
                else:
                    s_token_end = current_token_offset + len(string)
                tokens_out.append(Token(string, token_start, end=s_token_end))
            elif index == len(tokens_in) - 1:
                tokens_out.append(Token(string, current_token_offset, end=token_end))
            else:
                tokens_out.append(
                    Token(
                        string,
                        current_token_offset,
                        end=current_token_offset + len(string),
                    )
                )

            current_token_offset += len(string)

        return tokens_out

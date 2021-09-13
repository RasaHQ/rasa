from __future__ import annotations
from typing import List, Text, Dict, Any

from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.tokenizers.tokenizer import Token, TokenizerGraphComponent
from rasa.shared.nlu.training_data.message import Message

from rasa.shared.utils.io import DEFAULT_ENCODING
from rasa.nlu.tokenizers._mitie_tokenizer import MitieTokenizer

# This is a workaround around until we have all components migrated to `GraphComponent`.
MitieTokenizer = MitieTokenizer


class MitieTokenizerGraphComponent(TokenizerGraphComponent):
    """Tokenizes messages using the `mitie` library.."""

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns default config (see parent class for full docstring)."""
        return {
            # Flag to check whether to split intents
            "intent_tokenization_flag": False,
            # Symbol on which intent should be split
            "intent_split_symbol": "_",
            # Regular expression to detect tokens
            "token_pattern": None,
        }

    @staticmethod
    def required_packages() -> List[Text]:
        """Any extra python dependencies required for this component to run."""
        return ["mitie"]

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> MitieTokenizerGraphComponent:
        """Creates a new component (see parent class for full docstring)."""
        return cls(config)

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        """Tokenizes the text of the provided attribute of the incoming message."""
        import mitie

        text = message.get(attribute)

        encoded_sentence = text.encode(DEFAULT_ENCODING)
        tokenized = mitie.tokenize_with_offsets(encoded_sentence)
        tokens = [
            self._token_from_offset(token, offset, encoded_sentence)
            for token, offset in tokenized
        ]

        return self._apply_token_pattern(tokens)

    def _token_from_offset(
        self, text: bytes, offset: int, encoded_sentence: bytes
    ) -> Token:
        return Token(
            text.decode(DEFAULT_ENCODING),
            self._byte_to_char_offset(encoded_sentence, offset),
        )

    @staticmethod
    def _byte_to_char_offset(text: bytes, byte_offset: int) -> int:
        return len(text[:byte_offset].decode(DEFAULT_ENCODING))

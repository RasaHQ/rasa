import logging
from typing import Any, Dict, List, Text

from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.shared.nlu.training_data.message import Message

logger = logging.getLogger(__name__)


class HFTransformersTokenizer(Tokenizer):
    defaults = {
        # Pre-Trained weights to be loaded(string)
        "model_weights": "bert-base-cased",
        # an optional path to a specific directory to download
        # and cache the pre-trained model weights.
        "cache_dir": None,
    }

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:
        """HuaggingFace's Transformers based tokenizer."""
        from transformers import AutoTokenizer

        super().__init__(component_config)

        self.model_weights = self.component_config["model_weights"]
        self.cache_dir = self.component_config["cache_dir"]

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_weights, cache_dir=self.cache_dir, use_fast=True
        )

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["transformers"]

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        text = message.get(attribute)

        encoded_input = self.tokenizer(
            text, return_offsets_mapping=True, add_special_tokens=False
        )

        tokens_text_in = self.tokenizer.convert_ids_to_tokens(
            encoded_input["input_ids"]
        )

        tokens = []

        for token_text, position in zip(
            tokens_text_in, encoded_input["offset_mapping"]
        ):
            token = Token(token_text, position[0], position[1])
            tokens.append(token)

        return self._apply_token_pattern(tokens)

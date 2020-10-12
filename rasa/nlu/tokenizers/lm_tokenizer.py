from typing import Text, List, Any, Dict, Type

from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.nlu.components import Component
from rasa.nlu.utils.hugging_face.hf_transformers import HFTransformersNLP
from rasa.shared.nlu.training_data.message import Message

from rasa.nlu.constants import LANGUAGE_MODEL_DOCS, TOKENS


class LanguageModelTokenizer(Tokenizer):
    """Tokenizer using transformer based language models.

    Uses the output of HFTransformersNLP component to set the tokens
    for dense featurizable attributes of each message object.
    """

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return [HFTransformersNLP]

    defaults = {
        # Flag to check whether to split intents
        "intent_tokenization_flag": False,
        # Symbol on which intent should be split
        "intent_split_symbol": "_",
    }

    def get_doc(self, message: Message, attribute: Text) -> Dict[Text, Any]:
        return message.get(LANGUAGE_MODEL_DOCS[attribute])

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        doc = self.get_doc(message, attribute)

        return doc[TOKENS]

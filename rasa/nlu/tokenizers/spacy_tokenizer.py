import typing
from typing import Text, List

from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.nlu.training_data import Message

from rasa.nlu.constants import TOKENS_NAMES, SPACY_DOCS, DENSE_FEATURIZABLE_ATTRIBUTES

if typing.TYPE_CHECKING:
    from spacy.tokens.doc import Doc  # pytype: disable=import-error


class SpacyTokenizer(Tokenizer):

    provides = [TOKENS_NAMES[attribute] for attribute in DENSE_FEATURIZABLE_ATTRIBUTES]

    requires = [SPACY_DOCS[attribute] for attribute in DENSE_FEATURIZABLE_ATTRIBUTES]

    defaults = {
        # Flag to check whether to split intents
        "intent_tokenization_flag": False,
        # Symbol on which intent should be split
        "intent_split_symbol": "_",
    }

    def get_doc(self, message: Message, attribute: Text) -> "Doc":
        return message.get(SPACY_DOCS[attribute])

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        doc = self.get_doc(message, attribute)

        return [Token(t.text, t.idx, lemma=t.lemma_) for t in doc]

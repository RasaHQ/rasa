import typing
from typing import Dict, Text, List, Any, Optional

from rasa.nlu.tokenizers.tokenizer import Token, TokenizerGraphComponent
from rasa.shared.nlu.training_data.message import Message

from rasa.nlu.constants import SPACY_DOCS

from rasa.nlu.tokenizers._spacy_tokenizer import SpacyTokenizer

if typing.TYPE_CHECKING:
    from spacy.tokens.doc import Doc

# This is a workaround around until we have all components migrated to `GraphComponent`.
SpacyTokenizer = SpacyTokenizer

POS_TAG_KEY = "pos"


class SpacyTokenizerGraphComponent(TokenizerGraphComponent):
    """Tokenizer that uses SpaCy."""

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """The component's default config (see parent class for full docstring)."""
        return {
            # Flag to check whether to split intents
            "intent_tokenization_flag": False,
            # Symbol on which intent should be split
            "intent_split_symbol": "_",
            # Regular expression to detect tokens
            "token_pattern": None,
        }

    @classmethod
    def required_packages(cls) -> List[Text]:
        """Any extra python dependencies required for this component to run."""
        return ["spacy"]

    def _get_doc(self, message: Message, attribute: Text) -> Optional["Doc"]:
        return message.get(SPACY_DOCS[attribute])

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        """Tokenizes the text of the provided attribute of the incoming message."""
        doc = self._get_doc(message, attribute)
        if not doc:
            return []

        tokens = [
            Token(
                t.text, t.idx, lemma=t.lemma_, data={POS_TAG_KEY: self._tag_of_token(t)}
            )
            for t in doc
            if t.text and t.text.strip()
        ]

        return self._apply_token_pattern(tokens)

    @staticmethod
    def _tag_of_token(token: Any) -> Text:
        import spacy

        if spacy.about.__version__ > "2" and token._.has("tag"):
            return token._.get("tag")
        else:
            return token.tag_

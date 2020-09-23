from typing import List, Text

from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.shared.nlu.training_data.message import Message


class CharacterTokenizer(Tokenizer):
    # the following language should can be tokenized using the CharacterTokenizer
    supported_language_list = ["zh", "ja", "th"]

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        text = message.get(attribute)

        words = [i for i in text]

        return self._convert_words_to_tokens(words, text)

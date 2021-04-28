import rasa.nlu.config
import rasa.shared.utils.components
from rasa.nlu.constants import TOKENS_NAMES
from rasa.nlu.tokenizers.tokenizer import Tokenizer
from rasa.shared.nlu.constants import TEXT, INTENT
from rasa.shared.nlu.training_data.message import Message


def test_augmenter_create_tokenizer():
    tokenizer_config = rasa.nlu.config.load("data/test_nlu_paraphrasing/config.yml")

    tokenizer = rasa.shared.utils.components.get_tokenizer_from_nlu_config(
        tokenizer_config
    )

    assert isinstance(tokenizer, Tokenizer)

    # Test Config has a simple WhitespaceTokenizer
    expected_tokens = ["xxx", "yyy", "zzz"]
    message = Message(data={TEXT: "xxx yyy zzz", INTENT: "abc"})

    tokenizer.process(message)
    tokens = [token.text for token in message.get(TOKENS_NAMES[TEXT])]

    assert tokens == expected_tokens
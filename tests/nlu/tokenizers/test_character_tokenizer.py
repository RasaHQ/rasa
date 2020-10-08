import pytest

from rasa.nlu.components import UnsupportedLanguageError
from rasa.nlu.config import RasaNLUModelConfig
from rasa.shared.nlu.constants import TEXT
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.tokenizers.character_tokenizer import CharacterTokenizer


@pytest.mark.parametrize(
    "text, expected_tokens, expected_indices",
    [
        ("早上好", ["早", "上", "好"], [(0, 1), (1, 2), (2, 3)]),
        (
            "おはようございます",
            ["お", "は", "よ", "う", "ご", "ざ", "い", "ま", "す"],
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)],
        ),
        (
            "สวัสดีตอนเช้า",
            ["ส", "ว", "ั", "ส", "ด", "ี", "ต", "อ", "น", "เ", "ช", "้", "า"],
            [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (12, 13),
            ],
        ),
    ],
)
def test_whitespace(text, expected_tokens, expected_indices):

    tk = CharacterTokenizer()

    tokens = tk.tokenize(Message.build(text=text), attribute=TEXT)

    assert [t.text for t in tokens] == expected_tokens
    assert [t.start for t in tokens] == [i[0] for i in expected_indices]
    assert [t.end for t in tokens] == [i[1] for i in expected_indices]


@pytest.mark.parametrize("language, error", [("en", True), ("zh", False)])
def test_character_language_suuport(language, error, component_builder):
    config = RasaNLUModelConfig(
        {
            "language": language,
            "pipeline": [
                {"name": "rasa.nlu.tokenizers.character_tokenizer.CharacterTokenizer"}
            ],
        }
    )

    if error:
        with pytest.raises(UnsupportedLanguageError):
            component_builder.create_component(
                {"name": "rasa.nlu.tokenizers.character_tokenizer.CharacterTokenizer"},
                config,
            )
    else:
        component_builder.create_component(
            {"name": "rasa.nlu.tokenizers.character_tokenizer.CharacterTokenizer"},
            config,
        )

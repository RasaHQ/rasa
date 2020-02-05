import pytest

from rasa.nlu.training_data import Message, TrainingData
from rasa.nlu.constants import TEXT_ATTRIBUTE, INTENT_ATTRIBUTE, TOKENS_NAMES
from rasa.nlu.tokenizers.lm_tokenizer import LanguageModelTokenizer
from rasa.nlu.utils.hugging_face.hf_transformers import HFTransformersNLP


@pytest.mark.parametrize(
    "model_name, texts, expected_tokens, expected_indices",
    [
        (
            "bert",
            [
                "Good evening.",
                "you're",
                "r. n. b.",
                "rock & roll",
                "here is the sentence I want embeddings for.",
            ],
            [
                ["good", "evening"],
                ["you", "re"],
                ["r", "n", "b"],
                ["rock", "&", "roll"],
                [
                    "here",
                    "is",
                    "the",
                    "sentence",
                    "i",
                    "want",
                    "em",
                    "bed",
                    "ding",
                    "s",
                    "for",
                ],
            ],
            [
                [(0, 4), (5, 12)],
                [(0, 3), (4, 6)],
                [(0, 1), (3, 4), (6, 7)],
                [(0, 4), (5, 6), (7, 11)],
                [
                    (0, 4),
                    (5, 7),
                    (8, 11),
                    (12, 20),
                    (21, 22),
                    (23, 27),
                    (28, 30),
                    (30, 33),
                    (33, 37),
                    (37, 38),
                    (39, 42),
                ],
            ],
        ),
        (
            "gpt",
            [
                "Good evening.",
                "hello",
                "you're",
                "r. n. b.",
                "rock & roll",
                "here is the sentence I want embeddings for.",
            ],
            [
                ["good", "evening"],
                ["hello"],
                ["you", "re"],
                ["r", "n", "b"],
                ["rock", "&", "roll"],
                ["here", "is", "the", "sentence", "i", "want", "embe", "ddings", "for"],
            ],
            [
                [(0, 4), (5, 12)],
                [(0, 5)],
                [(0, 3), (4, 6)],
                [(0, 1), (3, 4), (6, 7)],
                [(0, 4), (5, 6), (7, 11)],
                [
                    (0, 4),
                    (5, 7),
                    (8, 11),
                    (12, 20),
                    (21, 22),
                    (23, 27),
                    (28, 32),
                    (32, 38),
                    (39, 42),
                ],
            ],
        ),
        (
            "gpt2",
            [
                "Good evening.",
                "hello",
                "you're",
                "r. n. b.",
                "rock & roll",
                "here is the sentence I want embeddings for.",
            ],
            [
                ["Good", "even", "ing"],
                ["hello"],
                ["you", "re"],
                ["r", "n", "b"],
                ["rock", "&", "roll"],
                [
                    "here",
                    "is",
                    "the",
                    "sent",
                    "ence",
                    "I",
                    "want",
                    "embed",
                    "d",
                    "ings",
                    "for",
                ],
            ],
            [
                [(0, 4), (5, 9), (9, 12)],
                [(0, 5)],
                [(0, 3), (4, 6)],
                [(0, 1), (3, 4), (6, 7)],
                [(0, 4), (5, 6), (7, 11)],
                [
                    (0, 4),
                    (5, 7),
                    (8, 11),
                    (12, 16),
                    (16, 20),
                    (21, 22),
                    (23, 27),
                    (28, 33),
                    (33, 34),
                    (34, 38),
                    (39, 42),
                ],
            ],
        ),
        (
            "xlnet",
            [
                "Good evening.",
                "hello",
                "you're",
                "r. n. b.",
                "rock & roll",
                "here is the sentence I want embeddings for.",
            ],
            [
                ["Good", "evening"],
                ["hello"],
                ["you", "re"],
                ["r", "n", "b"],
                ["rock", "&", "roll"],
                [
                    "here",
                    "is",
                    "the",
                    "sentence",
                    "I",
                    "want",
                    "embed",
                    "ding",
                    "s",
                    "for",
                ],
            ],
            [
                [(0, 4), (5, 12)],
                [(0, 5)],
                [(0, 3), (4, 6)],
                [(0, 1), (3, 4), (6, 7)],
                [(0, 4), (5, 6), (7, 11)],
                [
                    (0, 4),
                    (5, 7),
                    (8, 11),
                    (12, 20),
                    (21, 22),
                    (23, 27),
                    (28, 33),
                    (33, 37),
                    (37, 38),
                    (39, 42),
                ],
            ],
        ),
        (
            "distilbert",
            [
                "Good evening.",
                "you're",
                "r. n. b.",
                "rock & roll",
                "here is the sentence I want embeddings for.",
            ],
            [
                ["good", "evening"],
                ["you", "re"],
                ["r", "n", "b"],
                ["rock", "&", "roll"],
                [
                    "here",
                    "is",
                    "the",
                    "sentence",
                    "i",
                    "want",
                    "em",
                    "bed",
                    "ding",
                    "s",
                    "for",
                ],
            ],
            [
                [(0, 4), (5, 12)],
                [(0, 3), (4, 6)],
                [(0, 1), (3, 4), (6, 7)],
                [(0, 4), (5, 6), (7, 11)],
                [
                    (0, 4),
                    (5, 7),
                    (8, 11),
                    (12, 20),
                    (21, 22),
                    (23, 27),
                    (28, 30),
                    (30, 33),
                    (33, 37),
                    (37, 38),
                    (39, 42),
                ],
            ],
        ),
        (
            "roberta",
            [
                "Good evening.",
                "hello",
                "you're",
                "r. n. b.",
                "rock & roll",
                "here is the sentence I want embeddings for.",
            ],
            [
                ["Good", "even", "ing"],
                ["hello"],
                ["you", "re"],
                ["r", "n", "b"],
                ["rock", "&", "roll"],
                [
                    "here",
                    "is",
                    "the",
                    "sent",
                    "ence",
                    "I",
                    "want",
                    "embed",
                    "d",
                    "ings",
                    "for",
                ],
            ],
            [
                [(0, 4), (5, 9), (9, 12)],
                [(0, 5)],
                [(0, 3), (4, 6)],
                [(0, 1), (3, 4), (6, 7)],
                [(0, 4), (5, 6), (7, 11)],
                [
                    (0, 4),
                    (5, 7),
                    (8, 11),
                    (12, 16),
                    (16, 20),
                    (21, 22),
                    (23, 27),
                    (28, 33),
                    (33, 34),
                    (34, 38),
                    (39, 42),
                ],
            ],
        ),
    ],
)
def test_lm_tokenizer_edge_cases(model_name, texts, expected_tokens, expected_indices):

    transformers_config = {"model_name": model_name}

    transformers_nlp = HFTransformersNLP(transformers_config)
    lm_tokenizer = LanguageModelTokenizer()

    for text, gt_tokens, gt_indices in zip(texts, expected_tokens, expected_indices):

        message = Message.build(text=text)
        transformers_nlp.process(message)
        tokens = lm_tokenizer.tokenize(message, TEXT_ATTRIBUTE)

        assert [t.text for t in tokens] == gt_tokens
        assert [t.start for t in tokens] == [i[0] for i in gt_indices]
        assert [t.end for t in tokens] == [i[1] for i in gt_indices]


@pytest.mark.parametrize(
    "text, expected_tokens",
    [
        ("Forecast_for_LUNCH", ["Forecast_for_LUNCH"]),
        ("Forecast for LUNCH", ["Forecast for LUNCH"]),
        ("Forecast+for+LUNCH", ["Forecast", "for", "LUNCH"]),
    ],
)
def test_lm_tokenizer_custom_intent_symbol(text, expected_tokens):
    component_config = {"intent_tokenization_flag": True, "intent_split_symbol": "+"}

    transformers_config = {"model_name": "bert"}  # Test for one should be enough

    transformers_nlp = HFTransformersNLP(transformers_config)
    lm_tokenizer = LanguageModelTokenizer(component_config)

    message = Message(text)
    message.set(INTENT_ATTRIBUTE, text)

    td = TrainingData([message])

    transformers_nlp.train(td)
    lm_tokenizer.train(td)

    assert [
        t.text for t in message.get(TOKENS_NAMES[INTENT_ATTRIBUTE])
    ] == expected_tokens

import operator
from typing import Callable, Text

import pytest
from rasa.nlu.components import Component
from rasa.nlu.constants import TOKENS_NAMES
from rasa.shared.constants import LATEST_TRAINING_DATA_FORMAT_VERSION
from rasa.shared.exceptions import InvalidConfigException
from rasa.shared.nlu.training_data.message import Message
import rasa.shared.nlu.training_data.loading
import rasa.shared.utils.io
from rasa.shared.nlu.constants import (
    INTENT,
    TEXT,
    VOCABULARY,
)


def test_augmenter_create_tokenizer():
    from rasa.nlu.data_augmentation.augmenter import _create_tokenizer_from_config

    config_path = "data/test_nlu_paraphrasing/config.yml"

    tokenizer = _create_tokenizer_from_config(config_path=config_path)

    assert isinstance(tokenizer, Component)

    # Test Config has a simple WhitespaceTokenizer
    expected_tokens = ["xxx", "yyy", "zzz"]
    message = Message(data={TEXT: "xxx yyy zzz", INTENT: "abc"})

    tokenizer.process(message)
    tokens = [token.text for token in message.get(TOKENS_NAMES[TEXT])]

    assert tokens == expected_tokens


@pytest.mark.xfail(raises=InvalidConfigException)
def test_augmenter_create_tokenizer_empty_config():
    from rasa.nlu.data_augmentation.augmenter import _create_tokenizer_from_config

    config_path = "data/test_nlu_paraphrasing/empty_config.yml"

    # Should raise InvalidConfigException
    _ = _create_tokenizer_from_config(config_path=config_path)


@pytest.mark.parametrize(
    "intent_proportion, comparator_fn_1, comparator_fn_2",
    [
        (0.0, operator.le, operator.lt),
        (0.5, operator.ge, operator.le),
        (1.0, operator.ge, operator.le),
    ],
)
def test_augmenter_intent_collection(
    intent_proportion: float, comparator_fn_1: Callable, comparator_fn_2: Callable
):
    from rasa.nlu.data_augmentation.augmenter import (
        _collect_intents_for_data_augmentation,
    )

    report_file = "data/test_nlu_paraphrasing/nlu_classification_report.json"
    classification_report = rasa.shared.utils.io.read_json_file(report_file)
    nlu_training_data = rasa.shared.nlu.training_data.loading.load_data(
        "data/test_nlu_paraphrasing/nlu_train.yml"
    )

    intents_to_augment = _collect_intents_for_data_augmentation(
        nlu_training_data=nlu_training_data,
        intent_proportion=intent_proportion,
        classification_report_no_augmentation=classification_report,
    )

    num_intents = len(nlu_training_data.number_of_examples_per_intent)
    num_intents_to_augment_per_criterion = round(num_intents * intent_proportion)

    assert comparator_fn_1(
        len(intents_to_augment), num_intents_to_augment_per_criterion
    )
    assert comparator_fn_2(len(intents_to_augment), num_intents)


@pytest.mark.parametrize(
    "intent_to_augment, min_paraphrase_sim_score, max_paraphrase_sim_score, expected_len",
    [
        ("ask_transfer_charge", 0.98, 0.99, 0),
        ("check_earnings", 0.98, 0.99, 0),
        ("check_recipients", 0.98, 0.99, 0),
        ("ask_transfer_charge", 0.0, 0.99, 6),
        ("check_earnings", 0.0, 0.99, 18),
        ("check_recipients", 0.0, 0.99, 18),
        ("ask_transfer_charge", 0.0, 0.45, 2),
        ("check_earnings", 0.0, 0.45, 3),
        ("check_recipients", 0.0, 0.45, 8),
        ("ask_transfer_charge", 0.4, 0.5, 1),
        ("check_earnings", 0.4, 0.5, 5),
        ("check_recipients", 0.4, 0.5, 8),
    ],
)
def test_augmenter_paraphrase_pool_creation(
    intent_to_augment: Text,
    min_paraphrase_sim_score: float,
    max_paraphrase_sim_score: float,
    expected_len: int,
):
    from rasa.nlu.data_augmentation.augmenter import _create_paraphrase_pool

    paraphrases = rasa.shared.nlu.training_data.loading.load_data(
        "data/test_nlu_paraphrasing/paraphrases.yml"
    )

    pool = _create_paraphrase_pool(
        paraphrases=paraphrases,
        intents_to_augment={intent_to_augment},
        min_paraphrase_sim_score=min_paraphrase_sim_score,
        max_paraphrase_sim_score=max_paraphrase_sim_score,
    )
    assert len(pool.get(intent_to_augment, [])) == expected_len


@pytest.mark.parametrize(
    "augmentation_factor, intent, size_after_augmentation",
    [
        (0.0, "check_recipients", None),
        (0.0, "inform", None),
        (0.0, "search_transactions", None),
        (0.0, "check_balance", None),
        (0.0, "transfer_money", None),
        (0.1, "check_recipients", None),
        (0.1, "inform", None),
        (0.1, "search_transactions", None),
        (0.1, "check_balance", None),
        (0.1, "transfer_money", None),
        (0.5, "check_recipients", None),
        (0.5, "inform", 2),
        (0.5, "search_transactions", None),
        (0.5, "check_balance", None),
        (0.5, "transfer_money", None),
        (1.0, "check_recipients", 1),
        (1.0, "inform", 4),
        (1.0, "search_transactions", 1),
        (1.0, "check_balance", 1),
        (1.0, "transfer_money", 1),
        (2.0, "check_recipients", 2),
        (2.0, "inform", 8),
        (2.0, "search_transactions", 2),
        (2.0, "check_balance", 2),
        (2.0, "transfer_money", 2),
        (5.0, "check_recipients", 5),
        (5.0, "inform", 20),
        (5.0, "search_transactions", 5),
        (5.0, "check_balance", 5),
        (5.0, "transfer_money", 5),
        (-1.0, "check_recipients", None),
        (-1.0, "inform", None),
        (-1.0, "search_transactions", None),
        (-1.0, "check_balance", None),
        (-1.0, "transfer_money", None),
        (-666, "check_recipients", None),
        (-666, "inform", None),
        (-666, "search_transactions", None),
        (-666, "check_balance", None),
        (-666, "transfer_money", None),
    ],
)
def test_augmenter_augmentation_factor_resolution(
    augmentation_factor: float, intent: Text, size_after_augmentation: int
):
    from rasa.nlu.data_augmentation.augmenter import _resolve_augmentation_factor

    nlu_training_data = rasa.shared.nlu.training_data.loading.load_data(
        "data/test_nlu_paraphrasing/nlu_train.yml"
    )

    resolved_dict = _resolve_augmentation_factor(
        nlu_training_data=nlu_training_data, augmentation_factor=augmentation_factor
    )

    assert resolved_dict[intent] == size_after_augmentation


def test_augmenter_build_max_vocab_expansion_training_set():
    from rasa.nlu.data_augmentation.augmenter import (
        _create_augmented_training_data_max_vocab_expansion,
    )

    nlu_training_data = rasa.shared.nlu.training_data.loading.load_data(
        "data/test_nlu_paraphrasing/nlu_train.yml"
    )
    config_path = "data/test_nlu_paraphrasing/config.yml"

    intent_to_check = "check_earnings"
    three_new_words = "xxx yyy zzz"
    two_new_words = "account aaa bbb"
    one_new_word = "account deposited xxx"

    intents_to_augment = {intent_to_check}
    num_examples_for_intent = 1
    augmentation_factor = {intent_to_check: 2}
    should_have_num_examples_after_augmentation = (
        num_examples_for_intent + augmentation_factor[intent_to_check]
    )

    paraphrase_pool = {
        "check_earnings": [
            Message(
                data={
                    INTENT: intent_to_check,
                    TEXT: one_new_word,
                    VOCABULARY: set(one_new_word.split()),
                }
            ),
            Message(
                data={
                    INTENT: intent_to_check,
                    TEXT: three_new_words,
                    VOCABULARY: set(three_new_words.split()),
                }
            ),
            Message(
                data={
                    INTENT: intent_to_check,
                    TEXT: two_new_words,
                    VOCABULARY: set(two_new_words.split()),
                }
            ),
        ]
    }

    augmented_data_should_contain = {three_new_words: False, two_new_words: False}
    augmented_data_should_not_contain = {one_new_word: False}

    augmented_data = _create_augmented_training_data_max_vocab_expansion(
        nlu_training_data=nlu_training_data,
        paraphrase_pool=paraphrase_pool,
        intents_to_augment=intents_to_augment,
        augmentation_factor=augmentation_factor,
        config=config_path,
    )
    num_examples = 0
    for message in augmented_data.intent_examples:
        if message.get(INTENT) == intent_to_check:
            num_examples += 1
            if message.get(TEXT) in augmented_data_should_contain:
                augmented_data_should_contain[message.get(TEXT)] = True
            if message.get(TEXT) in augmented_data_should_not_contain:
                augmented_data_should_not_contain[message.get(TEXT)] = True

    assert num_examples == should_have_num_examples_after_augmentation
    assert all(augmented_data_should_contain.values())
    assert not all(augmented_data_should_not_contain.values())


def test_augmenter_build_random_sampling_training_set():
    from rasa.nlu.data_augmentation.augmenter import (
        _create_augmented_training_data_random_sampling,
    )

    nlu_training_data = rasa.shared.nlu.training_data.loading.load_data(
        "data/test_nlu_paraphrasing/nlu_train.yml"
    )

    intent_to_check = "check_earnings"
    three_new_words = "xxx yyy zzz"
    two_new_words = "account aaa bbb"
    one_new_word = "account deposited xxx"

    intents_to_augment = {intent_to_check}
    num_examples_for_intent = 1
    augmentation_factor = {intent_to_check: 2}
    should_have_num_examples_after_augmentation = (
        num_examples_for_intent + augmentation_factor[intent_to_check]
    )

    paraphrase_pool = {
        "check_earnings": [
            Message(
                data={
                    INTENT: intent_to_check,
                    TEXT: one_new_word,
                    VOCABULARY: set(one_new_word.split()),
                }
            ),
            Message(
                data={
                    INTENT: intent_to_check,
                    TEXT: three_new_words,
                    VOCABULARY: set(three_new_words.split()),
                }
            ),
            Message(
                data={
                    INTENT: intent_to_check,
                    TEXT: two_new_words,
                    VOCABULARY: set(two_new_words.split()),
                }
            ),
        ]
    }

    augmented_data = _create_augmented_training_data_random_sampling(
        nlu_training_data=nlu_training_data,
        paraphrase_pool=paraphrase_pool,
        intents_to_augment=intents_to_augment,
        augmentation_factor=augmentation_factor,
        random_seed=29306,
    )

    num_examples = 0
    num_augmented_examples = 0
    for m in augmented_data.intent_examples:
        if m.get(INTENT) == intent_to_check:
            num_examples += 1
            if m.get(TEXT) in {three_new_words, two_new_words, one_new_word}:
                num_augmented_examples += 1

    assert num_examples == should_have_num_examples_after_augmentation
    assert num_augmented_examples == augmentation_factor[intent_to_check]

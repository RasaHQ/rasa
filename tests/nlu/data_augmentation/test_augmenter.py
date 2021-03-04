import os
from typing import Callable

from _pytest.pytester import RunResult, Testdir
from rasa.shared.constants import LATEST_TRAINING_DATA_FORMAT_VERSION
from rasa.shared.nlu.training_data.message import Message
import rasa.shared.nlu.training_data.loading
import rasa.shared.utils.io


def test_augmenter_intent_collection(run: Callable[..., RunResult],):
    from rasa.nlu.data_augmentation.augmenter import (
        _collect_intents_for_data_augmentation,
    )

    data_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    report_file = os.path.join(
        data_root, "data/test_nlu_paraphrasing/nlu_classification_report.json"
    )
    classification_report = rasa.shared.utils.io.read_json_file(report_file)
    nlu_training_data = rasa.shared.nlu.training_data.loading.load_data(
        os.path.join(data_root, "data/test_nlu_paraphrasing/nlu_train.yml")
    )
    intent_proportion = 0.5

    intents_to_augment = _collect_intents_for_data_augmentation(
        nlu_training_data=nlu_training_data,
        intent_proportion=intent_proportion,
        classification_report=classification_report,
    )

    num_intents = len(nlu_training_data.number_of_examples_per_intent)
    num_intents_to_augment_per_criterion = round(num_intents * intent_proportion)
    assert len(intents_to_augment) >= num_intents_to_augment_per_criterion
    assert len(intents_to_augment) <= num_intents


def test_augmenter_paraphrase_pool_creation(run: Callable[..., RunResult],):
    from rasa.nlu.data_augmentation.augmenter import _create_paraphrase_pool

    data_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    paraphrases = rasa.shared.nlu.training_data.loading.load_data(
        os.path.join(data_root, "data/test_nlu_paraphrasing/paraphrases.yml")
    )

    # See the file `rasa/data/test_nlu_paraphrasing/nlu_train.yml` for available intents
    intents_to_augment = {"ask_transfer_charge", "check_earnings", "check_recipients"}

    # Max sim score in `rasa/data/test_nlu_paraphrasing/paraphrases.yml` is 0.975
    # Min sim score in `rasa/data/test_nlu_paraphrasing/paraphrases.yml` is 0.389

    pool = _create_paraphrase_pool(
        paraphrases=paraphrases,
        intents_to_augment=intents_to_augment,
        min_paraphrase_sim_score=0.98,
        max_paraphrase_sim_score=0.99,
    )
    assert len(pool) == 0

    pool = _create_paraphrase_pool(
        paraphrases=paraphrases,
        intents_to_augment=intents_to_augment,
        min_paraphrase_sim_score=0.0,
        max_paraphrase_sim_score=0.99,
    )
    assert len(pool["check_earnings"]) == 18
    assert len(pool["check_recipients"]) == 18
    assert len(pool["ask_transfer_charge"]) == 6

    pool = _create_paraphrase_pool(
        paraphrases=paraphrases,
        intents_to_augment=intents_to_augment,
        min_paraphrase_sim_score=0.0,
        max_paraphrase_sim_score=0.45,
    )

    assert len(pool["check_earnings"]) == 3
    assert len(pool["check_recipients"]) == 8
    assert len(pool["ask_transfer_charge"]) == 2

    pool = _create_paraphrase_pool(
        paraphrases=paraphrases,
        intents_to_augment=intents_to_augment,
        min_paraphrase_sim_score=0.4,
        max_paraphrase_sim_score=0.5,
    )
    assert len(pool["check_earnings"]) == 5
    assert len(pool["check_recipients"]) == 8
    assert len(pool["ask_transfer_charge"]) == 1


def test_augmenter_augmentation_factor_resolution(
    run: Callable[..., RunResult],
):
    from rasa.nlu.data_augmentation.augmenter import _resolve_augmentation_factor

    data_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    nlu_training_data = rasa.shared.nlu.training_data.loading.load_data(
        os.path.join(data_root, "data/test_nlu_paraphrasing/nlu_train.yml")
    )
    augmentation_factor = 2.0

    assert_dict = {
        "goodbye": 2,
        "greet": 2,
        "deny": 2,
        "human_handoff": 2,
        "affirm": 2,
        "help": 2,
        "ask_transfer_charge": 2,
        "check_earnings": 2,
        "thankyou": 2,
        "check_recipients": 2,
        "search_transactions": 2,
        "transfer_money": 2,
        "pay_cc": 2,
        "check_balance": 2,
        "inform": 8,
    }

    resolved_dict = _resolve_augmentation_factor(
        nlu_training_data=nlu_training_data, augmentation_factor=augmentation_factor
    )

    assert all([assert_dict[key] == resolved_dict[key] for key in assert_dict.keys()])


def test_augmenter_build_max_vocab_expansion_training_set(
    run: Callable[..., RunResult],
):
    from rasa.nlu.data_augmentation.augmenter import (
        _create_augmented_training_data_max_vocab_expansion,
    )

    data_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    nlu_training_data = rasa.shared.nlu.training_data.loading.load_data(
        os.path.join(data_root, "data/test_nlu_paraphrasing/nlu_train.yml")
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
                    "intent": intent_to_check,
                    "text": one_new_word,
                    "metadata": {"vocabulary": set(one_new_word.split())},
                }
            ),
            Message(
                data={
                    "intent": intent_to_check,
                    "text": three_new_words,
                    "metadata": {"vocabulary": set(three_new_words.split())},
                }
            ),
            Message(
                data={
                    "intent": intent_to_check,
                    "text": two_new_words,
                    "metadata": {"vocabulary": set(two_new_words.split())},
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
    )
    num_examples = 0
    for m in augmented_data.intent_examples:
        if m.get("intent") == intent_to_check:
            num_examples += 1
            if m.get("text") in augmented_data_should_contain:
                augmented_data_should_contain[m.get("text")] = True
            if m.get("text") in augmented_data_should_not_contain:
                augmented_data_should_not_contain[m.get("text")] = True

    assert num_examples == should_have_num_examples_after_augmentation
    assert all(augmented_data_should_contain.values())
    assert not all(augmented_data_should_not_contain.values())


def test_augmenter_build_random_sampling_training_set(
    run: Callable[..., RunResult],
):
    from rasa.nlu.data_augmentation.augmenter import (
        _create_augmented_training_data_random_sampling,
    )

    data_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    nlu_training_data = rasa.shared.nlu.training_data.loading.load_data(
        os.path.join(data_root, "data/test_nlu_paraphrasing/nlu_train.yml")
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
                    "intent": intent_to_check,
                    "text": one_new_word,
                    "metadata": {"vocabulary": set(one_new_word.split())},
                }
            ),
            Message(
                data={
                    "intent": intent_to_check,
                    "text": three_new_words,
                    "metadata": {"vocabulary": set(three_new_words.split())},
                }
            ),
            Message(
                data={
                    "intent": intent_to_check,
                    "text": two_new_words,
                    "metadata": {"vocabulary": set(two_new_words.split())},
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
        if m.get("intent") == intent_to_check:
            num_examples += 1
            if m.get("text") in {three_new_words, two_new_words, one_new_word}:
                num_augmented_examples += 1

    assert num_examples == should_have_num_examples_after_augmentation
    assert num_augmented_examples == augmentation_factor[intent_to_check]
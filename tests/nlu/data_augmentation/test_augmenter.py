import operator
import os
from typing import Callable, List, Set, Text

import pytest
from rasa.nlu.constants import TOKENS_NAMES, VOCABULARY
from rasa.nlu.tokenizers.tokenizer import Tokenizer
from rasa.constants import DEFAULT_RANDOM_SEED
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.data_augmentation import augmenter
import rasa.nlu.config
import rasa.shared.nlu.training_data.loading
import rasa.shared.utils.components
import rasa.shared.utils.io
from rasa.shared.nlu.constants import INTENT, TEXT, INTENT_REPORT_FILE_NAME


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
    report_file = (
        "data/test_nlu_paraphrasing/nlu_classification_report_no_augmentation.json"
    )
    classification_report = rasa.shared.utils.io.read_json_file(report_file)
    nlu_training_data = rasa.shared.nlu.training_data.loading.load_data(
        "data/test_nlu_paraphrasing/nlu_train.yml"
    )

    intents_to_augment = augmenter._collect_intents_for_data_augmentation(
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
def test_augmenter_paraphrase_pool_thresholds(
    intent_to_augment: Text,
    min_paraphrase_sim_score: float,
    max_paraphrase_sim_score: float,
    expected_len: int,
):
    paraphrases = rasa.shared.nlu.training_data.loading.load_data(
        "data/test_nlu_paraphrasing/paraphrases.yml"
    )

    pool = augmenter._create_paraphrase_pool(
        paraphrases=paraphrases,
        intents_to_augment={intent_to_augment},
        min_paraphrase_sim_score=min_paraphrase_sim_score,
        max_paraphrase_sim_score=max_paraphrase_sim_score,
    )
    assert len(pool.get(intent_to_augment, [])) == expected_len


@pytest.mark.parametrize(
    "intents_to_augment",
    [
        set(),
        {"check_recipients", "check_earnings"},
        {
            "ask_transfer_charge",
            "greet",
            "help",
            "deny",
            "check_recipients",
            "thankyou",
            "search_transactions",
            "check_balance",
            "pay_cc",
            "goodbye",
            "affirm",
            "inform",
            "check_earnings",
            "human_handoff",
            "transfer_money",
        },
    ],
)
def test_augmenter_paraphrase_pool_intents(intents_to_augment: Set[Text]):
    paraphrases = rasa.shared.nlu.training_data.loading.load_data(
        "data/test_nlu_paraphrasing/paraphrases.yml"
    )

    pool = augmenter._create_paraphrase_pool(
        paraphrases=paraphrases,
        intents_to_augment=intents_to_augment,
        min_paraphrase_sim_score=0.0,
        max_paraphrase_sim_score=1.0,
    )

    assert len(pool) == len(intents_to_augment)
    assert all([intent in pool for intent in intents_to_augment])


@pytest.mark.parametrize(
    "augmentation_factor, intent, size_after_augmentation",
    [
        (0.0, "check_recipients", 1),
        (0.0, "inform", 1),
        (0.0, "search_transactions", 1),
        (0.0, "check_balance", 1),
        (0.0, "transfer_money", 1),
        (0.1, "check_recipients", 1),
        (0.1, "inform", 1),
        (0.1, "search_transactions", 1),
        (0.1, "check_balance", 1),
        (0.1, "transfer_money", 1),
        (0.5, "check_recipients", 1),
        (0.5, "inform", 2),
        (0.5, "search_transactions", 1),
        (0.5, "check_balance", 1),
        (0.5, "transfer_money", 1),
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
    ],
)
def test_augmenter_augmentation_factor_resolution(
    augmentation_factor: float, intent: Text, size_after_augmentation: int
):
    nlu_training_data = rasa.shared.nlu.training_data.loading.load_data(
        "data/test_nlu_paraphrasing/nlu_train.yml"
    )

    resolved_dict = augmenter._resolve_augmentation_factor(
        nlu_training_data=nlu_training_data, augmentation_factor=augmentation_factor
    )

    assert resolved_dict[intent] == size_after_augmentation


def test_augmenter_build_max_vocab_expansion_training_set():
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

    augmented_data = augmenter._create_augmented_training_data_max_vocab_expansion(
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

    augmented_data = augmenter._create_augmented_training_data_random_sampling(
        nlu_training_data=nlu_training_data,
        paraphrase_pool=paraphrase_pool,
        intents_to_augment=intents_to_augment,
        augmentation_factor=augmentation_factor,
        random_seed=DEFAULT_RANDOM_SEED,
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


@pytest.mark.parametrize(
    "all_intents, significant_figures, expected_changed_intents",
    [
        (
            [
                "goodbye",
                "transfer_money",
                "check_recipients",
                "human_handoff",
                "greet",
                "check_balance",
                "inform",
                "search_transactions",
                "ask_transfer_charge",
                "thankyou",
                "pay_cc",
                "help",
                "deny",
                "affirm",
                "check_earnings",
            ],
            1,
            {"goodbye", "human_handoff", "check_recipients"},
        ),
        (
            [
                "goodbye",
                "transfer_money",
                "check_recipients",
                "human_handoff",
                "greet",
                "check_balance",
                "inform",
                "search_transactions",
                "ask_transfer_charge",
                "thankyou",
                "pay_cc",
                "help",
                "deny",
                "affirm",
                "check_earnings",
            ],
            0,
            set(),
        ),
        (
            [
                "goodbye",
                "transfer_money",
                "check_recipients",
                "human_handoff",
                "greet",
                "check_balance",
                "inform",
                "search_transactions",
                "ask_transfer_charge",
                "thankyou",
                "pay_cc",
                "help",
                "deny",
                "affirm",
                "check_earnings",
            ],
            2,
            {
                "check_recipients",
                "goodbye",
                "ask_transfer_charge",
                "human_handoff",
                "inform",
            },
        ),
        (
            [
                "goodbye",
                "transfer_money",
                "check_recipients",
                "human_handoff",
                "greet",
                "check_balance",
                "inform",
                "search_transactions",
                "ask_transfer_charge",
                "thankyou",
                "pay_cc",
                "help",
                "deny",
                "affirm",
                "check_earnings",
            ],
            3,
            {
                "check_recipients",
                "goodbye",
                "ask_transfer_charge",
                "human_handoff",
                "inform",
                "search_transactions",
            },
        ),
    ],
)
def test_get_intents_with_performance_changes(
    all_intents: List[Text],
    significant_figures: int,
    expected_changed_intents: Set[Text],
):
    classification_report_no_augmentation = rasa.shared.utils.io.read_json_file(
        "data/test_nlu_paraphrasing/nlu_classification_report_no_augmentation.json"
    )
    intent_report_with_augmentation = rasa.shared.utils.io.read_json_file(
        "data/test_nlu_paraphrasing/intent_report_with_augmentation-01.json"
    )

    changed_intents = augmenter._get_intents_with_performance_changes(
        classification_report_no_augmentation=classification_report_no_augmentation,
        intent_report_with_augmentation=intent_report_with_augmentation,
        all_intents=all_intents,
        significant_figures=significant_figures,
    )

    assert expected_changed_intents == changed_intents


def test_augmentation_summary_creation():
    classification_report_no_augmentation = rasa.shared.utils.io.read_json_file(
        "data/test_nlu_paraphrasing/nlu_classification_report_no_augmentation.json"
    )
    intent_report_with_augmentation = rasa.shared.utils.io.read_json_file(
        "data/test_nlu_paraphrasing/intent_report_with_augmentation-02.json"
    )

    intents_to_augment = {"check_recipients"}
    changed_intents = {"goodbye"}

    (
        changed_intent_summary,
        augmented_classification_report,
    ) = augmenter._create_augmentation_summary(
        intents_to_augment,
        changed_intents,
        classification_report_no_augmentation,
        intent_report_with_augmentation,
    )

    # Assess performance changes to augmented intents
    performance_changes = {
        f"{metric}_change": intent_report_with_augmentation["check_recipients"][metric]
        - classification_report_no_augmentation["check_recipients"][metric]
        for metric in ["precision", "recall", "f1-score"]
    }
    assert all(
        changed_intent_summary["check_recipients"][metric]
        == performance_changes[metric]
        for metric in ["precision_change", "recall_change", "f1-score_change"]
    )
    assert all(
        augmented_classification_report["check_recipients"][metric]
        == performance_changes[metric]
        for metric in ["precision_change", "recall_change", "f1-score_change"]
    )

    # Assess performance changes to affected intents
    performance_changes = {
        f"{metric}_change": intent_report_with_augmentation["goodbye"][metric]
        - classification_report_no_augmentation["goodbye"][metric]
        for metric in ["precision", "recall", "f1-score"]
    }
    assert all(
        changed_intent_summary["goodbye"][metric] == performance_changes[metric]
        for metric in ["precision_change", "recall_change", "f1-score_change"]
    )
    assert all(
        augmented_classification_report["goodbye"][metric]
        == performance_changes[metric]
        for metric in ["precision_change", "recall_change", "f1-score_change"]
    )

    # Assess that all the totals keys are in the summary (with respective keys
    # indicating performance changes)
    assert "accuracy" in changed_intent_summary
    assert "accuracy_change" in changed_intent_summary["accuracy"]
    assert "accuracy" in augmented_classification_report
    assert "accuracy_change" in augmented_classification_report["accuracy"]

    assert "weighted avg" in changed_intent_summary
    assert "precision_change" in changed_intent_summary["weighted avg"]
    assert "recall_change" in changed_intent_summary["weighted avg"]
    assert "f1-score_change" in changed_intent_summary["weighted avg"]
    assert "weighted avg" in augmented_classification_report
    assert "precision_change" in augmented_classification_report["weighted avg"]
    assert "recall_change" in augmented_classification_report["weighted avg"]
    assert "f1-score_change" in augmented_classification_report["weighted avg"]

    assert "macro avg" in changed_intent_summary
    assert "precision_change" in changed_intent_summary["macro avg"]
    assert "recall_change" in changed_intent_summary["macro avg"]
    assert "f1-score_change" in changed_intent_summary["macro avg"]
    assert "macro avg" in augmented_classification_report
    assert "precision_change" in augmented_classification_report["macro avg"]
    assert "recall_change" in augmented_classification_report["macro avg"]
    assert "f1-score_change" in augmented_classification_report["macro avg"]


def test_summary_report_creation():
    classification_report_no_augmentation = rasa.shared.utils.io.read_json_file(
        "data/test_nlu_paraphrasing/nlu_classification_report_no_augmentation.json"
    )
    intent_report_with_augmentation = rasa.shared.utils.io.read_json_file(
        "data/test_nlu_paraphrasing/intent_report_with_augmentation-02.json"
    )
    training_intents = [
        "goodbye",
        "transfer_money",
        "check_recipients",
        "human_handoff",
        "greet",
        "check_balance",
        "inform",
        "search_transactions",
        "ask_transfer_charge",
        "thankyou",
        "pay_cc",
        "help",
        "deny",
        "affirm",
        "check_earnings",
    ]
    intents_to_augment = {"check_recipients"}

    tmp_path = rasa.utils.io.create_temporary_directory()
    out_path = os.path.join(tmp_path, "augmentation_report")
    os.makedirs(out_path)

    # Contents of both reports tested by test_augmentation_summary_creation
    _ = augmenter._create_summary_report(
        intent_report_with_augmentation=intent_report_with_augmentation,
        classification_report_no_augmentation=classification_report_no_augmentation,
        intents_to_augment=intents_to_augment,
        training_intents=training_intents,
        output_directory=out_path,
    )

    assert os.path.exists(os.path.join(out_path, INTENT_REPORT_FILE_NAME))


def test_summary_plot_creation():
    intent_summary = rasa.shared.utils.io.read_json_file(
        "data/test_nlu_paraphrasing/intent_summary.json"
    )

    tmp_path = rasa.utils.io.create_temporary_directory()
    out_path = os.path.join(tmp_path, "augmentation_plot")
    os.makedirs(out_path)

    augmenter._plot_summary_report(intent_summary=intent_summary, output_directory=out_path)

    assert all(
        os.path.exists(os.path.join(out_path, output_file))
        for output_file in [
            "precision_changes.png",
            "recall_changes.png",
            "f1-score_changes.png",
        ]
    )

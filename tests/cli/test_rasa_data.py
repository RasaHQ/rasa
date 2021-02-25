import argparse
import filecmp
import os
from pathlib import Path
from unittest.mock import Mock
import pytest
from collections import namedtuple
from typing import Callable, Text, Dict, List

from _pytest.monkeypatch import MonkeyPatch
from _pytest.pytester import RunResult, Testdir
from rasa.cli import data
from rasa.core.constants import (
    DEFAULT_NLU_FALLBACK_AMBIGUITY_THRESHOLD,
    DEFAULT_NLU_FALLBACK_THRESHOLD,
    DEFAULT_CORE_FALLBACK_THRESHOLD,
)
from rasa.shared.constants import LATEST_TRAINING_DATA_FORMAT_VERSION, DEFAULT_DATA_PATH
from rasa.shared.core.domain import Domain
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.nlu.training_data.message import Message
from rasa.validator import Validator
import rasa.shared.nlu.training_data.loading
import rasa.shared.utils.io


def test_data_split_nlu(run_in_simple_project: Callable[..., RunResult]):
    responses_yml = (
        "responses:\n"
        "  chitchat/ask_name:\n"
        "  - text: my name is Sara, Rasa's documentation bot!\n"
        "  chitchat/ask_weather:\n"
        "  - text: the weather is great!\n"
    )

    with open("data/responses.yml", "w") as f:
        f.write(responses_yml)

    run_in_simple_project(
        "data",
        "split",
        "nlu",
        "-u",
        "data/nlu.yml",
        "--training-fraction",
        "0.75",
        "--random-seed",
        "12345",
    )

    folder = Path("train_test_split")
    assert folder.exists()

    nlu_files = [folder / "test_data.yml", folder / "training_data.yml"]
    nlg_files = [folder / "nlg_test_data.yml", folder / "nlg_training_data.yml"]
    for yml_file in nlu_files:
        assert yml_file.exists(), f"{yml_file} file does not exist"
        nlu_data = rasa.shared.utils.io.read_yaml_file(yml_file)
        assert "version" in nlu_data
        assert nlu_data.get("nlu")

    for yml_file in nlg_files:
        assert yml_file.exists(), f"{yml_file} file does not exist"


def test_data_convert_nlu(run_in_simple_project: Callable[..., RunResult]):
    run_in_simple_project(
        "data",
        "convert",
        "nlu",
        "--data",
        "data/nlu.yml",
        "--out",
        "out_nlu_data.json",
        "-f",
        "json",
    )

    assert os.path.exists("out_nlu_data.json")


def test_data_split_help(run: Callable[..., RunResult]):
    output = run("data", "split", "nlu", "--help")

    help_text = """usage: rasa data split nlu [-h] [-v] [-vv] [--quiet] [-u NLU]
                           [--training-fraction TRAINING_FRACTION]
                           [--random-seed RANDOM_SEED] [--out OUT]"""

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = set(output.outlines)
    for line in lines:
        assert line in printed_help


def test_data_convert_help(run: Callable[..., RunResult]):
    output = run("data", "convert", "nlu", "--help")

    help_text = """usage: rasa data convert nlu [-h] [-v] [-vv] [--quiet] [-f {json,md,yaml}]
                             --data DATA [--out OUT] [-l LANGUAGE]"""

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = set(output.outlines)
    for line in lines:
        assert line in printed_help


def test_data_validate_help(run: Callable[..., RunResult]):
    output = run("data", "validate", "--help")

    help_text = """usage: rasa data validate [-h] [-v] [-vv] [--quiet]
                          [--max-history MAX_HISTORY] [-c CONFIG]
                          [--fail-on-warnings] [-d DOMAIN] [--data DATA]"""

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = set(output.outlines)
    for line in lines:
        assert line in printed_help


def _text_is_part_of_output_error(text: Text, output: RunResult) -> bool:
    found_info_string = False
    for line in output.errlines:
        if text in line:
            found_info_string = True
    return found_info_string


def test_data_validate_stories_with_max_history_zero(monkeypatch: MonkeyPatch):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="Rasa commands")
    data.add_subparser(subparsers, parents=[])

    args = parser.parse_args(
        [
            "data",
            "validate",
            "stories",
            "--data",
            "examples/moodbot/data",
            "--max-history",
            0,
        ]
    )

    async def mock_from_importer(importer: TrainingDataImporter) -> Validator:
        return Mock()

    monkeypatch.setattr("rasa.validator.Validator.from_importer", mock_from_importer)

    with pytest.raises(argparse.ArgumentTypeError):
        data.validate_files(args)


def test_rasa_data_augment_nlu_intent_collection(run: Callable[..., RunResult],):
    from rasa.nlu.data_augmentation.augmenter import _collect_intents_for_data_augmentation

    data_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    report_file = os.path.join(
            data_root, "data/test_nlu_paraphrasing/nlu_classification_report.json"
        )
    classification_report = rasa.shared.utils.io.read_json_file(report_file)
    nlu_training_data = rasa.shared.nlu.training_data.loading.load_data(os.path.join(data_root, "data/test_nlu_paraphrasing/nlu_train.yml"))
    intent_proportion = 0.5

    intents_to_augment = _collect_intents_for_data_augmentation(
        nlu_training_data=nlu_training_data, intent_proportion=intent_proportion, classification_report=classification_report
    )

    num_intents = len(nlu_training_data.number_of_examples_per_intent)
    num_intents_to_augment_per_criterion = round(num_intents * intent_proportion)
    assert len(intents_to_augment) >= num_intents_to_augment_per_criterion
    assert len(intents_to_augment) <= num_intents


def test_rasa_data_augment_nlu_paraphrase_pool_creation(run: Callable[..., RunResult],):
    from rasa.nlu.data_augmentation.augmenter import _create_paraphrase_pool

    data_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    paraphrases = rasa.shared.nlu.training_data.loading.load_data(
        os.path.join(data_root, "data/test_nlu_paraphrasing/paraphrases.yml"))

    # See the file `rasa/data/test_nlu_paraphrasing/nlu_train.yml` for available intents
    intents_to_augment = {"ask_transfer_charge", "check_earnings", "check_recipients"}

    # Max sim score in `rasa/data/test_nlu_paraphrasing/paraphrases.yml` is 0.975
    # Min sim score in `rasa/data/test_nlu_paraphrasing/paraphrases.yml` is 0.389

    pool = _create_paraphrase_pool(paraphrases=paraphrases, intents_to_augment=intents_to_augment,
                                   min_paraphrase_sim_score=0.98, max_paraphrase_sim_score=0.99)
    assert len(pool) == 0

    pool = _create_paraphrase_pool(paraphrases=paraphrases, intents_to_augment=intents_to_augment,
                                   min_paraphrase_sim_score=0.0, max_paraphrase_sim_score=0.99)
    assert len(pool["check_earnings"]) == 18
    assert len(pool["check_recipients"]) == 18
    assert len(pool["ask_transfer_charge"]) == 6

    pool = _create_paraphrase_pool(paraphrases=paraphrases, intents_to_augment=intents_to_augment,
                                   min_paraphrase_sim_score=0.0, max_paraphrase_sim_score=0.45)

    assert len(pool["check_earnings"]) == 3
    assert len(pool["check_recipients"]) == 8
    assert len(pool["ask_transfer_charge"]) == 2

    pool = _create_paraphrase_pool(paraphrases=paraphrases, intents_to_augment=intents_to_augment,
                                   min_paraphrase_sim_score=0.4, max_paraphrase_sim_score=0.5)
    assert len(pool["check_earnings"]) == 5
    assert len(pool["check_recipients"]) == 8
    assert len(pool["ask_transfer_charge"]) == 1


def test_rasa_data_augment_nlu_augmentation_factor_resolution(run: Callable[..., RunResult],):
    from rasa.nlu.data_augmentation.augmenter import _resolve_augmentation_factor

    data_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    nlu_training_data = rasa.shared.nlu.training_data.loading.load_data(
        os.path.join(data_root, "data/test_nlu_paraphrasing/nlu_train.yml"))
    augmentation_factor = 2.

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
        "inform": 8
    }

    resolved_dict = _resolve_augmentation_factor(nlu_training_data=nlu_training_data, augmentation_factor=augmentation_factor)

    assert all([assert_dict[key] == resolved_dict[key] for key in assert_dict.keys()])


def test_rasa_data_augment_nlu_build_max_vocab_expansion_training_set(run: Callable[..., RunResult],):
    from rasa.nlu.data_augmentation.augmenter import _create_augmented_training_data_max_vocab_expansion

    data_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    nlu_training_data = rasa.shared.nlu.training_data.loading.load_data(
        os.path.join(data_root, "data/test_nlu_paraphrasing/nlu_train.yml"))

    intent_to_check = "check_earnings"
    three_new_words = "xxx yyy zzz"
    two_new_words = "account aaa bbb"
    one_new_word = "account deposited xxx"

    intents_to_augment = {intent_to_check}
    num_examples_for_intent = 1
    augmentation_factor = {
        intent_to_check: 2
    }
    should_have_num_examples_after_augmentation = num_examples_for_intent + augmentation_factor[intent_to_check]

    paraphrase_pool = {
        "check_earnings": [
            Message(data={"intent": intent_to_check, "text": one_new_word, "metadata": {"vocabulary": set(one_new_word.split())}}),
            Message(data={"intent": intent_to_check, "text": three_new_words, "metadata": {"vocabulary": set(three_new_words.split())}}),
            Message(data={"intent": intent_to_check, "text": two_new_words, "metadata": {"vocabulary": set(two_new_words.split())}}),
        ]
    }

    augmented_data_should_contain = {
        three_new_words: False,
        two_new_words: False
    }
    augmented_data_should_not_contain = {
         one_new_word: False
    }

    augmented_data = _create_augmented_training_data_max_vocab_expansion(nlu_training_data=nlu_training_data, paraphrase_pool=paraphrase_pool, intents_to_augment=intents_to_augment, augmentation_factor=augmentation_factor)
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


def test_rasa_data_augment_nlu_build_random_sampling_training_set(run: Callable[..., RunResult],):
    from rasa.nlu.data_augmentation.augmenter import _create_augmented_training_data_random_sampling

    data_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    nlu_training_data = rasa.shared.nlu.training_data.loading.load_data(
        os.path.join(data_root, "data/test_nlu_paraphrasing/nlu_train.yml"))

    intent_to_check = "check_earnings"
    three_new_words = "xxx yyy zzz"
    two_new_words = "account aaa bbb"
    one_new_word = "account deposited xxx"

    intents_to_augment = {intent_to_check}
    num_examples_for_intent = 1
    augmentation_factor = {
        intent_to_check: 2
    }
    should_have_num_examples_after_augmentation = num_examples_for_intent + augmentation_factor[intent_to_check]

    paraphrase_pool = {
        "check_earnings": [
            Message(data={"intent": intent_to_check, "text": one_new_word,
                          "metadata": {"vocabulary": set(one_new_word.split())}}),
            Message(data={"intent": intent_to_check, "text": three_new_words,
                          "metadata": {"vocabulary": set(three_new_words.split())}}),
            Message(data={"intent": intent_to_check, "text": two_new_words,
                          "metadata": {"vocabulary": set(two_new_words.split())}}),
        ]
    }

    augmented_data = _create_augmented_training_data_random_sampling(nlu_training_data=nlu_training_data, paraphrase_pool=paraphrase_pool, intents_to_augment=intents_to_augment, augmentation_factor=augmentation_factor, random_seed=29306)

    num_examples = 0
    num_augmented_examples = 0
    for m in augmented_data.intent_examples:
        if m.get("intent") == intent_to_check:
            num_examples += 1
            if m.get("text") in {three_new_words, two_new_words, one_new_word}:
                num_augmented_examples += 1

    assert num_examples == should_have_num_examples_after_augmentation
    assert num_augmented_examples == augmentation_factor[intent_to_check]


def test_rasa_data_augment_nlu(run: Callable[..., RunResult],):

    out_path = "data/augmentation_results"
    os.makedirs(out_path)

    data_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    run(
        "data",
        "augment",
        "nlu",
        "--nlu-training-data",
        os.path.join(data_root, "data/test_nlu_paraphrasing/nlu_train.yml"),
        "--nlu-evaluation-data",
        os.path.join(data_root, "data/test_nlu_paraphrasing/nlu_test.yml"),
        "--paraphrases",
        os.path.join(data_root, "data/test_nlu_paraphrasing/paraphrases.yml"),
        "--nlu-classification-report",
        os.path.join(
            data_root, "data/test_nlu_paraphrasing/nlu_classification_report.json"
        ),
        "--out",
        out_path,
        "-c",
        os.path.join(data_root, "data/test_nlu_paraphrasing/config.yml"),
        "--min-paraphrase-sim-score",
        "0.1",
    )

    out_report_diverse = os.path.join(
        out_path, "augmentation_max_vocab_expansion/intent_report.json"
    )
    out_errors_diverse = os.path.join(
        out_path, "augmentation_max_vocab_expansion/intent_errors.json"
    )
    out_plot_f1_diverse = os.path.join(
        out_path, "augmentation_max_vocab_expansion/f1-score_changes.png"
    )
    out_plot_precision_diverse = os.path.join(
        out_path, "augmentation_max_vocab_expansion/precision_changes.png"
    )
    out_plot_recall_diverse = os.path.join(
        out_path, "augmentation_max_vocab_expansion/recall_changes.png"
    )
    out_augmented_train_diverse = os.path.join(
        out_path, "augmentation_max_vocab_expansion/nlu_train_augmented_max_vocab_expansion.yml"
    )

    # Files from the diverse augmentation criterion exist
    assert os.path.exists(out_report_diverse)
    assert os.path.exists(out_errors_diverse)
    assert os.path.exists(out_plot_f1_diverse)
    assert os.path.exists(out_plot_precision_diverse)
    assert os.path.exists(out_plot_recall_diverse)
    assert os.path.exists(out_augmented_train_diverse)

    out_report_random = os.path.join(out_path, "augmentation_random/intent_report.json")
    out_errors_random = os.path.join(out_path, "augmentation_random/intent_errors.json")
    out_plot_f1_random = os.path.join(
        out_path, "augmentation_random/f1-score_changes.png"
    )
    out_plot_precision_random = os.path.join(
        out_path, "augmentation_random/precision_changes.png"
    )
    out_plot_recall_random = os.path.join(
        out_path, "augmentation_random/recall_changes.png"
    )
    out_augmented_train_random = os.path.join(
        out_path, "augmentation_random/nlu_train_augmented_random.yml"
    )

    # Files from the random augmentation criterion exist
    assert os.path.exists(out_report_random)
    assert os.path.exists(out_errors_random)
    assert os.path.exists(out_plot_f1_random)
    assert os.path.exists(out_plot_precision_random)
    assert os.path.exists(out_plot_recall_random)
    assert os.path.exists(out_augmented_train_random)


def test_validate_files_exit_early():
    with pytest.raises(SystemExit) as pytest_e:
        args = {
            "domain": "data/test_domains/duplicate_intents.yml",
            "data": None,
            "max_history": None,
            "config": None,
        }
        data.validate_files(namedtuple("Args", args.keys())(*args.values()))

    assert pytest_e.type == SystemExit
    assert pytest_e.value.code == 1


def test_rasa_data_convert_nlu_to_yaml(
    run_in_simple_project: Callable[..., RunResult],
):
    converted_data_folder = "converted_data"
    os.mkdir(converted_data_folder)

    converted_single_file_folder = "converted_single_file"
    os.mkdir(converted_single_file_folder)

    simple_nlu_md = """
    ## intent:greet
    - hey
    - hello
    """

    os.mkdir("data/nlu")
    with open("data/nlu/nlu.md", "w") as f:
        f.write(simple_nlu_md)

    run_in_simple_project(
        "data",
        "convert",
        "nlu",
        "-f",
        "yaml",
        "--data",
        "data",
        "--out",
        converted_data_folder,
    )

    run_in_simple_project(
        "data",
        "convert",
        "nlu",
        "-f",
        "yaml",
        "--data",
        "data/nlu/nlu.md",
        "--out",
        converted_single_file_folder,
    )

    assert filecmp.cmp(
        Path(converted_data_folder) / "nlu_converted.yml",
        Path(converted_single_file_folder) / "nlu_converted.yml",
    )


def test_rasa_data_convert_stories_to_yaml(
    run_in_simple_project: Callable[..., RunResult],
):
    converted_data_folder = "converted_data"
    os.mkdir(converted_data_folder)

    converted_single_file_folder = "converted_single_file"
    os.mkdir(converted_single_file_folder)

    simple_story_md = """
    ## happy path
    * greet OR goodbye
        - utter_greet
        - form{"name": null}
    """

    with open("data/stories.md", "w") as f:
        f.write(simple_story_md)

    run_in_simple_project(
        "data",
        "convert",
        "core",
        "-f",
        "yaml",
        "--data",
        "data",
        "--out",
        converted_data_folder,
    )

    run_in_simple_project(
        "data",
        "convert",
        "core",
        "-f",
        "yaml",
        "--data",
        "data/stories.md",
        "--out",
        converted_single_file_folder,
    )

    assert filecmp.cmp(
        Path(converted_data_folder) / "stories_converted.yml",
        Path(converted_single_file_folder) / "stories_converted.yml",
    )


def test_rasa_data_convert_test_stories_to_yaml(
    run_in_simple_project: Callable[..., RunResult]
):
    converted_data_folder = "converted_data"
    os.mkdir(converted_data_folder)

    simple_test_story_md = """
    ## happy path
    * greet: hi how are you?
        - utter_greet
    """

    with open("tests/test_stories.md", "w") as f:
        f.write(simple_test_story_md)

    run_in_simple_project(
        "data",
        "convert",
        "core",
        "-f",
        "yaml",
        "--data",
        "tests",
        "--out",
        converted_data_folder,
    )

    assert (Path(converted_data_folder) / "test_stories_converted.yml").exists()


def test_rasa_data_convert_nlg_to_yaml(
    run_in_simple_project: Callable[..., RunResult], run: Callable[..., RunResult]
):
    converted_data_folder = "converted_data"
    os.mkdir(converted_data_folder)

    converted_single_file_folder = "converted_single_file"
    os.mkdir(converted_single_file_folder)

    simple_nlg_md = (
        "## ask name\n"
        "* chitchat/ask_name\n"
        "- my name is Sara, Rasa's documentation bot!\n"
    )

    with open("data/responses.md", "w") as f:
        f.write(simple_nlg_md)

    run_in_simple_project(
        "data",
        "convert",
        "nlg",
        "-f",
        "yaml",
        "--data",
        "data",
        "--out",
        converted_data_folder,
    )

    run_in_simple_project(
        "data",
        "convert",
        "nlg",
        "-f",
        "yaml",
        "--data",
        "data/responses.md",
        "--out",
        converted_single_file_folder,
    )

    assert filecmp.cmp(
        Path(converted_data_folder) / "responses_converted.yml",
        Path(converted_single_file_folder) / "responses_converted.yml",
    )


def test_rasa_data_convert_responses(
    run_in_simple_project: Callable[..., RunResult], run: Callable[..., RunResult]
):
    converted_data_folder = "converted_data"
    os.mkdir(converted_data_folder)
    expected_data_folder = "expected_data"
    os.mkdir(expected_data_folder)

    domain = (
        "version: '2.0'\n"
        "session_config:\n"
        "  session_expiration_time: 60\n"
        "  carry_over_slots_to_new_session: true\n"
        "actions:\n"
        "- respond_chitchat\n"
        "- utter_greet\n"
        "- utter_cheer_up"
    )

    expected_domain = (
        "version: '2.0'\n"
        "session_config:\n"
        "  session_expiration_time: 60\n"
        "  carry_over_slots_to_new_session: true\n"
        "actions:\n"
        "- utter_chitchat\n"
        "- utter_greet\n"
        "- utter_cheer_up\n"
    )

    with open(f"{expected_data_folder}/domain.yml", "w") as f:
        f.write(expected_domain)

    with open("domain.yml", "w") as f:
        f.write(domain)

    stories = (
        "stories:\n"
        "- story: sad path\n"
        "  steps:\n"
        "  - intent: greet\n"
        "  - action: utter_greet\n"
        "  - intent: mood_unhappy\n"
        "- story: chitchat\n"
        "  steps:\n"
        "  - intent: chitchat\n"
        "  - action: respond_chitchat\n"
    )

    expected_stories = (
        'version: "2.0"\n'
        "stories:\n"
        "- story: sad path\n"
        "  steps:\n"
        "  - intent: greet\n"
        "  - action: utter_greet\n"
        "  - intent: mood_unhappy\n"
        "- story: chitchat\n"
        "  steps:\n"
        "  - intent: chitchat\n"
        "  - action: utter_chitchat\n"
    )

    with open(f"{expected_data_folder}/stories.yml", "w") as f:
        f.write(expected_stories)

    with open("data/stories.yml", "w") as f:
        f.write(stories)

    run_in_simple_project(
        "data",
        "convert",
        "responses",
        "--data",
        "data",
        "--out",
        converted_data_folder,
    )

    assert filecmp.cmp(
        Path(converted_data_folder) / "domain_converted.yml",
        Path(expected_data_folder) / "domain.yml",
    )

    assert filecmp.cmp(
        Path(converted_data_folder) / "stories_converted.yml",
        Path(expected_data_folder) / "stories.yml",
    )


def test_rasa_data_convert_nlu_lookup_tables_to_yaml(
    run_in_simple_project: Callable[..., RunResult]
):
    converted_data_folder = "converted_data"
    os.mkdir(converted_data_folder)

    simple_nlu_md = """
    ## lookup:products.txt
      data/nlu/lookups/products.txt
    """

    os.mkdir("data/nlu")
    with open("data/nlu/nlu.md", "w") as f:
        f.write(simple_nlu_md)

    simple_lookup_table_txt = "core\n nlu\n x\n"
    os.mkdir("data/nlu/lookups")
    with open("data/nlu/lookups/products.txt", "w") as f:
        f.write(simple_lookup_table_txt)

    run_in_simple_project(
        "data",
        "convert",
        "nlu",
        "-f",
        "yaml",
        "--data",
        "data",
        "--out",
        converted_data_folder,
    )

    assert len(os.listdir(converted_data_folder)) == 1


def test_convert_config(
    run: Callable[..., RunResult], tmp_path: Path, default_domain_path: Text
):
    deprecated_config = {
        "policies": [{"name": "MappingPolicy"}, {"name": "FallbackPolicy"}],
        "pipeline": [{"name": "WhitespaceTokenizer"}],
    }
    config_file = tmp_path / "config.yml"
    rasa.shared.utils.io.write_yaml(deprecated_config, config_file)

    domain = Domain.from_dict(
        {
            "intents": [{"greet": {"triggers": "action_greet"}}, "leave"],
            "actions": ["action_greet"],
        }
    )
    domain_file = tmp_path / "domain.yml"
    domain.persist(domain_file)

    rules_file = tmp_path / "rules.yml"

    result = run(
        "data",
        "convert",
        "config",
        "--config",
        str(config_file),
        "--domain",
        str(domain_file),
        "--out",
        str(rules_file),
    )

    assert result.ret == 0
    new_config = rasa.shared.utils.io.read_yaml_file(config_file)
    new_domain = rasa.shared.utils.io.read_yaml_file(domain_file)
    new_rules = rasa.shared.utils.io.read_yaml_file(rules_file)

    assert new_config == {
        "policies": [
            {
                "name": "RulePolicy",
                "core_fallback_action_name": "action_default_fallback",
                "core_fallback_threshold": DEFAULT_CORE_FALLBACK_THRESHOLD,
            }
        ],
        "pipeline": [
            {"name": "WhitespaceTokenizer"},
            {
                "name": "FallbackClassifier",
                "ambiguity_threshold": DEFAULT_NLU_FALLBACK_AMBIGUITY_THRESHOLD,
                "threshold": DEFAULT_NLU_FALLBACK_THRESHOLD,
            },
        ],
    }
    assert new_domain["intents"] == ["greet", "leave"]
    assert new_rules == {
        "rules": [
            {
                "rule": "Rule to map `greet` intent to `action_greet` "
                "(automatic conversion)",
                "steps": [{"intent": "greet"}, {"action": "action_greet"}],
            },
            {
                "rule": "Rule to handle messages with low NLU confidence "
                "(automated conversion from 'FallbackPolicy')",
                "steps": [
                    {"intent": "nlu_fallback"},
                    {"action": "action_default_fallback"},
                ],
            },
        ],
        "version": LATEST_TRAINING_DATA_FORMAT_VERSION,
    }

    domain_backup = tmp_path / "domain.yml.bak"
    assert domain_backup.exists()

    config_backup = tmp_path / "config.yml.bak"
    assert config_backup.exists()


def test_convert_config_if_nothing_to_migrate(
    run_in_simple_project: Callable[..., RunResult], tmp_path: Path
):
    result = run_in_simple_project("data", "convert", "config")

    assert result.ret == 1

    output = "\n".join(result.outlines)
    assert "No policies were found which need migration" in output


def test_convert_config_with_output_file_containing_data(
    run_in_simple_project: Callable[..., RunResult], tmp_path: Path, testdir: Testdir
):
    deprecated_config = {
        "policies": [{"name": "FallbackPolicy"}],
        "pipeline": [{"name": "WhitespaceTokenizer"}],
    }
    config_file = tmp_path / "config.yml"
    rasa.shared.utils.io.write_yaml(deprecated_config, config_file)

    output_file = testdir.tmpdir / DEFAULT_DATA_PATH / "rules.yml"
    # the default project already contains rules training data
    assert output_file.exists()
    existing_rules = rasa.shared.utils.io.read_yaml_file(output_file)["rules"]
    assert existing_rules

    result = run_in_simple_project(
        "data", "convert", "config", "--config", str(config_file)
    )

    assert result.ret == 0

    new_rules = rasa.shared.utils.io.read_yaml_file(output_file)["rules"]
    expected_new_rule = {
        "rule": "Rule to handle messages with low NLU confidence "
        "(automated conversion from 'FallbackPolicy')",
        "steps": [{"intent": "nlu_fallback"}, {"action": "action_default_fallback"}],
    }

    assert expected_new_rule in new_rules
    assert all(existing in new_rules for existing in existing_rules)

    backup_file = testdir.tmpdir / DEFAULT_DATA_PATH / "rules.yml.bak"
    assert backup_file.exists()


def test_convert_config_with_invalid_config_path(run: Callable[..., RunResult]):
    result = run("data", "convert", "config", "--config", "invalid path")

    assert result.ret == 1

    output = "\n".join(result.outlines)
    assert "Please provide a valid path" in output


def test_convert_config_with_missing_nlu_pipeline_config(
    run_in_simple_project: Callable[..., RunResult], tmp_path: Path
):
    deprecated_config = {"policies": [{"name": "FallbackPolicy"}]}
    config_file = tmp_path / "config.yml"
    rasa.shared.utils.io.write_yaml(deprecated_config, config_file)

    result = run_in_simple_project(
        "data", "convert", "config", "--config", str(config_file)
    )

    assert result.ret == 1

    output = "\n".join(result.outlines)
    assert "The model configuration has to include an NLU pipeline" in output


def test_convert_config_with_missing_nlu_pipeline_config_if_no_fallbacks(
    run_in_simple_project: Callable[..., RunResult], tmp_path: Path
):
    deprecated_config = {"policies": [{"name": "MappingPolicy"}]}
    config_file = tmp_path / "config.yml"
    rasa.shared.utils.io.write_yaml(deprecated_config, config_file)

    result = run_in_simple_project(
        "data", "convert", "config", "--config", str(config_file)
    )

    assert result.ret == 0


@pytest.mark.parametrize(
    "policy_config, expected_error_message",
    [
        (
            [{"name": "FallbackPolicy"}, {"name": "TwoStageFallbackPolicy"}],
            "two configured policies for handling fallbacks",
        )
    ],
)
def test_convert_config_with_invalid_policy_config(
    run_in_simple_project: Callable[..., RunResult],
    tmp_path: Path,
    policy_config: List[Dict],
    expected_error_message: Text,
):
    deprecated_config = {
        "policies": policy_config,
        "pipeline": [{"name": "WhitespaceTokenizer"}],
    }
    config_file = tmp_path / "config.yml"
    rasa.shared.utils.io.write_yaml(deprecated_config, config_file)

    result = run_in_simple_project(
        "data", "convert", "config", "--config", str(config_file)
    )

    assert result.ret == 1

    output = "\n".join(result.outlines)
    assert expected_error_message in output


def test_warning_for_form_migration(
    run_in_simple_project: Callable[..., RunResult], tmp_path: Path
):
    deprecated_config = {
        "policies": [{"name": "FallbackPolicy"}, {"name": "FormPolicy"}],
        "pipeline": [{"name": "WhitespaceTokenizer"}],
    }
    config_file = tmp_path / "config.yml"
    rasa.shared.utils.io.write_yaml(deprecated_config, config_file)

    result = run_in_simple_project(
        "data", "convert", "config", "--config", str(config_file)
    )

    assert result.ret == 0

    output = "\n".join(result.outlines)
    assert "you have to migrate the 'FormPolicy' manually" in output


@pytest.mark.parametrize(
    "two_stage_config, expected_error_message",
    [
        (
            {"deny_suggestion_intent_name": "something else"},
            "has to use the intent 'out_of_scope'",
        ),
        (
            {"fallback_nlu_action_name": "something else"},
            "has to use the action 'action_default_fallback",
        ),
    ],
)
def test_convert_config_with_two_stage_fallback_policy(
    run_in_simple_project: Callable[..., RunResult],
    tmp_path: Path,
    two_stage_config: Dict,
    expected_error_message: Text,
):
    deprecated_config = {
        "policies": [
            {"name": "MappingPolicy"},
            {"name": "TwoStageFallbackPolicy", **two_stage_config},
        ],
        "pipeline": [{"name": "WhitespaceTokenizer"}],
    }
    config_file = tmp_path / "config.yml"
    rasa.shared.utils.io.write_yaml(deprecated_config, config_file)

    result = run_in_simple_project(
        "data", "convert", "config", "--config", str(config_file)
    )

    assert result.ret == 1

    output = "\n".join(result.outlines)
    assert expected_error_message in output


def test_convert_config_with_invalid_domain_path(run: Callable[..., RunResult]):
    result = run("data", "convert", "config", "--domain", "invalid path")

    assert result.ret == 1

    output = "\n".join(result.outlines)
    assert "path to a valid model configuration" in output


def test_convert_config_with_default_rules_directory(
    run_in_simple_project: Callable[..., RunResult], tmp_path: Path
):
    deprecated_config = {
        "policies": [{"name": "FallbackPolicy"}],
        "pipeline": [{"name": "WhitespaceTokenizer"}],
    }
    config_file = tmp_path / "config.yml"
    rasa.shared.utils.io.write_yaml(deprecated_config, config_file)

    result = run_in_simple_project(
        "data",
        "convert",
        "config",
        "--config",
        str(config_file),
        "--out",
        str(tmp_path),
    )

    assert result.ret == 1

    output = "\n".join(result.outlines)
    assert "needs to be the path to a file" in output

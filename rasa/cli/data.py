import argparse
import collections
import logging
import operator
import os
import random
import shutil
from functools import reduce
from pathlib import Path
from typing import Dict, List, Set, Text, Tuple, TYPE_CHECKING

import rasa.shared.core.domain
from rasa import telemetry
from rasa.cli import SubParsersAction
from rasa.cli.arguments import data as arguments
from rasa.cli.arguments import default_arguments
import rasa.cli.utils
from rasa.core.training.converters.responses_prefix_converter import (
    DomainResponsePrefixConverter,
    StoryResponsePrefixConverter,
)
from rasa.model import get_model
import rasa.nlu.convert
from rasa.nlu.model import Interpreter
import rasa.nlu.test
from rasa.nlu.test import (
    create_intent_report,
    get_eval_data,
    remove_pretrained_extractors,
    extract_intent_errors_from_results,
)
from rasa.shared.constants import (
    DEFAULT_DATA_PATH,
    DOCS_URL_MIGRATION_GUIDE,
)
import rasa.shared.data
from rasa.shared.core.constants import (
    USER_INTENT_OUT_OF_SCOPE,
    ACTION_DEFAULT_FALLBACK_NAME,
)
from rasa.shared.core.training_data.story_reader.yaml_story_reader import (
    YAMLStoryReader,
)
from rasa.shared.core.training_data.story_writer.yaml_story_writer import (
    YAMLStoryWriter,
)
from rasa.shared.importers.rasa import RasaFileImporter
import rasa.shared.nlu.training_data.loading
import rasa.shared.nlu.training_data.util
from rasa.shared.nlu.training_data.training_data import TrainingData
import rasa.shared.utils.cli
from rasa.train import train_nlu
import rasa.utils.common
import rasa.utils.plotting
from rasa.utils.converter import TrainingDataConverter
from rasa.validator import Validator
from rasa.shared.core.domain import Domain, InvalidDomain
import rasa.shared.utils.io
import rasa.core.config
from rasa.core.policies.form_policy import FormPolicy
from rasa.core.policies.fallback import FallbackPolicy
from rasa.core.policies.two_stage_fallback import TwoStageFallbackPolicy
from rasa.core.policies.mapping_policy import MappingPolicy

if TYPE_CHECKING:
    from rasa.shared.core.training_data.structures import StoryStep

logger = logging.getLogger(__name__)


def add_subparser(
    subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]
) -> None:
    """Add all data parsers.

    Args:
        subparsers: subparser we are going to attach to
        parents: Parent parsers, needed to ensure tree structure in argparse
    """
    data_parser = subparsers.add_parser(
        "data",
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="Utils for the Rasa training files.",
    )
    data_parser.set_defaults(func=lambda _: data_parser.print_help(None))

    data_subparsers = data_parser.add_subparsers()

    _add_data_convert_parsers(data_subparsers, parents)
    _add_data_split_parsers(data_subparsers, parents)
    _add_data_validate_parsers(data_subparsers, parents)
    _add_data_suggest_parsers(data_subparsers, parents)


def _add_data_convert_parsers(
    data_subparsers, parents: List[argparse.ArgumentParser]
) -> None:
    convert_parser = data_subparsers.add_parser(
        "convert",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="Converts Rasa data between different formats.",
    )
    convert_parser.set_defaults(func=lambda _: convert_parser.print_help(None))

    convert_subparsers = convert_parser.add_subparsers()
    convert_nlu_parser = convert_subparsers.add_parser(
        "nlu",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="Converts NLU data between formats.",
    )
    convert_nlu_parser.set_defaults(func=_convert_nlu_data)

    arguments.set_convert_arguments(convert_nlu_parser, data_type="Rasa NLU")

    convert_nlg_parser = convert_subparsers.add_parser(
        "nlg",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help=(
            "Converts NLG data between formats. If you're migrating from 1.x, "
            "please run `rasa data convert responses` to adapt the training data "
            "to the new response selector format."
        ),
    )
    convert_nlg_parser.set_defaults(func=_convert_nlg_data)

    arguments.set_convert_arguments(convert_nlg_parser, data_type="Rasa NLG")

    convert_core_parser = convert_subparsers.add_parser(
        "core",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="Converts Core data between formats.",
    )
    convert_core_parser.set_defaults(func=_convert_core_data)

    arguments.set_convert_arguments(convert_core_parser, data_type="Rasa Core")

    migrate_config_parser = convert_subparsers.add_parser(
        "config",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="Migrate model configuration between Rasa Open Source versions.",
    )
    migrate_config_parser.set_defaults(func=_migrate_model_config)
    default_arguments.add_config_param(migrate_config_parser)
    default_arguments.add_domain_param(migrate_config_parser)
    default_arguments.add_out_param(
        migrate_config_parser,
        default=os.path.join(DEFAULT_DATA_PATH, "rules.yml"),
        help_text="Path to the file which should contain any rules which are created "
        "as part of the migration. If the file doesn't exist, it will be created.",
    )

    convert_responses_parser = convert_subparsers.add_parser(
        "responses",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help=(
            "Convert retrieval intent responses between Rasa Open Source versions. "
            "Please also run `rasa data convert nlg` to convert training data files "
            "to the right format."
        ),
    )
    convert_responses_parser.set_defaults(func=_migrate_responses)
    arguments.set_convert_arguments(convert_responses_parser, data_type="Rasa stories")
    default_arguments.add_domain_param(convert_responses_parser)


def _add_data_split_parsers(
    data_subparsers, parents: List[argparse.ArgumentParser]
) -> None:
    split_parser = data_subparsers.add_parser(
        "split",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="Splits Rasa data into training and test data.",
    )
    split_parser.set_defaults(func=lambda _: split_parser.print_help(None))

    split_subparsers = split_parser.add_subparsers()
    nlu_split_parser = split_subparsers.add_parser(
        "nlu",
        parents=parents,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Performs a split of your NLU data into training and test data "
        "according to the specified percentages.",
    )
    nlu_split_parser.set_defaults(func=split_nlu_data)

    arguments.set_split_arguments(nlu_split_parser)


def _add_data_validate_parsers(
    data_subparsers, parents: List[argparse.ArgumentParser]
) -> None:
    validate_parser = data_subparsers.add_parser(
        "validate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="Validates domain and data files to check for possible mistakes.",
    )
    _append_story_structure_arguments(validate_parser)
    validate_parser.set_defaults(func=validate_files)
    arguments.set_validator_arguments(validate_parser)

    validate_subparsers = validate_parser.add_subparsers()
    story_structure_parser = validate_subparsers.add_parser(
        "stories",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="Checks for inconsistencies in the story files.",
    )
    _append_story_structure_arguments(story_structure_parser)
    story_structure_parser.set_defaults(func=validate_stories)
    arguments.set_validator_arguments(story_structure_parser)


def _add_data_suggest_parsers(
    data_subparsers, parents: List[argparse.ArgumentParser]
) -> None:
    suggest_parser = data_subparsers.add_parser(
        "suggest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=parents,
        help="Suggests data to be added to the training data.",
    )
    # suggest_parser.set_defaults(func=???)
    arguments.set_suggest_arguments(suggest_parser)

    suggest_subparsers = suggest_parser.add_subparsers()
    nlu_suggest_parser = suggest_subparsers.add_parser(
        "nlu",
        parents=parents,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Suggests data to be added to your NLU training data.",
    )
    nlu_suggest_parser.set_defaults(func=suggest_nlu_data)

    arguments.set_suggest_arguments(nlu_suggest_parser)


def _append_story_structure_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-history",
        type=int,
        default=None,
        help="Number of turns taken into account for story structure validation.",
    )


def split_nlu_data(args: argparse.Namespace) -> None:
    """Load data from a file path and split the NLU data into test and train examples.

    Args:
        args: Commandline arguments
    """
    data_path = rasa.cli.utils.get_validated_path(args.nlu, "nlu", DEFAULT_DATA_PATH)
    data_path = rasa.shared.data.get_nlu_directory(data_path)

    nlu_data = rasa.shared.nlu.training_data.loading.load_data(data_path)
    extension = rasa.shared.nlu.training_data.util.get_file_format_extension(data_path)

    train, test = nlu_data.train_test_split(args.training_fraction, args.random_seed)

    train.persist(args.out, filename=f"training_data{extension}")
    test.persist(args.out, filename=f"test_data{extension}")

    telemetry.track_data_split(args.training_fraction, "nlu")


def _create_paraphrase_pool(
    paraphrases: TrainingData, pooled_intents: Set, paraphrase_quality_threshold: float
) -> Dict[Text, List]:
    paraphrase_pool = collections.defaultdict(list)
    for p in paraphrases.intent_examples:
        if p.data["intent"] in pooled_intents:
            paraphrases_for_example = p.data["metadata"]["example"]["paraphrases"].split("\n")
            paraphrase_scores = p.data["metadata"]["example"]["scores"].split("\n")

            for paraphrase, score in zip(paraphrases_for_example, paraphrase_scores):
                if paraphrase == "" or float(score) < paraphrase_quality_threshold: continue

                paraphrase_pool[p.data["intent"]].append((p, set(paraphrase.lower().split()), paraphrase))

    return paraphrase_pool


def _create_training_data_pool(
    nlu_training_data: TrainingData, pooled_intents: Set
) -> Tuple[Dict[Text, List], Dict[Text, Set]]:
    training_data_pool = collections.defaultdict(list)
    training_data_vocab_per_intent = collections.defaultdict(set)
    for m in nlu_training_data.intent_examples:
        training_data_pool[m.data["intent"]].append(m)
        if m.data["intent"] in pooled_intents:
            training_data_vocab_per_intent[m.data["intent"]] |= set(
                m.data["text"].lower().split()
            )
    return training_data_pool, training_data_vocab_per_intent


def _build_diverse_augmentation_pool(
    paraphrase_pool: Dict[Text, List], training_data_vocab_per_intent: Dict[Text, List]
) -> Dict[Text, List]:
    max_vocab_expansion = collections.defaultdict(list)
    for intent in paraphrase_pool.keys():
        for p, vocab, paraphrase in paraphrase_pool[intent]:
            num_new_words = len(vocab - training_data_vocab_per_intent[intent])
            max_vocab_expansion[intent].append((num_new_words, p, paraphrase))
        max_vocab_expansion[intent] = sorted(
            max_vocab_expansion[intent], key=operator.itemgetter(0), reverse=True
        )
    return max_vocab_expansion


def _build_random_augmentation_pool(
    paraphrase_pool: Dict[Text, List]
) -> Dict[Text, List]:
    shuffled_paraphrases = {}
    for intent in paraphrase_pool.keys():
        shuffled_paraphrases[intent] = random.sample(
            paraphrase_pool[intent], len(paraphrase_pool[intent])
        )
    return shuffled_paraphrases


def _build_augmentation_training_sets(
    nlu_training_data: TrainingData,
    training_data_pool: Dict[Text, Set],
    random_expansion: Dict[Text, List],
    max_vocab_expansion: Dict[Text, List],
    pooled_intents: Set,
    avg_size: int,
) -> Tuple[TrainingData, TrainingData]:
    augmented_training_set_diverse = []
    augmented_training_set_random = []
    for intent in nlu_training_data.intents:
        augmented_training_set_diverse.extend(training_data_pool[intent])
        augmented_training_set_random.extend(training_data_pool[intent])
        if intent in pooled_intents:
            # Handle augmentation based on random sampling
            augmented_training_set_random.extend(
                list(map(lambda item: item[0], random_expansion[intent]))
            )

            # Handle augmentation based on the diversity criterion (maximum vocabulary expansion)
            diff = avg_size - nlu_training_data.number_of_examples_per_intent[intent]
            if diff > 0:
                augmentation_examples = list(
                    map(lambda item: item[1], max_vocab_expansion[intent][:diff])
                )
                augmented_training_set_diverse.extend(augmentation_examples)

    return (
        TrainingData(training_examples=augmented_training_set_diverse),
        TrainingData(training_examples=augmented_training_set_random),
    )


def _get_intents_with_performance_changes(
    classification_report: Dict[Text, Dict[Text, float]],
    intent_report: Dict[Text, float],
    all_intents: List[Text],
    significant_figures: int = 2,
) -> Set[Text]:
    changed_intents = set()
    for intent_key in all_intents:
        for metric in ["precision", "recall", "f1-score"]:
            rounded_original = round(
                classification_report[intent_key][metric], significant_figures
            )
            rounded_augmented = round(
                intent_report[intent_key][metric], significant_figures
            )
            if (rounded_original != rounded_augmented):
                changed_intents.add(intent_key)

    return changed_intents


def _create_augmentation_summary(
    pooled_intents: Set,
    changed_intents: Set,
    classification_report: Dict[Text, Dict[Text, float]],
    intent_report: Dict[Text, float]
) -> Tuple[
    Dict[Text, Dict[Text, float]],
    Dict[Text, float],
]:

    intent_summary = collections.defaultdict(dict)
    for intent in (
        pooled_intents | changed_intents | {"micro avg", "macro avg", "weighted avg"}
    ):
        if intent not in classification_report.keys():
            continue

        intent_results_original = classification_report[intent]
        intent_results = intent_report[intent]

        # Record performance changes for augmentation based on the diversity criterion
        precision_change = (
            intent_results["precision"] - intent_results_original["precision"]
        )
        recall_change = (
            intent_results["recall"] - intent_results_original["recall"]
        )
        f1_change = (
            intent_results["f1-score"] - intent_results_original["f1-score"]
        )

        intent_results["precision_change"] = intent_summary[intent][
            "precision_change"
        ] = precision_change
        intent_results["recall_change"] = intent_summary[intent][
            "recall_change"
        ] = recall_change
        intent_results["f1-score_change"] = intent_summary[intent][
            "f1-score_change"
        ] = f1_change
        intent_report[intent] = intent_results

    return (
        intent_summary,
        intent_report
    )


def _plot_summary_reports(
    intent_summary_diverse: Dict[Text, Dict[Text, float]],
    intent_summary_random: Dict[Text, Dict[Text, float]],
    changed_intents_diverse: Set[Text],
    changed_intents_random: Set[Text],
    output_directory_diverse: Text,
    output_directory_random: Text,
):

    for metric in ["precision", "recall", "f1-score"]:
        output_file_diverse = os.path.join(
            output_directory_diverse, f"{metric}_changes.png"
        )
        rasa.utils.plotting.plot_intent_augmentation_summary(
            augmentation_summary=intent_summary_diverse,
            changed_intents=changed_intents_diverse,
            metric=metric,
            output_file=output_file_diverse,
        )

        output_file_random = os.path.join(
            output_directory_random, f"{metric}_changes.png"
        )
        rasa.utils.plotting.plot_intent_augmentation_summary(
            augmentation_summary=intent_summary_random,
            changed_intents=changed_intents_random,
            metric=metric,
            output_file=output_file_random,
        )


def suggest_nlu_data(args: argparse.Namespace) -> None:
    """Load NLU training & evaluation data, paraphrases and classification report and suggest additional training
     examples.

    Args:
        args: Commandline arguments
    """
    nlu_training_data = rasa.shared.nlu.training_data.loading.load_data(
        args.nlu_training_data
    )
    nlu_evaluation_data = rasa.shared.nlu.training_data.loading.load_data(
        args.nlu_evaluation_data
    )
    paraphrases = rasa.shared.nlu.training_data.loading.load_data(args.paraphrases)
    classification_report = rasa.shared.utils.io.read_json_file(
        args.nlu_classification_report
    )
    random.seed(args.random_seed)

    # Determine low data, low performing and frequently confused intents
    num_intents = len(nlu_training_data.intents)
    avg_size = int(
        reduce(
            lambda acc, num: acc + (num / num_intents),
            nlu_training_data.number_of_examples_per_intent.values(),
            0,
        )
    )

    low_data_intents = sorted(
        nlu_training_data.number_of_examples_per_intent.items(),
        key=operator.itemgetter(1),
    )[: args.num_intents]
    low_precision_intents = sorted(
        map(
            lambda k: (k, classification_report[k]["precision"]),
            nlu_training_data.intents,
        ),
        key=operator.itemgetter(1),
    )[: args.num_intents]
    low_recall_intents = sorted(
        map(
            lambda k: (k, classification_report[k]["recall"]), nlu_training_data.intents
        ),
        key=operator.itemgetter(1),
    )[: args.num_intents]
    low_f1_intents = sorted(
        map(
            lambda k: (k, classification_report[k]["f1-score"]),
            nlu_training_data.intents,
        ),
        key=operator.itemgetter(1),
    )[: args.num_intents]
    freq_confused_intents = sorted(
        map(
            lambda k: (k, sum(classification_report[k]["confused_with"].values())),
            nlu_training_data.intents,
        ),
        key=operator.itemgetter(1),
        reverse=True,
    )[: args.num_intents]

    pooled_intents = (
        set(map(lambda tp: tp[0], low_data_intents))
        | set(map(lambda tp: tp[0], low_precision_intents))
        | set(map(lambda tp: tp[0], low_recall_intents))
        | set(map(lambda tp: tp[0], low_f1_intents))
        | set(map(lambda tp: tp[0], freq_confused_intents))
    )

    # Retrieve paraphrase pool and training data pool
    paraphrase_pool = _create_paraphrase_pool(paraphrases, pooled_intents, args.paraphrase_score_threshold)
    training_data_pool, training_data_vocab_per_intent = _create_training_data_pool(
        nlu_training_data, pooled_intents
    )

    # Build augmentation pools based on the maximum vocabulary expansion criterion ("diverse") and random sampling
    max_vocab_expansion = _build_diverse_augmentation_pool(
        paraphrase_pool, training_data_vocab_per_intent
    )
    random_expansion = _build_random_augmentation_pool(paraphrase_pool)

    # Build augmentation training set
    augmented_data_diverse, augmented_data_random = _build_augmentation_training_sets(
        nlu_training_data,
        training_data_pool,
        random_expansion,
        max_vocab_expansion,
        pooled_intents,
        avg_size,
    )

    # Store training data files
    output_directory_diverse = os.path.join(args.out, "augmentation_diverse")
    if not os.path.exists(output_directory_diverse):
        os.makedirs(output_directory_diverse)
    output_directory_random = os.path.join(args.out, "augmentation_random")
    if not os.path.exists(output_directory_random):
        os.makedirs(output_directory_random)

    out_file_diverse = os.path.join(
        output_directory_diverse, "train_augmented_diverse.yml"
    )
    augmented_data_diverse.persist_nlu(filename=out_file_diverse)
    out_file_random = os.path.join(
        output_directory_random, "train_augmented_random.yml"
    )
    augmented_data_random.persist_nlu(filename=out_file_random)

    # Train NLU models on diverse and random augmentation sets
    model_path_diverse = train_nlu(
        config=args.config,
        nlu_data=out_file_diverse,
        output=output_directory_diverse,
        domain=args.domain,
    )

    model_path_random = train_nlu(
        config=args.config,
        nlu_data=out_file_random,
        output=output_directory_random,
        domain=args.domain,
    )

    # Evaluate NLU models on NLU evaluation data
    unpacked_model_path_diverse = get_model(model_path_diverse)
    nlu_model_path_diverse = os.path.join(unpacked_model_path_diverse, "nlu")

    interpreter = Interpreter.load(nlu_model_path_diverse)
    interpreter.pipeline = remove_pretrained_extractors(interpreter.pipeline)

    (intent_results, *_) = get_eval_data(interpreter, nlu_evaluation_data)
    intent_report_diverse = create_intent_report(
        intent_results=intent_results,
        add_confused_labels_to_report=True,
        metrics_as_dict=True,
    )
    intent_errors_diverse = extract_intent_errors_from_results(
        intent_results=intent_results
    )
    rasa.shared.utils.io.dump_obj_as_json_to_file(
        os.path.join(output_directory_diverse, "intent_errors.json"),
        intent_errors_diverse,
    )

    unpacked_model_random = get_model(model_path_random)
    nlu_model_path_random = os.path.join(unpacked_model_random, "nlu")

    interpreter = Interpreter.load(nlu_model_path_random)
    interpreter.pipeline = remove_pretrained_extractors(interpreter.pipeline)

    (intent_results, *_) = get_eval_data(interpreter, nlu_evaluation_data)
    intent_report_random = create_intent_report(
        intent_results=intent_results,
        add_confused_labels_to_report=True,
        metrics_as_dict=True,
    )
    intent_errors_random = extract_intent_errors_from_results(
        intent_results=intent_results
    )
    rasa.shared.utils.io.dump_obj_as_json_to_file(
        os.path.join(output_directory_random, "intent_errors.json"),
        intent_errors_random,
    )

    # Retrieve intents for which performance has changed
    changed_intents_diverse = _get_intents_with_performance_changes(
        classification_report, intent_report_diverse.report, nlu_training_data.intents
    ) - pooled_intents

    changed_intents_random = _get_intents_with_performance_changes(
        classification_report, intent_report_random.report, nlu_training_data.intents
    ) - pooled_intents

    # Create and update result reports
    report_tuple = _create_augmentation_summary(
        pooled_intents,
        changed_intents_diverse,
        classification_report,
        intent_report_diverse.report
    )

    intent_summary_diverse = report_tuple[0]
    intent_report_diverse.report.update(report_tuple[1])

    report_tuple = _create_augmentation_summary(
        pooled_intents,
        changed_intents_random,
        classification_report,
        intent_report_random.report
    )
    intent_summary_random = report_tuple[0]
    intent_report_random.report.update(report_tuple[1])

    # Store reports to file
    rasa.shared.utils.io.dump_obj_as_json_to_file(
        os.path.join(output_directory_diverse, "intent_report.json"),
        intent_report_diverse.report,
    )
    rasa.shared.utils.io.dump_obj_as_json_to_file(
        os.path.join(output_directory_random, "intent_report.json"),
        intent_report_random.report,
    )

    # Plot the summary reports
    _plot_summary_reports(
        intent_summary_diverse,
        intent_summary_random,
        changed_intents_diverse,
        changed_intents_random,
        output_directory_diverse,
        output_directory_random
    )

    telemetry.track_data_suggest()


def validate_files(args: argparse.Namespace, stories_only: bool = False) -> None:
    """Validates either the story structure or the entire project.

    Args:
        args: Commandline arguments
        stories_only: If `True`, only the story structure is validated.
    """
    file_importer = RasaFileImporter(
        domain_path=args.domain, training_data_paths=args.data
    )

    validator = rasa.utils.common.run_in_loop(Validator.from_importer(file_importer))

    if stories_only:
        all_good = _validate_story_structure(validator, args)
    else:
        all_good = (
            _validate_domain(validator)
            and _validate_nlu(validator, args)
            and _validate_story_structure(validator, args)
        )

    telemetry.track_validate_files(all_good)
    if not all_good:
        rasa.shared.utils.cli.print_error_and_exit(
            "Project validation completed with errors."
        )


def validate_stories(args: argparse.Namespace) -> None:
    """Validates that training data file content conforms to training data spec.

    Args:
        args: Commandline arguments
    """
    validate_files(args, stories_only=True)


def _validate_domain(validator: Validator) -> bool:
    return validator.verify_domain_validity()


def _validate_nlu(validator: Validator, args: argparse.Namespace) -> bool:
    return validator.verify_nlu(not args.fail_on_warnings)


def _validate_story_structure(validator: Validator, args: argparse.Namespace) -> bool:
    # Check if a valid setting for `max_history` was given
    if isinstance(args.max_history, int) and args.max_history < 1:
        raise argparse.ArgumentTypeError(
            f"The value of `--max-history {args.max_history}` is not a positive integer."
        )

    return validator.verify_story_structure(
        not args.fail_on_warnings, max_history=args.max_history
    )


def _convert_nlu_data(args: argparse.Namespace) -> None:
    from rasa.nlu.training_data.converters.nlu_markdown_to_yaml_converter import (
        NLUMarkdownToYamlConverter,
    )

    if args.format in ["json", "md"]:
        rasa.nlu.convert.convert_training_data(
            args.data, args.out, args.format, args.language
        )
        telemetry.track_data_convert(args.format, "nlu")
    elif args.format == "yaml":
        rasa.utils.common.run_in_loop(
            _convert_to_yaml(args.out, args.data, NLUMarkdownToYamlConverter())
        )
        telemetry.track_data_convert(args.format, "nlu")
    else:
        rasa.shared.utils.cli.print_error_and_exit(
            "Could not recognize output format. Supported output formats: 'json', "
            "'md', 'yaml'. Specify the desired output format with '--format'."
        )


def _convert_core_data(args: argparse.Namespace) -> None:
    from rasa.core.training.converters.story_markdown_to_yaml_converter import (
        StoryMarkdownToYamlConverter,
    )

    if args.format == "yaml":
        rasa.utils.common.run_in_loop(
            _convert_to_yaml(args.out, args.data, StoryMarkdownToYamlConverter())
        )
        telemetry.track_data_convert(args.format, "core")
    else:
        rasa.shared.utils.cli.print_error_and_exit(
            "Could not recognize output format. Supported output formats: "
            "'yaml'. Specify the desired output format with '--format'."
        )


def _convert_nlg_data(args: argparse.Namespace) -> None:
    from rasa.nlu.training_data.converters.nlg_markdown_to_yaml_converter import (
        NLGMarkdownToYamlConverter,
    )

    if args.format == "yaml":
        rasa.utils.common.run_in_loop(
            _convert_to_yaml(args.out, args.data, NLGMarkdownToYamlConverter())
        )
        telemetry.track_data_convert(args.format, "nlg")
    else:
        rasa.shared.utils.cli.print_error_and_exit(
            "Could not recognize output format. Supported output formats: "
            "'yaml'. Specify the desired output format with '--format'."
        )


def _migrate_responses(args: argparse.Namespace) -> None:
    """Migrate retrieval intent responses to the new 2.0 format.
    It does so modifying the stories and domain files.
    """
    if args.format == "yaml":
        rasa.utils.common.run_in_loop(
            _convert_to_yaml(args.out, args.domain, DomainResponsePrefixConverter())
        )
        rasa.utils.common.run_in_loop(
            _convert_to_yaml(args.out, args.data, StoryResponsePrefixConverter())
        )
        telemetry.track_data_convert(args.format, "responses")
    else:
        rasa.shared.utils.cli.print_error_and_exit(
            "Could not recognize output format. Supported output formats: "
            "'yaml'. Specify the desired output format with '--format'."
        )


async def _convert_to_yaml(
    out_path: Text, data_path: Text, converter: TrainingDataConverter
) -> None:

    output = Path(out_path)
    if not os.path.exists(output):
        rasa.shared.utils.cli.print_error_and_exit(
            f"The output path '{output}' doesn't exist. Please make sure to specify "
            f"an existing directory and try again."
        )

    training_data = Path(data_path)
    if not os.path.exists(training_data):
        rasa.shared.utils.cli.print_error_and_exit(
            f"The training data path {training_data} doesn't exist "
            f"and will be skipped."
        )

    num_of_files_converted = 0

    if os.path.isfile(training_data):
        if await _convert_file_to_yaml(training_data, output, converter):
            num_of_files_converted += 1
    elif os.path.isdir(training_data):
        for root, _, files in os.walk(training_data, followlinks=True):
            for f in sorted(files):
                source_path = Path(os.path.join(root, f))
                if await _convert_file_to_yaml(source_path, output, converter):
                    num_of_files_converted += 1

    if num_of_files_converted:
        rasa.shared.utils.cli.print_info(
            f"Converted {num_of_files_converted} file(s), saved in '{output}'."
        )
    else:
        rasa.shared.utils.cli.print_warning(
            f"Didn't convert any files under '{training_data}' path. "
            "Did you specify the correct file/directory?"
        )


async def _convert_file_to_yaml(
    source_file: Path, target_dir: Path, converter: TrainingDataConverter
) -> bool:
    """Converts a single training data file to `YAML` format.

    Args:
        source_file: Training data file to be converted.
        target_dir: Target directory for the converted file.
        converter: Converter to be used.

    Returns:
        `True` if file was converted, `False` otherwise.
    """
    if not rasa.shared.data.is_valid_filetype(source_file):
        return False

    if converter.filter(source_file):
        await converter.convert_and_write(source_file, target_dir)
        return True

    rasa.shared.utils.cli.print_warning(f"Skipped file: '{source_file}'.")

    return False


def _migrate_model_config(args: argparse.Namespace) -> None:
    """Migrates old "rule-like" policies to the new `RulePolicy`.

    Updates the config, domain, and generates the required rules.

    Args:
        args: The commandline args with the required paths.
    """
    configuration_file = Path(args.config)
    model_configuration = _get_configuration(configuration_file)

    domain_file = Path(args.domain)
    domain = _get_domain(domain_file)

    rule_output_file = _get_rules_path(args.out)

    (
        model_configuration,
        domain,
        new_rules,
    ) = rasa.core.config.migrate_mapping_policy_to_rules(model_configuration, domain)

    model_configuration, fallback_rule = rasa.core.config.migrate_fallback_policies(
        model_configuration
    )

    if new_rules:
        _backup(domain_file)
        domain.persist_clean(domain_file)

    if fallback_rule:
        new_rules.append(fallback_rule)

    if new_rules:
        _backup(configuration_file)
        rasa.shared.utils.io.write_yaml(model_configuration, configuration_file)
        _dump_rules(rule_output_file, new_rules)

    telemetry.track_data_convert("yaml", "config")

    _print_success_message(new_rules, rule_output_file)


def _get_configuration(path: Path) -> Dict:
    config = {}
    try:
        config = rasa.shared.utils.io.read_config_file(path)
    except Exception:
        rasa.shared.utils.cli.print_error_and_exit(
            f"'{path}' is not a path to a valid model configuration. "
            f"Please provide a valid path."
        )

    policy_names = [p.get("name") for p in config.get("policies", [])]

    _assert_config_needs_migration(policy_names)
    _assert_nlu_pipeline_given(config, policy_names)
    _assert_two_stage_fallback_policy_is_migratable(config)
    _assert_only_one_fallback_policy_present(policy_names)

    if FormPolicy.__name__ in policy_names:
        _warn_about_manual_forms_migration()

    return config


def _assert_config_needs_migration(policies: List[Text]) -> None:
    migratable_policies = {
        MappingPolicy.__name__,
        FallbackPolicy.__name__,
        TwoStageFallbackPolicy.__name__,
    }

    if not migratable_policies.intersection((set(policies))):
        rasa.shared.utils.cli.print_error_and_exit(
            f"No policies were found which need migration. This command can migrate "
            f"'{MappingPolicy.__name__}', '{FallbackPolicy.__name__}' and "
            f"'{TwoStageFallbackPolicy.__name__}'."
        )


def _warn_about_manual_forms_migration() -> None:
    rasa.shared.utils.cli.print_warning(
        f"Your model configuration contains the '{FormPolicy.__name__}'. "
        f"Note that this command does not migrate the '{FormPolicy.__name__}' and "
        f"you have to migrate the '{FormPolicy.__name__}' manually. "
        f"Please see the migration guide for further details: "
        f"{DOCS_URL_MIGRATION_GUIDE}"
    )


def _assert_nlu_pipeline_given(config: Dict, policy_names: List[Text]) -> None:
    if not config.get("pipeline") and any(
        policy in policy_names
        for policy in [FallbackPolicy.__name__, TwoStageFallbackPolicy.__name__]
    ):
        rasa.shared.utils.cli.print_error_and_exit(
            "The model configuration has to include an NLU pipeline. This is required "
            "in order to migrate the fallback policies."
        )


def _assert_two_stage_fallback_policy_is_migratable(config: Dict) -> None:
    two_stage_fallback_config = next(
        (
            policy_config
            for policy_config in config.get("policies", [])
            if policy_config.get("name") == TwoStageFallbackPolicy.__name__
        ),
        None,
    )
    if not two_stage_fallback_config:
        return

    if (
        two_stage_fallback_config.get(
            "deny_suggestion_intent_name", USER_INTENT_OUT_OF_SCOPE
        )
        != USER_INTENT_OUT_OF_SCOPE
    ):
        rasa.shared.utils.cli.print_error_and_exit(
            f"The TwoStageFallback in Rasa Open Source 2.0 has to use the intent "
            f"'{USER_INTENT_OUT_OF_SCOPE}' to recognize when users deny suggestions. "
            f"Please change the parameter 'deny_suggestion_intent_name' to "
            f"'{USER_INTENT_OUT_OF_SCOPE}' before migrating the model configuration. "
        )

    if (
        two_stage_fallback_config.get(
            "fallback_nlu_action_name", ACTION_DEFAULT_FALLBACK_NAME
        )
        != ACTION_DEFAULT_FALLBACK_NAME
    ):
        rasa.shared.utils.cli.print_error_and_exit(
            f"The Two-Stage Fallback in Rasa Open Source 2.0 has to use the action "
            f"'{ACTION_DEFAULT_FALLBACK_NAME}' for cases when the user denies the "
            f"suggestion multiple times. "
            f"Please change the parameter 'fallback_nlu_action_name' to "
            f"'{ACTION_DEFAULT_FALLBACK_NAME}' before migrating the model "
            f"configuration. "
        )


def _assert_only_one_fallback_policy_present(policies: List[Text]) -> None:
    if (
        FallbackPolicy.__name__ in policies
        and TwoStageFallbackPolicy.__name__ in policies
    ):
        rasa.shared.utils.cli.print_error_and_exit(
            "Your policy configuration contains two configured policies for handling "
            "fallbacks. Please decide on one."
        )


def _get_domain(path: Path) -> Domain:
    try:
        return Domain.from_path(path)
    except InvalidDomain:
        rasa.shared.utils.cli.print_error_and_exit(
            f"'{path}' is not a path to a valid domain file. "
            f"Please provide a valid domain."
        )


def _get_rules_path(path: Text) -> Path:
    rules_file = Path(path)

    if rules_file.is_dir():
        rasa.shared.utils.cli.print_error_and_exit(
            f"'{rules_file}' needs to be the path to a file."
        )

    if not rules_file.is_file():
        rasa.shared.utils.cli.print_info(
            f"Output file '{rules_file}' did not exist and will be created."
        )
        rasa.shared.utils.io.create_directory_for_file(rules_file)

    return rules_file


def _dump_rules(path: Path, new_rules: List["StoryStep"]) -> None:
    existing_rules = []
    if path.exists():
        rules_reader = YAMLStoryReader()
        existing_rules = rules_reader.read_from_file(path)
        _backup(path)

    if existing_rules:
        rasa.shared.utils.cli.print_info(
            f"Found existing rules in the output file '{path}'. The new rules will "
            f"be appended to the existing rules."
        )

    rules_writer = YAMLStoryWriter()
    rules_writer.dump(path, existing_rules + new_rules)


def _backup(path: Path) -> None:
    backup_file = path.parent / f"{path.name}.bak"
    shutil.copy(path, backup_file)


def _print_success_message(new_rules: List["StoryStep"], output_file: Path) -> None:
    if len(new_rules) > 1:
        suffix = "rule"
        verb = "was"
    else:
        suffix = "rules"
        verb = "were"

    rasa.shared.utils.cli.print_success(
        f"Finished migrating your policy configuration 🎉.\n"
        f"The migration generated {len(new_rules)} {suffix} which {verb} added to "
        f"'{output_file}'."
    )

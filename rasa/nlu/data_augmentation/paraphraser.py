import collections
import operator
import os
import random
from typing import Any, Dict, List, Set, Text, Tuple

from rasa.model import get_model
from rasa.shared.core.domain import Domain
import rasa.shared.utils.io
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.train import train_nlu
from rasa.nlu.model import Interpreter
from rasa.nlu.test import (
    create_intent_report,
    get_eval_data,
    remove_pretrained_extractors,
    extract_intent_errors_from_results,
)
import rasa.utils.plotting


def collect_intents_for_data_augmentation(
    nlu_training_data: TrainingData,
    num_intents_to_augment: int,
    classification_report: Dict[Text, Dict[Text, Any]],
) -> Set[Text]:
    """Analyses the training dataset and extracts:
        * The `num_intents_to_augment` intents with the least data
        * The `num_intents_to_augment` with lowest precision (according to `classification_report`)
        * The `num_intents_to_augment` with lowest recall (according to `classification_report`)
        * The `num_intents_to_augment` with lowest f1-score (according to `classification_report`)
        * The `num_intents_to_augment` most frequently confused intents (according to `classification_report`)

    Args:
        nlu_training_data: The existing NLU training data.
        num_intents_to_augment: The number of intents to pick per criterion.
        classification_report: An existing classification report (without data augmentation).

    Returns:
        The set of intent names for which data augmentation will be performed.
    """

    # Determine low data, low performing and frequently confused intents
    low_data_intents = sorted(
        nlu_training_data.number_of_examples_per_intent.items(),
        key=operator.itemgetter(1),
    )[:num_intents_to_augment]
    low_precision_intents = sorted(
        [
            (intent, classification_report[intent]["precision"])
            for intent in nlu_training_data.intents
        ],
        key=operator.itemgetter(1),
    )[:num_intents_to_augment]
    low_recall_intents = sorted(
        [
            (intent, classification_report[intent]["recall"])
            for intent in nlu_training_data.intents
        ],
        key=operator.itemgetter(1),
    )[:num_intents_to_augment]
    low_f1_intents = sorted(
        [
            (intent, classification_report[intent]["f1-score"])
            for intent in nlu_training_data.intents
        ],
        key=operator.itemgetter(1),
    )[:num_intents_to_augment]
    freq_confused_intents = sorted(
        [
            (intent, sum(classification_report[intent]["confused_with"].values()))
            for intent in nlu_training_data.intents
        ],
        key=operator.itemgetter(1),
        reverse=True,
    )[:num_intents_to_augment]

    pooled_intents = {
        tp[0]
        for tp in low_data_intents
        + low_precision_intents
        + low_recall_intents
        + low_f1_intents
        + freq_confused_intents
    }

    return pooled_intents


def create_paraphrase_pool(
    paraphrases: TrainingData,
    pooled_intents: Set[Text],
    paraphrase_quality_threshold: float,
) -> Dict[Text, List]:
    """Determines all suitable paraphrases for data augmentation for the given intents.

    Args:
        paraphrases: The paraphrases for data augmentation.
        pooled_intents: The intents for which to perform data augmentation.
        paraphrase_quality_threshold: Accept/Reject threshold for individual paraphrases.

    Returns:
        The pool of suitable paraphrases for data augmentation.
    """

    paraphrase_pool = collections.defaultdict(list)
    for p in paraphrases.intent_examples:
        if p.data["intent"] not in pooled_intents:
            continue

        paraphrases_for_example = p.data["metadata"]["example"]["paraphrases"].split(
            "\n"
        )
        paraphrase_scores = p.data["metadata"]["example"]["scores"].split("\n")

        for paraphrase, score in zip(paraphrases_for_example, paraphrase_scores):
            if paraphrase == "" or float(score) < paraphrase_quality_threshold:
                continue

            paraphrase_pool[p.data["intent"]].append(
                (p, set(paraphrase.lower().split()), paraphrase)
            )

    return paraphrase_pool


def create_training_data_pool(
    nlu_training_data: TrainingData, pooled_intents: Set
) -> Tuple[Dict[Text, List], Dict[Text, Set]]:
    """Determines the existing available training data for the given set of intents and extracts their vocabulary.

    Args:
        nlu_training_data: The existing NLU training data.
        pooled_intents: The intents for which to perform data augmentation.

    Returns:
        The available existing training data for the given set of intents and their vocabulary
    """
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
    paraphrase_pool: Dict[Text, List], training_data_vocab_per_intent: Dict[Text, Set]
) -> Dict[Text, List]:
    """Selects paraphrases for data augmentation on the basis of maximum vocabulary expansion between the existing training data for a given intent and the generated paraphrases.

    Args:
        paraphrase_pool: Paraphrases for intents that should be augmented.
        training_data_vocab_per_intent: Vocabulary per intent.

    Returns:
        Paraphrases, sorted (DESC) by vocabulary expansion, for data augmentation.
    """
    max_vocab_expansion = collections.defaultdict(list)
    for intent in paraphrase_pool.keys():
        for p, vocab, paraphrase in paraphrase_pool[intent]:
            num_new_words = len(vocab - training_data_vocab_per_intent[intent])
            max_vocab_expansion[intent].append((p, num_new_words, paraphrase))
        max_vocab_expansion[intent] = sorted(
            max_vocab_expansion[intent], key=operator.itemgetter(1), reverse=True
        )
    return max_vocab_expansion


def _build_random_augmentation_pool(
    paraphrase_pool: Dict[Text, List]
) -> Dict[Text, List]:
    """Randomly selects paraphrases for data augmentation from the generated pool.

    Args:
        paraphrase_pool: Paraphrases for intents that should be augmented.

    Returns:
        Paraphrases for data augmentation.
    """
    shuffled_paraphrases = {}
    for intent in paraphrase_pool.keys():
        shuffled_paraphrases[intent] = random.sample(
            paraphrase_pool[intent], len(paraphrase_pool[intent])
        )
    return shuffled_paraphrases


def _build_augmented_training_data(
    nlu_training_data: TrainingData,
    training_data_pool: Dict[Text, List],
    data_augmentation_pool: Dict[Text, List],
) -> TrainingData:
    """Creates a TrainingData object from the existing training data and the data augmentation pool.

    Args:
        nlu_training_data: Existing NLU training data.
        data_augmentation_pool: Paraphrases chosen for data augmentation.

    Return:
        Augmented TrainingData.

    """
    augmented_training_set = []
    for intent in nlu_training_data.intents:
        augmented_training_set.extend(training_data_pool[intent])
        if intent in data_augmentation_pool.keys():
            augmented_training_set.extend(
                [item[0] for item in data_augmentation_pool[intent]]
            )

    return TrainingData(training_examples=augmented_training_set)


def _train_test_nlu_model(
    output_directory: Text,
    nlu_training_file: Text,
    config: Text,
    nlu_evaluation_data: TrainingData,
) -> Dict[Text, float]:
    """Runs the NLU train/test loop using the given augmented training data.

    Args:
         output_directory: Directory to store the evaluation reports and augmented training data in.
         nlu_training_file: Augmented NLU training data file.
         config: NLU model config.
         nlu_evaluation_data: NLU evaluation data.

    Returns:
        Classification report of the NLU model trained on the augmented training data.
    """

    # Train NLU model
    model_path = train_nlu(
        config=config, nlu_data=nlu_training_file, output=output_directory
    )

    # Load and Evaluate NLU Model
    unpacked_model_path = get_model(model_path)
    nlu_model_path = os.path.join(unpacked_model_path, "nlu")

    interpreter = Interpreter.load(nlu_model_path)
    interpreter.pipeline = remove_pretrained_extractors(interpreter.pipeline)

    (intent_results, *_) = get_eval_data(interpreter, nlu_evaluation_data)
    intent_report = create_intent_report(
        intent_results=intent_results,
        add_confused_labels_to_report=True,
        metrics_as_dict=True,
    )
    intent_errors = extract_intent_errors_from_results(intent_results=intent_results)
    rasa.shared.utils.io.dump_obj_as_json_to_file(
        os.path.join(output_directory, "intent_errors.json"), intent_errors,
    )

    return intent_report.report


def _create_augmentation_summary(
    pooled_intents: Set[Text],
    changed_intents: Set[Text],
    classification_report: Dict[Text, Dict[Text, float]],
    intent_report: Dict[Text, float],
) -> Tuple[
    Dict[Text, Dict[Text, float]], Dict[Text, float],
]:
    """Creates a summary report of the effect of data augmentation and modifies the original classification report with that information.

    Args:
        pooled_intents: The intents that have been selected for data augmentation.
        changed_intents: The intents that have been affected by data augmentation.
        classification_report: Classification report of the model run *without* data augmentation.
        intent_report: Report of the model run *with* data augmentation.

    Returns:
        A tuple representing a summary of the changed intents as well as a modified version of the original classification report with performance changes for all affected intents.
    """

    intent_summary = collections.defaultdict(dict)
    for intent in (
        pooled_intents | changed_intents | {"micro avg", "macro avg", "weighted avg"}
    ):
        if intent not in classification_report:
            continue

        intent_results_original = classification_report[intent]
        intent_results = intent_report[intent]

        # Record performance changes for augmentation based on the diversity criterion
        precision_change = (
            intent_results["precision"] - intent_results_original["precision"]
        )
        recall_change = intent_results["recall"] - intent_results_original["recall"]
        f1_change = intent_results["f1-score"] - intent_results_original["f1-score"]

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

    return (intent_summary, intent_report)


def _create_summary_report(
    intent_report: Dict[Text, float],
    classification_report: Dict[Text, Dict[Text, Any]],
    training_intents: List[Text],
    pooled_intents: Set[Text],
    output_directory: Text,
) -> Tuple[Dict[Text, Dict[Text, float]], Set[Text]]:
    """Creates a summary of the effect of data augmentation.

    Args:
        intent_report: Report of the model run *with* data augmentation.
        classification_report: Classification report of the model run *without* data augmentation.
        training_intents: All intents in the training data (non-augmented).
        pooled_intents: The intents that have been selected for data augmentation.
        output_directory: Directory to store the output reports in.

    Returns:
        Tuple representing the data augmentation summary as well as the set of changed intents.
    """

    # Retrieve intents for which performance has changed
    changed_intents = (
        _get_intents_with_performance_changes(
            classification_report, intent_report, training_intents,
        )
        - pooled_intents
    )

    # Create and update result reports
    report_tuple = _create_augmentation_summary(
        pooled_intents, changed_intents, classification_report, intent_report,
    )

    intent_summary = report_tuple[0]
    intent_report.update(report_tuple[1])

    # Store report to file
    rasa.shared.utils.io.dump_obj_as_json_to_file(
        os.path.join(output_directory, "intent_report.json"), intent_report,
    )

    return intent_summary, changed_intents


def _plot_summary_report(
    intent_summary: Dict[Text, Dict[Text, float]],
    changed_intents: Set[Text],
    output_directory: Text,
) -> None:
    """
    Plots the data augmentation summary.

    Args:
        intent_summary: Summary report of the effect of data augmentation.
        changed_intents: Set of intents that were affected by data augmentation.
        output_directory: Directory to store the summary plot in.
    """
    for metric in ["precision", "recall", "f1-score"]:
        output_file_diverse = os.path.join(output_directory, f"{metric}_changes.png")
        rasa.utils.plotting.plot_intent_augmentation_summary(
            augmentation_summary=intent_summary,
            changed_intents=changed_intents,
            metric=metric,
            output_file=output_file_diverse,
        )


def _get_intents_with_performance_changes(
    classification_report: Dict[Text, Dict[Text, Any]],
    intent_report: Dict[Text, float],
    all_intents: List[Text],
    significant_figures: int = 2,
) -> Set[Text]:
    """
    Extracts the intents whose performance has changed.

    Args:
        classification_report: Classification report of the model run *without* data augmentation.
        intent_report: Report of the model run *with* data augmentation.
        all_intents: List of all intents.
        significant_figures: Significant figures to be taken into account when assessing whether the performance of an intent has changed.

    Returns:
        Set of intents that have changed - i.e. that were affected by data augmentation.
    """
    changed_intents = set()
    for intent_key in all_intents:
        for metric in ["precision", "recall", "f1-score"]:
            rounded_original = round(
                classification_report[intent_key][metric], significant_figures
            )
            rounded_augmented = round(
                intent_report[intent_key][metric], significant_figures
            )
            if rounded_original != rounded_augmented:
                changed_intents.add(intent_key)

    return changed_intents


def run_data_augmentation_max_vocab_expansion(
    nlu_training_data: TrainingData,
    paraphrase_pool: Dict[Text, List],
    training_data_pool: Dict[Text, List],
    pooled_intents: Set[Text],
    training_data_vocab_per_intent: Dict[Text, Set],
    output_directory: Text,
    config: Text,
    nlu_evaluation_data: TrainingData,
    classification_report: Dict[Text, Dict[Text, float]],
) -> None:
    """
    Runs the NLU train/test cycle with data augmentation (maximum vocabulary expansion criterion) and generates the reports and plots summarising the impact of data augmentation on model performance.

    Args:
        nlu_training_data: NLU training data (without data augmentation).
        paraphrase_pool: Available paraphrases for data augmentation.
        training_data_pool: Training examples grouped by intent.
        pooled_intents: The intents that have been selected for data augmentation.
        training_data_vocab_per_intent: Training data vocabulary per intent.
        output_directory: Directory to store the output files in.
        config: NLU model config.
        nlu_evaluation_data: NLU evaluation data.
        classification_report: Classification report of the model run *without* data augmentation.
    """
    # Build augmentation pool based on the maximum vocabulary expansion criterion ("diverse")
    max_vocab_expansion = _build_diverse_augmentation_pool(
        paraphrase_pool, training_data_vocab_per_intent
    )

    # Build new augmented training data
    augmented_training_data = _build_augmented_training_data(
        nlu_training_data=nlu_training_data,
        training_data_pool=training_data_pool,
        data_augmentation_pool=max_vocab_expansion,
    )

    # Store augmented training data to file
    rasa.shared.utils.io.create_directory(output_directory)
    nlu_training_file = os.path.join(output_directory, "train_augmented_diverse.yml")
    augmented_training_data.persist_nlu(filename=nlu_training_file)

    # Run NLU train/test loop
    intent_report = _train_test_nlu_model(
        output_directory=output_directory,
        nlu_training_file=nlu_training_file,
        config=config,
        nlu_evaluation_data=nlu_evaluation_data,
    )

    # Create data augmentation summary
    (intent_summary, changed_intents) = _create_summary_report(
        intent_report=intent_report,
        classification_report=classification_report,
        training_intents=nlu_training_data.intents,
        pooled_intents=pooled_intents,
        output_directory=output_directory,
    )

    # Plot data augmentation summary
    _plot_summary_report(
        intent_summary=intent_summary,
        changed_intents=changed_intents,
        output_directory=output_directory,
    )


def run_data_augmentation_random_sampling(
    nlu_training_data: TrainingData,
    paraphrase_pool: Dict[Text, List],
    training_data_pool: Dict[Text, List],
    pooled_intents: Set[Text],
    output_directory: Text,
    config: Text,
    nlu_evaluation_data: TrainingData,
    classification_report: Dict[Text, Dict[Text, float]],
) -> None:
    """Runs the NLU train/test cycle with data augmentation (random sampling) and generates the reports and plots summarising the impact of data augmentation on model performance.

    Args:
        nlu_training_data: NLU training data (without data augmentation).
        paraphrase_pool: Available paraphrases for data augmentation.
        training_data_pool: Training examples grouped by intent.
        pooled_intents: The intents that have been selected for data augmentation.
        output_directory: Directory to store the output files in.
        config: NLU model config.
        nlu_evaluation_data: NLU evaluation data.
        classification_report: Classification report of the model run *without* data augmentation.
    """
    # Build augmentation pools based on random sampling
    random_expansion = _build_random_augmentation_pool(paraphrase_pool)

    # Build new augmented training data
    augmented_training_data = _build_augmented_training_data(
        nlu_training_data=nlu_training_data,
        training_data_pool=training_data_pool,
        data_augmentation_pool=random_expansion,
    )

    # Store augmented training data to file
    rasa.shared.utils.io.create_directory(output_directory)
    nlu_training_file = os.path.join(output_directory, "train_augmented_random.yml")
    augmented_training_data.persist_nlu(filename=nlu_training_file)

    # Run NLU train/test loop
    intent_report = _train_test_nlu_model(
        output_directory=output_directory,
        nlu_training_file=nlu_training_file,
        config=config,
        nlu_evaluation_data=nlu_evaluation_data,
    )

    # Create data augmentation summary
    (intent_summary, changed_intents) = _create_summary_report(
        intent_report=intent_report,
        classification_report=classification_report,
        training_intents=nlu_training_data.intents,
        pooled_intents=pooled_intents,
        output_directory=output_directory,
    )

    # Plot data augmentation summary
    _plot_summary_report(
        intent_summary=intent_summary,
        changed_intents=changed_intents,
        output_directory=output_directory,
    )

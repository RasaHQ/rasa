import collections
import operator
import os
import random
from typing import Any, Dict, List, Set, Text, Tuple
import logging

from rasa.model import get_model
from rasa.shared.core.domain import Domain
import rasa.shared.utils.io
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.train import train_nlu
from rasa.nlu.model import Interpreter
from rasa.nlu.test import (
    create_intent_report,
    get_eval_data,
    remove_pretrained_extractors,
    get_intent_errors,
)
from rasa.shared.nlu.constants import (
    INTENT,
    METADATA,
    METADATA_EXAMPLE,
    METADATA_VOCABULARY,
    TEXT,
)
import rasa.utils.plotting

logger = logging.getLogger(__name__)


def _collect_intents_for_data_augmentation(
    nlu_training_data: TrainingData,
    intent_proportion: float,
    classification_report: Dict[Text, Dict[Text, Any]],
) -> Set[Text]:
    """Collects intents for which to perform data augmentation.

    It analyses the training datasets and extracts:
        * The `num_intents_to_augment` intents with the least number of training examples.
        * The `num_intents_to_augment` with lowest precision (according to `classification_report`)
        * The `num_intents_to_augment` with lowest recall (according to `classification_report`)
        * The `num_intents_to_augment` with lowest f1-score (according to `classification_report`)
        * The `num_intents_to_augment` most frequently confused intents (according to `classification_report`)

    For all of the intents matching the above criteria paraphrases will be used for data augmentation.

    Args:
        nlu_training_data: The existing NLU training data.
        intent_proportion: The proportion of intents (out of all intents)
            considered for data augmentation. The actual number of intents
            considered for data augmentation is determined on the basis of several
            factors, such as their current performance statistics or the number of
            available training examples.
        classification_report: An existing classification report (without data augmentation).

    Returns:
        The set of intent names for which data augmentation will be performed.
    """
    num_intents = len(nlu_training_data.number_of_examples_per_intent)
    num_intents_to_augment = round(num_intents * intent_proportion)

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
        intent_score_tuple[0]
        for intent_score_tuple in low_data_intents
        + low_precision_intents
        + low_recall_intents
        + low_f1_intents
        + freq_confused_intents
    }

    return pooled_intents


def _create_paraphrase_pool(
    paraphrases: TrainingData,
    intents_to_augment: Set[Text],
    min_paraphrase_sim_score: float,
    max_paraphrase_sim_score: float,
) -> Dict[Text, List]:
    """Determines all suitable paraphrases for data augmentation for the given intents.

    Args:
        paraphrases: The paraphrases for data augmentation.
        intents_to_augment: The intents for which to perform data augmentation.
        min_paraphrase_sim_score: Accept/Reject minimum similarity threshold for individual paraphrases.
        max_paraphrase_sim_score: Accept/Reject maximum similarity threshold for individual paraphrases.

    Returns:
        The pool of suitable paraphrases for data augmentation.
    """

    paraphrase_pool = collections.defaultdict(list)
    for paraphrase_msg in paraphrases.intent_examples:

        paraphrases_for_example = (
            paraphrase_msg.get(METADATA, {})
            .get(METADATA_EXAMPLE, {})
            .get("paraphrases", {})
        )
        if (
            paraphrase_msg.get(INTENT) not in intents_to_augment
            or not paraphrases_for_example
        ):
            continue

        paraphrase_scores = (
            paraphrase_msg.get(METADATA, {}).get(METADATA_EXAMPLE, {}).get("scores", {})
        )

        if not paraphrase_scores:
            logger.debug(
                f"""Skipping "{paraphrase_msg.get(TEXT)}" as no similarity scores were found for its paraphrases."""
            )

        for paraphrase, score in zip(paraphrases_for_example, paraphrase_scores):
            if (
                paraphrase == ""
                or float(score) < min_paraphrase_sim_score
                or float(score) > max_paraphrase_sim_score
            ):
                continue

            # Create Message-compatible data representation for the paraphrases
            data = {
                TEXT: paraphrase,
                INTENT: paraphrase_msg.get(INTENT),
                METADATA: {METADATA_VOCABULARY: set(paraphrase.lower().split())},
            }
            paraphrase_pool[paraphrase_msg.get(INTENT)].append(Message(data=data))

    return paraphrase_pool


def _resolve_augmentation_factor(
    nlu_training_data: TrainingData, augmentation_factor: float
) -> Dict[Text, int]:
    """Calculates how many paraphrases should maximally be added to the training data.

    Args:
        nlu_training_data: The existing NLU training data.
        augmentation_factor: Factor - as a multiple of the number of training data per intent - to determine the amount
            of paraphrases used for data augmentation.

    Returns:
        A dictionary specifying how many paraphrases may maximally be added per intent.
    """
    aug_factor = {}
    for key, val in nlu_training_data.number_of_examples_per_intent.items():
        augmentation_size = int(round(val * augmentation_factor))
        # Use `None` if the user passes e.g. -1 (indicating that all paraphrases should be used), because `None` d
        # oesn't affect slicing, i.e. my_list == my_list[:] == my_list[:None]
        aug_factor[key] = augmentation_size if augmentation_size > 0 else None

    return aug_factor


def _create_augmented_training_data_max_vocab_expansion(
    nlu_training_data: TrainingData,
    paraphrase_pool: Dict[Text, List],
    intents_to_augment: Set[Text],
    augmentation_factor: Dict[Text, int],
) -> TrainingData:
    """Selects paraphrases for data augmentation on the basis of maximum vocabulary expansion between the existing
        training data for a given intent and the generated paraphrases.

    Args:
        nlu_training_data: NLU training data (without augmentation).
        paraphrase_pool: Paraphrases for intents that should be augmented.
        intents_to_augment: Intents that should be augmented.
        augmentation_factor: Amount of data augmentation per intent.

    Returns:
        Augmented training data based on the maximum vocabulary expansion strategy
    """

    # Extract intent-level vocabulary for all intents that should be augmented
    intent_vocab = collections.defaultdict(set)
    filtered_training_data = nlu_training_data.filter_training_examples(
        condition=lambda ex: ex.get(INTENT) in intents_to_augment
    )
    for message in filtered_training_data.intent_examples:
        intent = message.get(INTENT)
        intent_vocab[intent] |= set(message.get(TEXT, "").lower().split())

    # Select paraphrases that maximise vocabulary expansion
    new_training_data = []
    for intent in paraphrase_pool.keys():
        max_vocab_expansion = []
        for message in paraphrase_pool[intent]:
            paraphrase_vocab = message.get(METADATA, {}).get(METADATA_VOCABULARY, set())
            num_new_words = len(paraphrase_vocab - intent_vocab[intent])

            max_vocab_expansion.append((num_new_words, message))

        # Creates `Message` objects from the list of all paraphrases, sorted by their vocabulary expansion
        new_training_data.extend(
            [
                Message(data={TEXT: item[1].get(TEXT), INTENT: item[1].get(INTENT)})
                for item in sorted(
                    max_vocab_expansion, key=operator.itemgetter(0), reverse=True
                )[: augmentation_factor[intent]]
            ]
        )

    augmented_training_data = TrainingData(
        training_examples=new_training_data + nlu_training_data.intent_examples
    )

    return augmented_training_data


def _create_augmented_training_data_random_sampling(
    nlu_training_data: TrainingData,
    paraphrase_pool: Dict[Text, List],
    intents_to_augment: Set[Text],
    augmentation_factor: Dict[Text, int],
    random_seed: int,
) -> TrainingData:
    """Randomly selects paraphrases for data augmentation from the generated paraphrase pool.

    Args:
        nlu_training_data: NLU training data (without augmentation).
        paraphrase_pool: Paraphrases for intents that should be augmented.
        intents_to_augment: Intents that should be augmented.
        augmentation_factor: Amount of data augmentation per intent.
        random_seed: Random seed for sampling paraphrases.

    Returns:
        Paraphrases for data augmentation.
    """

    random.seed(random_seed)
    new_training_data = []
    for intent in intents_to_augment:
        random.shuffle(paraphrase_pool[intent])

        # Creates `Message` objects from the randomly shuffled list of paraphrases
        new_training_data.extend(
            [
                Message(data={TEXT: item.get(TEXT), INTENT: item.get(INTENT)})
                for item in paraphrase_pool[intent][: augmentation_factor[intent]]
            ]
        )

    augmented_training_data = TrainingData(
        training_examples=new_training_data + nlu_training_data.intent_examples
    )

    return augmented_training_data


def _train_test_nlu_model(
    output_directory: Text,
    nlu_training_file: Text,
    config: Text,
    nlu_evaluation_data: TrainingData,
) -> Dict[Text, float]:
    """Runs the NLU train/test loop using the given augmented training data.

    Performs training a new NLU model with the augmented training set and subsequently evaluates the model
    on the test data. The trained model will be stored to the given output directory.

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
        intent_results=intent_results, output_directory=None, report_as_dict=True,
    )
    intent_errors = get_intent_errors(intent_results=intent_results)
    rasa.shared.utils.io.dump_obj_as_json_to_file(
        os.path.join(output_directory, "intent_errors.json"), intent_errors,
    )

    return intent_report["report"]


def _create_augmentation_summary(
    intents_to_augment: Set[Text],
    changed_intents: Set[Text],
    classification_report: Dict[Text, Dict[Text, float]],
    intent_report: Dict[Text, float],
) -> Tuple[
    Dict[Text, Dict[Text, float]], Dict[Text, float],
]:
    """Creates a summary report of the effect of data augmentation and modifies the original classification report
    with that information.

    Args:
        intents_to_augment: The intents that have been selected for data augmentation.
        changed_intents: The intents that have been affected by data augmentation.
        classification_report: Classification report of the model run *without* data augmentation.
        intent_report: Report of the model run *with* data augmentation.

    Returns:
        A tuple representing a summary of the changed intents as well as a modified version of the original
            classification report with performance changes for all affected intents.
    """

    intent_summary = collections.defaultdict(dict)

    # accuracy is the only non-dict like thing in the classification report, so it receives extra treatment
    if "accuracy" in classification_report:
        accuracy_change = intent_report["accuracy"] - classification_report["accuracy"]
        acc_dict = {
            "accuracy_change": accuracy_change,
            "accuracy": intent_report["accuracy"],
        }

        intent_report["accuracy"] = acc_dict
        intent_summary["accuracy"] = {"accuracy_change": accuracy_change}

    intents_affected_by_augmentation = (
        intents_to_augment
        | changed_intents
        | {"micro avg", "macro avg", "weighted avg"}
    )
    for intent in intents_affected_by_augmentation:
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
) -> Dict[Text, Dict[Text, float]]:
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

    return intent_summary


def _plot_summary_report(
    intent_summary: Dict[Text, Dict[Text, float]], output_directory: Text,
) -> None:
    """Plots the data augmentation summary.

    Args:
        intent_summary: Summary report of the effect of data augmentation.
        output_directory: Directory to store the summary plot in.
    """
    for metric in ["precision", "recall", "f1-score"]:
        output_file = os.path.join(output_directory, f"{metric}_changes.png")
        rasa.utils.plotting.plot_intent_augmentation_summary(
            augmentation_summary=intent_summary, metric=metric, output_file=output_file,
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
        significant_figures: Significant figures to be taken into account when assessing whether the performance of an
            intent has changed.

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


def _run_data_augmentation(
    nlu_training_data: TrainingData,
    intents_to_augment: Set[Text],
    nlu_training_file: Text,
    output_directory: Text,
    config: Text,
    nlu_evaluation_data: TrainingData,
    classification_report: Dict[Text, Dict[Text, float]],
) -> None:
    """
    Runs the NLU train/test cycle with data augmentation.

    Also, generate reports and plots summarising the impact of data augmentation on model performance.

    Args:
        nlu_training_data: The augmented NLU training data.
        intents_to_augment: The intents that were chosen for data augmentation.
        nlu_training_file: The file for storing the augmented NLU training data.
        output_directory: Directory to store the output files in.
        config: NLU model config.
        nlu_evaluation_data: NLU evaluation data.
        classification_report: Classification report of the model run *without* data augmentation.
    """

    # Store augmented training data to file
    nlu_training_data.persist_nlu(filename=nlu_training_file)

    # Run NLU train/test loop
    intent_report = _train_test_nlu_model(
        output_directory=output_directory,
        nlu_training_file=nlu_training_file,
        config=config,
        nlu_evaluation_data=nlu_evaluation_data,
    )

    # Create data augmentation summary
    intent_summary = _create_summary_report(
        intent_report=intent_report,
        classification_report=classification_report,
        training_intents=nlu_training_data.intents,
        pooled_intents=intents_to_augment,
        output_directory=output_directory,
    )

    # Plot data augmentation summary
    _plot_summary_report(
        intent_summary=intent_summary, output_directory=output_directory,
    )


def augment_nlu_data(
    nlu_training_data: TrainingData,
    nlu_evaluation_data: TrainingData,
    paraphrases: TrainingData,
    classification_report: Dict[Text, Dict[Text, float]],
    config: Text,
    intent_proportion: float,
    random_seed: int,
    min_paraphrase_sim_score: float,
    max_paraphrase_sim_score: float,
    output_directory: Text,
    augmentation_factor: float,
) -> None:
    """Performs data augmentation for NLU by evaluating two augmentation strategies.

    Args:
        nlu_training_data: NLU training data (without data augmentation).
        nlu_evaluation_data: NLU evaluation data.
        paraphrases: The generated paraphrases with similarity scores obtained
            from https://github.com/RasaHQ/paraphraser.
        classification_report: Classification report of the model run *without* data augmentation.
        config: NLU model config.
        intent_proportion: The proportion of intents (out of all intents) considered for data augmentation.
            The actual number of intents considered for data augmentation is determined on the basis of several factors,
            such as their current performance statistics or the number of available training examples.
        random_seed: Random seed for sampling the paraphrases.
        min_paraphrase_sim_score: Minimum required similarity for a generated paraphrase to be considered for data
            augmentation.
        max_paraphrase_sim_score: Maximum similarity for a generated paraphrase to be considered for data augmentation.
        output_directory: Directory to store the output files in.
        augmentation_factor: Factor - as a multiple of the number of training data per intent - to determine the amount
            of paraphrases used for data augmentation.
    """
    # Determine intents for which to perform data augmentation
    intents_to_augment = _collect_intents_for_data_augmentation(
        nlu_training_data=nlu_training_data,
        intent_proportion=intent_proportion,
        classification_report=classification_report,
    )

    logger.info(
        f"Picked {len(intents_to_augment)} intents for augmentation - {intents_to_augment}"
    )

    # Retrieve paraphrase pool and training data pool
    paraphrase_pool = _create_paraphrase_pool(
        paraphrases=paraphrases,
        intents_to_augment=intents_to_augment,
        min_paraphrase_sim_score=min_paraphrase_sim_score,
        max_paraphrase_sim_score=max_paraphrase_sim_score,
    )
    augmentation_factor_per_intent = _resolve_augmentation_factor(
        nlu_training_data=nlu_training_data, augmentation_factor=augmentation_factor
    )

    # Run data augmentation with diverse augmentation
    logger.info(
        "Running augmentation strategy by maximising vocabulary expansion per intent..."
    )
    max_vocab_expansion_training_file_path = os.path.join(
        output_directory, "augmentation_max_vocab_expansion"
    )
    rasa.shared.utils.io.create_directory(max_vocab_expansion_training_file_path)

    nlu_training_file_diverse = os.path.join(
        max_vocab_expansion_training_file_path,
        "nlu_train_augmented_max_vocab_expansion.yml",
    )
    nlu_max_vocab_augmentation_data = _create_augmented_training_data_max_vocab_expansion(
        nlu_training_data=nlu_training_data,
        paraphrase_pool=paraphrase_pool,
        intents_to_augment=intents_to_augment,
        augmentation_factor=augmentation_factor_per_intent,
    )

    _run_data_augmentation(
        nlu_training_data=nlu_max_vocab_augmentation_data,
        nlu_evaluation_data=nlu_evaluation_data,
        nlu_training_file=nlu_training_file_diverse,
        output_directory=max_vocab_expansion_training_file_path,
        classification_report=classification_report,
        intents_to_augment=intents_to_augment,
        config=config,
    )

    # Run data augmentation with random sampling augmentation
    logger.info("Running augmentation by picking random paraphrases...")
    random_training_file_path = os.path.join(output_directory, "augmentation_random")
    rasa.shared.utils.io.create_directory(random_training_file_path)

    nlu_training_file_random = os.path.join(
        random_training_file_path, "nlu_train_augmented_random.yml"
    )
    nlu_random_augmentation_data = _create_augmented_training_data_random_sampling(
        nlu_training_data=nlu_training_data,
        paraphrase_pool=paraphrase_pool,
        intents_to_augment=intents_to_augment,
        augmentation_factor=augmentation_factor_per_intent,
        random_seed=random_seed,
    )

    _run_data_augmentation(
        nlu_training_data=nlu_random_augmentation_data,
        nlu_evaluation_data=nlu_evaluation_data,
        nlu_training_file=nlu_training_file_random,
        output_directory=random_training_file_path,
        classification_report=classification_report,
        intents_to_augment=intents_to_augment,
        config=config,
    )

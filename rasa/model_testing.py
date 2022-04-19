import copy
import logging
import os
from typing import (
    Text,
    Dict,
    Optional,
    List,
    Any,
    Iterable,
    Tuple,
    Union,
)
from pathlib import Path

from rasa.core.agent import Agent
from rasa.engine.storage.local_model_storage import LocalModelStorage
import rasa.shared.utils.cli
import rasa.shared.utils.common
import rasa.shared.utils.io
import rasa.utils.common
from rasa.constants import RESULTS_FILE, NUMBER_OF_TRAINING_STORIES_FILE
from rasa.exceptions import ModelNotFound
from rasa.shared.constants import DEFAULT_RESULTS_PATH
import rasa.shared.nlu.training_data.loading
from rasa.shared.data import TrainingType
from rasa.shared.nlu.training_data.training_data import TrainingData
import rasa.model


logger = logging.getLogger(__name__)


class ClassificationReportException(Exception):
    """Raised when clf_report doesn't correctly set accuracy and/or micro avg.

    sklearn.metrics.classification_report should provide either accuracy or micro avg.
    """


async def test_core_models_in_directory(
    model_directory: Text,
    stories: Text,
    output: Text,
    use_conversation_test_files: bool = False,
) -> None:
    """Evaluates a directory with multiple Core models using test data.

    Args:
        model_directory: Directory containing multiple model files.
        stories: Path to a conversation test file.
        output: Output directory to store results to.
        use_conversation_test_files: `True` if conversation test files should be used
            for testing instead of regular Core story files.
    """
    from rasa.core.test import compare_models_in_dir

    model_directory = _get_sanitized_model_directory(model_directory)

    await compare_models_in_dir(
        model_directory,
        stories,
        output,
        use_conversation_test_files=use_conversation_test_files,
    )

    story_n_path = os.path.join(model_directory, NUMBER_OF_TRAINING_STORIES_FILE)
    number_of_stories = rasa.shared.utils.io.read_json_file(story_n_path)
    plot_core_results(output, number_of_stories)


def plot_core_results(output_directory: Text, number_of_examples: List[int]) -> None:
    """Plot core model comparison graph.

    Args:
        output_directory: path to the output directory
        number_of_examples: number of examples per run
    """
    import rasa.utils.plotting as plotting_utils

    graph_path = os.path.join(output_directory, "core_model_comparison_graph.pdf")

    plotting_utils.plot_curve(
        output_directory,
        number_of_examples,
        x_label_text="Number of stories present during training",
        y_label_text="Number of correct test stories",
        graph_path=graph_path,
    )


def _get_sanitized_model_directory(model_directory: Text) -> Text:
    """Adjusts the `--model` argument of `rasa test core` when called with
    `--evaluate-model-directory`.

    By default rasa uses the latest model for the `--model` parameter. However, for
    `--evaluate-model-directory` we need a directory. This function checks if the
    passed parameter is a model or an individual model file.

    Args:
        model_directory: The model_directory argument that was given to
        `test_core_models_in_directory`.

    Returns: The adjusted model_directory that should be used in
        `test_core_models_in_directory`.
    """

    p = Path(model_directory)
    if p.is_file():
        if model_directory != rasa.model.get_latest_model():
            rasa.shared.utils.cli.print_warning(
                "You passed a file as '--model'. Will use the directory containing "
                "this file instead."
            )
        model_directory = str(p.parent)

    return model_directory


async def test_core_models(
    models: List[Text],
    stories: Text,
    output: Text,
    use_conversation_test_files: bool = False,
) -> None:
    """Compares multiple Core models based on test data.

    Args:
        models: A list of models files.
        stories: Path to test data.
        output: Path to output directory for test results.
        use_conversation_test_files: `True` if conversation test files should be used
            for testing instead of regular Core story files.
    """
    from rasa.core.test import compare_models

    await compare_models(
        models, stories, output, use_conversation_test_files=use_conversation_test_files
    )


async def test_core(
    model: Optional[Text] = None,
    stories: Optional[Text] = None,
    output: Text = DEFAULT_RESULTS_PATH,
    additional_arguments: Optional[Dict] = None,
    use_conversation_test_files: bool = False,
) -> None:
    """Tests a trained Core model against a set of test stories."""
    try:
        model = rasa.model.get_local_model(model)
    except ModelNotFound:
        rasa.shared.utils.cli.print_error(
            "Unable to test: could not find a model. Use 'rasa train' to train a "
            "Rasa model and provide it via the '--model' argument."
        )
        return

    metadata = LocalModelStorage.metadata_from_archive(model)
    if metadata.training_type == TrainingType.NLU:
        rasa.shared.utils.cli.print_error(
            "Unable to test: no core model found. Use 'rasa train' to train a "
            "Rasa model and provide it via the '--model' argument."
        )
    elif metadata.training_type == TrainingType.CORE and use_conversation_test_files:
        rasa.shared.utils.cli.print_warning(
            "No NLU model found. Using default 'RegexMessageHandler' for end-to-end "
            "evaluation. If you added actual user messages to your test stories "
            "this will likely lead to the tests failing. In that case, you need "
            "to train a NLU model first, e.g. using `rasa train`."
        )

    if additional_arguments is None:
        additional_arguments = {}

    if output:
        rasa.shared.utils.io.create_directory(output)

    _agent = Agent.load(model_path=model)

    if not _agent.is_ready():
        rasa.shared.utils.cli.print_error(
            "Unable to test: processor not loaded. Use 'rasa train' to train a "
            "Rasa model and provide it via the '--model' argument."
        )
        return

    from rasa.core.test import test as core_test

    kwargs = rasa.shared.utils.common.minimal_kwargs(
        additional_arguments, core_test, ["stories", "agent", "e2e"]
    )

    await core_test(
        stories, _agent, e2e=use_conversation_test_files, out_directory=output, **kwargs
    )


async def test_nlu(
    model: Optional[Text],
    nlu_data: Optional[Text],
    output_directory: Text = DEFAULT_RESULTS_PATH,
    additional_arguments: Optional[Dict] = None,
    domain_path: Optional[Text] = None,
) -> None:
    """Tests the NLU Model."""
    from rasa.nlu.test import run_evaluation

    rasa.shared.utils.io.create_directory(output_directory)

    try:
        model = rasa.model.get_local_model(model)
    except ModelNotFound:
        rasa.shared.utils.cli.print_error(
            "Could not find any model. Use 'rasa train nlu' to train a "
            "Rasa model and provide it via the '--model' argument."
        )
        return

    metadata = LocalModelStorage.metadata_from_archive(model)

    if os.path.exists(model) and metadata.training_type != TrainingType.CORE:
        kwargs = rasa.shared.utils.common.minimal_kwargs(
            additional_arguments, run_evaluation, ["data_path", "model"]
        )
        _agent = Agent.load(model_path=model)
        await run_evaluation(
            nlu_data,
            _agent.processor,
            output_directory=output_directory,
            domain_path=domain_path,
            **kwargs,
        )
    else:
        rasa.shared.utils.cli.print_error(
            "Could not find any model. Use 'rasa train nlu' to train a "
            "Rasa model and provide it via the '--model' argument."
        )


async def compare_nlu_models(
    configs: List[Text],
    test_data: TrainingData,
    output: Text,
    runs: int,
    exclusion_percentages: List[int],
) -> None:
    """Trains multiple models, compares them and saves the results."""

    from rasa.nlu.test import drop_intents_below_freq
    from rasa.nlu.utils import write_json_to_file
    from rasa.utils.io import create_path
    from rasa.nlu.test import compare_nlu

    test_data = drop_intents_below_freq(test_data, cutoff=5)

    create_path(output)

    bases = [os.path.basename(nlu_config) for nlu_config in configs]
    model_names = [os.path.splitext(base)[0] for base in bases]

    f1_score_results: Dict[Text, List[List[float]]] = {
        model_name: [[] for _ in range(runs)] for model_name in model_names
    }

    training_examples_per_run = await compare_nlu(
        configs,
        test_data,
        exclusion_percentages,
        f1_score_results,
        model_names,
        output,
        runs,
    )

    f1_path = os.path.join(output, RESULTS_FILE)
    write_json_to_file(f1_path, f1_score_results)

    plot_nlu_results(output, training_examples_per_run)


def plot_nlu_results(output_directory: Text, number_of_examples: List[int]) -> None:
    """Plot NLU model comparison graph.

    Args:
        output_directory: path to the output directory
        number_of_examples: number of examples per run
    """
    import rasa.utils.plotting as plotting_utils

    graph_path = os.path.join(output_directory, "nlu_model_comparison_graph.pdf")

    plotting_utils.plot_curve(
        output_directory,
        number_of_examples,
        x_label_text="Number of intent examples present during training",
        y_label_text="Label-weighted average F1 score on test set",
        graph_path=graph_path,
    )


async def perform_nlu_cross_validation(
    config: Dict[Text, Any],
    data: TrainingData,
    output: Text,
    additional_arguments: Optional[Dict[Text, Any]],
) -> None:
    """Runs cross-validation on test data.

    Args:
        config: The model configuration.
        data: The data which is used for the cross-validation.
        output: Output directory for the cross-validation results.
        additional_arguments: Additional arguments which are passed to the
            cross-validation, like number of `disable_plotting`.
    """
    from rasa.nlu.test import (
        drop_intents_below_freq,
        cross_validate,
        log_results,
        log_entity_results,
    )

    additional_arguments = additional_arguments or {}
    folds = int(additional_arguments.get("folds", 3))

    data = drop_intents_below_freq(data, cutoff=folds)
    kwargs = rasa.shared.utils.common.minimal_kwargs(
        additional_arguments, cross_validate
    )

    results, entity_results, response_selection_results = await cross_validate(
        data, folds, config, output, **kwargs
    )
    logger.info(f"CV evaluation (n={folds})")

    if any(results):
        logger.info("Intent evaluation results")
        log_results(results.train, "train")
        log_results(results.test, "test")
    if any(entity_results):
        logger.info("Entity evaluation results")
        log_entity_results(entity_results.train, "train")
        log_entity_results(entity_results.test, "test")
    if any(response_selection_results):
        logger.info("Response Selection evaluation results")
        log_results(response_selection_results.train, "train")
        log_results(response_selection_results.test, "test")


def get_evaluation_metrics(
    targets: Iterable[Any],
    predictions: Iterable[Any],
    output_dict: bool = False,
    exclude_label: Optional[Text] = None,
) -> Tuple[Union[Text, Dict[Text, Dict[Text, float]]], float, float, float]:
    """Compute the f1, precision, accuracy and summary report from sklearn.

    Args:
        targets: target labels
        predictions: predicted labels
        output_dict: if True sklearn returns a summary report as dict, if False the
          report is in string format
        exclude_label: labels to exclude from evaluation

    Returns:
        Report from sklearn, precision, f1, and accuracy values.
    """
    from sklearn import metrics

    targets = clean_labels(targets)
    predictions = clean_labels(predictions)

    labels = get_unique_labels(targets, exclude_label)
    if not labels:
        logger.warning("No labels to evaluate. Skip evaluation.")
        return {}, 0.0, 0.0, 0.0

    report = metrics.classification_report(
        targets, predictions, labels=labels, output_dict=output_dict
    )
    precision = metrics.precision_score(
        targets, predictions, labels=labels, average="weighted"
    )
    f1 = metrics.f1_score(targets, predictions, labels=labels, average="weighted")
    accuracy = metrics.accuracy_score(targets, predictions)

    if output_dict:
        report = make_classification_report_complete(report, accuracy)

    return report, precision, f1, accuracy


def make_classification_report_complete(report: dict, accuracy: float) -> dict:
    """Completes the sklearn classification report with accuracy xor micro avg.

    Args:
        report: Report generated by metrics.classification_report with output_dict=True
        accuracy: Model accuracy

    Raises:
        Exception: When sklearn.metrics.classification_report
            behaves different to our expectation.

    Returns:
        report: Report generated by metrics.classification_report
            enhanced with accuracy xor micro avg.
    """
    report = copy.deepcopy(report)
    if "accuracy" in report and "micro avg" not in report:
        # micro avg corresponds to accuracy in this case
        # and is the same for all metrics
        acc = report["accuracy"]
        support = report["macro avg"]["support"]
        report["micro avg"] = {
            "precision": acc,
            "recall": acc,
            "f1-score": acc,
            "support": support,
        }
    elif "accuracy" not in report and "micro avg" in report:
        # Due to provided labels, micro avg can have recall != precision
        # The accuracy therefore has to be inferred separately
        report["accuracy"] = accuracy
    else:
        raise ClassificationReportException(
            "This cannot happen according to classification_report's docs"
        )
    return report


def clean_labels(labels: Iterable[Text]) -> List[Text]:
    """Remove `None` labels. sklearn metrics do not support them.

    Args:
        labels: list of labels

    Returns:
        Cleaned labels.
    """
    return [label if label is not None else "" for label in labels]


def get_unique_labels(
    targets: Iterable[Text], exclude_label: Optional[Text]
) -> List[Text]:
    """Get unique labels. Exclude 'exclude_label' if specified.

    Args:
        targets: labels
        exclude_label: label to exclude

    Returns:
         Unique labels.
    """
    labels = set(targets)
    if exclude_label and exclude_label in labels:
        labels.remove(exclude_label)
    return list(labels)

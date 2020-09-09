import asyncio
import logging
import os
import typing
from typing import Text, Dict, Optional, List, Any, Iterable, Tuple, Union
from pathlib import Path

import rasa.utils.io as io_utils
from rasa.constants import (
    DEFAULT_RESULTS_PATH,
    RESULTS_FILE,
    NUMBER_OF_TRAINING_STORIES_FILE,
)
import rasa.cli.utils as cli_utils
import rasa.utils.common as utils
from rasa.exceptions import ModelNotFound

if typing.TYPE_CHECKING:
    from rasa.core.agent import Agent

logger = logging.getLogger(__name__)


def test_core_models_in_directory(
    model_directory: Text, stories: Text, output: Text
) -> None:
    from rasa.core.test import compare_models_in_dir

    model_directory = _get_sanitized_model_directory(model_directory)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(compare_models_in_dir(model_directory, stories, output))

    story_n_path = os.path.join(model_directory, NUMBER_OF_TRAINING_STORIES_FILE)
    number_of_stories = io_utils.read_json_file(story_n_path)
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
    import rasa.model

    p = Path(model_directory)
    if p.is_file():
        if model_directory != rasa.model.get_latest_model():
            cli_utils.print_warning(
                "You passed a file as '--model'. Will use the directory containing "
                "this file instead."
            )
        model_directory = str(p.parent)

    return model_directory


def test_core_models(models: List[Text], stories: Text, output: Text):
    from rasa.core.test import compare_models

    loop = asyncio.get_event_loop()
    loop.run_until_complete(compare_models(models, stories, output))


def test(
    model: Text,
    stories: Text,
    nlu_data: Text,
    output: Text = DEFAULT_RESULTS_PATH,
    additional_arguments: Optional[Dict] = None,
):
    if additional_arguments is None:
        additional_arguments = {}

    test_core(model, stories, output, additional_arguments)
    test_nlu(model, nlu_data, output, additional_arguments)


def test_core(
    model: Optional[Text] = None,
    stories: Optional[Text] = None,
    output: Text = DEFAULT_RESULTS_PATH,
    additional_arguments: Optional[Dict] = None,
) -> None:
    import rasa.model
    from rasa.core.interpreter import RegexInterpreter
    from rasa.core.agent import Agent

    if additional_arguments is None:
        additional_arguments = {}

    if output:
        io_utils.create_directory(output)

    try:
        unpacked_model = rasa.model.get_model(model)
    except ModelNotFound:
        cli_utils.print_error(
            "Unable to test: could not find a model. Use 'rasa train' to train a "
            "Rasa model and provide it via the '--model' argument."
        )
        return

    _agent = Agent.load(unpacked_model)

    if _agent.policy_ensemble is None:
        cli_utils.print_error(
            "Unable to test: could not find a Core model. Use 'rasa train' to train a "
            "Rasa model and provide it via the '--model' argument."
        )

    if isinstance(_agent.interpreter, RegexInterpreter):
        cli_utils.print_warning(
            "No NLU model found. Using default 'RegexInterpreter' for end-to-end "
            "evaluation. If you added actual user messages to your test stories "
            "this will likely lead to the tests failing. In that case, you need "
            "to train a NLU model first, e.g. using `rasa train`."
        )

    from rasa.core.test import test

    kwargs = utils.minimal_kwargs(additional_arguments, test, ["stories", "agent"])

    _test_core(stories, _agent, output, **kwargs)


def _test_core(
    stories: Optional[Text], agent: "Agent", output_directory: Text, **kwargs: Any
) -> None:
    from rasa.core.test import test

    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        test(stories, agent, out_directory=output_directory, **kwargs)
    )


def test_nlu(
    model: Optional[Text],
    nlu_data: Optional[Text],
    output_directory: Text = DEFAULT_RESULTS_PATH,
    additional_arguments: Optional[Dict] = None,
):
    from rasa.nlu.test import run_evaluation
    from rasa.model import get_model

    try:
        unpacked_model = get_model(model)
    except ModelNotFound:
        cli_utils.print_error(
            "Could not find any model. Use 'rasa train nlu' to train a "
            "Rasa model and provide it via the '--model' argument."
        )
        return

    io_utils.create_directory(output_directory)

    nlu_model = os.path.join(unpacked_model, "nlu")

    if os.path.exists(nlu_model):
        kwargs = utils.minimal_kwargs(
            additional_arguments, run_evaluation, ["data_path", "model"]
        )
        run_evaluation(nlu_data, nlu_model, output_directory=output_directory, **kwargs)
    else:
        cli_utils.print_error(
            "Could not find any model. Use 'rasa train nlu' to train a "
            "Rasa model and provide it via the '--model' argument."
        )


def compare_nlu_models(
    configs: List[Text],
    nlu: Text,
    output: Text,
    runs: int,
    exclusion_percentages: List[int],
):
    """Trains multiple models, compares them and saves the results."""

    from rasa.nlu.test import drop_intents_below_freq
    from rasa.nlu.training_data import load_data
    from rasa.nlu.utils import write_json_to_file
    from rasa.utils.io import create_path
    from rasa.nlu.test import compare_nlu

    data = load_data(nlu)
    data = drop_intents_below_freq(data, cutoff=5)

    create_path(output)

    bases = [os.path.basename(nlu_config) for nlu_config in configs]
    model_names = [os.path.splitext(base)[0] for base in bases]

    f1_score_results = {
        model_name: [[] for _ in range(runs)] for model_name in model_names
    }

    training_examples_per_run = compare_nlu(
        configs,
        data,
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


def perform_nlu_cross_validation(
    config: Text,
    nlu: Text,
    output: Text,
    additional_arguments: Optional[Dict[Text, Any]],
):
    import rasa.nlu.config
    from rasa.nlu.test import (
        drop_intents_below_freq,
        cross_validate,
        log_results,
        log_entity_results,
    )

    additional_arguments = additional_arguments or {}
    folds = int(additional_arguments.get("folds", 3))
    nlu_config = rasa.nlu.config.load(config)
    data = rasa.nlu.training_data.load_data(nlu)
    data = drop_intents_below_freq(data, cutoff=folds)
    kwargs = utils.minimal_kwargs(additional_arguments, cross_validate)
    results, entity_results, response_selection_results = cross_validate(
        data, folds, nlu_config, output, **kwargs
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

    return report, precision, f1, accuracy


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

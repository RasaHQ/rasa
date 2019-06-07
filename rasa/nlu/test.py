import itertools
from collections import defaultdict, namedtuple

import os
import logging
import numpy as np
from typing import Iterable, Iterator, Tuple, List, Set, Optional, Text, Union, Dict
from tqdm import tqdm

from rasa.nlu import config, training_data, utils
from rasa.nlu.components import ComponentBuilder
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.extractors.crf_entity_extractor import CRFEntityExtractor
from rasa.nlu.model import Interpreter, Trainer, TrainingData
from rasa.nlu.components import Component
from rasa.nlu.tokenizers import Token

logger = logging.getLogger(__name__)

PRETRAINED_EXTRACTORS = {"DucklingHTTPExtractor", "SpacyEntityExtractor"}

ENTITY_PROCESSORS = {"EntitySynonymMapper"}

CVEvaluationResult = namedtuple("Results", "train test")

IntentEvaluationResult = namedtuple(
    "IntentEvaluationResult",
    "intent_target " "intent_prediction " "message " "confidence",
)

EntityEvaluationResult = namedtuple(
    "EntityEvaluationResult", "entity_targets " "entity_predictions " "tokens"
)

IntentMetrics = Dict[Text, List[float]]
EntityMetrics = Dict[Text, Dict[Text, List[float]]]


def plot_confusion_matrix(
    cm: np.array,
    classes: np.array,
    normalize: bool = False,
    title: Text = "Confusion matrix",
    cmap=None,
    zmin: int = 1,
    out: Optional[Text] = None,
) -> None:  # pragma: no cover
    """Print and plot the confusion matrix for the intent classification.
    Normalization can be applied by setting `normalize=True`."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    zmax = cm.max()
    plt.clf()
    if not cmap:
        cmap = plt.cm.Blues
    plt.imshow(
        cm,
        interpolation="nearest",
        cmap=cmap,
        aspect="auto",
        norm=LogNorm(vmin=zmin, vmax=zmax),
    )
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        logger.info("Normalized confusion matrix: \n{}".format(cm))
    else:
        logger.info("Confusion matrix, without normalization: \n{}".format(cm))

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    # save confusion matrix to file before showing it
    if out:
        fig = plt.gcf()
        fig.set_size_inches(20, 20)
        fig.savefig(out, bbox_inches="tight")


def plot_histogram(
    hist_data: List[List[float]], out: Optional[Text] = None
) -> None:  # pragma: no cover
    """Plot a histogram of the confidence distribution of the predictions in
    two columns.
    Wine-ish colour for the confidences of hits.
    Blue-ish colour for the confidences of misses.
    Saves the plot to a file."""
    import matplotlib.pyplot as plt

    colors = ["#009292", "#920000"]  #
    bins = [0.05 * i for i in range(1, 21)]

    plt.xlim([0, 1])
    plt.hist(hist_data, bins=bins, color=colors)
    plt.xticks(bins)
    plt.title("Intent Prediction Confidence Distribution")
    plt.xlabel("Confidence")
    plt.ylabel("Number of Samples")
    plt.legend(["hits", "misses"])

    if out:
        fig = plt.gcf()
        fig.set_size_inches(10, 10)
        fig.savefig(out, bbox_inches="tight")


def log_evaluation_table(
    report: Text, precision: float, f1: float, accuracy: float
) -> None:  # pragma: no cover
    """Log the sklearn evaluation metrics."""

    logger.info("F1-Score:  {}".format(f1))
    logger.info("Precision: {}".format(precision))
    logger.info("Accuracy:  {}".format(accuracy))
    logger.info("Classification report: \n{}".format(report))


def get_evaluation_metrics(
    targets: Iterable[Optional[Text]],
    predictions: Iterable[Optional[Text]],
    output_dict: bool = False,
) -> Tuple[Union[Text, Dict[Text, Dict[Text, float]]], float, float, float]:
    """Compute the f1, precision, accuracy and summary report from sklearn."""
    from sklearn import metrics

    targets = clean_intent_labels(targets)
    predictions = clean_intent_labels(predictions)

    report = metrics.classification_report(
        targets, predictions, output_dict=output_dict
    )
    precision = metrics.precision_score(targets, predictions, average="weighted")
    f1 = metrics.f1_score(targets, predictions, average="weighted")
    accuracy = metrics.accuracy_score(targets, predictions)

    return report, precision, f1, accuracy


def remove_empty_intent_examples(
    intent_results: List[IntentEvaluationResult]
) -> List[IntentEvaluationResult]:
    """Remove those examples without an intent."""

    filtered = []
    for r in intent_results:
        # substitute None values with empty string
        # to enable sklearn evaluation
        if r.intent_prediction is None:
            r = r._replace(intent_prediction="")

        if r.intent_target != "" and r.intent_target is not None:
            filtered.append(r)

    return filtered


def clean_intent_labels(labels: Iterable[Optional[Text]]) -> List[Text]:
    """Get rid of `None` intents. sklearn metrics do not support them."""
    return [l if l is not None else "" for l in labels]


def drop_intents_below_freq(td: TrainingData, cutoff: int = 5) -> TrainingData:
    """Remove intent groups with less than cutoff instances."""

    logger.debug("Raw data intent examples: {}".format(len(td.intent_examples)))
    keep_examples = [
        ex
        for ex in td.intent_examples
        if td.examples_per_intent[ex.get("intent")] >= cutoff
    ]

    return TrainingData(keep_examples, td.entity_synonyms, td.regex_features)


def collect_nlu_successes(
    intent_results: List[IntentEvaluationResult], successes_filename: Text
) -> None:
    """Log messages which result in successful predictions
    and save them to file"""

    successes = [
        {
            "text": r.message,
            "intent": r.intent_target,
            "intent_prediction": {
                "name": r.intent_prediction,
                "confidence": r.confidence,
            },
        }
        for r in intent_results
        if r.intent_target == r.intent_prediction
    ]

    if successes:
        utils.write_json_to_file(successes_filename, successes)
        logger.info(
            "Model prediction successes saved to {}.".format(successes_filename)
        )
        logger.debug(
            "\n\nSuccessfully predicted the following intents: \n{}".format(successes)
        )
    else:
        logger.info("Your model made no successful predictions")


def collect_nlu_errors(
    intent_results: List[IntentEvaluationResult], errors_filename: Text
) -> None:
    """Log messages which result in wrong predictions and save them to file"""

    errors = [
        {
            "text": r.message,
            "intent": r.intent_target,
            "intent_prediction": {
                "name": r.intent_prediction,
                "confidence": r.confidence,
            },
        }
        for r in intent_results
        if r.intent_target != r.intent_prediction
    ]

    if errors:
        utils.write_json_to_file(errors_filename, errors)
        logger.info("Model prediction errors saved to {}.".format(errors_filename))
        logger.debug(
            "\n\nThese intent examples could not be classified "
            "correctly: \n{}".format(errors)
        )
    else:
        logger.info("Your model made no errors")


def plot_intent_confidences(
    intent_results: List[IntentEvaluationResult], intent_hist_filename: Optional[Text]
) -> None:
    import matplotlib.pyplot as plt

    # create histogram of confidence distribution, save to file and display
    plt.gcf().clear()
    pos_hist = [
        r.confidence for r in intent_results if r.intent_target == r.intent_prediction
    ]

    neg_hist = [
        r.confidence for r in intent_results if r.intent_target != r.intent_prediction
    ]

    plot_histogram([pos_hist, neg_hist], intent_hist_filename)


def evaluate_intents(
    intent_results: List[IntentEvaluationResult],
    report_folder: Optional[Text],
    successes_filename: Optional[Text],
    errors_filename: Optional[Text],
    confmat_filename: Optional[Text],
    intent_hist_filename: Optional[Text],
) -> Dict:  # pragma: no cover
    """Creates a confusion matrix and summary statistics for intent predictions.
    Log samples which could not be classified correctly and save them to file.
    Creates a confidence histogram which is saved to file.
    Wrong and correct prediction confidences will be
    plotted in separate bars of the same histogram plot.
    Only considers those examples with a set intent.
    Others are filtered out. Returns a dictionary of containing the
    evaluation result."""

    # remove empty intent targets
    num_examples = len(intent_results)
    intent_results = remove_empty_intent_examples(intent_results)

    logger.info(
        "Intent Evaluation: Only considering those "
        "{} examples that have a defined intent out "
        "of {} examples".format(len(intent_results), num_examples)
    )

    target_intents, predicted_intents = _targets_predictions_from(intent_results)

    if report_folder:
        report, precision, f1, accuracy = get_evaluation_metrics(
            target_intents, predicted_intents, output_dict=True
        )

        report_filename = os.path.join(report_folder, "intent_report.json")

        utils.write_json_to_file(report_filename, report)
        logger.info("Classification report saved to {}.".format(report_filename))

    else:
        report, precision, f1, accuracy = get_evaluation_metrics(
            target_intents, predicted_intents
        )
        if isinstance(report, str):
            log_evaluation_table(report, precision, f1, accuracy)

    if successes_filename:
        # save classified samples to file for debugging
        collect_nlu_successes(intent_results, successes_filename)

    if errors_filename:
        # log and save misclassified samples to file for debugging
        collect_nlu_errors(intent_results, errors_filename)

    if confmat_filename:
        from sklearn.metrics import confusion_matrix
        from sklearn.utils.multiclass import unique_labels
        import matplotlib.pyplot as plt

        cnf_matrix = confusion_matrix(target_intents, predicted_intents)
        labels = unique_labels(target_intents, predicted_intents)
        plot_confusion_matrix(
            cnf_matrix,
            classes=labels,
            title="Intent Confusion matrix",
            out=confmat_filename,
        )
        plt.show(block=False)

        plot_intent_confidences(intent_results, intent_hist_filename)

        plt.show(block=False)

    predictions = [
        {
            "text": res.message,
            "intent": res.intent_target,
            "predicted": res.intent_prediction,
            "confidence": res.confidence,
        }
        for res in intent_results
    ]

    return {
        "predictions": predictions,
        "report": report,
        "precision": precision,
        "f1_score": f1,
        "accuracy": accuracy,
    }


def merge_labels(
    aligned_predictions: List[Dict], extractor: Optional[Text] = None
) -> np.array:
    """Concatenates all labels of the aligned predictions.
    Takes the aligned prediction labels which are grouped for each message
    and concatenates them."""

    if extractor:
        label_lists = [ap["extractor_labels"][extractor] for ap in aligned_predictions]
    else:
        label_lists = [ap["target_labels"] for ap in aligned_predictions]

    flattened = list(itertools.chain(*label_lists))
    return np.array(flattened)


def substitute_labels(labels: List[Text], old: Text, new: Text) -> List[Text]:
    """Replaces label names in a list of labels."""
    return [new if label == old else label for label in labels]


def evaluate_entities(
    entity_results: List[EntityEvaluationResult],
    extractors: Set[Text],
    report_folder: Optional[Text],
) -> Dict:  # pragma: no cover
    """Creates summary statistics for each entity extractor.
    Logs precision, recall, and F1 per entity type for each extractor."""

    aligned_predictions = align_all_entity_predictions(entity_results, extractors)
    merged_targets = merge_labels(aligned_predictions)
    merged_targets = substitute_labels(merged_targets, "O", "no_entity")

    result = {}

    for extractor in extractors:
        merged_predictions = merge_labels(aligned_predictions, extractor)
        merged_predictions = substitute_labels(merged_predictions, "O", "no_entity")
        logger.info("Evaluation for entity extractor: {} ".format(extractor))
        if report_folder:
            report, precision, f1, accuracy = get_evaluation_metrics(
                merged_targets, merged_predictions, output_dict=True
            )

            report_filename = extractor + "_report.json"
            extractor_report_filename = os.path.join(report_folder, report_filename)

            utils.write_json_to_file(extractor_report_filename, report)
            logger.info(
                "Classification report for '{}' saved to '{}'."
                "".format(extractor, extractor_report_filename)
            )

        else:
            report, precision, f1, accuracy = get_evaluation_metrics(
                merged_targets, merged_predictions
            )
            if isinstance(report, str):
                log_evaluation_table(report, precision, f1, accuracy)

        result[extractor] = {
            "report": report,
            "precision": precision,
            "f1_score": f1,
            "accuracy": accuracy,
        }

    return result


def is_token_within_entity(token: Token, entity: Dict) -> bool:
    """Checks if a token is within the boundaries of an entity."""
    return determine_intersection(token, entity) == len(token.text)


def does_token_cross_borders(token: Token, entity: Dict) -> bool:
    """Checks if a token crosses the boundaries of an entity."""

    num_intersect = determine_intersection(token, entity)
    return 0 < num_intersect < len(token.text)


def determine_intersection(token: Token, entity: Dict) -> int:
    """Calculates how many characters a given token and entity share."""

    pos_token = set(range(token.offset, token.end))
    pos_entity = set(range(entity["start"], entity["end"]))
    return len(pos_token.intersection(pos_entity))


def do_entities_overlap(entities: List[Dict]) -> bool:
    """Checks if entities overlap.
    I.e. cross each others start and end boundaries.
    :param entities: list of entities
    :return: boolean
    """

    sorted_entities = sorted(entities, key=lambda e: e["start"])
    for i in range(len(sorted_entities) - 1):
        curr_ent = sorted_entities[i]
        next_ent = sorted_entities[i + 1]
        if (
            next_ent["start"] < curr_ent["end"]
            and next_ent["entity"] != curr_ent["entity"]
        ):
            logger.warn("Overlapping entity {} with {}".format(curr_ent, next_ent))
            return True

    return False


def find_intersecting_entites(token: Token, entities: List[Dict]) -> List[Dict]:
    """Finds the entities that intersect with a token.
    :param token: a single token
    :param entities: entities found by a single extractor
    :return: list of entities
    """

    candidates = []
    for e in entities:
        if is_token_within_entity(token, e):
            candidates.append(e)
        elif does_token_cross_borders(token, e):
            candidates.append(e)
            logger.debug(
                "Token boundary error for token {}({}, {}) "
                "and entity {}"
                "".format(token.text, token.offset, token.end, e)
            )
    return candidates


def pick_best_entity_fit(token: Token, candidates: List[Dict]) -> Text:
    """Determines the token label given intersecting entities.
    :param token: a single token
    :param candidates: entities found by a single extractor
    :return: entity type
    """

    if len(candidates) == 0:
        return "O"
    elif len(candidates) == 1:
        return candidates[0]["entity"]
    else:
        best_fit = np.argmax([determine_intersection(token, c) for c in candidates])
        return candidates[best_fit]["entity"]


def determine_token_labels(
    token: Token, entities: List[Dict], extractors: Optional[Set[Text]]
) -> Text:
    """Determines the token label given entities that do not overlap.
    Args:
        token: a single token
        entities: entities found by a single extractor
        extractors: list of extractors
    Returns:
        entity type
    """

    if len(entities) == 0:
        return "O"
    if not do_extractors_support_overlap(extractors) and do_entities_overlap(entities):
        raise ValueError("The possible entities should not overlap")

    candidates = find_intersecting_entites(token, entities)
    return pick_best_entity_fit(token, candidates)


def do_extractors_support_overlap(extractors: Optional[Set[Text]]) -> bool:
    """Checks if extractors support overlapping entities"""
    if extractors is None:
        return False
    return CRFEntityExtractor.name not in extractors


def align_entity_predictions(
    result: EntityEvaluationResult, extractors: Set[Text]
) -> Dict:
    """Aligns entity predictions to the message tokens.
    Determines for every token the true label based on the
    prediction targets and the label assigned by each
    single extractor.
    :param result: entity prediction result
    :param extractors: the entity extractors that should be considered
    :return: dictionary containing the true token labels and token labels
             from the extractors
    """

    true_token_labels = []
    entities_by_extractors = {
        extractor: [] for extractor in extractors
    }  # type: Dict[Text, List]
    for p in result.entity_predictions:
        entities_by_extractors[p["extractor"]].append(p)
    extractor_labels = {
        extractor: [] for extractor in extractors
    }  # type: Dict[Text, List]
    for t in result.tokens:
        true_token_labels.append(determine_token_labels(t, result.entity_targets, None))
        for extractor, entities in entities_by_extractors.items():
            extracted = determine_token_labels(t, entities, set([extractor]))
            extractor_labels[extractor].append(extracted)

    return {
        "target_labels": true_token_labels,
        "extractor_labels": dict(extractor_labels),
    }


def align_all_entity_predictions(
    entity_results: List[EntityEvaluationResult], extractors: Set[Text]
) -> List[Dict]:
    """ Aligns entity predictions to the message tokens for the whole dataset
        using align_entity_predictions
    :param entity_results: list of entity prediction results
    :param extractors: the entity extractors that should be considered
    :return: list of dictionaries containing the true token labels and token
             labels from the extractors
    """
    aligned_predictions = []
    for result in entity_results:
        aligned_predictions.append(align_entity_predictions(result, extractors))

    return aligned_predictions


def get_eval_data(
    interpreter: Interpreter, test_data: TrainingData
) -> Tuple[
    List[IntentEvaluationResult], List[EntityEvaluationResult]
]:  # pragma: no cover
    """Runs the model for the test set and extracts targets and predictions.

    Returns intent results (intent targets and predictions, the original
    messages and the confidences of the predictions), as well as entity
    results(entity_targets, entity_predictions, and tokens)."""

    logger.info("Running model for predictions:")

    intent_results, entity_results = [], []

    intent_labels = [e.get("intent") for e in test_data.intent_examples]
    should_eval_intents = (
        is_intent_classifier_present(interpreter) and len(set(intent_labels)) >= 2
    )
    should_eval_entities = is_entity_extractor_present(interpreter)

    for example in tqdm(test_data.training_examples):
        result = interpreter.parse(example.text, only_output_properties=False)

        if should_eval_intents:
            intent_prediction = result.get("intent", {}) or {}
            intent_results.append(
                IntentEvaluationResult(
                    example.get("intent", ""),
                    intent_prediction.get("name"),
                    result.get("text", {}),
                    intent_prediction.get("confidence"),
                )
            )

        if should_eval_entities:
            entity_results.append(
                EntityEvaluationResult(
                    example.get("entities", []),
                    result.get("entities", []),
                    result.get("tokens", []),
                )
            )

    return intent_results, entity_results


def get_entity_extractors(interpreter: Interpreter) -> Set[Text]:
    """Finds the names of entity extractors used by the interpreter.
    Processors are removed since they do not
    detect the boundaries themselves."""

    extractors = set([c.name for c in interpreter.pipeline if "entities" in c.provides])
    return extractors - ENTITY_PROCESSORS


def is_entity_extractor_present(interpreter: Interpreter) -> bool:
    """Checks whether entity extractor is present"""

    extractors = get_entity_extractors(interpreter)
    return extractors != []


def is_intent_classifier_present(interpreter: Interpreter) -> bool:
    """Checks whether intent classifier is present"""

    intent_classifiers = [
        c.name for c in interpreter.pipeline if "intent" in c.provides
    ]
    return intent_classifiers != []


def remove_pretrained_extractors(pipeline: List[Component]) -> List[Component]:
    """Removes pretrained extractors from the pipeline so that entities
       from pre-trained extractors are not predicted upon parsing"""
    pipeline = [c for c in pipeline if c.name not in PRETRAINED_EXTRACTORS]
    return pipeline


def run_evaluation(
    data_path: Text,
    model_path: Text,
    report: Optional[Text] = None,
    successes: Optional[Text] = None,
    errors: Optional[Text] = "errors.json",
    confmat: Optional[Text] = None,
    histogram: Optional[Text] = None,
    component_builder: Optional[ComponentBuilder] = None,
) -> Dict:  # pragma: no cover
    """
    Evaluate intent classification and entity extraction.

    :param data_path: path to the test data
    :param model_path: path to the model
    :param report: path to folder where reports are stored
    :param successes: path to file that will contain success cases
    :param errors: path to file that will contain error cases
    :param confmat: path to file that will show the confusion matrix
    :param histogram: path fo file that will show a histogram
    :param component_builder: component builder

    :return: dictionary containing evaluation results
    """

    # get the metadata config from the package data
    interpreter = Interpreter.load(model_path, component_builder)

    interpreter.pipeline = remove_pretrained_extractors(interpreter.pipeline)
    test_data = training_data.load_data(data_path, interpreter.model_metadata.language)

    result = {
        "intent_evaluation": None,
        "entity_evaluation": None,
    }  # type: Dict[Text, Optional[Dict]]

    if report:
        utils.create_dir(report)

    intent_results, entity_results = get_eval_data(interpreter, test_data)

    if intent_results:
        logger.info("Intent evaluation results:")
        result["intent_evaluation"] = evaluate_intents(
            intent_results, report, successes, errors, confmat, histogram
        )

    if entity_results:
        logger.info("Entity evaluation results:")
        extractors = get_entity_extractors(interpreter)
        result["entity_evaluation"] = evaluate_entities(
            entity_results, extractors, report
        )

    return result


def generate_folds(
    n: int, td: TrainingData
) -> Iterator[Tuple[TrainingData, TrainingData]]:
    """Generates n cross validation folds for training data td."""

    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n, shuffle=True)
    x = td.intent_examples
    y = [example.get("intent") for example in x]
    for i_fold, (train_index, test_index) in enumerate(skf.split(x, y)):
        logger.debug("Fold: {}".format(i_fold))
        train = [x[i] for i in train_index]
        test = [x[i] for i in test_index]
        yield (
            TrainingData(
                training_examples=train,
                entity_synonyms=td.entity_synonyms,
                regex_features=td.regex_features,
            ),
            TrainingData(
                training_examples=test,
                entity_synonyms=td.entity_synonyms,
                regex_features=td.regex_features,
            ),
        )


def combine_result(
    intent_metrics: IntentMetrics,
    entity_metrics: EntityMetrics,
    interpreter: Interpreter,
    data: TrainingData,
    intent_results: Optional[List[IntentEvaluationResult]] = None,
    entity_results: Optional[List[EntityEvaluationResult]] = None,
) -> Tuple[IntentMetrics, EntityMetrics]:
    """Collects intent and entity metrics for crossvalidation folds.
    If `intent_results` or `entity_results` is provided as a list, prediction results
    are also collected.
    """

    (
        intent_current_metrics,
        entity_current_metrics,
        current_intent_results,
        current_entity_results,
    ) = compute_metrics(interpreter, data)

    if intent_results is not None:
        intent_results += current_intent_results

    if entity_results is not None:
        entity_results += current_entity_results

    for k, v in intent_current_metrics.items():
        intent_metrics[k] = v + intent_metrics[k]

    for extractor, extractor_metric in entity_current_metrics.items():
        entity_metrics[extractor] = {
            k: v + entity_metrics[extractor][k] for k, v in extractor_metric.items()
        }

    return intent_metrics, entity_metrics


def cross_validate(
    data: TrainingData,
    n_folds: int,
    nlu_config: Union[RasaNLUModelConfig, Text],
    report: Optional[Text] = None,
    successes: Optional[Text] = None,
    errors: Optional[Text] = "errors.json",
    confmat: Optional[Text] = None,
    histogram: Optional[Text] = None,
) -> Tuple[CVEvaluationResult, CVEvaluationResult]:
    """Stratified cross validation on data.

    Args:
        data: Training Data
        n_folds: integer, number of cv folds
        nlu_config: nlu config file
        report: path to folder where reports are stored
        successes: path to file that will contain success cases
        errors: path to file that will contain error cases
        confmat: path to file that will show the confusion matrix
        histogram: path fo file that will show a histogram

    Returns:
        dictionary with key, list structure, where each entry in list
              corresponds to the relevant result for one fold
    """
    from collections import defaultdict

    if isinstance(nlu_config, str):
        nlu_config = config.load(nlu_config)

    if report:
        utils.create_dir(report)

    trainer = Trainer(nlu_config)
    trainer.pipeline = remove_pretrained_extractors(trainer.pipeline)

    intent_train_metrics = defaultdict(list)  # type: IntentMetrics
    intent_test_metrics = defaultdict(list)  # type: IntentMetrics
    entity_train_metrics = defaultdict(lambda: defaultdict(list))  # type: EntityMetrics
    entity_test_metrics = defaultdict(lambda: defaultdict(list))  # type: EntityMetrics

    intent_test_results = []  # type: List[IntentEvaluationResult]
    entity_test_results = []  # type: List[EntityEvaluationResult]
    intent_classifier_present = False
    extractors = set()  # type: Set[Text]

    for train, test in generate_folds(n_folds, data):
        interpreter = trainer.train(train)

        # calculate train accuracy
        combine_result(intent_train_metrics, entity_train_metrics, interpreter, train)
        # calculate test accuracy
        combine_result(
            intent_test_metrics,
            entity_test_metrics,
            interpreter,
            test,
            intent_test_results,
            entity_test_results,
        )

        if not extractors:
            extractors = get_entity_extractors(interpreter)

        if is_intent_classifier_present(interpreter):
            intent_classifier_present = True

    if intent_classifier_present:
        logger.info("Accumulated test folds intent evaluation results:")
        evaluate_intents(
            intent_test_results, report, successes, errors, confmat, histogram
        )

    if extractors:
        logger.info("Accumulated test folds entity evaluation results:")
        evaluate_entities(entity_test_results, extractors, report)

    return (
        CVEvaluationResult(dict(intent_train_metrics), dict(intent_test_metrics)),
        CVEvaluationResult(dict(entity_train_metrics), dict(entity_test_metrics)),
    )


def _targets_predictions_from(
    intent_results: List[IntentEvaluationResult]
) -> Iterator[Iterable[Optional[Text]]]:
    return zip(*[(r.intent_target, r.intent_prediction) for r in intent_results])


def compute_metrics(
    interpreter: Interpreter, corpus: TrainingData
) -> Tuple[
    IntentMetrics,
    EntityMetrics,
    List[IntentEvaluationResult],
    List[EntityEvaluationResult],
]:
    """Computes metrics for intent classification and entity extraction.
    Returns intent and entity metrics, and prediction results.
    """

    intent_results, entity_results = get_eval_data(interpreter, corpus)

    intent_results = remove_empty_intent_examples(intent_results)

    intent_metrics = _compute_intent_metrics(intent_results)
    entity_metrics = _compute_entity_metrics(entity_results, interpreter)

    return (intent_metrics, entity_metrics, intent_results, entity_results)


def _compute_intent_metrics(
    intent_results: List[IntentEvaluationResult]
) -> IntentMetrics:
    """Computes intent evaluation metrics for a given corpus and
    returns the results
    """
    # compute fold metrics
    targets, predictions = _targets_predictions_from(intent_results)
    _, precision, f1, accuracy = get_evaluation_metrics(targets, predictions)

    return {"Accuracy": [accuracy], "F1-score": [f1], "Precision": [precision]}


def _compute_entity_metrics(
    entity_results: List[EntityEvaluationResult], interpreter: Interpreter
) -> EntityMetrics:
    """Computes entity evaluation metrics and returns the results"""

    entity_metric_results = defaultdict(
        lambda: defaultdict(list)
    )  # type: EntityMetrics
    extractors = get_entity_extractors(interpreter)

    if not extractors:
        return entity_metric_results

    aligned_predictions = align_all_entity_predictions(entity_results, extractors)

    merged_targets = merge_labels(aligned_predictions)
    merged_targets = substitute_labels(merged_targets, "O", "no_entity")

    for extractor in extractors:
        merged_predictions = merge_labels(aligned_predictions, extractor)
        merged_predictions = substitute_labels(merged_predictions, "O", "no_entity")
        _, precision, f1, accuracy = get_evaluation_metrics(
            merged_targets, merged_predictions
        )
        entity_metric_results[extractor]["Accuracy"].append(accuracy)
        entity_metric_results[extractor]["F1-score"].append(f1)
        entity_metric_results[extractor]["Precision"].append(precision)

    return entity_metric_results


def return_results(results: IntentMetrics, dataset_name: Text) -> None:
    """Returns results of crossvalidation
    :param results: dictionary of results returned from cv
    :param dataset_name: string of which dataset the results are from, e.g.
                    test/train
    """

    for k, v in results.items():
        logger.info(
            "{} {}: {:.3f} ({:.3f})".format(dataset_name, k, np.mean(v), np.std(v))
        )


def return_entity_results(results: EntityMetrics, dataset_name: Text) -> None:
    """Returns entity results of crossvalidation
    :param results: dictionary of dictionaries of results returned from cv
    :param dataset_name: string of which dataset the results are from, e.g.
                    test/train
    """
    for extractor, result in results.items():
        logger.info("Entity extractor: {}".format(extractor))
        return_results(result, dataset_name)


if __name__ == "__main__":
    raise RuntimeError(
        "Calling `rasa.nlu.test` directly is no longer supported. Please use "
        "`rasa test` to test a combined Core and NLU model or `rasa test nlu` "
        "to test an NLU model."
    )

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import itertools
import json
import logging
import shutil
from collections import defaultdict
from collections import namedtuple
from typing import List, Optional, Text

import numpy as np

from rasa_nlu import training_data, utils, config
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Interpreter
from rasa_nlu.model import Trainer, TrainingData

logger = logging.getLogger(__name__)

duckling_extractors = {"ner_duckling", "ner_duckling_http"}

known_duckling_dimensions = {"amount-of-money", "distance", "duration",
                             "email", "number",
                             "ordinal", "phone-number", "timezone",
                             "temperature", "time", "url", "volume"}

entity_processors = {"ner_synonyms"}

CVEvaluationResult = namedtuple('Results', 'train test')

IntentEvaluationResult = namedtuple('IntentEvaluationResult',
                                    'target '
                                    'prediction '
                                    'message '
                                    'confidence')


def create_argument_parser():
    import argparse
    parser = argparse.ArgumentParser(
            description='evaluate a Rasa NLU pipeline with cross '
                        'validation or on external data')

    parser.add_argument('-d', '--data', required=True,
                        help="file containing training/evaluation data")

    parser.add_argument('--mode', default="evaluation",
                        help="evaluation|crossvalidation (evaluate "
                             "pretrained model or train model "
                             "by crossvalidation)")

    # todo: make the two different modes two subparsers
    parser.add_argument('-c', '--config',
                        help="model configuration file (crossvalidation only)")

    parser.add_argument('-m', '--model', required=False,
                        help="path to model (evaluation only)")

    parser.add_argument('-f', '--folds', required=False, default=10,
                        help="number of CV folds (crossvalidation only)")

    parser.add_argument('--errors', required=False, default="errors.json",
                        help="output path for the json with wrong predictions")

    parser.add_argument('--histogram', required=False, default="hist.png",
                        help="output path for the confidence histogram")

    parser.add_argument('--confmat', required=False, default="confmat.png",
                        help="output path for the confusion matrix plot")

    utils.add_logging_option_arguments(parser, default=logging.INFO)

    return parser


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=None,
                          zmin=1,
                          out=None):  # pragma: no cover
    """Print and plot the confusion matrix for the intent classification.

    Normalization can be applied by setting `normalize=True`."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    zmax = cm.max()
    plt.clf()
    if not cmap:
        cmap = plt.cm.Blues
    plt.imshow(cm, interpolation='nearest', cmap=cmap,
               aspect='auto', norm=LogNorm(vmin=zmin, vmax=zmax))
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        logger.info("Normalized confusion matrix: \n{}".format(cm))
    else:
        logger.info("Confusion matrix, without normalization: \n{}".format(cm))

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # save confusion matrix to file before showing it
    if out:
        fig = plt.gcf()
        fig.set_size_inches(20, 20)
        fig.savefig(out, bbox_inches='tight')


def plot_histogram(hist_data,  # type: List[List[float]]
                   out=None  # type: Optional[Text]
                   ):  # pragma: no cover
    """Plot a histogram of the confidence distribution of the predictions in
    two columns.
    Wine-ish colour for the confidences of hits.
    Blue-ish colour for the confidences of misses.
    Saves the plot to a file."""
    import matplotlib.pyplot as plt

    colors = ['#009292', '#920000']  #
    bins = [0.05 * i for i in range(1, 21)]

    plt.xlim([0, 1])
    plt.hist(hist_data, bins=bins, color=colors)
    plt.xticks(bins)
    plt.title('Intent Prediction Confidence Distribution')
    plt.xlabel('Confidence')
    plt.ylabel('Number of Samples')
    plt.legend(['hits', 'misses'])

    if out:
        fig = plt.gcf()
        fig.set_size_inches(10, 10)
        fig.savefig(out, bbox_inches='tight')


def log_evaluation_table(report,  # type: Text
                         precision,  # type: float
                         f1,  # type: float
                         accuracy  # type: float
                         ):  # pragma: no cover
    """Log the sklearn evaluation metrics."""

    logger.info("F1-Score:  {}".format(f1))
    logger.info("Precision: {}".format(precision))
    logger.info("Accuracy:  {}".format(accuracy))
    logger.info("Classification report: \n{}".format(report))


def get_evaluation_metrics(targets, predictions):  # pragma: no cover
    """Compute the f1, precision, accuracy and summary report from sklearn."""
    from sklearn import metrics

    targets = clean_intent_labels(targets)
    predictions = clean_intent_labels(predictions)

    report = metrics.classification_report(targets, predictions)
    precision = metrics.precision_score(targets, predictions,
                                        average='weighted')
    f1 = metrics.f1_score(targets, predictions, average='weighted')
    accuracy = metrics.accuracy_score(targets, predictions)

    return report, precision, f1, accuracy


def remove_empty_intent_examples(intent_results):
    """Remove those examples without an intent."""

    filtered = []
    for r in intent_results:
        # substitute None values with empty string
        # to enable sklearn evaluation
        if r.prediction is None:
            r = r._replace(prediction="")

        if r.target != "" and r.target is not None:
            filtered.append(r)

    return filtered


def clean_intent_labels(labels):
    """Get rid of `None` intents. sklearn metrics do not support them."""
    return [l if l is not None else "" for l in labels]


def drop_intents_below_freq(td, cutoff=5):
    # type: (TrainingData, int) -> TrainingData
    """Remove intent groups with less than cutoff instances."""

    logger.debug(
            "Raw data intent examples: {}".format(len(td.intent_examples)))
    keep_examples = [ex
                     for ex in td.intent_examples
                     if td.examples_per_intent[ex.get("intent")] >= cutoff]

    return TrainingData(keep_examples, td.entity_synonyms, td.regex_features)


def save_nlu_errors(errors, filename):
    """Write out nlu classification errors to a file."""

    utils.write_to_file(filename,
                        json.dumps(errors, indent=4, ensure_ascii=False))
    logger.info("Model prediction errors saved to {}.".format(filename))


def collect_nlu_errors(intent_results):  # pragma: no cover
    """Log messages which result in wrong predictions and save them to file"""

    # it could be interesting to include entity-errors later
    # therefore we start with a "intent_errors" key
    intent_errors = [{"text": r.message,
                      "intent": r.target,
                      "intent_prediction": {
                          "name": r.prediction,
                          "confidence": r.confidence
                      }}
                     for r in intent_results if r.target != r.prediction]

    if intent_errors:
        logger.info("There were some nlu intent classification errors. "
                    "Use `--verbose` to show them in the log.")
        logger.debug("\n\nThese intent examples could not be classified "
                     "correctly \n{}".format(intent_errors))

        return {'intent_errors': intent_errors}
    else:
        logger.info("No prediction errors were found. You are AWESOME!")
        return None


def plot_intent_confidences(intent_results, intent_hist_filename):
    import matplotlib.pyplot as plt
    # create histogram of confidence distribution, save to file and display
    plt.gcf().clear()
    pos_hist = [
        r.confidence
        for r in intent_results if r.target == r.prediction]

    neg_hist = [
        r.confidence
        for r in intent_results if r.target != r.prediction]

    plot_histogram([pos_hist, neg_hist], intent_hist_filename)


def evaluate_intents(intent_results,
                     errors_filename,
                     confmat_filename,
                     intent_hist_filename):  # pragma: no cover
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

    logger.info("Intent Evaluation: Only considering those "
                "{} examples that have a defined intent out "
                "of {} examples".format(len(intent_results), num_examples))

    targets, predictions = _targets_predictions_from(intent_results)

    report, precision, f1, accuracy = get_evaluation_metrics(targets,
                                                             predictions)

    log_evaluation_table(report, precision, f1, accuracy)

    # log and save misclassified samples to file for debugging
    errors = collect_nlu_errors(intent_results)

    if errors and errors_filename:
        save_nlu_errors(errors, errors_filename)

    if confmat_filename:
        from sklearn.metrics import confusion_matrix
        from sklearn.utils.multiclass import unique_labels
        import matplotlib.pyplot as plt

        cnf_matrix = confusion_matrix(targets, predictions)
        labels = unique_labels(targets, predictions)
        plot_confusion_matrix(cnf_matrix, classes=labels,
                              title='Intent Confusion matrix',
                              out=confmat_filename)
        plt.show()

        plot_intent_confidences(intent_results,
                                intent_hist_filename)

        plt.show()

    predictions = [
        {
            "text": res.message,
            "intent": res.target,
            "predicted": res.prediction,
            "confidence": res.confidence
        } for res in intent_results
    ]

    return {
        "predictions": predictions,
        "report": report,
        "precision": precision,
        "f1_score": f1,
        "accuracy": accuracy
    }


def merge_labels(aligned_predictions, extractor=None):
    """Concatenates all labels of the aligned predictions.

    Takes the aligned prediction labels which are grouped for each message
    and concatenates them."""

    if extractor:
        label_lists = [ap["extractor_labels"][extractor]
                       for ap in aligned_predictions]
    else:
        label_lists = [ap["target_labels"]
                       for ap in aligned_predictions]

    flattened = list(itertools.chain(*label_lists))
    return np.array(flattened)


def substitute_labels(labels, old, new):
    """Replaces label names in a list of labels."""
    return [new if label == old else label for label in labels]


def evaluate_entities(targets,
                      predictions,
                      tokens,
                      extractors):  # pragma: no cover
    """Creates summary statistics for each entity extractor.

    Logs precision, recall, and F1 per entity type for each extractor."""

    aligned_predictions = align_all_entity_predictions(targets, predictions,
                                                       tokens, extractors)
    merged_targets = merge_labels(aligned_predictions)
    merged_targets = substitute_labels(merged_targets, "O", "no_entity")

    result = {}

    for extractor in extractors:
        merged_predictions = merge_labels(aligned_predictions, extractor)
        merged_predictions = substitute_labels(
                merged_predictions, "O", "no_entity")
        logger.info("Evaluation for entity extractor: {} ".format(extractor))
        report, precision, f1, accuracy = get_evaluation_metrics(
                merged_targets, merged_predictions)
        log_evaluation_table(report, precision, f1, accuracy)
        result[extractor] = {
            "report": report,
            "precision": precision,
            "f1_score": f1,
            "accuracy": accuracy
        }

    return result


def is_token_within_entity(token, entity):
    """Checks if a token is within the boundaries of an entity."""
    return determine_intersection(token, entity) == len(token.text)


def does_token_cross_borders(token, entity):
    """Checks if a token crosses the boundaries of an entity."""

    num_intersect = determine_intersection(token, entity)
    return 0 < num_intersect < len(token.text)


def determine_intersection(token, entity):
    """Calculates how many characters a given token and entity share."""

    pos_token = set(range(token.offset, token.end))
    pos_entity = set(range(entity["start"], entity["end"]))
    return len(pos_token.intersection(pos_entity))


def do_entities_overlap(entities):
    """Checks if entities overlap.

    I.e. cross each others start and end boundaries.

    :param entities: list of entities
    :return: boolean
    """

    sorted_entities = sorted(entities, key=lambda e: e["start"])
    for i in range(len(sorted_entities) - 1):
        curr_ent = sorted_entities[i]
        next_ent = sorted_entities[i + 1]
        if (next_ent["start"] < curr_ent["end"]
                and next_ent["entity"] != curr_ent["entity"]):
            return True

    return False


def find_intersecting_entites(token, entities):
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
            logger.debug("Token boundary error for token {}({}, {}) "
                         "and entity {}"
                         "".format(token.text, token.offset, token.end, e))
    return candidates


def pick_best_entity_fit(token, candidates):
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
        best_fit = np.argmax([determine_intersection(token, c)
                              for c in candidates])
        return candidates[best_fit]["entity"]


def determine_token_labels(token, entities):
    """Determines the token label given entities that do not overlap.

    :param token: a single token
    :param entities: entities found by a single extractor
    :return: entity type
    """

    if len(entities) == 0:
        return "O"

    if do_entities_overlap(entities):
        raise ValueError("The possible entities should not overlap")

    candidates = find_intersecting_entites(token, entities)
    return pick_best_entity_fit(token, candidates)


def align_entity_predictions(targets, predictions, tokens, extractors):
    """Aligns entity predictions to the message tokens.

    Determines for every token the true label based on the
    prediction targets and the label assigned by each
    single extractor.

    :param targets: list of target entities
    :param predictions: list of predicted entities
    :param tokens: original message tokens
    :param extractors: the entity extractors that should be considered
    :return: dictionary containing the true token labels and token labels
             from the extractors
    """

    true_token_labels = []
    entities_by_extractors = {extractor: [] for extractor in extractors}
    for p in predictions:
        entities_by_extractors[p["extractor"]].append(p)
    extractor_labels = defaultdict(list)
    for t in tokens:
        true_token_labels.append(determine_token_labels(t, targets))
        for extractor, entities in entities_by_extractors.items():
            extracted = determine_token_labels(t, entities)
            extractor_labels[extractor].append(extracted)

    return {"target_labels": true_token_labels,
            "extractor_labels": dict(extractor_labels)}


def align_all_entity_predictions(targets, predictions, tokens, extractors):
    """ Aligns entity predictions to the message tokens for the whole dataset
        using align_entity_predictions

    :param targets: list of lists of target entities
    :param predictions: list of lists of predicted entities
    :param tokens: list of original message tokens
    :param extractors: the entity extractors that should be considered
    :return: list of dictionaries containing the true token labels and token
             labels from the extractors
    """

    aligned_predictions = []
    for ts, ps, tks in zip(targets, predictions, tokens):
        aligned_predictions.append(align_entity_predictions(ts, ps, tks,
                                                            extractors))

    return aligned_predictions


def get_intent_targets(test_data):  # pragma: no cover
    """Extracts intent targets from the test data."""
    return [e.get("intent", "") for e in test_data.training_examples]


def get_entity_targets(test_data):
    """Extracts entity targets from the test data."""
    return [e.get("entities", []) for e in test_data.training_examples]


def extract_intent(result):  # pragma: no cover
    """Extracts the intent from a parsing result."""
    return result.get('intent', {}).get('name')


def extract_entities(result):  # pragma: no cover
    """Extracts entities from a parsing result."""
    return result.get('entities', [])


def extract_message(result):  # pragma: no cover
    """Extracts the original message from a parsing result."""
    return result.get('text', {})


def extract_confidence(result):  # pragma: no cover
    """Extracts the confidence from a parsing result."""
    return result.get('intent', {}).get('confidence')


def get_intent_predictions(targets, interpreter,
                           test_data):  # pragma: no cover
    """Runs the model for the test set and extracts intent predictions.
        Returns intent predictions, the original messages
        and the confidences of the predictions"""
    intent_results = []
    for e, target in zip(test_data.training_examples, targets):
        res = interpreter.parse(e.text, only_output_properties=False)
        intent_results.append(IntentEvaluationResult(
                target,
                extract_intent(res),
                extract_message(res),
                extract_confidence(res)))

    return intent_results


def get_entity_predictions(interpreter, test_data):  # pragma: no cover
    """Runs the model for the test set and extracts entity
    predictions and tokens."""
    entity_predictions, tokens = [], []
    for e in test_data.training_examples:
        res = interpreter.parse(e.text, only_output_properties=False)
        entity_predictions.append(extract_entities(res))
        try:
            tokens.append(res["tokens"])
        except KeyError:
            logger.debug("No tokens present, which is fine if you don't have a"
                         " tokenizer in your pipeline")
    return entity_predictions, tokens


def get_entity_extractors(interpreter):
    """Finds the names of entity extractors used by the interpreter.

    Processors are removed since they do not
    detect the boundaries themselves."""

    extractors = set([c.name for c in interpreter.pipeline
                      if "entities" in c.provides])
    return extractors - entity_processors


def is_intent_classifier_present(interpreter):
    """Checks whether intent classifier is present"""

    intent_classifier = [c.name for c in interpreter.pipeline
                         if "intent" in c.provides]
    return intent_classifier != []


def combine_extractor_and_dimension_name(extractor, dim):
    """Joins the duckling extractor name with a dimension's name."""
    return "{} ({})".format(extractor, dim)


def get_duckling_dimensions(interpreter, duckling_extractor_name):
    """Gets the activated dimensions of a duckling extractor.

    If there are no activated dimensions, it uses all known
    dimensions as a fallback."""

    component = find_component(interpreter, duckling_extractor_name)
    if component.component_config["dimensions"]:
        return component.component_config["dimensions"]
    else:
        return known_duckling_dimensions


def find_component(interpreter, component_name):
    """Finds a component in a pipeline."""

    for c in interpreter.pipeline:
        if c.name == component_name:
            return c
    return None


def remove_duckling_extractors(extractors):
    """Removes duckling exctractors"""
    used_duckling_extractors = duckling_extractors.intersection(extractors)
    for duckling_extractor in used_duckling_extractors:
        logger.info("Skipping evaluation of {}".format(duckling_extractor))
        extractors.remove(duckling_extractor)

    return extractors


def remove_duckling_entities(entity_predictions):
    """Removes duckling entity predictions"""

    patched_entity_predictions = []
    for entities in entity_predictions:
        patched_entities = []
        for e in entities:
            if e["extractor"] not in duckling_extractors:
                patched_entities.append(e)
        patched_entity_predictions.append(patched_entities)

    return patched_entity_predictions


def run_evaluation(data_path, model,
                   errors_filename='errors.json',
                   confmat_filename=None,
                   intent_hist_filename=None,
                   component_builder=None):  # pragma: no cover
    """Evaluate intent classification and entity extraction."""

    # get the metadata config from the package data
    if isinstance(model, Interpreter):
        interpreter = model
    else:
        interpreter = Interpreter.load(model, component_builder)
    test_data = training_data.load_data(data_path,
                                        interpreter.model_metadata.language)
    extractors = get_entity_extractors(interpreter)
    entity_predictions, tokens = get_entity_predictions(interpreter,
                                                        test_data)

    if duckling_extractors.intersection(extractors):
        entity_predictions = remove_duckling_entities(entity_predictions)
        extractors = remove_duckling_extractors(extractors)

    result = {
        "intent_evaluation": None,
        "entity_evaluation": None
    }

    if is_intent_classifier_present(interpreter):
        intent_targets = get_intent_targets(test_data)
        intent_results = get_intent_predictions(
                intent_targets, interpreter, test_data)

        logger.info("Intent evaluation results:")
        result['intent_evaluation'] = evaluate_intents(intent_results,
                                                       errors_filename,
                                                       confmat_filename,
                                                       intent_hist_filename)

    if extractors:
        entity_targets = get_entity_targets(test_data)

        logger.info("Entity evaluation results:")
        result['entity_evaluation'] = evaluate_entities(entity_targets,
                                                        entity_predictions,
                                                        tokens,
                                                        extractors)

    return result


def generate_folds(n, td):
    """Generates n cross validation folds for training data td."""

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n, shuffle=True)
    x = td.intent_examples
    y = [example.get("intent") for example in x]
    for i_fold, (train_index, test_index) in enumerate(skf.split(x, y)):
        logger.debug("Fold: {}".format(i_fold))
        train = [x[i] for i in train_index]
        test = [x[i] for i in test_index]
        yield (TrainingData(training_examples=train,
                            entity_synonyms=td.entity_synonyms,
                            regex_features=td.regex_features),
               TrainingData(training_examples=test,
                            entity_synonyms=td.entity_synonyms,
                            regex_features=td.regex_features))


def combine_intent_result(results, interpreter, data):
    """Combines intent result for crossvalidation folds"""

    current_result = compute_intent_metrics(interpreter, data)

    return {k: v + results[k] for k, v in current_result.items()}


def combine_entity_result(results, interpreter, data):
    """Combines entity result for crossvalidation folds"""

    current_result = compute_entity_metrics(interpreter, data)

    for k, v in current_result.items():
        results[k] = {key: val + results[k][key] for key, val in v.items()}

    return results


def run_cv_evaluation(data, n_folds, nlu_config):
    # type: (TrainingData, int, RasaNLUModelConfig) -> CVEvaluationResult
    """Stratified cross validation on data

    :param data: Training Data
    :param n_folds: integer, number of cv folds
    :param nlu_config: nlu config file
    :return: dictionary with key, list structure, where each entry in list
              corresponds to the relevant result for one fold
    """
    from collections import defaultdict
    import tempfile

    trainer = Trainer(nlu_config)
    train_results = defaultdict(list)
    test_results = defaultdict(list)
    entity_train_results = defaultdict(lambda: defaultdict(list))
    entity_test_results = defaultdict(lambda: defaultdict(list))
    tmp_dir = tempfile.mkdtemp()

    for train, test in generate_folds(n_folds, data):
        interpreter = trainer.train(train)

        # calculate train accuracy
        train_results = combine_intent_result(train_results, interpreter,
                                              train)
        test_results = combine_intent_result(test_results, interpreter, test)
        # calculate test accuracy
        entity_train_results = combine_entity_result(entity_train_results,
                                                     interpreter, train)
        entity_test_results = combine_entity_result(entity_test_results,
                                                    interpreter, test)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    return (CVEvaluationResult(dict(train_results), dict(test_results)),
            CVEvaluationResult(dict(entity_train_results),
                               dict(entity_test_results)))


def _targets_predictions_from(intent_results):
    return zip(*[(r.target, r.prediction) for r in intent_results])


def compute_intent_metrics(interpreter, corpus):
    """Computes intent evaluation metrics for a given corpus and
    returns the results
    """
    if not is_intent_classifier_present(interpreter):
        return {}
    intent_targets = get_intent_targets(corpus)
    intent_results = get_intent_predictions(intent_targets, interpreter,
                                            corpus)
    intent_results = remove_empty_intent_examples(intent_results)

    # compute fold metrics
    targets, predictions = _targets_predictions_from(intent_results)
    _, precision, f1, accuracy = get_evaluation_metrics(targets, predictions)

    return {"Accuracy": [accuracy], "F1-score": [f1], "Precision": [precision]}


def compute_entity_metrics(interpreter, corpus):
    """Computes entity evaluation metrics for a given corpus and
    returns the results
    """
    entity_results = defaultdict(lambda: defaultdict(list))
    extractors = get_entity_extractors(interpreter)
    entity_predictions, tokens = get_entity_predictions(interpreter, corpus)

    if duckling_extractors.intersection(extractors):
        entity_predictions = remove_duckling_entities(entity_predictions)
        extractors = remove_duckling_extractors(extractors)

    if not extractors:
        return entity_results

    entity_targets = get_entity_targets(corpus)

    aligned_predictions = align_all_entity_predictions(entity_targets,
                                                       entity_predictions,
                                                       tokens, extractors)

    merged_targets = merge_labels(aligned_predictions)
    merged_targets = substitute_labels(merged_targets, "O", "no_entity")

    for extractor in extractors:
        merged_predictions = merge_labels(aligned_predictions, extractor)
        merged_predictions = substitute_labels(merged_predictions, "O",
                                               "no_entity")
        _, precision, f1, accuracy = get_evaluation_metrics(merged_targets,
                                                            merged_predictions)
        entity_results[extractor]["Accuracy"].append(accuracy)
        entity_results[extractor]["F1-score"].append(f1)
        entity_results[extractor]["Precision"].append(precision)

    return entity_results


def return_results(results, dataset_name):
    """Returns results of crossvalidation
    :param results: dictionary of results returned from cv
    :param dataset_name: string of which dataset the results are from, e.g.
                    test/train
    """

    for k, v in results.items():
        logger.info("{} {}: {:.3f} ({:.3f})".format(dataset_name, k,
                                                    np.mean(v),
                                                    np.std(v)))


def return_entity_results(results, dataset_name):
    """Returns entity results of crossvalidation
    :param results: dictionary of dictionaries of results returned from cv
    :param dataset_name: string of which dataset the results are from, e.g.
                    test/train
    """
    for extractor, result in results.items():
        logger.info("Entity extractor: {}".format(extractor))
        return_results(result, dataset_name)


def main():
    parser = create_argument_parser()
    cmdline_args = parser.parse_args()
    utils.configure_colored_logging(cmdline_args.loglevel)

    if cmdline_args.mode == "crossvalidation":

        # TODO: move parsing into sub parser
        # manual check argument dependency
        if cmdline_args.model is not None:
            parser.error("Crossvalidation will train a new model "
                         "- do not specify external model.")

        if cmdline_args.config is None:
            parser.error("Crossvalidation will train a new model "
                         "you need to specify a model configuration.")

        nlu_config = config.load(cmdline_args.config)
        data = training_data.load_data(cmdline_args.data)
        data = drop_intents_below_freq(data, cutoff=5)
        results, entity_results = run_cv_evaluation(
                data, int(cmdline_args.folds), nlu_config)
        logger.info("CV evaluation (n={})".format(cmdline_args.folds))

        if any(results):
            logger.info("Intent evaluation results")
            return_results(results.train, "train")
            return_results(results.test, "test")
        if any(entity_results):
            logger.info("Entity evaluation results")
            return_entity_results(entity_results.train, "train")
            return_entity_results(entity_results.test, "test")

    elif cmdline_args.mode == "evaluation":
        run_evaluation(cmdline_args.data,
                       cmdline_args.model,
                       cmdline_args.errors,
                       cmdline_args.confmat,
                       cmdline_args.histogram)

    logger.info("Finished evaluation")


if __name__ == '__main__':  # pragma: no cover
    main()

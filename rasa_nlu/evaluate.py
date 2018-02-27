from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import itertools
import logging
import os
import numpy as np

from collections import defaultdict, namedtuple

from typing import Dict, Text, List

from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Interpreter
from rasa_nlu.model import Trainer, TrainingData
from rasa_nlu import training_data, utils

logger = logging.getLogger(__name__)

duckling_extractors = {"ner_duckling", "ner_duckling_http"}

known_duckling_dimensions = {"amount-of-money", "distance", "duration",
                             "email", "number",
                             "ordinal", "phone-number", "timezone",
                             "temperature", "time", "url", "volume"}

entity_processors = {"ner_synonyms"}

CVEvaluationResult = namedtuple('Results', 'train test')


def create_argparser():  # pragma: no cover
    import argparse
    parser = argparse.ArgumentParser(
            description='evaluate a Rasa NLU pipeline with cross '
                        'validation or on external data')

    parser.add_argument('-d', '--data', required=True,
                        help="file containing training/evaluation data")
    parser.add_argument('--mode', required=False, default="evaluation",
                        help="evaluation|crossvalidation (evaluate "
                             "pretrained model or train model "
                             "by crossvalidation)")
    parser.add_argument('-c', '--config', required=True,
                        help="config file")

    parser.add_argument('-m', '--model', required=False,
                        help="path to model (evaluation only)")
    parser.add_argument('-f', '--folds', required=False, default=10,
                        help="number of CV folds (crossvalidation only)")
    return parser


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=None,
                          zmin=1):  # pragma: no cover
    """Print and plot the confusion matrix for the intent classification.

    Normalization can be applied by setting `normalize=True`."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    zmax = cm.max()
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=cmap if cmap else plt.cm.Blues,
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


def log_evaluation_table(targets, predictions):  # pragma: no cover
    """Logs the sklearn evaluation metrics"""
    report, precision, f1, accuracy = get_evaluation_metrics(targets,
                                                             predictions)

    logger.info("Intent Evaluation Results")
    logger.info("F1-Score:  {}".format(f1))
    logger.info("Precision: {}".format(precision))
    logger.info("Accuracy:  {}".format(accuracy))
    logger.info("Classification report: \n{}".format(report))


def get_evaluation_metrics(targets, predictions):  # pragma: no cover
    """Computes the f1, precision and accuracy sklearn evaluation metrics

    and fetches a summary report.
    """
    from sklearn import metrics

    report = metrics.classification_report(targets, predictions)
    precision = metrics.precision_score(targets, predictions, average='weighted')
    f1 = metrics.f1_score(targets, predictions, average='weighted')
    accuracy = metrics.accuracy_score(targets, predictions)

    return report, precision, f1, accuracy


def remove_empty_intent_examples(targets, predictions):
    """Removes those examples without intent."""

    targets = np.array(targets)
    mask = targets != ""
    targets = targets[mask]
    predictions = np.array(predictions)[mask]
    return targets, predictions


def clean_intent_labels(labels):
    """Gets rid of `None` intents, since sklearn metrics does not support it

    anymore.
    """
    return [l if l is not None else "" for l in labels]


def drop_intents_below_freq(td, cutoff=5):
    # type: (TrainingData, int) -> TrainingData
    """Remove intent groups with less than cutoff instances."""

    logger.debug("Raw data intent examples: {}".format(len(td.intent_examples)))
    keep_examples = [ex
                     for ex in td.intent_examples
                     if td.examples_per_intent[ex.get("intent")] >= cutoff]

    return TrainingData(keep_examples, td.entity_synonyms, td.regex_features)


def evaluate_intents(targets, predictions):  # pragma: no cover
    """Creates a confusion matrix and summary statistics for intent predictions.

    Only considers those examples with a set intent.
    Others are filtered out."""
    from sklearn.metrics import confusion_matrix
    from sklearn.utils.multiclass import unique_labels
    import matplotlib.pyplot as plt

    # remove empty intent targets
    num_examples = len(targets)
    targets, predictions = remove_empty_intent_examples(targets, predictions)
    logger.info("Intent Evaluation: Only considering those "
                "{} examples that have a defined intent out "
                "of {} examples".format(targets.size, num_examples))
    log_evaluation_table(targets, predictions)

    cnf_matrix = confusion_matrix(targets, predictions)
    labels = unique_labels(targets, predictions)
    plot_confusion_matrix(cnf_matrix,
                          classes=labels,
                          title='Intent Confusion matrix')

    plt.show()


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


def evaluate_entities(targets, predictions, tokens, extractors):  # pragma: no cover
    """Creates summary statistics for each entity extractor.

    Logs precision, recall, and F1 per entity type for each extractor."""

    aligned_predictions = []
    for ts, ps, tks in zip(targets, predictions, tokens):
        aligned_predictions.append(align_entity_predictions(ts, ps, tks,
                                                            extractors))

    merged_targets = merge_labels(aligned_predictions)
    merged_targets = substitute_labels(merged_targets, "O", "no_entity")

    for extractor in extractors:
        merged_predictions = merge_labels(aligned_predictions, extractor)
        merged_predictions = substitute_labels(merged_predictions, "O", "no_entity")
        logger.info("Evaluation for entity extractor: {} ".format(extractor))
        log_evaluation_table(merged_targets, merged_predictions)


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


def get_targets(test_data):  # pragma: no cover
    """Extracts targets from the test data."""

    intent_targets = [e.get("intent", "")
                      for e in test_data.training_examples]

    entity_targets = [e.get("entities", [])
                      for e in test_data.training_examples]

    return intent_targets, entity_targets


def extract_intent(result):  # pragma: no cover
    """Extracts the intent from a parsing result."""
    return result.get('intent', {}).get('name')


def extract_entities(result):  # pragma: no cover
    """Extracts entities from a parsing result."""
    return result.get('entities', [])


def get_predictions(interpreter, test_data):  # pragma: no cover
    """Runs the model for the test set and extracts predictions and tokens."""
    intent_predictions, entity_predictions, tokens = [], [], []
    for e in test_data.training_examples:
        res = interpreter.parse(e.text, only_output_properties=False)
        intent_predictions.append(extract_intent(res))
        entity_predictions.append(extract_entities(res))
        tokens.append(res["tokens"])
    return intent_predictions, entity_predictions, tokens


def get_entity_extractors(interpreter):
    """Finds the names of entity extractors used by the interpreter.

    Processors are removed since they do not
    detect the boundaries themselves."""

    extractors = set([c.name for c in interpreter.pipeline
                      if "entities" in c.provides])
    return extractors - entity_processors


def combine_extractor_and_dimension_name(extractor, dim):
    """Joins the duckling extractor name with a dimension's name."""
    return "{} ({})".format(extractor, dim)


def get_duckling_dimensions(interpreter, duckling_extractor_name):
    """Gets the activated dimensions of a duckling extractor.

    If there are no activated dimensions, it uses all known
    dimensions as a fallback."""

    component = find_component(interpreter, duckling_extractor_name)
    if component.dimensions:
        return component.dimensions
    else:
        return known_duckling_dimensions


def find_component(interpreter, component_name):
    """Finds a component in a pipeline."""

    for c in interpreter.pipeline:
        if c.name == component_name:
            return c
    return None


def patch_duckling_extractors(interpreter, extractors):  # pragma: no cover
    """Removes the basic duckling extractor from the set of extractors and
    adds dimension-suffixed ones.

    :param interpreter: a rasa nlu interpreter object
    :param extractors: a set of entity extractor names used in the interpreter
    """

    extractors = extractors.copy()
    used_duckling_extractors = duckling_extractors.intersection(extractors)
    for duckling_extractor in used_duckling_extractors:
        extractors.remove(duckling_extractor)
        for dim in get_duckling_dimensions(interpreter, duckling_extractor):
            new_extractor_name = combine_extractor_and_dimension_name(
                    duckling_extractor, dim)
            extractors.add(new_extractor_name)
    return extractors


def patch_duckling_entity(entity):
    """Patches a single entity by combining extractor and dimension name."""

    if entity["extractor"] in duckling_extractors:
        entity = entity.copy()
        entity["extractor"] = combine_extractor_and_dimension_name(
                entity["extractor"], entity["entity"])

    return entity


def patch_duckling_entities(entity_predictions):
    """Adds the duckling dimension as a suffix to the extractor name.

    As a result, there is only is one prediction per
    token per extractor name."""

    patched_entity_predictions = []
    for entities in entity_predictions:
        patched_entities = []
        for e in entities:
            patched_entities.append(patch_duckling_entity(e))
        patched_entity_predictions.append(patched_entities)

    return patched_entity_predictions


def run_evaluation(config, model_path,
                   component_builder=None):  # pragma: no cover
    """Evaluate intent classification and entity extraction."""

    # get the metadata config from the package data
    test_data = training_data.load_data(config['data'], config['language'])
    interpreter = Interpreter.load(model_path, config, component_builder)
    intent_targets, entity_targets = get_targets(test_data)
    intent_predictions, entity_predictions, tokens = get_predictions(
            interpreter, test_data)
    extractors = get_entity_extractors(interpreter)

    if extractors.intersection(duckling_extractors):
        entity_predictions = patch_duckling_entities(entity_predictions)
        extractors = patch_duckling_extractors(interpreter, extractors)

    evaluate_intents(intent_targets, intent_predictions)
    evaluate_entities(entity_targets, entity_predictions, tokens, extractors)


def generate_folds(n, td):
    """Generates n cross validation folds for training data td."""

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n, random_state=2018, shuffle=True)
    x = td.intent_examples
    y = [example.get("intent") for example in x]
    for i_fold, (train_index, test_index) in enumerate(skf.split(x, y)):
        logger.debug("Fold: {}".format(i_fold))
        train = [x[i] for i in train_index]
        test = [x[i] for i in test_index]
        yield train, test


def run_cv_evaluation(td, n_folds, nlu_config):
    # type: (TrainingData, int, RasaNLUConfig) -> CVEvaluationResult
    """Stratified cross validation on data

    :param td: Training Data
    :param n_folds: integer, number of cv folds
    :param nlu_config: nlu config file
    :return: dictionary with key, list structure, where each entry in list
              corresponds to the relevant result for one fold
    """
    from sklearn import metrics
    from collections import defaultdict
    import tempfile

    trainer = Trainer(nlu_config)
    train_results = defaultdict(list)
    test_results = defaultdict(list)

    tmp_dir = tempfile.mkdtemp()

    for train, test in generate_folds(n_folds, td):
        trainer.train(TrainingData(training_examples=train,
                                   entity_synonyms=td.entity_synonyms,
                                   regex_features=td.regex_features))
        model_dir = trainer.persist(tmp_dir)
        interpreter = Interpreter.load(model_dir, nlu_config)

        # calculate train accuracy
        compute_metrics(interpreter, train, train_results)
        # calculate test accuracy
        compute_metrics(interpreter, test, test_results)

        utils.remove_model(model_dir)

    os.rmdir(os.path.join(tmp_dir, "default"))
    os.rmdir(tmp_dir)

    return CVEvaluationResult(dict(train_results), dict(test_results))


def compute_metrics(interpreter, corpus, results):
    """Computes evaluation metrics for a given corpus and

    appends them to results.
    """
    y = [e.get("intent") for e in corpus]

    preds = []
    for e in corpus:
        res = interpreter.parse(e.text)
        if res.get('intent'):
            preds.append(res['intent'].get('name'))
        else:
            preds.append(None)

    y = clean_intent_labels(y)
    preds = clean_intent_labels(preds)

    # compute fold metrics
    _, precision, f1, accuracy = get_evaluation_metrics(y, preds)

    results["Accuracy"].append(accuracy)
    results["F1-score"].append(f1)
    results["Precision"].append(precision)


if __name__ == '__main__':  # pragma: no cover
    parser = create_argparser()
    args = parser.parse_args()

    # manual check argument dependency
    if args.mode == "crossvalidation":
        if args.model is not None:
            parser.error("Crossvalidation will train a new model "
                         "- do not specify external model")

    nlu_config = RasaNLUConfig(args.config, os.environ, vars(args))
    logging.basicConfig(level=nlu_config['log_level'])

    if args.mode == "crossvalidation":
        td = training_data.load_data(args.data)
        td = drop_intents_below_freq(td, cutoff=5)
        results = run_cv_evaluation(td, int(args.folds), nlu_config)
        logger.info("CV evaluation (n={})".format(args.folds))
        for k, v in results.train.items():
            logger.info("train {}: {:.3f} ({:.3f})".format(k, np.mean(v), np.std(v)))
        for k, v in results.test.items():
            logger.info("test {}: {:.3f} ({:.3f})".format(k, np.mean(v), np.std(v)))

    elif args.mode == "evaluation":
        run_evaluation(nlu_config, args.model)

    logger.info("Finished evaluation")

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import itertools
import logging
import os
import numpy as np

from collections import defaultdict

from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.converters import load_data
from rasa_nlu.model import Interpreter
from rasa_nlu.model import Metadata
from rasa_nlu.model import Trainer, TrainingData

logger = logging.getLogger(__name__)

duckling_extractors = {"ner_duckling", "ner_duckling_http"}
known_duckling_dimensions = {"amount-of-money", "distance", "duration", "email", "number",
                             "ordinal", "phone-number", "timezone", "temperature", "time", "url", "volume"}
entity_processors = {"ner_synonyms"}

def create_argparser():  # pragma: no cover
    import argparse
    parser = argparse.ArgumentParser(
            description='evaluate a Rasa NLU pipeline with cross validation or on external data')

    parser.add_argument('-d', '--data', required=True,
                        help="file containing training/evaluation data")
    parser.add_argument('--mode', required=False, default="evaluation",
                        help="evaluation|crossvalidation (evaluate pretrained model or train model by crossvalidation)")
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

    Normalization can be applied by setting `normalize=True`.
    """
    import numpy as np
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


def log_evaluation_table(test_y, preds):  # pragma: no cover
    from sklearn import metrics

    report = metrics.classification_report(test_y, preds)
    precision = metrics.precision_score(test_y, preds, average='weighted')
    f1 = metrics.f1_score(test_y, preds, average='weighted')
    accuracy = metrics.accuracy_score(test_y, preds)

    logger.info("Intent Evaluation Results")
    logger.info("F1-Score:  {}".format(f1))
    logger.info("Precision: {}".format(precision))
    logger.info("Accuracy:  {}".format(accuracy))
    logger.info("Classification report: \n{}".format(report))

def remove_empty_intent_examples(targets, predictions):
    """Removes those examples without intent."""
    targets = np.array(targets)
    mask = targets != ""
    targets = targets[mask]
    predictions = np.array(predictions)[mask]
    return targets, predictions


def prepare_data(data, cutoff = 5):
    """Remove intent groups with less than cutoff instances."""
    data = data.sorted_intent_examples()
    logger.info("Raw data intent examples: {}".format(len(data)))

    # count intents
    list_intents = []
    n_intents = []
    for intent, group in itertools.groupby(data, lambda e: e.get("intent")):
        size = len(list(group))
        n_intents.append(size)
        list_intents.append(intent)

    # only include intents with enough traing data
    prep_data = []
    good_intents = [list_intents[i] for i, _ in enumerate(list_intents) if n_intents[i] >= cutoff]
    for ind, _ in enumerate(data):
        if data[ind].get("intent") in good_intents:
            prep_data.append(data[ind])

    return prep_data


def evaluate_intents(targets, predictions):  # pragma: no cover
    """Creates a confusion matrix and summary statistics for intent predictions.

    Only considers those examples with a set intent. Others are filtered out.
    """
    from sklearn.metrics import confusion_matrix
    from sklearn.utils.multiclass import unique_labels
    import matplotlib.pyplot as plt

    # remove empty intent targets
    num_examples = len(targets)
    targets, predictions = remove_empty_intent_examples(targets, predictions)
    logger.info("Intent Evaluation: Only considering those {} examples that "
                "have a defined intent out of {} examples".format(targets.size, num_examples))
    log_evaluation_table(targets, predictions)

    cnf_matrix = confusion_matrix(targets, predictions)
    plot_confusion_matrix(cnf_matrix, classes=unique_labels(targets, predictions),
                          title='Intent Confusion matrix')

    plt.show()


def merge_labels(aligned_predictions, extractor=None):
    """Concatenates all labels of the aligned predictions.

    Takes the aligned prediction labels which are grouped for each message
    and concatenates them.
    """
    if extractor:
        label_lists = [ap["extractor_labels"][extractor] for ap in aligned_predictions]
    else:
        label_lists = [ap["target_labels"] for ap in aligned_predictions]

    flattened = list(itertools.chain(*label_lists))
    return np.array(flattened)


def substitute_labels(labels, old, new):
    """Replaces label names in a list of labels."""
    return [new if label == old else label for label in labels]


def evaluate_entities(targets, predictions, tokens, extractors):  # pragma: no cover
    """Creates summary statistics for each entity extractor.

    Logs precision, recall, and F1 per entity type for each extractor.
    """
    aligned_predictions = []
    for ts, ps, tks in zip(targets, predictions, tokens):
        aligned_predictions.append(align_entity_predictions(ts, ps, tks, extractors))

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
    return num_intersect > 0 and num_intersect < len(token.text)


def determine_intersection(token, entity):
    """Calculates how many characters a given token and entity share."""
    pos_token = set(range(token.offset, token.end))
    pos_entity = set(range(entity["start"], entity["end"]))
    return len(pos_token.intersection(pos_entity))


def do_entities_overlap(entities):
    """Checks if entities overlap, i.e. cross each others start and end boundaries.

    :param entities: list of entities
    :return: boolean
    """
    sorted_entities = sorted(entities, key=lambda e: e["start"])
    for i in range(len(sorted_entities) - 1):
        if sorted_entities[i + 1]["start"] < sorted_entities[i]["end"] \
                and sorted_entities[i + 1]["entity"] != sorted_entities[i]["entity"]:
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
            logger.debug("Token boundary error for token {}({}, {}) and entity {}".format(
                    token.text, token.offset, token.end, e))
    return candidates


def pick_best_entity_fit(token, candidates):
    """Determines the token label given intersecting entities.

    :param token: a single token
    :param entities: entities found by a single extractor
    :return: entity type
    """
    if len(candidates) == 0:
        return "O"
    elif len(candidates) == 1:
        return candidates[0]["entity"]
    else:
        best_fit = np.argmax([determine_intersection(token, c) for c in candidates])
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

    Determines for every token the true label based on the prediction targets and
    the label assigned by each single extractor.
    :param targets: list of target entities
    :param predictions: list of predicted entities
    :param tokens: original message tokens
    :param extractors: the entity extractors that should be considered
    :return: dictionary containing the true token labels and token labels from the extractors
    """
    true_token_labels = []
    entities_by_extractors = {extractor: [] for extractor in extractors}
    for p in predictions:
        entities_by_extractors[p["extractor"]].append(p)
    extractor_labels = defaultdict(list)
    for t in tokens:
        true_token_labels.append(determine_token_labels(t, targets))
        for extractor, entities in entities_by_extractors.items():
            extractor_labels[extractor].append(determine_token_labels(t, entities))

    return {"target_labels": true_token_labels, "extractor_labels": dict(extractor_labels)}


def get_targets(test_data):  # pragma: no cover
    """Extracts targets from the test data."""
    intent_targets = [e.get("intent", "") for e in test_data.training_examples]
    entity_targets = [e.get("entities", []) for e in test_data.training_examples]

    return intent_targets, entity_targets


def extract_intent(result):  # pragma: no cover
    """Extracts the intent from a parsing result."""
    return result['intent'].get('name') if 'intent' in result else None


def extract_entities(result):  # pragma: no cover
    """Extracts entities from a parsing result."""
    return result['entities'] if 'entities' in result else []


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

    Processors are removed since they do not detect the boundaries themselves.
    """
    extractors = set([c.name for c in interpreter.pipeline if "entities" in c.provides])
    return extractors - entity_processors


def combine_extractor_and_dimension_name(extractor, dim):
    """Joins the duckling extractor name with a dimension's name."""
    return "{} ({})".format(extractor, dim)


def get_duckling_dimensions(interpreter, duckling_extractor_name):
    """Gets the activated dimensions of a duckling extractor, or all known dimensions as a fallback."""
    component = find_component(interpreter, duckling_extractor_name)
    return component.dimensions if component.dimensions else known_duckling_dimensions


def find_component(interpreter, component_name):
    """Finds a component in a pipeline."""
    return [c for c in interpreter.pipeline if c.name == component_name][0]


def patch_duckling_extractors(interpreter, extractors):  # pragma: no cover
    """Removes the basic duckling extractor from the set of extractors and adds dimension-suffixed ones.

    :param interpreter: a rasa nlu interpreter object
    :param extractors: a set of entity extractor names used in the interpreter
    """
    extractors = extractors.copy()
    used_duckling_extractors = duckling_extractors.intersection(extractors)
    for duckling_extractor in used_duckling_extractors:
        extractors.remove(duckling_extractor)
        for dim in get_duckling_dimensions(interpreter, duckling_extractor):
            new_extractor_name = combine_extractor_and_dimension_name(duckling_extractor, dim)
            extractors.add(new_extractor_name)
    return extractors


def patch_duckling_entity(entity):
    """Patches a single entity by combining extractor and dimension name."""
    if entity["extractor"] in duckling_extractors:
        entity = entity.copy()
        entity["extractor"] = combine_extractor_and_dimension_name(entity["extractor"], entity["entity"])

    return entity


def patch_duckling_entities(entity_predictions):
    """Adds the duckling dimension as a suffix to the extractor name.

    As a result, there is only is one prediction per token per extractor name.
    """
    patched_entity_predictions = []
    for entities in entity_predictions:
        patched_entities = []
        for e in entities:
            patched_entities.append(patch_duckling_entity(e))
        patched_entity_predictions.append(patched_entities)

    return patched_entity_predictions


def run_evaluation(config, model_path, component_builder=None):  # pragma: no cover
    """Evaluate intent classification and entity extraction."""
    # get the metadata config from the package data
    test_data = load_data(config['data'], config['language'])
    interpreter = Interpreter.load(model_path, config, component_builder)
    intent_targets, entity_targets = get_targets(test_data)
    intent_predictions, entity_predictions, tokens = get_predictions(interpreter, test_data)
    extractors = get_entity_extractors(interpreter)

    if extractors.intersection(duckling_extractors):
        entity_predictions = patch_duckling_entities(entity_predictions)
        extractors = patch_duckling_extractors(interpreter, extractors)

    evaluate_intents(intent_targets, intent_predictions)
    evaluate_entities(entity_targets, entity_predictions, tokens, extractors)


def run_cv_evaluation(data, n_folds, nlu_config):
    from sklearn import metrics
    from sklearn.model_selection import StratifiedKFold
    from collections import defaultdict
    # type: (List[rasa_nlu.training_data.Message], int, RasaNLUConfig) -> Dict[Text, List[float]]
    """Stratified cross validation on data

    :param data: list of rasa_nlu.training_data.Message objects
    :param n_folds: integer, number of cv folds
    :param nlu_config: nlu config file
    :return: dictionary with key, list structure, where each entry in list
              corresponds to the relevant result for one fold

    """
    trainer = Trainer(nlu_config)
    results = defaultdict(list)

    y_true = [e.get("intent") for e in data]

    skf = StratifiedKFold(n_splits=n_folds, random_state=11, shuffle=True)
    counter = 1
    logger.info("Evaluation started")
    for train_index, test_index in skf.split(data, y_true):

        train = [data[i] for i in train_index]
        test = [data[i] for i in test_index]

        logger.debug("Fold: {}".format(counter))
        logger.debug("Training ...")
        trainer.train(TrainingData(training_examples=train))
        model_directory = trainer.persist("projects/")  # Returns the directory the model is stored in

        logger.debug("Evaluation ...")
        interpreter = Interpreter.load(model_directory, nlu_config)
        test_y = [e.get("intent") for e in test]

        preds = []
        for e in test:
            res = interpreter.parse(e.text)
            if res.get('intent'):
                preds.append(res['intent'].get('name'))
            else:
                preds.append(None)

        # compute fold metrics
        results["Accuracy"].append(metrics.accuracy_score(test_y, preds))
        results["F1-score"].append(metrics.f1_score(test_y, preds, average='weighted'))
        results["Precision"] = metrics.precision_score(test_y, preds, average='weighted')

        # increase fold counter
        counter += 1

    return dict(results)

if __name__ == '__main__':  # pragma: no cover
    parser = create_argparser()
    args = parser.parse_args()

    # manual check argument dependency
    if args.mode == "crossvalidation":
        if args.model is not None:
            parser.error("Crossvalidation will train a new model \
                         - do not specify external model")

    nlu_config = RasaNLUConfig(args.config, os.environ, vars(args))
    logging.basicConfig(level=nlu_config['log_level'])

    if args.mode == "crossvalidation":
        data = load_data(args.data)
        data = prepare_data(data, cutoff = 5)
        results = run_cv_evaluation(data, int(args.folds), nlu_config)
        logger.info("CV evaluation (n={})".format(args.folds))
        for k,v in results.items():
            logger.info("{}: {:.3f} ({:.3f})".format(k, np.mean(v), np.std(v)))
    elif args.mode == "evaluation":
        run_evaluation(nlu_config, args.model)

    logger.info("Finished evaluation")

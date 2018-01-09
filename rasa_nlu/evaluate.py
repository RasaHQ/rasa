from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import itertools
import logging
import os

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.converters import load_data
from rasa_nlu.model import Interpreter
from rasa_nlu.model import Metadata
from rasa_nlu.model import Trainer, TrainingData

logger = logging.getLogger(__name__)


def create_argparser():
    import argparse
    parser = argparse.ArgumentParser(
            description='evaluate a Rasa NLU pipeline with cross validation or on external data \n
            ')

    parser.add_argument('-d', '--data', default=None,
                        help="file containing training/evaluation data")
    parser.add_argument('--mode', required=True,
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
                          zmin=1):
    """Print and plot the confusion matrix for the intent classification.

    Normalization can be applied by setting `normalize=True`."""
    import numpy as np

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


def log_evaluation_table(test_y, preds):
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


def run_model_evaluation(config, model_path, component_builder=None):
    from sklearn.metrics import confusion_matrix
    from sklearn.utils.multiclass import unique_labels

    # get the metadata config from the package data
    test_data = load_data(config['data'], config['language'])
    interpreter = Interpreter.load(model_path, config, component_builder)

    test_y = [e.get("intent") for e in test_data.training_examples]

    preds = []
    for e in test_data.training_examples:
        res = interpreter.parse(e.text)
        if res.get('intent'):
            preds.append(res['intent'].get('name'))
        else:
            preds.append(None)

    log_evaluation_table(test_y, preds)

    cnf_matrix = confusion_matrix(test_y, preds)
    plot_confusion_matrix(cnf_matrix, classes=unique_labels(test_y, preds),
                          title='Intent Confusion matrix')

    plt.show()
    return


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

if __name__ == '__main__':
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
        run_cv_evaluation(data, int(args.folds), nlu_config)
    elif args.mode == "evaluation":
        run_model_evaluation(nlu_config, args.model)

    logger.info("Finished evaluation")

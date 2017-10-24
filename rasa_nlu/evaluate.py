from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import itertools
import logging
import os

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.converters import load_data
from rasa_nlu.model import Interpreter
from rasa_nlu.model import Metadata

logger = logging.getLogger(__name__)


def create_argparser():
    parser = argparse.ArgumentParser(
            description='evaluate a trained Rasa NLU pipeline')

    parser.add_argument('-d', '--data', default=None,
                        help="file containing evaluation data")
    parser.add_argument('-m', '--model', required=True,
                        help="path to model")
    parser.add_argument('-c', '--config', required=True,
                        help="config file")
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


def run_intent_evaluation(config, model_path, component_builder=None):
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


if __name__ == '__main__':
    parser = create_argparser()
    args = parser.parse_args()
    nlu_config = RasaNLUConfig(args.config, os.environ, vars(args))
    logging.basicConfig(level=nlu_config['log_level'])

    run_intent_evaluation(nlu_config, args.model)
    logger.info("Finished evaluation")

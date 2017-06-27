from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import os

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.utils.multiclass import unique_labels

from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.converters import load_data
from rasa_nlu.model import Interpreter
from rasa_nlu.model import Metadata

logger = logging.getLogger(__name__)


def create_argparser():
    parser = argparse.ArgumentParser(description='evaluate a trained Rasa NLU pipeline')

    parser.add_argument('-d', '--data', default=None, help="file containing evaluation data")
    parser.add_argument('-m', '--model', required=True, help="path to model")
    parser.add_argument('-c', '--config', required=True, help="config file")
    return parser


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    import numpy as np
    import matplotlib.pyplot as plt

    plt.imshow(cm, interpolation='nearest', cmap=cmap if cmap else plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def do_evaluation(config, model_path, component_builder=None):
    # get the metadata config from the package data
    test_data = load_data(config['data'])
    metadata = Metadata.load(model_path)
    interpreter = Interpreter.load(metadata, config, component_builder)

    test_y = [e["refinement"] for e in test_data.training_examples]

    preds = []
    for e in test_data.training_examples:
        res = interpreter.parse(e["text"])
        preds.append(res['faq']['name'])

    print('f1 = {}'.format(f1_score(test_y, preds, average='weighted')))
    print('precision = {}'.format(precision_score(test_y, preds, average='weighted')))
    print('accuracy_score = {}'.format(accuracy_score(test_y, preds)))
    print(classification_report(test_y, preds))

    cnf_matrix = confusion_matrix(test_y, preds)
    import matplotlib.pyplot as plt
    plot_confusion_matrix(cnf_matrix, classes=unique_labels(test_y, preds),
                          title='Confusion matrix')

    plt.show()
    return


if __name__ == '__main__':
    parser = create_argparser()
    args = parser.parse_args()
    config = RasaNLUConfig(args.config, os.environ, vars(args))
    logging.basicConfig(level=config['log_level'])

    do_evaluation(config, args.model)
    logger.info("Finished evaluation")


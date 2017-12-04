from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from rasa_nlu.converters import load_data
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Trainer, TrainingData, Interpreter

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn import metrics
import numpy as np
import logging
import argparse
import os
from itertools import groupby
from collections import defaultdict

logger = logging.getLogger(__name__)

def create_argparser():
    parser = argparse.ArgumentParser(
            description='cross evaluate a Rasa NLU pipeline')

    parser.add_argument('-d', '--data', default=None,
                        help="file containing training data")
    parser.add_argument('-c', '--config', required=True,
                        help="config file")
    return parser


def prepare_data(data, cutoff = 5):
    """remove intent groups with less than cutoff instances."""
    data = data.sorted_intent_examples()
    logger.info("Raw data intent examples: {}".format(len(data)))

    # count intents
    list_intents = []
    n_intents = []
    for intent, group in groupby(data, lambda e: e.get("intent")):
        size = len(list(group))
        n_intents.append(size)
        list_intents.append(intent)

    # remove small intents groups
    ind_delete = []
    for ind in range(len(data)):
        if data[ind].get("intent") in [list_intents[i] for i in range(len(list_intents)) if n_intents[i] < cutoff]:
            ind_delete.append(ind)

    # make sure to sort first, so that index change is not affecting deletions
    for index in sorted(ind_delete, reverse=True):
        del data[index]

    return data


def run_cv_evaluation(data, n_splits, nlu_config, component_builder=None):
    """stratified cross validation on data

    params:
        :data: TODO
        :n_splits: TODO
        :nlu_config:
        :component_builder: TODO

    returns:
        :results: dictionary with key, list structure, where each entry in list
                  corresponds to the relevant result for one fold

    """
    trainer = Trainer(nlu_config)
    results = defaultdict(list)

    y_true = [e.get("intent") for e in data]

    skf = StratifiedKFold(n_splits=n_splits, random_state=11, shuffle=True)
    counter = 1
    for train_index, test_index in skf.split(data, y_true):

        #  train_index = list(train_index)
        #  test_index = list(test_index)
        #  test_index = list(train_index) # TODO debugging
        train = [data[i] for i in train_index]
        test = [data[i] for i in test_index]

        logger.info("Fold: {}".format(counter))
        logger.info("Training ...")
        trainer.train(TrainingData(training_examples=train))
        model_directory = trainer.persist("projects/")  # Returns the directory the model is stored in

        logger.info("Evaluation ...")
        interpreter = Interpreter.load(model_directory, nlu_config, component_builder)
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

    return results

if __name__ == '__main__':
    parser = create_argparser()
    args = parser.parse_args()
    nlu_config = RasaNLUConfig(args.config, os.environ, vars(args))
    logging.basicConfig(level=nlu_config['log_level'])

    n_splits = 5
    data = load_data(args.data)
    data = prepare_data(data)

    results = run_cv_evaluation(data, n_splits, nlu_config)
    logger.info(results)
    for key, value in results.items():
        logger.info("{0}: {1}".format(key, np.mean(value)))

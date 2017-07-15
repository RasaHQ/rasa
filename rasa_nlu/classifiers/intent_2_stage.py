from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import io
import json
import re
import spacy
import csv
import random
import numpy as np
from builtins import map
from typing import Any
from typing import Dict
from typing import Text
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


import rasa_nlu.converters as converters
from rasa_nlu.training_data import TrainingData, Message
from rasa_nlu.featurizers.spacy_featurizer import SpacyFeaturizer
from rasa_nlu.featurizers.ngram_featurizer import NGramFeaturizer

from rasa_nlu.components import Component

MAX_CV_FOLDS = 5

class Intent2Stage(Component):

    name = "intent_2_stage"

    provides = ["intent"]

    requires = ["intent"]

    def __init__(self, clf=None):
        #self.neg_train_data = neg_train_data
        self.clf = clf

    def neg_format(self, file_path):
        """
        Clean data from fb-page-comment-scraper and return it as a list of sentences.  
        """
        # import sys  
        # reload(sys)  
        # sys.setdefaultencoding('utf8')
        
        # Need this to direct prints to jupyter notebook rather than the terminal (lines above work too but with strange printing)
        import sys
        stdout = sys.stdout
        reload(sys)
        sys.setdefaultencoding('utf-8')
        sys.stdout = stdout
        
        dic = {'\n': '', '\[\[[a-zA-Z]+\]\]': '', '[^a-zA-Z0-9]{4,}': ' ', 'http\S+': ''}
        # this dictionary removes (line breaks, [[something (eg. PHOTO)]], strings of 4 or more non-alphanumeric characters, links)

        def replace_all(text, dic):
            for i, j in dic.iteritems():
                text = re.sub(i, j, text)
            return(text)

        with open(file_path, 'rU') as infile:
        # read the file as a dictionary for each row ({header : value})
            reader = csv.DictReader(infile)
            data = {}
            for row in reader:
                for header, value in row.items():
                    try:
                        data[header].append(value)
                    except KeyError:
                        data[header] = [value]

        raw = data['comment_message']
        neg_train_data = []
        for comment in raw:
            comment = replace_all(comment, dic)
            sents = [sent for sent in re.split('[.?!]+', comment) if re.match('.*[a-zA-Z]+', sent)]
            if (len(sents) > 0 & len(sents) < 3):
                #if type(comment) != "<type 'unicode'>":
                #    unicode(comment, "utf-8")
                if not(isinstance(comment, unicode)):
                    comment = unicode(comment, "utf-8")
                neg_train_data.append(comment)
        return neg_train_data


    def neg_featurize(self, neg_train_data):
        """
        Use the previously trained featurizers in the pipeline to featurize the negative training data (a set of sentences)
        """

        X_neg = []
        for example in neg_train_data:
            m = Message(example)
            self.partially_process(m)
            #print("message: {}; intent: {}".format(example, m.get("intent")))
            X_neg.append(m.get("text_features"))

        X_neg = np.array(X_neg)
        return X_neg 

    def train(self, training_data, config, **kwargs):
        #def train(self, training_data, intent_features, num_threads, max_number_of_ngrams):
        """ 
        Train an SVC on both +/- training sets. intent_features is the positive class training set. Currently, 
        the negative training set is explicitly created inside this function
        """

        intent_features = np.stack([example.get("text_features") for example in training_data.intent_examples])
        raw_data = self.neg_format('/home/sarenne/facebook-page-post-scraper/unnutzeswissen_facebook_comments.csv')
        neg_train_data = self.neg_featurize(raw_data)
        split = 0.5

        print("positive raw data shape: {}".format(intent_features.shape))
        print("negative raw data shape: {}".format(neg_train_data.shape))

        idx = np.random.randint(neg_train_data.shape[0], size=int(intent_features.shape[0] * split)) # pick the number of positive training samples that you have (ie. intent_features.shape[0])
        # Shuffle +/- training data and corresponding labels to get final training set (X, y)
        #train_neg = np.concatenate((neg_train_data, np.array([np.zeros(neg_train_data.shape[0])]).T), axis=1) -- this line works with manual dataset
        train_neg = np.concatenate((neg_train_data[idx, :], np.array([np.zeros(idx.shape[0])]).T), axis=1) 
        train_pos = np.concatenate((intent_features, np.array([np.ones(intent_features.shape[0])]).T), axis=1)
        print("shape of negative training data: {}\nshape of positive training data: {}".format(train_neg.shape, train_pos.shape))
        print("pos example intent: {}".format(training_data.intent_examples[0].get("intent")))
        train = np.concatenate((train_neg, train_pos), axis=0)
        np.take(train, np.random.permutation(train.shape[0]), axis=0, out=train)
        X = train[:,:-1]
        y = train[:, -1]
        print("X shape: {}\ny shape: {}".format(X.shape, y.shape))

        tuned_parameters = [{'C': [1, 2, 5, 10, 20, 100], 'kernel': [str('linear')]}]
        #cv_splits = max(2, min(MAX_CV_FOLDS, np.min(np.bincount(y)) // 5))  # aim for 5 examples in each fold

        self.clf = GridSearchCV(SVC(C=1, probability=True),
                                    param_grid=tuned_parameters, n_jobs=config["num_threads"],
                                    scoring='f1_weighted', verbose=1)
        # self.clf = GridSearchCV(SVC(C=1, probability=True), scoring='f1_weighted',
        #                             n_jobs=num_threads, verbose=1)
        self.clf.fit(X, y)

    def make_test_set(self, path, split):
        """
        create a test set by appending a random set of negative examples to an existing json file os positive examples. 
        `split` indicates the proportion of negative examples in the test set (ie 0.5 = 50% negative examples).
        """

        test_examples = []
        # load positive examples from json path
        with open(path, "r") as outfile:
             pos_examples = json.load(outfile)["rasa_nlu_data"]["common_examples"]

        test_examples.extend(pos_examples)

        # pick random set of negative examples (size from `split`) and append to the positive examples to create test_examples
        neg_examples = []
        raw_data = self.neg_format('/home/sarenne/facebook-page-post-scraper/unnutzeswissen_facebook_comments.csv')
        for sentence in raw_data:
            example = {
                "text": sentence,
                "intent": "out_of_scope",
                "entities": []
                }
            neg_examples.append(example)
        neg_num = int((len(pos_examples)/(1 - split)) - len(pos_examples))
        print("# neg examples: {}\n# pos examples: {}".format(neg_num, len(pos_examples)))
        idx = np.random.randint(len(neg_examples), size=neg_num)
        neg_random = [neg_examples[i] for i in idx]
        test_examples.extend(neg_random)
        #print("pos examples: {}\n------\nneg examples: {}\n----------\ntest_examples: {}".format(pos_examples, neg_random, test_examples))

        # overwrite the json path with new set of test examples
        with open(path, "w") as outfile: 
            json.dump({"rasa_nlu_data": {"common_examples":test_examples}}, outfile, indent=4)

    def predict_prob(self, X):
        # type: (np.ndarray) -> np.ndarray
        """Given a bow vector of an input text, predict the intent label. Returns probabilities for all labels.

        :param X: bow of input text
        :return: vector of probabilities containing one entry for each label"""
        return self.clf.predict_proba(X)

    def predict(self, X):
        # type: (np.ndarray) -> Tuple[np.ndarray, np.ndarray]
        """Given a bow vector of an input text, predict most probable label. Returns only the most likely label.

        :param X: bow of input text
        :return: tuple of first, the most probable label and second, its probability"""

        import numpy as np

        pred_result = self.predict_prob(X)
        # sort the probabilities retrieving the indices of the elements in sorted order
        sorted_indices = np.fliplr(np.argsort(pred_result, axis=1))
        #print("pred result: {}\nsorted indices: {}".format(pred_result[:, sorted_indices], sorted_indices))
        return sorted_indices, pred_result[:, sorted_indices]

    def process(self, message, **kwargs):
        # def process(self, intent_features, intent):
        # type: (Text) -> Dict[Text, Any]
        if self.clf:
            X = message.get("text_features").reshape(1, -1)
            intents, probabilities = self.predict(X)
            # `predict` returns a matrix as it is supposed to work for multiple examples as well, hence we need to flatten
            intents, probabilities = intents.flatten(), probabilities.flatten()

            if intents.size > 0 and probabilities.size > 0:
                ranking = list(zip(list(intents), list(probabilities)))
                if intents[0] == 0:
                    message.set("intent", {
                            "name": "out_of_scope",
                            "confidence": probabilities[0],
                        }, add_to_output=True)       

    @classmethod
    def load(cls, model_dir=None, model_metadata=None, cached_component=None, **kwargs):
        #def load(cls, model_dir, intent_2_stage):
        # type: (Text, Text) -> SklearnIntentClassifier
        import cloudpickle

        if model_dir and model_metadata.get("intent_2_stage"):
            classifier_file = os.path.join(model_dir, model_metadata.get("intent_2_stage"))
            with io.open(classifier_file, 'rb') as f:   # pragma: no test
                if PY3:
                    return cloudpickle.load(f, encoding="latin-1")
                else:
                    return cloudpickle.load(f)
        else:
            return Intent2Stage()

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]
        """Persist this model into the passed directory. Returns the metadata necessary to load the model again."""

        import cloudpickle

        classifier_file = os.path.join(model_dir, "intent_classifier.pkl")
        with io.open(classifier_file, 'wb') as f:
            cloudpickle.dump(self, f)

        return {
            "intent_2_stage": "intent_classifier.pkl"
        }


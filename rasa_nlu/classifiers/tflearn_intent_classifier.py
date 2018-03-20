from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from typing import Any
from typing import Dict
from typing import Text

from rasa_nlu.components import Component
from rasa_nlu.training_data import Message

import os
import tensorflow as tf
import pickle
import tflearn
import numpy as np
import random
from nltk.stem.lancaster import LancasterStemmer


class TflearnIntentClassifier(Component):

    name = "intent_classifier_tflearn"

    provides = ["intent", "intent_ranking"]

    def __init__(self, clf=None):
        self.model = None
        self.words = []
        self.classes = []
        self.train_x = {}
        self.train_y = {}
        self.clf = clf

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["nltk", "tensorflow", "pickle", "numpy", "tflearn"]

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUConfig, **Any) -> None

        """Train the intent classifier on a data set."""

        stemmer = LancasterStemmer()
        documents = []
        ignore_words = ['?', '¡', '!', '.', ',', ':']

        for e in training_data.intent_examples:
            # get tokens
            w = []
            for token in e.get("tokens"):
                w.append(token.text)
            # add to our words list
            self.words.extend(w)
            # add to documents in our corpus
            documents.append((w, e.get("intent")))
            # add to our classes list
            if e.get("intent") not in self.classes:
                self.classes.append(e.get("intent"))

        if self.classes:
            # stem and lower each word and remove duplicates
            self.words = [stemmer.stem(w.lower()) for w in self.words if w not in ignore_words]
            self.words = sorted(list(set(self.words)))

            # remove duplicates
            self.classes = sorted(list(set(self.classes)))

            # create our training data
            training = []

            # create an empty array for our output
            output_empty = [0] * len(self.classes)

            # training set, bag of words for each sentence
            for doc in documents:
                # initialize our bag of words
                bag = []
                # list of tokenized words for the pattern
                pattern_words = doc[0]
                # stem each word
                pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
                # create our bag of words array
                for w in self.words:
                    bag.append(1) if w in pattern_words else bag.append(0)

                # output is a '0' for each tag and '1' for current tag
                output_row = list(output_empty)
                output_row[self.classes.index(doc[1])] = 1

                training.append([bag, output_row])

            # shuffle our features and turn into np.array
            random.shuffle(training)
            training = np.array(training)

            # create train and test lists
            self.train_x = list(training[:, 0])
            self.train_y = list(training[:, 1])

            # reset underlying graph data
            tf.reset_default_graph()
            # Build neural network
            net = tflearn.input_data(shape=[None, len(self.train_x[0])])
            net = tflearn.fully_connected(net, 32)
            net = tflearn.fully_connected(net, 32)
            net = tflearn.fully_connected(net, len(self.train_y[0]), activation='softmax')
            net = tflearn.regression(net)
            self.model = tflearn.DNN(net)

            # Start training (apply gradient descent algorithm)
            self.model.fit(self.train_x,
                           self.train_y,
                           n_epoch=len(self.classes) + 100,
                           batch_size=int(len(self.classes) / 10),
                           show_metric=False)

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        if not self.clf:
            # component is either not trained or didn't receive enough training data
            intent = None
            intent_ranking = []
        else:
            threshold = 0

            model = self.clf.get_model()
            results = model.predict([self.bow(message.get("tokens"), self.clf.get_words())])[0]
            results = [[i, r] for i, r in enumerate(results) if r > threshold]
            results.sort(key=lambda x: x[1], reverse=True)
            return_list = []
            for r in results:
                classes = self.clf.get_classes()
                return_list.append((classes[r[0]], r[1]))

            if len(return_list) == 0:
                intent = {"name": None, "confidence": 0.0}
                intent_ranking = []
            else:
                intent = {"name": return_list[0][0], "confidence": return_list[0][1]}
                intent_ranking = [{"name": intent_name, "confidence": score} for intent_name, score in return_list]

        message.set("intent", intent, add_to_output=True)
        message.set("intent_ranking", intent_ranking, add_to_output=True)

    @classmethod
    def load(cls, model_dir=None, model_metadata=None, cached_component=None, **kwargs):
        # type: (Text, Metadata, Optional[Component], **Any) -> SklearnIntentClassifier
        if model_dir and model_metadata.get("intent_classifier_tflearn"):
            classifier_file = os.path.join(model_dir,
                                           model_metadata.get("intent_classifier_tflearn").split(',')[0])
            trained_intents_file = os.path.join(model_dir,
                                                model_metadata.get("intent_classifier_tflearn").split(',')[1])

            if os.path.isfile(trained_intents_file):
                # restore all of our data structures
                data = pickle.load(open(trained_intents_file, "rb"))
                words = data['words']
                classes = data['classes']
                train_x = data['train_x']
                train_y = data['train_y']

                with tf.Graph().as_default():
                    # Build neural network
                    net = tflearn.input_data(shape=[None, len(train_x[0])])
                    net = tflearn.fully_connected(net, 32)
                    net = tflearn.fully_connected(net, 32)
                    net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
                    net = tflearn.regression(net)
                    model = tflearn.DNN(net)

                    # load our saved model
                    model.load(classifier_file)
                    return cls(TfModel(model, words, classes))
        else:
            return cls()

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]
        """Persist this model into the passed directory. Returns the metadata necessary to load the model again."""

        classifier_file = os.path.join(model_dir, "intents-model.tflearn")
        trained_intents_file = os.path.join(model_dir, "trained-intents")
        if classifier_file and trained_intents_file and self.model:
            self.model.save(classifier_file)

            # save all of our data structures
            pickle.dump({'words': self.words,
                        'classes': self.classes,
                        'train_x': self.train_x,
                        'train_y': self.train_y},
                        open(trained_intents_file, "wb"))
        
        return {
            "intent_classifier_tflearn": "intents-model.tflearn,trained-intents"
        }

    def clean_up_sentence(self, tokens):
        stemmer = LancasterStemmer()
        # stem each word
        sentence_words = [stemmer.stem(word.text.lower()) for word in tokens]
        return sentence_words

    # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
    def bow(self, tokens, words):
        # tokenize the pattern
        sentence_words = self.clean_up_sentence(tokens)
        # bag of words
        bag = [0] * len(words)
        for s in sentence_words:
            for i, w in enumerate(words):
                if w == s:
                    bag[i] = 1

        return np.array(bag)


class TfModel:
    def __init__(self, model=None, words=None, classes=None):
        self.model = model
        self.words = words
        self.classes = classes

    def get_model(self):
        return self.model

    def get_words(self):
        return self.words

    def get_classes(self):
        return self.classes

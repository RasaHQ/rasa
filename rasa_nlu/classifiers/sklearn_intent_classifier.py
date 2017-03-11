import os

from rasa_nlu.components import Component


class SklearnIntentClassifier(Component):
    """Intent classifier using the sklearn framework"""

    name = "intent_sklearn"

    def __init__(self, clf=None, le=None):
        """Construct a new intent classifier using the sklearn framework.

        :param uses_probabilities: defines if the model should be trained
                                           to be able to predict label probabilities
        :param max_num_threads: number of threads used during training time
        :type uses_probabilities: bool"""
        if le is not None:
            self.le = le
        else:
            from sklearn.preprocessing import LabelEncoder
            self.le = LabelEncoder()
        self.clf = clf

    def transform_labels_str2num(self, labels):
        """Transforms a list of strings into numeric label representation.

        :param labels: List of labels to convert to numeric representation
        :type labels: list of str"""

        y = self.le.fit_transform(labels)
        return y

    def transform_labels_num2str(self, y):
        """Transforms a list of strings into numeric label representation.

        :param labels: List of labels to convert to numeric representation
        :type labels: list of str"""

        labels = self.le.inverse_transform(y)
        return labels

    def train(self, training_data, intent_features, num_threads):
        """Train the intent classifier on a data set."""

        from sklearn.model_selection import GridSearchCV
        from sklearn.svm import SVC

        labels = [e["intent"] for e in training_data.intent_examples]

        if len(set(labels)) < 2:
            raise Exception("Can not train an intent classifier. Need at least 2 different classes.")
        y = self.transform_labels_str2num(labels)
        X = intent_features

        tuned_parameters = [{'C': [1, 2, 5, 10, 20, 100], 'kernel': ['linear']}]
        self.clf = GridSearchCV(SVC(C=1, probability=True),
                                param_grid=tuned_parameters, n_jobs=num_threads,
                                cv=2, scoring='f1_weighted')

        self.clf.fit(X, y)

    def process(self, intent_features):
        """Returns the most likely intent and its probability for the input text."""

        X = intent_features.reshape(1, -1)
        intent_ids, probabilities = self.predict(X)
        intents = self.transform_labels_num2str(intent_ids)
        intent, score = intents[0], probabilities[0]

        return {
            "intent": {
                "name": intent,
                "confidence": score,
            }
        }

    def predict_prob(self, X):
        """Given a bow vector of an input text, predict the intent label. Returns probabilities for all labels.

        :param X: bow of input text
        :return: vector of probabilities containing one entry for each label"""

        import numpy as np

        if hasattr(self, 'uses_probabilities') and self.uses_probabilities:
            return self.clf.predict_proba(X)
        else:
            y_pred_indices = self.clf.predict(X)
            # convert representation to one-hot. all labels are zero, only the predicted label gets assigned prob=1
            y_pred = np.zeros((np.size(X, 0), len(self.le.classes_)))
            y_pred[np.arange(y_pred.shape[0]), y_pred_indices] = 1
            return y_pred

    def predict(self, X):
        """Given a bow vector of an input text, predict most probable label. Returns only the most likely label.

        :param X: bow of input text
        :return: tuple of first, the most probable label and second, its probability"""

        import numpy as np

        pred_result = self.predict_prob(X)
        max_indicies = np.argmax(pred_result, axis=1)
        # retrieve the index of the intent with the highest probability
        max_values = pred_result[:, max_indicies].flatten()
        return max_indicies, max_values

    @classmethod
    def load(cls, model_dir, intent_classifier):
        import cloudpickle
        if model_dir and intent_classifier:
            classifier_file = os.path.join(model_dir, intent_classifier)
            with open(classifier_file, 'rb') as f:
                return cloudpickle.load(f)
        else:
            return SklearnIntentClassifier()

    def persist(self, model_dir):
        """Persist this model into the passed directory. Returns the metadata necessary to load the model again."""

        import cloudpickle

        classifier_file = os.path.join(model_dir, "intent_classifier.pkl")
        with open(classifier_file, 'wb') as f:
            cloudpickle.dump(self, f)

        return {
            "intent_classifier": "intent_classifier.pkl"
        }

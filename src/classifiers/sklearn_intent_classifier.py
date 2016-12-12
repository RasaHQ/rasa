from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import logging


class SklearnIntentClassifier(object):
    def __init__(self):
        self.le = LabelEncoder()
        self.tuned_parameters = [
            {'C': [1, 2, 5, 10, 20, 100], 'kernel': ['linear']}
        ]
        self.score = 'f1'  # 'precision'
        self.clf = GridSearchCV(SVC(C=1), self.tuned_parameters, cv=2, scoring='%s_weighted' % self.score)

    def transform_labels(self, labels):
        y = self.le.fit_transform(labels)
        return y

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=0)
        self.clf.fit(X_train, y_train)

        # Test the trained model
        logging.info("Score of intent model on test data: %s " % self.clf.score(X_test, y_test))

    def predict(self, X, to_labels=True):
        y_pred = self.clf.predict(X)
        if (to_labels):
            return self.le.inverse_transform(y_pred)
        return y_pred

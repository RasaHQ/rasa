import pytest
import copy

from rasa.nlu.classifiers.keyword_intent_classifier import KeywordIntentClassifier
from rasa.nlu.classifiers.embedding_intent_classifier import EmbeddingIntentClassifier
from rasa.nlu.classifiers.mitie_intent_classifier import MitieIntentClassifier
from rasa.nlu.classifiers.sklearn_intent_classifier import SklearnIntentClassifier
from rasa.nlu.training_data import load_data

from tests.nlu.conftest import DEFAULT_DATA_PATH


@pytest.fixture(scope="module")
def training_data():
    return load_data(DEFAULT_DATA_PATH)


@pytest.fixture(scope="module")
def test_data():
    return load_data(DEFAULT_DATA_PATH)


class ClassifierTestCollection():
    """Tests every classifier needs to fulfill.

    Each classifier can have additional tests in its own class."""

    @pytest.fixture(scope="module")
    def classifier_class(self):
        return NotImplementedError

    @pytest.fixture(scope="class")
    def filename(self, classifier_class):
        return "component_0_"+classifier_class.name

    @pytest.fixture(scope="module")
    def trained_classifier(self, classifier_class, training_data,
                           component_config, **kwargs):
        classifier_params = classifier_class.__init__.__code__.co_varnames
        train_params = {}
        for p in classifier_params:
            arg = kwargs.pop(p, None)
            if arg is not None:
                train_params.update(arg)
        classifier = self._create_classifier(classifier_class, component_config
                                             , **kwargs)
        classifier.train(training_data, {}, **train_params)
        return classifier

    @pytest.fixture(scope="module")
    def component_config(self):
        return {}

    def _create_classifier(self, classifier_class, component_config, **kwargs):
        classifier = classifier_class(component_config, **kwargs)
        return classifier

    def test_persist_and_load(self, test_data, trained_classifier, filename, tmpdir):
        meta = trained_classifier.persist(filename, tmpdir)
        loaded = trained_classifier.__class__.load(meta, tmpdir)
        predicted = copy.copy(test_data)
        actual = copy.copy(test_data)
        for m1, m2 in zip(predicted.training_examples, actual.training_examples):
            loaded.process(m1)
            trained_classifier.process(m2)
            assert m1.get("intent") == m2.get("intent")


class TestKeywordClassifier(ClassifierTestCollection):
    @pytest.fixture(scope="module")
    def classifier_class(self):
        return KeywordIntentClassifier


class TestEmbeddingIntentClassifier(ClassifierTestCollection):
    @pytest.fixture(scope="module")
    def classifier_class(self):
        return EmbeddingIntentClassifier


class TestMitieIntentClassifier(ClassifierTestCollection):
    @pytest.fixture(scope="module")
    def classifier_class(self):
        return MitieIntentClassifier


class TestSklearnIntentClassifier(ClassifierTestCollection):
    @pytest.fixture(scope="module")
    def classifier_class(self):
        return SklearnIntentClassifier


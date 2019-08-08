import pytest
import copy
import logging

from rasa.nlu.classifiers.keyword_intent_classifier import KeywordIntentClassifier

# TODO: add tests for other classifers
# from rasa.nlu.classifiers.embedding_intent_classifier import EmbeddingIntentClassifier
# from rasa.nlu.classifiers.mitie_intent_classifier import MitieIntentClassifier
# from rasa.nlu.classifiers.sklearn_intent_classifier import SklearnIntentClassifier
from rasa.nlu.training_data.formats.rasa import RasaReader
from rasa.nlu.training_data import load_data
from rasa.nlu.training_data.message import Message
from tests.nlu.conftest import DEFAULT_DATA_PATH


@pytest.fixture(scope="module")
def training_data():
    return load_data(DEFAULT_DATA_PATH)


class ClassifierTestCollection:
    """Tests every classifier needs to fulfill.

    Each classifier can have additional tests in its own class."""

    @pytest.fixture(scope="module")
    def classifier_class(self):
        return NotImplementedError

    @pytest.fixture(scope="class")
    def filename(self, classifier_class):
        return "component_0_" + classifier_class.name

    @pytest.fixture(scope="module")
    def trained_classifier(
        self, classifier_class, training_data, component_config, **kwargs
    ):
        return self._train_classifier(
            classifier_class, training_data, component_config, **kwargs
        )

    def _train_classifier(
        self, classifier_class, training_data, component_config, **kwargs
    ):
        # this ugly line is here because the kwargs of this function contain kwargs
        # for both the classifier init and the training, getting the names of the
        # classifiers kwargs we can separate them from the training kwargs
        classifier_params = classifier_class.__init__.__code__.co_varnames
        train_params = {}
        for p in classifier_params:
            arg = kwargs.pop(p, None)
            if arg is not None:
                train_params.update(arg)
        classifier = self._create_classifier(
            classifier_class, component_config, **kwargs
        )
        classifier.train(training_data, {}, **train_params)
        return classifier

    @pytest.fixture(scope="module")
    def component_config(self):
        return {}

    def _create_classifier(self, classifier_class, component_config, **kwargs):
        classifier = classifier_class(component_config, **kwargs)
        return classifier

    def test_persist_and_load(
        self, training_data, trained_classifier, filename, tmpdir
    ):
        meta = trained_classifier.persist(filename, tmpdir)
        loaded = trained_classifier.__class__.load(meta, tmpdir)
        predicted = copy.copy(training_data)
        actual = copy.copy(training_data)
        for m1, m2 in zip(predicted.training_examples, actual.training_examples):
            loaded.process(m1)
            trained_classifier.process(m2)
            assert m1.get("intent") == m2.get("intent")

    @pytest.mark.parametrize(
        "input, intent",
        [
            ("hey there joe", "greet"),
            ("hello weiouaosdhalkh", "greet"),
            ("show me chines restaurants in the north of town", "restaurant_search"),
            ("great", "affirm"),
            ("bye bye birdie", "goodbye"),
            ("show me a mexican place", None),
            ("i", None),
            ("in", None),
            ("eet", None),
        ],
    )
    def test_classification(self, trained_classifier, input, intent):
        text = Message(input)
        trained_classifier.process(text)
        assert text.get("intent").get("name", "NO_INTENT") == intent


class TestKeywordClassifier(ClassifierTestCollection):
    @pytest.fixture(scope="module")
    def classifier_class(self):
        return KeywordIntentClassifier

    @pytest.mark.parametrize(
        "input, intents",
        [
            ("hey there ok", ["greet", "affirm"]),
            ("ok hey there stop", ["affirm", "greet", "goodbye"]),
        ],
    )
    def test_multiple_matches(self, caplog, trained_classifier, input, intents):
        text = Message(input)
        with caplog.at_level(logging.DEBUG):
            trained_classifier.process(text)
        assert len(caplog.records) == 1
        for intent in intents:
            assert intent in caplog.records[0].getMessage()

    def test_ambiguous_data(
        self, caplog, classifier_class, training_data, component_config, **kwargs
    ):
        json_data = {
            "rasa_nlu_data": {
                "common_examples": [
                    {"text": "good", "intent": "affirm", "entities": []},
                    {"text": "good morning", "intent": "greet", "entities": []},
                    {"text": "see you", "intent": "goodbye", "entities": []},
                    {"text": "nice to see you", "intent": "greet", "entities": []},
                ]
            }
        }
        rasa_reader = RasaReader()
        data = rasa_reader.read_from_json(json_data)

        with caplog.at_level(logging.WARNING):
            self._train_classifier(classifier_class, data, component_config, **kwargs)
        assert len(caplog.records) == 2

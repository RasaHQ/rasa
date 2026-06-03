import inspect
import os.path
import tempfile
from typing import Dict, Text, Any, Optional
from unittest.mock import Mock
from _pytest.monkeypatch import MonkeyPatch

import rasa.shared.utils.io
from rasa.core.policies.ted_policy import TEDPolicy
from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.training import fingerprinting
from rasa.nlu.classifiers.diet_classifier import DIETClassifier
from rasa.nlu.selectors.response_selector import ResponseSelector
from tests.engine.training.test_components import FingerprintableText


def test_fingerprint_stays_same():
    key1 = fingerprinting.calculate_fingerprint_key(
        TEDPolicy, TEDPolicy.get_default_config(), {"input": FingerprintableText("Hi")}
    )
    key2 = fingerprinting.calculate_fingerprint_key(
        TEDPolicy, TEDPolicy.get_default_config(), {"input": FingerprintableText("Hi")}
    )

    assert key1 == key2


def test_fingerprint_changes_due_to_class():
    key1 = fingerprinting.calculate_fingerprint_key(
        DIETClassifier,
        TEDPolicy.get_default_config(),
        {"input": FingerprintableText("Hi")},
    )
    key2 = fingerprinting.calculate_fingerprint_key(
        ResponseSelector,
        TEDPolicy.get_default_config(),
        {"input": FingerprintableText("Hi")},
    )

    assert key1 != key2


def test_fingerprint_changes_due_to_config():
    key1 = fingerprinting.calculate_fingerprint_key(
        TEDPolicy, {}, {"input": FingerprintableText("Hi")}
    )
    key2 = fingerprinting.calculate_fingerprint_key(
        ResponseSelector,
        TEDPolicy.get_default_config(),
        {"input": FingerprintableText("Hi")},
    )

    assert key1 != key2


def test_fingerprint_changes_due_to_inputs():
    key1 = fingerprinting.calculate_fingerprint_key(
        TEDPolicy, {}, {"input": FingerprintableText("Hi")}
    )
    key2 = fingerprinting.calculate_fingerprint_key(
        ResponseSelector,
        TEDPolicy.get_default_config(),
        {"input": FingerprintableText("bye")},
    )

    assert key1 != key2


def test_fingerprint_changes_due_to_changed_source(monkeypatch: MonkeyPatch):
    key1 = fingerprinting.calculate_fingerprint_key(
        TEDPolicy, {}, {"input": FingerprintableText("Hi")}
    )

    get_source_mock = Mock(return_value="other implementation")
    monkeypatch.setattr(inspect, inspect.getsource.__name__, get_source_mock)

    key2 = fingerprinting.calculate_fingerprint_key(
        TEDPolicy, {}, {"input": FingerprintableText("Hi")}
    )

    assert key1 != key2

    get_source_mock.assert_called_once_with(TEDPolicy)


def test_fingerprint_changes_when_external_file_changes():
    tmp_file = tempfile.mktemp()

    class MinimalComponent(GraphComponent):
        @classmethod
        def create(
            cls,
            config: Dict[Text, Any],
            model_storage: ModelStorage,
            resource: Resource,
            execution_context: ExecutionContext,
        ) -> "MinimalComponent":
            return MinimalComponent()

        @classmethod
        def fingerprint_addon(cls, config: Dict[str, Any]) -> Optional[str]:
            if not os.path.exists(tmp_file):
                return None
            else:
                return rasa.shared.utils.io.get_text_hash(open(tmp_file, "r").read())

    with open(tmp_file, "w") as external_data:
        external_data.write("This is a test.")

    fingerprint_1 = fingerprinting.calculate_fingerprint_key(MinimalComponent, {}, {})

    fingerprint_2 = fingerprinting.calculate_fingerprint_key(MinimalComponent, {}, {})

    assert fingerprint_1 == fingerprint_2

    # overwrite the original external data
    with open(tmp_file, "w") as external_data:
        external_data.write("This is a test for changes in external data.")

    fingerprint_3 = fingerprinting.calculate_fingerprint_key(MinimalComponent, {}, {})

    assert fingerprint_3 != fingerprint_1

    os.remove(tmp_file)

import inspect
from unittest.mock import Mock
from _pytest.monkeypatch import MonkeyPatch

from rasa.core.policies.ted_policy import TEDPolicyGraphComponent
from rasa.engine.training import fingerprinting
from rasa.nlu.classifiers.diet_classifier import DIETClassifierGraphComponent
from rasa.nlu.selectors.response_selector import ResponseSelectorGraphComponent
from tests.engine.training.test_components import FingerprintableText


def test_fingerprint_stays_same():
    key1 = fingerprinting.calculate_fingerprint_key(
        TEDPolicyGraphComponent,
        TEDPolicyGraphComponent.get_default_config(),
        {"input": FingerprintableText("Hi")},
    )
    key2 = fingerprinting.calculate_fingerprint_key(
        TEDPolicyGraphComponent,
        TEDPolicyGraphComponent.get_default_config(),
        {"input": FingerprintableText("Hi")},
    )

    assert key1 == key2


def test_fingerprint_changes_due_to_class():
    key1 = fingerprinting.calculate_fingerprint_key(
        DIETClassifierGraphComponent,
        TEDPolicyGraphComponent.get_default_config(),
        {"input": FingerprintableText("Hi")},
    )
    key2 = fingerprinting.calculate_fingerprint_key(
        ResponseSelectorGraphComponent,
        TEDPolicyGraphComponent.get_default_config(),
        {"input": FingerprintableText("Hi")},
    )

    assert key1 != key2


def test_fingerprint_changes_due_to_config():
    key1 = fingerprinting.calculate_fingerprint_key(
        TEDPolicyGraphComponent, {}, {"input": FingerprintableText("Hi")},
    )
    key2 = fingerprinting.calculate_fingerprint_key(
        ResponseSelectorGraphComponent,
        TEDPolicyGraphComponent.get_default_config(),
        {"input": FingerprintableText("Hi")},
    )

    assert key1 != key2


def test_fingerprint_changes_due_to_inputs():
    key1 = fingerprinting.calculate_fingerprint_key(
        TEDPolicyGraphComponent, {}, {"input": FingerprintableText("Hi")},
    )
    key2 = fingerprinting.calculate_fingerprint_key(
        ResponseSelectorGraphComponent,
        TEDPolicyGraphComponent.get_default_config(),
        {"input": FingerprintableText("bye")},
    )

    assert key1 != key2


def test_fingerprint_changes_due_to_changed_source(monkeypatch: MonkeyPatch):
    key1 = fingerprinting.calculate_fingerprint_key(
        TEDPolicyGraphComponent, {}, {"input": FingerprintableText("Hi")},
    )

    get_source_mock = Mock(return_value="other implementation")
    monkeypatch.setattr(inspect, inspect.getsource.__name__, get_source_mock)

    key2 = fingerprinting.calculate_fingerprint_key(
        TEDPolicyGraphComponent, {}, {"input": FingerprintableText("Hi")},
    )

    assert key1 != key2

    get_source_mock.assert_called_once_with(TEDPolicyGraphComponent)

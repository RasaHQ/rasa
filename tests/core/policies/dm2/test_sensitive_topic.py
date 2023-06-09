import os
from unittest import mock
from rasa.core.policies.dm2.sensitive_topic import (
    SensitiveTopicDetector,
    SensitiveTopicDetectorStub,
    CONFIG_KEY_USE_STUB,
    CONFIG_KEY_ACTION,
)
import logging

logger = logging.getLogger("test")


@mock.patch.object(os, "getenv", lambda x: None)
def test_detector_no_key_fallback(caplog):
    caplog.set_level("WARNING")
    detector = SensitiveTopicDetector({})
    assert "No OPENAI_API_KEY" in caplog.text
    assert detector._use_stub


@mock.patch.object(os, "getenv", lambda x: None)
def test_detector_use_stub(caplog):
    caplog.set_level("WARNING")
    detector = SensitiveTopicDetector({CONFIG_KEY_USE_STUB: True})
    assert "No OPENAI_API_KEY" not in caplog.text
    assert detector._use_stub


@mock.patch.object(os, "getenv", lambda x: None)
def test_detector_stub():
    detector = SensitiveTopicDetector(
        {
            CONFIG_KEY_USE_STUB: True,
            CONFIG_KEY_ACTION: "action",
        }
    )
    assert detector.action() == "action"
    assert not detector.check("Normal message")
    assert detector.check("I hear voices in my head")

    detector = SensitiveTopicDetectorStub(
        {CONFIG_KEY_USE_STUB: True}, positive=("suspicious Message",)
    )
    assert detector.check("suspicious message")
    assert not detector.check("normal message")

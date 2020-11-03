from pathlib import Path
from typing import List, Text

import numpy as np
import pytest
from rasa.core.policies.ted_policy import TEDPolicy
from rasa.shared.core.constants import ACTION_LISTEN_NAME
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import ActionExecuted, UserUttered
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.interpreter import RegexInterpreter

UTTER_GREET_ACTION = "utter_greet"
GREET_INTENT_NAME = "greet"
DOMAIN_YAML = f"""
intents:
- {GREET_INTENT_NAME}
actions:
- {UTTER_GREET_ACTION}
"""


def test_diagnostics():
    domain = Domain.from_yaml(DOMAIN_YAML)
    policy = TEDPolicy()
    GREET_RULE = DialogueStateTracker.from_events(
        "greet rule",
        evts=[
            UserUttered(intent={"name": GREET_INTENT_NAME}),
            ActionExecuted(UTTER_GREET_ACTION),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(intent={"name": GREET_INTENT_NAME}),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
    )
    policy.train([GREET_RULE], domain, RegexInterpreter())
    (
        action_probabilities,
        diagnostic_data,
    ) = policy.predict_action_probabilities(
        GREET_RULE, domain, RegexInterpreter()
    )

    assert isinstance(diagnostic_data, dict)
    assert "attention_weights" in diagnostic_data
    assert isinstance(diagnostic_data.get("attention_weights"), np.ndarray)

from rasa.core.policies.ted_policy import TEDPolicy
from rasa.shared.core.constants import ACTION_LISTEN_NAME
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import ActionExecuted, UserUttered
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.interpreter import RegexInterpreter
import tensorflow as tf


UTTER_GREET_ACTION = "utter_greet"
GREET_INTENT_NAME = "greet"
DOMAIN_YAML = f"""
intents:
- {GREET_INTENT_NAME}
actions:
- {UTTER_GREET_ACTION}
"""


if __name__ == "__main__":
    domain = Domain.from_yaml(DOMAIN_YAML)
    policy = TEDPolicy()
    tracker = DialogueStateTracker.from_events(
        "greet rule",
        evts=[
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(intent={"name": GREET_INTENT_NAME}),
            ActionExecuted(UTTER_GREET_ACTION),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(intent={"name": GREET_INTENT_NAME}),
            ActionExecuted(UTTER_GREET_ACTION),
            ActionExecuted(UTTER_GREET_ACTION),
            ActionExecuted(UTTER_GREET_ACTION),
        ],
    )
    policy.train([tracker], domain, RegexInterpreter())
    prediction = policy.predict_action_probabilities(
        tracker, domain, RegexInterpreter()
    )

    print(f"{prediction.diagnostic_data.get('attention_weights')}")

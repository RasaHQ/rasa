from pathlib import Path

from rasa.core.actions.action import ACTION_LISTEN_NAME
from rasa.core.constants import REQUESTED_SLOT, RULE_SNIPPET_ACTION_NAME
from rasa.core.domain import Domain
from rasa.core.events import (
    ActionExecuted,
    UserUttered,
    Form,
    SlotSet,
    ActionExecutionRejected,
)
from rasa.core.policies.rule_policy import RulePolicy
from rasa.core.trackers import DialogueStateTracker
from rasa.core.training.generator import TrackerWithCachedStates

UTTER_GREET_ACTION = "utter_greet"
GREET_RULE = DialogueStateTracker.from_events(
    "bla",
    evts=[
        ActionExecuted(RULE_SNIPPET_ACTION_NAME),
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered("haha", {"name": "greet"}),
        ActionExecuted(UTTER_GREET_ACTION),
    ],
)


def test_faq_rule():
    domain = Domain.from_yaml(
        f"""
intents:
- greet
actions:
- {UTTER_GREET_ACTION}
- {RULE_SNIPPET_ACTION_NAME}
    """
    )

    policy = RulePolicy()
    policy.train([GREET_RULE], domain)
    new_conversation = DialogueStateTracker.from_events(
        "bla2", GREET_RULE.applied_events()[1:-1]
    )
    action_probabilities = policy.predict_action_probabilities(new_conversation, domain)

    assert max(action_probabilities) == 1

    index_of_predicted_action = action_probabilities.index(max(action_probabilities))
    prediction_action_name = domain.action_names[index_of_predicted_action]
    assert prediction_action_name == UTTER_GREET_ACTION


async def test_predict_form_action_if_in_form(tmp_path: Path):
    form_name = "some_form"

    domain = Domain.from_yaml(
        f"""
    intents:
    - greet
    actions:
    - {UTTER_GREET_ACTION}
    - {RULE_SNIPPET_ACTION_NAME}
    - some-action
    slots:
      requested_slot:
        type: text
    forms:
    - {form_name}
"""
    )

    policy = RulePolicy()
    policy.train([GREET_RULE], domain)

    form_conversation = DialogueStateTracker.from_events(
        "in a form",
        evts=[
            Form(form_name),
            SlotSet(REQUESTED_SLOT, "some value"),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered("haha", {"name": "greet"}),
        ],
        slots=domain.slots,
    )

    action_probabilities = policy.predict_action_probabilities(
        form_conversation, domain
    )
    assert max(action_probabilities) == 1

    index_of_predicted_action = action_probabilities.index(max(action_probabilities))
    prediction_action_name = domain.action_names[index_of_predicted_action]
    assert prediction_action_name == form_name


async def test_form_unhappy_path(tmp_path: Path):
    form_name = "some_form"

    domain = Domain.from_yaml(
        f"""
        intents:
        - greet
        actions:
        - {UTTER_GREET_ACTION}
        - {RULE_SNIPPET_ACTION_NAME}
        - some-action
        slots:
          requested_slot:
            type: text
        forms:
        - {form_name}
    """
    )

    policy = RulePolicy()
    policy.train([GREET_RULE], domain)

    unhappy_form_conversation = DialogueStateTracker.from_events(
        "in a form",
        evts=[
            Form(form_name),
            SlotSet(REQUESTED_SLOT, "some value"),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered("haha", {"name": "greet"}),
            Form(form_name),
            ActionExecutionRejected(form_name),
        ],
        slots=domain.slots,
    )

    action_probabilities = policy.predict_action_probabilities(
        unhappy_form_conversation, domain
    )
    assert max(action_probabilities) == 1

    index_of_predicted_action = action_probabilities.index(max(action_probabilities))
    prediction_action_name = domain.action_names[index_of_predicted_action]
    assert prediction_action_name == UTTER_GREET_ACTION

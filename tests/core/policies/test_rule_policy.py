from rasa.core.actions.action import ACTION_LISTEN_NAME
from rasa.core.constants import REQUESTED_SLOT
from rasa.core.domain import Domain
from rasa.core.events import ActionExecuted, UserUttered, Form, SlotSet
from rasa.core.policies.rule_policy import RulePolicy
from rasa.core.trackers import DialogueStateTracker


SNIPPET_ACTION_NAME = "..."
UTTER_GREET_ACTION = "utter_greet"
GREET_RULE = DialogueStateTracker.from_events(
    "bla",
    evts=[
        ActionExecuted(SNIPPET_ACTION_NAME),
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
- {SNIPPET_ACTION_NAME}
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


# def test_predict_form_action_if_in_form():
#     form_name = "some_form"
#     form_rule = DialogueStateTracker.from_events(
#         "lala",
#         evts=[
#             ActionExecuted(form_name),
#             Form(form_name),
#             ActionExecuted("..."),
#             ActionExecuted(ACTION_LISTEN_NAME),
#             UserUttered("haha", {"name": "greet"}),
#             ActionExecuted(form_name),
#         ],
#     )
#
#     domain = Domain.from_yaml(
#         f"""
#     intents:
#     - greet
#     actions:
#     - {UTTER_GREET_ACTION}
#     - {SNIPPET_ACTION_NAME}
#
#     forms:
#     - {form_name}
# """
#     )
#
#     policy = RulePolicy()
#     policy.train([form_rule], domain)
#
#     form_conversation = DialogueStateTracker.from_events(
#         "in a form",
#         evts=[
#             # ActionExecuted(form_name),
#             Form(form_name),
#             # SlotSet(REQUESTED_SLOT, "some value"),
#             ActionExecuted(ACTION_LISTEN_NAME),
#             UserUttered("haha", {"name": "greet"}),
#         ],
#     )
#
#     action_probabilities = policy.predict_action_probabilities(
#         form_conversation, domain
#     )
#     assert max(action_probabilities) == 1
#
#     index_of_predicted_action = action_probabilities.index(max(action_probabilities))
#     prediction_action_name = domain.action_names[index_of_predicted_action]
#     assert prediction_action_name == form_name

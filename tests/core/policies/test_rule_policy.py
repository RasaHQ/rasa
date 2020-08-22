from typing import List, Text

import pytest
from rasa.constants import DEFAULT_NLU_FALLBACK_INTENT_NAME

from rasa.core import training
from rasa.core.actions.action import (
    ACTION_LISTEN_NAME,
    ACTION_DEFAULT_FALLBACK_NAME,
    ActionDefaultFallback,
    ACTION_RESTART_NAME,
    ACTION_BACK_NAME,
    ACTION_SESSION_START_NAME,
    RULE_SNIPPET_ACTION_NAME,
)
from rasa.core.channels import CollectingOutputChannel
from rasa.core.constants import (
    REQUESTED_SLOT,
    USER_INTENT_RESTART,
    USER_INTENT_BACK,
    USER_INTENT_SESSION_START,
)
from rasa.core.domain import Domain
from rasa.core.events import (
    ActionExecuted,
    UserUttered,
    ActiveLoop,
    SlotSet,
    ActionExecutionRejected,
    FormValidation,
)
from rasa.core.interpreter import RegexInterpreter
from rasa.core.nlg import TemplatedNaturalLanguageGenerator
from rasa.core.policies.rule_policy import RulePolicy
from rasa.core.trackers import DialogueStateTracker
from rasa.core.training.generator import TrackerWithCachedStates

UTTER_GREET_ACTION = "utter_greet"
GREET_INTENT_NAME = "greet"
GREET_RULE = DialogueStateTracker.from_events(
    "bla",
    evts=[
        ActionExecuted(RULE_SNIPPET_ACTION_NAME),
        ActionExecuted(ACTION_LISTEN_NAME),
        # Greet is a FAQ here and gets triggered in any context
        UserUttered("haha", {"name": GREET_INTENT_NAME}),
        ActionExecuted(UTTER_GREET_ACTION),
        ActionExecuted(ACTION_LISTEN_NAME),
    ],
)
GREET_RULE.is_rule_tracker = True


def _form_submit_rule(
    domain: Domain, submit_action_name: Text, form_name: Text
) -> DialogueStateTracker:
    return TrackerWithCachedStates.from_events(
        "bla",
        domain=domain,
        slots=domain.slots,
        evts=[
            ActiveLoop(form_name),
            # Any events in between
            ActionExecuted(RULE_SNIPPET_ACTION_NAME),
            # Form runs and deactivates itself
            ActionExecuted(form_name),
            ActiveLoop(None),
            SlotSet(REQUESTED_SLOT, None),
            ActionExecuted(submit_action_name),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
        is_rule_tracker=True,
    )


def _form_activation_rule(
    domain: Domain, form_name: Text, activation_intent_name: Text
) -> DialogueStateTracker:
    return TrackerWithCachedStates.from_events(
        "bla",
        domain=domain,
        slots=domain.slots,
        evts=[
            ActionExecuted(RULE_SNIPPET_ACTION_NAME),
            # The intent `other_intent` activates the form
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered("haha", {"name": activation_intent_name}),
            ActionExecuted(form_name),
            ActiveLoop(form_name),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
        is_rule_tracker=True,
    )


def test_rule_policy_has_max_history_none():
    policy = RulePolicy()
    assert policy.featurizer.max_history is None


def test_faq_rule():
    domain = Domain.from_yaml(
        f"""
intents:
- {GREET_INTENT_NAME}
actions:
- {UTTER_GREET_ACTION}
    """
    )

    policy = RulePolicy()
    policy.train([GREET_RULE], domain, RegexInterpreter())
    # remove first ... action and utter_greet and last action_listen from greet rule
    new_conversation = DialogueStateTracker.from_events(
        "simple greet",
        evts=[
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered("haha", {"name": GREET_INTENT_NAME}),
        ],
    )
    action_probabilities = policy.predict_action_probabilities(new_conversation, domain)

    assert_predicted_action(action_probabilities, domain, UTTER_GREET_ACTION)


def assert_predicted_action(
    action_probabilities: List[float],
    domain: Domain,
    expected_action_name: Text,
    confidence: float = 1.0,
) -> None:
    assert max(action_probabilities) == confidence
    index_of_predicted_action = action_probabilities.index(max(action_probabilities))
    prediction_action_name = domain.action_names[index_of_predicted_action]
    assert prediction_action_name == expected_action_name


async def test_predict_form_action_if_in_form():
    form_name = "some_form"

    domain = Domain.from_yaml(
        f"""
    intents:
    - {GREET_INTENT_NAME}
    actions:
    - {UTTER_GREET_ACTION}
    - some-action
    slots:
      {REQUESTED_SLOT}:
        type: unfeaturized
    forms:
    - {form_name}
"""
    )

    policy = RulePolicy()
    policy.train([GREET_RULE], domain, RegexInterpreter())

    form_conversation = DialogueStateTracker.from_events(
        "in a form",
        evts=[
            # We are in an activate form
            ActionExecuted(form_name),
            ActiveLoop(form_name),
            SlotSet(REQUESTED_SLOT, "some value"),
            ActionExecuted(ACTION_LISTEN_NAME),
            # User sends message as response to a requested slot
            UserUttered("haha", {"name": GREET_INTENT_NAME}),
        ],
        slots=domain.slots,
    )

    # RulePolicy triggers form again
    action_probabilities = policy.predict_action_probabilities(
        form_conversation, domain
    )
    assert_predicted_action(action_probabilities, domain, form_name)


async def test_predict_form_action_if_multiple_turns():
    form_name = "some_form"
    other_intent = "bye"
    domain = Domain.from_yaml(
        f"""
    intents:
    - {GREET_INTENT_NAME}
    - {other_intent}
    actions:
    - {UTTER_GREET_ACTION}
    - some-action
    slots:
      {REQUESTED_SLOT}:
        type: unfeaturized
    forms:
    - {form_name}
"""
    )

    policy = RulePolicy()
    policy.train([GREET_RULE], domain, RegexInterpreter())

    form_conversation = DialogueStateTracker.from_events(
        "in a form",
        evts=[
            # We are in an active form
            ActionExecuted(form_name),
            ActiveLoop(form_name),
            SlotSet(REQUESTED_SLOT, "some value"),
            # User responds to slot request
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered("haha", {"name": GREET_INTENT_NAME}),
            # Form validates input and requests another slot
            ActionExecuted(form_name),
            SlotSet(REQUESTED_SLOT, "some other"),
            # User responds to 2nd slot request
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered("haha", {"name": other_intent}),
        ],
        slots=domain.slots,
    )

    # RulePolicy triggers form again
    action_probabilities = policy.predict_action_probabilities(
        form_conversation, domain
    )
    assert_predicted_action(action_probabilities, domain, form_name)


async def test_predict_action_listen_after_form():
    form_name = "some_form"

    domain = Domain.from_yaml(
        f"""
        intents:
        - {GREET_INTENT_NAME}
        actions:
        - {UTTER_GREET_ACTION}
        - some-action
        slots:
          {REQUESTED_SLOT}:
            type: unfeaturized
        forms:
        - {form_name}
    """
    )

    policy = RulePolicy()
    policy.train([GREET_RULE], domain, RegexInterpreter())

    form_conversation = DialogueStateTracker.from_events(
        "in a form",
        evts=[
            # We are in an activate form
            ActionExecuted(form_name),
            ActiveLoop(form_name),
            SlotSet(REQUESTED_SLOT, "some value"),
            ActionExecuted(ACTION_LISTEN_NAME),
            # User sends message as response to a requested slot
            UserUttered("haha", {"name": GREET_INTENT_NAME}),
            # Form is running again
            ActionExecuted(form_name),
        ],
        slots=domain.slots,
    )

    # RulePolicy predicts action listen
    action_probabilities = policy.predict_action_probabilities(
        form_conversation, domain
    )
    assert_predicted_action(action_probabilities, domain, ACTION_LISTEN_NAME)


async def test_dont_predict_form_if_already_finished():
    form_name = "some_form"

    domain = Domain.from_yaml(
        f"""
    intents:
    - {GREET_INTENT_NAME}
    actions:
    - {UTTER_GREET_ACTION}
    - some-action
    slots:
      {REQUESTED_SLOT}:
        type: unfeaturized
    forms:
    - {form_name}
"""
    )

    policy = RulePolicy()
    policy.train([GREET_RULE], domain, RegexInterpreter())

    form_conversation = DialogueStateTracker.from_events(
        "in a form",
        evts=[
            # We are in an activate form
            ActionExecuted(form_name),
            ActiveLoop(form_name),
            SlotSet(REQUESTED_SLOT, "some value"),
            ActionExecuted(ACTION_LISTEN_NAME),
            # User sends message as response to a requested slot
            UserUttered("haha", {"name": GREET_INTENT_NAME}),
            # Form is happy and deactivates itself
            ActionExecuted(form_name),
            ActiveLoop(None),
            SlotSet(REQUESTED_SLOT, None),
            # User sends another message. Form is already done. Shouldn't get triggered
            # again
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered("haha", {"name": GREET_INTENT_NAME}),
        ],
        slots=domain.slots,
    )

    # RulePolicy triggers form again
    action_probabilities = policy.predict_action_probabilities(
        form_conversation, domain
    )
    assert_predicted_action(action_probabilities, domain, UTTER_GREET_ACTION)


async def test_form_unhappy_path():
    form_name = "some_form"

    domain = Domain.from_yaml(
        f"""
        intents:
        - {GREET_INTENT_NAME}
        actions:
        - {UTTER_GREET_ACTION}
        - some-action
        slots:
          {REQUESTED_SLOT}:
            type: unfeaturized
        forms:
        - {form_name}
    """
    )

    policy = RulePolicy()
    policy.train([GREET_RULE], domain, RegexInterpreter())

    unhappy_form_conversation = DialogueStateTracker.from_events(
        "in a form",
        evts=[
            # We are in an active form
            ActionExecuted(form_name),
            ActiveLoop(form_name),
            SlotSet(REQUESTED_SLOT, "some value"),
            # User responds to slot request
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered("haha", {"name": GREET_INTENT_NAME}),
            # Form isn't happy with the answer and rejects execution
            ActionExecutionRejected(form_name),
        ],
        slots=domain.slots,
    )

    # RulePolicy doesn't trigger form but FAQ
    action_probabilities = policy.predict_action_probabilities(
        unhappy_form_conversation, domain
    )

    assert_predicted_action(action_probabilities, domain, UTTER_GREET_ACTION)


async def test_form_unhappy_path_from_general_rule():
    form_name = "some_form"

    domain = Domain.from_yaml(
        f"""
        intents:
        - {GREET_INTENT_NAME}
        actions:
        - {UTTER_GREET_ACTION}
        - some-action
        slots:
          {REQUESTED_SLOT}:
            type: unfeaturized
        forms:
        - {form_name}
    """
    )

    policy = RulePolicy()
    # RulePolicy should memorize that unhappy_rule overrides GREET_RULE
    policy.train([GREET_RULE], domain, RegexInterpreter())

    # Check that RulePolicy predicts action to handle unhappy path
    conversation_events = [
        ActionExecuted(form_name),
        ActiveLoop(form_name),
        SlotSet(REQUESTED_SLOT, "some value"),
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered("haha", {"name": GREET_INTENT_NAME}),
        ActionExecutionRejected(form_name),
    ]

    action_probabilities = policy.predict_action_probabilities(
        DialogueStateTracker.from_events(
            "casd", evts=conversation_events, slots=domain.slots
        ),
        domain,
    )
    # check that general rule action is predicted
    assert_predicted_action(action_probabilities, domain, UTTER_GREET_ACTION)

    # Check that RulePolicy triggers form again after handling unhappy path
    conversation_events.append(ActionExecuted(UTTER_GREET_ACTION))
    action_probabilities = policy.predict_action_probabilities(
        DialogueStateTracker.from_events(
            "casd", evts=conversation_events, slots=domain.slots
        ),
        domain,
    )
    # check that action_listen from general rule is overwritten by form action
    assert_predicted_action(action_probabilities, domain, form_name)


async def test_form_unhappy_path_from_in_form_rule():
    form_name = "some_form"
    handle_rejection_action_name = "utter_handle_rejection"

    domain = Domain.from_yaml(
        f"""
        intents:
        - {GREET_INTENT_NAME}
        actions:
        - {UTTER_GREET_ACTION}
        - {handle_rejection_action_name}
        - some-action
        slots:
          {REQUESTED_SLOT}:
            type: unfeaturized
        forms:
        - {form_name}
    """
    )

    unhappy_rule = TrackerWithCachedStates.from_events(
        "bla",
        domain=domain,
        slots=domain.slots,
        evts=[
            # We are in an active form
            ActiveLoop(form_name),
            SlotSet(REQUESTED_SLOT, "bla"),
            ActionExecuted(RULE_SNIPPET_ACTION_NAME),
            ActionExecuted(ACTION_LISTEN_NAME),
            # When a user says "hi", and the form is unhappy,
            # we want to run a specific action
            UserUttered("haha", {"name": GREET_INTENT_NAME}),
            ActionExecuted(handle_rejection_action_name),
            ActionExecuted(form_name),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
        is_rule_tracker=True,
    )

    policy = RulePolicy()
    # RulePolicy should memorize that unhappy_rule overrides GREET_RULE
    policy.train([GREET_RULE, unhappy_rule], domain, RegexInterpreter())

    # Check that RulePolicy predicts action to handle unhappy path
    conversation_events = [
        ActionExecuted(form_name),
        ActiveLoop(form_name),
        SlotSet(REQUESTED_SLOT, "some value"),
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered("haha", {"name": GREET_INTENT_NAME}),
        ActionExecutionRejected(form_name),
    ]

    action_probabilities = policy.predict_action_probabilities(
        DialogueStateTracker.from_events(
            "casd", evts=conversation_events, slots=domain.slots
        ),
        domain,
    )
    assert_predicted_action(action_probabilities, domain, handle_rejection_action_name)

    # Check that RulePolicy triggers form again after handling unhappy path
    conversation_events.append(ActionExecuted(handle_rejection_action_name))
    action_probabilities = policy.predict_action_probabilities(
        DialogueStateTracker.from_events(
            "casd", evts=conversation_events, slots=domain.slots
        ),
        domain,
    )
    assert_predicted_action(action_probabilities, domain, form_name)


async def test_form_unhappy_path_from_story():
    form_name = "some_form"
    handle_rejection_action_name = "utter_handle_rejection"

    domain = Domain.from_yaml(
        f"""
        intents:
        - {GREET_INTENT_NAME}
        actions:
        - {UTTER_GREET_ACTION}
        - {handle_rejection_action_name}
        - some-action
        slots:
          {REQUESTED_SLOT}:
            type: unfeaturized
        forms:
        - {form_name}
    """
    )

    unhappy_story = TrackerWithCachedStates.from_events(
        "bla",
        domain=domain,
        slots=domain.slots,
        evts=[
            # We are in an active form
            ActionExecuted(form_name),
            ActiveLoop(form_name),
            UserUttered("haha", {"name": GREET_INTENT_NAME}),
            ActionExecuted(UTTER_GREET_ACTION),
            # After our bot says "hi", we want to run a specific action
            ActionExecuted(handle_rejection_action_name),
            ActionExecuted(form_name),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
    )

    policy = RulePolicy()
    policy.train([GREET_RULE, unhappy_story], domain, RegexInterpreter())

    # Check that RulePolicy predicts action to handle unhappy path
    conversation_events = [
        ActionExecuted(form_name),
        ActiveLoop(form_name),
        SlotSet(REQUESTED_SLOT, "some value"),
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered("haha", {"name": GREET_INTENT_NAME}),
        ActionExecutionRejected(form_name),
    ]

    action_probabilities = policy.predict_action_probabilities(
        DialogueStateTracker.from_events(
            "casd", evts=conversation_events, slots=domain.slots
        ),
        domain,
    )
    assert_predicted_action(action_probabilities, domain, UTTER_GREET_ACTION)

    # Check that RulePolicy doesn't trigger form or action_listen
    # after handling unhappy path
    conversation_events.append(ActionExecuted(handle_rejection_action_name))
    action_probabilities = policy.predict_action_probabilities(
        DialogueStateTracker.from_events(
            "casd", evts=conversation_events, slots=domain.slots
        ),
        domain,
    )
    assert max(action_probabilities) == policy._core_fallback_threshold


async def test_form_unhappy_path_no_validation_from_rule():
    form_name = "some_form"
    handle_rejection_action_name = "utter_handle_rejection"

    domain = Domain.from_yaml(
        f"""
        intents:
        - {GREET_INTENT_NAME}
        actions:
        - {UTTER_GREET_ACTION}
        - {handle_rejection_action_name}
        - some-action
        slots:
          {REQUESTED_SLOT}:
            type: unfeaturized
        forms:
        - {form_name}
    """
    )

    unhappy_rule = TrackerWithCachedStates.from_events(
        "bla",
        domain=domain,
        slots=domain.slots,
        evts=[
            # We are in an active form
            ActiveLoop(form_name),
            SlotSet(REQUESTED_SLOT, "bla"),
            ActionExecuted(RULE_SNIPPET_ACTION_NAME),
            ActionExecuted(ACTION_LISTEN_NAME),
            # When a user says "hi", and the form is unhappy,
            # we want to run a specific action
            UserUttered("haha", {"name": GREET_INTENT_NAME}),
            ActionExecuted(handle_rejection_action_name),
            # Next user utterance is an answer to the previous question
            # and shouldn't be validated by the form
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered("haha", {"name": GREET_INTENT_NAME}),
            ActionExecuted(form_name),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
        is_rule_tracker=True,
    )

    policy = RulePolicy()
    # RulePolicy should memorize that unhappy_rule overrides GREET_RULE
    policy.train([GREET_RULE, unhappy_rule], domain, RegexInterpreter())

    # Check that RulePolicy predicts action to handle unhappy path
    conversation_events = [
        ActionExecuted(form_name),
        ActiveLoop(form_name),
        SlotSet(REQUESTED_SLOT, "some value"),
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered("haha", {"name": GREET_INTENT_NAME}),
        ActionExecutionRejected(form_name),
    ]

    action_probabilities = policy.predict_action_probabilities(
        DialogueStateTracker.from_events(
            "casd", evts=conversation_events, slots=domain.slots
        ),
        domain,
    )
    assert_predicted_action(action_probabilities, domain, handle_rejection_action_name)

    # Check that RulePolicy predicts action_listen
    conversation_events.append(ActionExecuted(handle_rejection_action_name))
    action_probabilities = policy.predict_action_probabilities(
        DialogueStateTracker.from_events(
            "casd", evts=conversation_events, slots=domain.slots
        ),
        domain,
    )
    assert_predicted_action(action_probabilities, domain, ACTION_LISTEN_NAME)

    # Check that RulePolicy triggers form again after handling unhappy path
    conversation_events.append(ActionExecuted(ACTION_LISTEN_NAME))
    tracker = DialogueStateTracker.from_events(
        "casd", evts=conversation_events, slots=domain.slots
    )
    action_probabilities = policy.predict_action_probabilities(tracker, domain)
    assert_predicted_action(action_probabilities, domain, form_name)
    # check that RulePolicy added FormValidation False event based on the training rule
    assert tracker.events[-1] == FormValidation(False)


async def test_form_unhappy_path_no_validation_from_story():
    form_name = "some_form"
    handle_rejection_action_name = "utter_handle_rejection"

    domain = Domain.from_yaml(
        f"""
        intents:
        - {GREET_INTENT_NAME}
        actions:
        - {UTTER_GREET_ACTION}
        - {handle_rejection_action_name}
        - some-action
        slots:
          {REQUESTED_SLOT}:
            type: unfeaturized
        forms:
        - {form_name}
    """
    )

    unhappy_story = TrackerWithCachedStates.from_events(
        "bla",
        domain=domain,
        slots=domain.slots,
        evts=[
            # We are in an active form
            ActionExecuted(form_name),
            ActiveLoop(form_name),
            # When a user says "hi", and the form is unhappy,
            # we want to run a specific action
            UserUttered("haha", {"name": GREET_INTENT_NAME}),
            ActionExecuted(handle_rejection_action_name),
            ActionExecuted(ACTION_LISTEN_NAME),
            # Next user utterance is an answer to the previous question
            # and shouldn't be validated by the form
            UserUttered("haha", {"name": GREET_INTENT_NAME}),
            ActionExecuted(form_name),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
    )

    policy = RulePolicy()
    policy.train([unhappy_story], domain, RegexInterpreter())

    # Check that RulePolicy predicts no validation to handle unhappy path
    conversation_events = [
        ActionExecuted(form_name),
        ActiveLoop(form_name),
        SlotSet(REQUESTED_SLOT, "some value"),
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered("haha", {"name": GREET_INTENT_NAME}),
        ActionExecutionRejected(form_name),
        ActionExecuted(handle_rejection_action_name),
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered("haha", {"name": GREET_INTENT_NAME}),
    ]

    tracker = DialogueStateTracker.from_events(
        "casd", evts=conversation_events, slots=domain.slots
    )
    action_probabilities = policy.predict_action_probabilities(tracker, domain)
    # there is no rule for next action
    assert max(action_probabilities) == policy._core_fallback_threshold
    # check that RulePolicy added FormValidation False event based on the training story
    assert tracker.events[-1] == FormValidation(False)


async def test_form_unhappy_path_without_rule():
    form_name = "some_form"
    other_intent = "bye"
    domain = Domain.from_yaml(
        f"""
        intents:
        - {GREET_INTENT_NAME}
        - {other_intent}
        actions:
        - {UTTER_GREET_ACTION}
        - some-action
        slots:
          {REQUESTED_SLOT}:
            type: unfeaturized
        forms:
        - {form_name}
    """
    )

    policy = RulePolicy()
    policy.train([GREET_RULE], domain, RegexInterpreter())

    conversation_events = [
        ActionExecuted(form_name),
        ActiveLoop(form_name),
        SlotSet(REQUESTED_SLOT, "some value"),
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered("haha", {"name": other_intent}),
        ActiveLoop(form_name),
        ActionExecutionRejected(form_name),
    ]

    # Unhappy path is not handled. No rule matches. Let's hope ML fixes our problems 🤞
    action_probabilities = policy.predict_action_probabilities(
        DialogueStateTracker.from_events(
            "casd", evts=conversation_events, slots=domain.slots
        ),
        domain,
    )

    assert max(action_probabilities) == policy._core_fallback_threshold


async def test_form_activation_rule():
    form_name = "some_form"
    other_intent = "bye"
    domain = Domain.from_yaml(
        f"""
        intents:
        - {GREET_INTENT_NAME}
        - {other_intent}
        actions:
        - {UTTER_GREET_ACTION}
        - some-action
        slots:
          {REQUESTED_SLOT}:
            type: unfeaturized
        forms:
        - {form_name}
    """
    )

    form_activation_rule = _form_activation_rule(domain, form_name, other_intent)
    policy = RulePolicy()
    policy.train([GREET_RULE, form_activation_rule], domain, RegexInterpreter())

    conversation_events = [
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered("haha", {"name": other_intent}),
    ]

    # RulePolicy correctly predicts the form action
    action_probabilities = policy.predict_action_probabilities(
        DialogueStateTracker.from_events(
            "casd", evts=conversation_events, slots=domain.slots
        ),
        domain,
    )

    assert_predicted_action(action_probabilities, domain, form_name)


async def test_failing_form_activation_due_to_no_rule():
    form_name = "some_form"
    other_intent = "bye"
    domain = Domain.from_yaml(
        f"""
        intents:
        - {GREET_INTENT_NAME}
        - {other_intent}
        actions:
        - {UTTER_GREET_ACTION}
        - some-action
        slots:
          {REQUESTED_SLOT}:
            type: unfeaturized
        forms:
        - {form_name}
    """
    )

    policy = RulePolicy()
    policy.train([GREET_RULE], domain, RegexInterpreter())

    conversation_events = [
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered("haha", {"name": other_intent}),
    ]

    # RulePolicy has no matching rule since no rule for form activation is given
    action_probabilities = policy.predict_action_probabilities(
        DialogueStateTracker.from_events(
            "casd", evts=conversation_events, slots=domain.slots
        ),
        domain,
    )

    assert max(action_probabilities) == policy._core_fallback_threshold


def test_form_submit_rule():
    form_name = "some_form"
    submit_action_name = "utter_submit"
    domain = Domain.from_yaml(
        f"""
        intents:
        - {GREET_INTENT_NAME}
        actions:
        - {UTTER_GREET_ACTION}
        - some-action
        - {submit_action_name}
        slots:
          {REQUESTED_SLOT}:
            type: unfeaturized
        forms:
        - {form_name}
    """
    )

    form_submit_rule = _form_submit_rule(domain, submit_action_name, form_name)

    policy = RulePolicy()
    policy.train([GREET_RULE, form_submit_rule], domain, RegexInterpreter())

    form_conversation = DialogueStateTracker.from_events(
        "in a form",
        evts=[
            # Form was activated
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered("haha", {"name": GREET_INTENT_NAME}),
            ActionExecuted(form_name),
            ActiveLoop(form_name),
            SlotSet(REQUESTED_SLOT, "some value"),
            ActionExecuted(ACTION_LISTEN_NAME),
            # User responds and fills requested slot
            UserUttered("haha", {"name": GREET_INTENT_NAME}),
            ActionExecuted(form_name),
            # Form get's deactivated
            ActiveLoop(None),
            SlotSet(REQUESTED_SLOT, None),
        ],
        slots=domain.slots,
    )

    # RulePolicy predicts action which handles submit
    action_probabilities = policy.predict_action_probabilities(
        form_conversation, domain
    )
    assert_predicted_action(action_probabilities, domain, submit_action_name)


def test_immediate_submit():
    form_name = "some_form"
    submit_action_name = "utter_submit"
    entity = "some_entity"
    slot = "some_slot"
    domain = Domain.from_yaml(
        f"""
        intents:
        - {GREET_INTENT_NAME}
        actions:
        - {UTTER_GREET_ACTION}
        - some-action
        - {submit_action_name}
        slots:
          {REQUESTED_SLOT}:
            type: unfeaturized
          {slot}:
            type: unfeaturized
        forms:
        - {form_name}
        entities:
        - {entity}
    """
    )

    form_activation_rule = _form_activation_rule(domain, form_name, GREET_INTENT_NAME)
    form_submit_rule = _form_submit_rule(domain, submit_action_name, form_name)

    policy = RulePolicy()
    policy.train(
        [GREET_RULE, form_activation_rule, form_submit_rule], domain, RegexInterpreter()
    )

    form_conversation = DialogueStateTracker.from_events(
        "in a form",
        evts=[
            # Form was activated
            ActionExecuted(ACTION_LISTEN_NAME),
            # The same intent which activates the form also deactivates it
            UserUttered(
                "haha",
                {"name": GREET_INTENT_NAME},
                entities=[{"entity": entity, "value": "Bruce Wayne"}],
            ),
            SlotSet(slot, "Bruce"),
            ActionExecuted(form_name),
            SlotSet("bla", "bla"),
            ActiveLoop(None),
            SlotSet(REQUESTED_SLOT, None),
        ],
        slots=domain.slots,
    )

    # RulePolicy predicts action which handles submit
    action_probabilities = policy.predict_action_probabilities(
        form_conversation, domain
    )
    assert_predicted_action(action_probabilities, domain, submit_action_name)


@pytest.fixture(scope="session")
def trained_rule_policy_domain() -> Domain:
    return Domain.load("examples/rules/domain.yml")


@pytest.fixture(scope="session")
async def trained_rule_policy(trained_rule_policy_domain: Domain) -> RulePolicy:
    trackers = await training.load_data(
        "examples/rules/data/rules.yml", trained_rule_policy_domain
    )

    rule_policy = RulePolicy()
    rule_policy.train(trackers, trained_rule_policy_domain, RegexInterpreter())

    return rule_policy


async def test_rule_policy_slot_filling_from_text(
    trained_rule_policy: RulePolicy, trained_rule_policy_domain: Domain
):
    form_conversation = DialogueStateTracker.from_events(
        "in a form",
        evts=[
            ActionExecuted(ACTION_LISTEN_NAME),
            # User responds and fills requested slot
            UserUttered("/activate_q_form", {"name": "activate_q_form"}),
            ActionExecuted("loop_q_form"),
            ActiveLoop("loop_q_form"),
            SlotSet(REQUESTED_SLOT, "some_slot"),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered("/bla", {"name": GREET_INTENT_NAME}),
            ActionExecuted("loop_q_form"),
            SlotSet("some_slot", "/bla"),
            ActiveLoop(None),
            SlotSet(REQUESTED_SLOT, None),
        ],
        slots=trained_rule_policy_domain.slots,
    )

    # RulePolicy predicts action which handles submit
    action_probabilities = trained_rule_policy.predict_action_probabilities(
        form_conversation, trained_rule_policy_domain
    )
    assert_predicted_action(
        action_probabilities, trained_rule_policy_domain, "utter_stop"
    )


async def test_one_stage_fallback_rule():
    domain = Domain.from_yaml(
        f"""
        intents:
        - {GREET_INTENT_NAME}
        - {DEFAULT_NLU_FALLBACK_INTENT_NAME}
        actions:
        - {UTTER_GREET_ACTION}
    """
    )

    fallback_recover_rule = TrackerWithCachedStates.from_events(
        "bla",
        domain=domain,
        slots=domain.slots,
        evts=[
            ActionExecuted(RULE_SNIPPET_ACTION_NAME),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered("haha", {"name": DEFAULT_NLU_FALLBACK_INTENT_NAME}),
            ActionExecuted(ACTION_DEFAULT_FALLBACK_NAME),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
        is_rule_tracker=True,
    )

    greet_rule_which_only_applies_at_start = TrackerWithCachedStates.from_events(
        "bla",
        domain=domain,
        evts=[
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered("haha", {"name": GREET_INTENT_NAME}),
            ActionExecuted(UTTER_GREET_ACTION),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
        is_rule_tracker=True,
    )
    policy = RulePolicy()
    policy.train(
        [greet_rule_which_only_applies_at_start, fallback_recover_rule],
        domain,
        RegexInterpreter(),
    )

    # RulePolicy predicts fallback action
    conversation_events = [
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered("dasdakl;fkasd", {"name": DEFAULT_NLU_FALLBACK_INTENT_NAME}),
    ]
    tracker = DialogueStateTracker.from_events(
        "casd", evts=conversation_events, slots=domain.slots
    )
    action_probabilities = policy.predict_action_probabilities(tracker, domain)
    assert_predicted_action(action_probabilities, domain, ACTION_DEFAULT_FALLBACK_NAME)

    # Fallback action reverts fallback events, next action is `ACTION_LISTEN`
    conversation_events += await ActionDefaultFallback().run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.templates),
        tracker,
        domain,
    )

    # Rasa is back on track when user rephrased intent
    conversation_events += [
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered("haha", {"name": GREET_INTENT_NAME}),
    ]
    tracker = DialogueStateTracker.from_events(
        "casd", evts=conversation_events, slots=domain.slots
    )

    action_probabilities = policy.predict_action_probabilities(tracker, domain)
    assert_predicted_action(action_probabilities, domain, UTTER_GREET_ACTION)


@pytest.mark.parametrize(
    "intent_name, expected_action_name",
    [
        (USER_INTENT_RESTART, ACTION_RESTART_NAME),
        (USER_INTENT_BACK, ACTION_BACK_NAME),
        (USER_INTENT_SESSION_START, ACTION_SESSION_START_NAME),
    ],
)
def test_default_actions(intent_name: Text, expected_action_name: Text):
    domain = Domain.from_yaml(
        f"""
intents:
- {GREET_INTENT_NAME}
actions:
- {UTTER_GREET_ACTION}
    """
    )
    policy = RulePolicy()
    policy.train([GREET_RULE], domain, RegexInterpreter())
    new_conversation = DialogueStateTracker.from_events(
        "bla2",
        evts=[
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered("haha", {"name": GREET_INTENT_NAME}),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered("haha", {"name": intent_name}),
        ],
    )
    action_probabilities = policy.predict_action_probabilities(new_conversation, domain)

    assert_predicted_action(action_probabilities, domain, expected_action_name)


@pytest.mark.parametrize(
    "rule_policy, expected_confidence, expected_prediction",
    [
        (RulePolicy(), 0.3, ACTION_DEFAULT_FALLBACK_NAME),
        (
            RulePolicy(
                core_fallback_threshold=0.1,
                core_fallback_action_name="my_core_fallback",
            ),
            0.1,
            "my_core_fallback",
        ),
    ],
)
def test_predict_core_fallback(
    rule_policy: RulePolicy, expected_confidence: float, expected_prediction: Text
):
    other_intent = "other"
    domain = Domain.from_yaml(
        f"""
    intents:
    - {GREET_INTENT_NAME}
    - {other_intent}
    actions:
    - {UTTER_GREET_ACTION}
    - my_core_fallback
        """
    )
    rule_policy.train([GREET_RULE], domain, RegexInterpreter())

    new_conversation = DialogueStateTracker.from_events(
        "bla2",
        evts=[
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered("haha", {"name": other_intent}),
        ],
    )

    action_probabilities = rule_policy.predict_action_probabilities(
        new_conversation, domain
    )

    assert_predicted_action(
        action_probabilities, domain, expected_prediction, expected_confidence
    )


def test_predict_nothing_if_fallback_disabled():
    other_intent = "other"
    domain = Domain.from_yaml(
        f"""
    intents:
    - {GREET_INTENT_NAME}
    - {other_intent}
    actions:
    - {UTTER_GREET_ACTION}
        """
    )
    policy = RulePolicy(enable_fallback_prediction=False)
    policy.train([GREET_RULE], domain, RegexInterpreter())
    new_conversation = DialogueStateTracker.from_events(
        "bla2",
        evts=[
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered("haha", {"name": other_intent}),
        ],
    )
    action_probabilities = policy.predict_action_probabilities(new_conversation, domain)

    assert max(action_probabilities) == 0

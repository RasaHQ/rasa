from pathlib import Path
from typing import List, Text

import pytest

from rasa.core.policies.policy import PolicyPrediction
from rasa.shared.constants import DEFAULT_NLU_FALLBACK_INTENT_NAME

from rasa.core import training
from rasa.core.actions.action import ActionDefaultFallback
from rasa.core.channels import CollectingOutputChannel
from rasa.shared.core.constants import (
    USER_INTENT_RESTART,
    USER_INTENT_BACK,
    USER_INTENT_SESSION_START,
    ACTION_LISTEN_NAME,
    ACTION_RESTART_NAME,
    ACTION_SESSION_START_NAME,
    ACTION_DEFAULT_FALLBACK_NAME,
    ACTION_BACK_NAME,
    RULE_SNIPPET_ACTION_NAME,
    REQUESTED_SLOT,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import (
    ActionExecuted,
    UserUttered,
    ActiveLoop,
    SlotSet,
    ActionExecutionRejected,
    LoopInterrupted,
)
from rasa.shared.nlu.interpreter import RegexInterpreter
from rasa.core.nlg import TemplatedNaturalLanguageGenerator
from rasa.core.policies.rule_policy import RulePolicy, InvalidRule
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.generator import TrackerWithCachedStates

UTTER_GREET_ACTION = "utter_greet"
GREET_INTENT_NAME = "greet"
GREET_RULE = DialogueStateTracker.from_events(
    "greet rule",
    evts=[
        ActionExecuted(RULE_SNIPPET_ACTION_NAME),
        ActionExecuted(ACTION_LISTEN_NAME),
        # Greet is a FAQ here and gets triggered in any context
        UserUttered(intent={"name": GREET_INTENT_NAME}),
        ActionExecuted(UTTER_GREET_ACTION),
        ActionExecuted(ACTION_LISTEN_NAME),
    ],
)
GREET_RULE.is_rule_tracker = True


def _form_submit_rule(
    domain: Domain, submit_action_name: Text, form_name: Text
) -> TrackerWithCachedStates:
    return TrackerWithCachedStates.from_events(
        "form submit rule",
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
) -> TrackerWithCachedStates:
    return TrackerWithCachedStates.from_events(
        "form activation rule",
        domain=domain,
        slots=domain.slots,
        evts=[
            ActionExecuted(RULE_SNIPPET_ACTION_NAME),
            # The intent `other_intent` activates the form
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(intent={"name": activation_intent_name}),
            ActionExecuted(form_name),
            ActiveLoop(form_name),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
        is_rule_tracker=True,
    )


def test_restrict_multiple_user_inputs_in_rules():
    domain = Domain.from_yaml(
        f"""
intents:
- {GREET_INTENT_NAME}
actions:
- {UTTER_GREET_ACTION}
    """
    )
    policy = RulePolicy()
    greet_events = [
        UserUttered(intent={"name": GREET_INTENT_NAME}),
        ActionExecuted(UTTER_GREET_ACTION),
        ActionExecuted(ACTION_LISTEN_NAME),
    ]

    forbidden_rule = DialogueStateTracker.from_events(
        "bla",
        evts=[
            ActionExecuted(RULE_SNIPPET_ACTION_NAME),
            ActionExecuted(ACTION_LISTEN_NAME),
        ]
        + greet_events * (policy.ALLOWED_NUMBER_OF_USER_INPUTS + 1),
    )
    forbidden_rule.is_rule_tracker = True
    with pytest.raises(InvalidRule):
        policy.train([forbidden_rule], domain, RegexInterpreter())


def test_incomplete_rules_due_to_slots():
    some_action = "some_action"
    some_slot = "some_slot"
    domain = Domain.from_yaml(
        f"""
intents:
- {GREET_INTENT_NAME}
actions:
- {some_action}
slots:
  {some_slot}:
    type: text
    """
    )
    policy = RulePolicy()
    complete_rule = TrackerWithCachedStates.from_events(
        "complete_rule",
        domain=domain,
        slots=domain.slots,
        evts=[
            ActionExecuted(RULE_SNIPPET_ACTION_NAME),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(intent={"name": GREET_INTENT_NAME}),
            ActionExecuted(some_action),
            SlotSet(some_slot, "bla"),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
        is_rule_tracker=True,
    )
    incomplete_rule = TrackerWithCachedStates.from_events(
        "incomplete_rule",
        domain=domain,
        slots=domain.slots,
        evts=[
            ActionExecuted(RULE_SNIPPET_ACTION_NAME),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(intent={"name": GREET_INTENT_NAME}),
            ActionExecuted(some_action),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
        is_rule_tracker=True,
    )

    with pytest.raises(InvalidRule) as execinfo:
        policy.train([complete_rule, incomplete_rule], domain, RegexInterpreter())
    assert all(
        name in execinfo.value.message
        for name in {some_action, incomplete_rule.sender_id}
    )

    fixed_incomplete_rule = TrackerWithCachedStates.from_events(
        "fixed_incomplete_rule",
        domain=domain,
        slots=domain.slots,
        evts=[
            ActionExecuted(RULE_SNIPPET_ACTION_NAME),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(intent={"name": GREET_INTENT_NAME}),
            ActionExecuted(some_action),
            ActionExecuted(RULE_SNIPPET_ACTION_NAME),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
        is_rule_tracker=True,
    )
    policy.train([complete_rule, fixed_incomplete_rule], domain, RegexInterpreter())


def test_no_incomplete_rules_due_to_slots_after_listen():
    some_action = "some_action"
    some_slot = "some_slot"
    domain = Domain.from_yaml(
        f"""
intents:
- {GREET_INTENT_NAME}
actions:
- {some_action}
entities:
- {some_slot}
slots:
  {some_slot}:
    type: text
    """
    )
    policy = RulePolicy()
    complete_rule = TrackerWithCachedStates.from_events(
        "complete_rule",
        domain=domain,
        slots=domain.slots,
        evts=[
            ActionExecuted(RULE_SNIPPET_ACTION_NAME),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(
                intent={"name": GREET_INTENT_NAME},
                entities=[{"entity": some_slot, "value": "bla"}],
            ),
            SlotSet(some_slot, "bla"),
            ActionExecuted(some_action),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
        is_rule_tracker=True,
    )
    potentially_incomplete_rule = TrackerWithCachedStates.from_events(
        "potentially_incomplete_rule",
        domain=domain,
        slots=domain.slots,
        evts=[
            ActionExecuted(RULE_SNIPPET_ACTION_NAME),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(intent={"name": GREET_INTENT_NAME}),
            ActionExecuted(some_action),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
        is_rule_tracker=True,
    )
    policy.train(
        [complete_rule, potentially_incomplete_rule], domain, RegexInterpreter()
    )


def test_incomplete_rules_due_to_loops():
    some_form = "some_form"
    domain = Domain.from_yaml(
        f"""
intents:
- {GREET_INTENT_NAME}
forms:
- {some_form}
    """
    )
    policy = RulePolicy()
    complete_rule = TrackerWithCachedStates.from_events(
        "complete_rule",
        domain=domain,
        slots=domain.slots,
        evts=[
            ActionExecuted(RULE_SNIPPET_ACTION_NAME),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(intent={"name": GREET_INTENT_NAME}),
            ActionExecuted(some_form),
            ActiveLoop(some_form),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
        is_rule_tracker=True,
    )
    incomplete_rule = TrackerWithCachedStates.from_events(
        "incomplete_rule",
        domain=domain,
        slots=domain.slots,
        evts=[
            ActionExecuted(RULE_SNIPPET_ACTION_NAME),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(intent={"name": GREET_INTENT_NAME}),
            ActionExecuted(some_form),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
        is_rule_tracker=True,
    )

    with pytest.raises(InvalidRule) as execinfo:
        policy.train([complete_rule, incomplete_rule], domain, RegexInterpreter())
    assert all(
        name in execinfo.value.message
        for name in {some_form, incomplete_rule.sender_id}
    )

    fixed_incomplete_rule = TrackerWithCachedStates.from_events(
        "fixed_incomplete_rule",
        domain=domain,
        slots=domain.slots,
        evts=[
            ActionExecuted(RULE_SNIPPET_ACTION_NAME),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(intent={"name": GREET_INTENT_NAME}),
            ActionExecuted(some_form),
            ActionExecuted(RULE_SNIPPET_ACTION_NAME),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
        is_rule_tracker=True,
    )
    policy.train([complete_rule, fixed_incomplete_rule], domain, RegexInterpreter())


def test_contradicting_rules():
    utter_anti_greet_action = "utter_anti_greet"
    domain = Domain.from_yaml(
        f"""
intents:
- {GREET_INTENT_NAME}
actions:
- {UTTER_GREET_ACTION}
- {utter_anti_greet_action}
    """
    )
    policy = RulePolicy()
    anti_greet_rule = TrackerWithCachedStates.from_events(
        "anti greet rule",
        domain=domain,
        slots=domain.slots,
        evts=[
            ActionExecuted(RULE_SNIPPET_ACTION_NAME),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(intent={"name": GREET_INTENT_NAME}),
            ActionExecuted(utter_anti_greet_action),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
    )
    anti_greet_rule.is_rule_tracker = True

    with pytest.raises(InvalidRule) as execinfo:
        policy.train([GREET_RULE, anti_greet_rule], domain, RegexInterpreter())
    assert all(
        name in execinfo.value.message
        for name in {
            UTTER_GREET_ACTION,
            GREET_RULE.sender_id,
            utter_anti_greet_action,
            anti_greet_rule.sender_id,
        }
    )


def test_contradicting_rules_and_stories():
    utter_anti_greet_action = "utter_anti_greet"
    domain = Domain.from_yaml(
        f"""
intents:
- {GREET_INTENT_NAME}
actions:
- {UTTER_GREET_ACTION}
- {utter_anti_greet_action}
    """
    )
    policy = RulePolicy()
    anti_greet_story = TrackerWithCachedStates.from_events(
        "anti greet story",
        domain=domain,
        slots=domain.slots,
        evts=[
            ActionExecuted(RULE_SNIPPET_ACTION_NAME),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(intent={"name": GREET_INTENT_NAME}),
            ActionExecuted(utter_anti_greet_action),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
    )

    with pytest.raises(InvalidRule) as execinfo:
        policy.train([GREET_RULE, anti_greet_story], domain, RegexInterpreter())

    assert all(
        name in execinfo.value.message
        for name in {utter_anti_greet_action, anti_greet_story.sender_id}
    )


def test_rule_policy_has_max_history_none():
    policy = RulePolicy()
    assert policy.featurizer.max_history is None


def test_all_policy_attributes_are_persisted(tmpdir: Path):
    priority = 5
    lookup = {"a": "b"}
    core_fallback_threshold = 0.451231
    core_fallback_action_name = "action_some_fallback"
    enable_fallback_prediction = False

    policy = RulePolicy(
        priority=priority,
        lookup=lookup,
        core_fallback_threshold=core_fallback_threshold,
        core_fallback_action_name=core_fallback_action_name,
        enable_fallback_prediction=enable_fallback_prediction,
    )
    policy.persist(tmpdir)

    persisted_policy = RulePolicy.load(tmpdir)
    assert persisted_policy.priority == priority
    assert persisted_policy.lookup == lookup
    assert persisted_policy._core_fallback_threshold == core_fallback_threshold
    assert persisted_policy._fallback_action_name == core_fallback_action_name
    assert persisted_policy._enable_fallback_prediction == enable_fallback_prediction


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
    prediction = policy.predict_action_probabilities(
        new_conversation, domain, RegexInterpreter()
    )

    assert_predicted_action(prediction, domain, UTTER_GREET_ACTION)


def assert_predicted_action(
    prediction: PolicyPrediction,
    domain: Domain,
    expected_action_name: Text,
    confidence: float = 1.0,
) -> None:
    assert prediction.max_confidence == confidence
    index_of_predicted_action = prediction.max_confidence_index
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
    prediction = policy.predict_action_probabilities(
        form_conversation, domain, RegexInterpreter()
    )
    assert_predicted_action(prediction, domain, form_name)


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
    prediction = policy.predict_action_probabilities(
        form_conversation, domain, RegexInterpreter()
    )
    assert_predicted_action(prediction, domain, form_name)


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
    prediction = policy.predict_action_probabilities(
        form_conversation, domain, RegexInterpreter()
    )
    assert_predicted_action(prediction, domain, ACTION_LISTEN_NAME)


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
    prediction = policy.predict_action_probabilities(
        form_conversation, domain, RegexInterpreter()
    )
    assert_predicted_action(prediction, domain, UTTER_GREET_ACTION)


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
    prediction = policy.predict_action_probabilities(
        unhappy_form_conversation, domain, RegexInterpreter()
    )

    assert_predicted_action(prediction, domain, UTTER_GREET_ACTION)


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

    prediction = policy.predict_action_probabilities(
        DialogueStateTracker.from_events(
            "casd", evts=conversation_events, slots=domain.slots
        ),
        domain,
        RegexInterpreter(),
    )
    # check that general rule action is predicted
    assert_predicted_action(prediction, domain, UTTER_GREET_ACTION)

    # Check that RulePolicy triggers form again after handling unhappy path
    conversation_events.append(ActionExecuted(UTTER_GREET_ACTION))
    prediction = policy.predict_action_probabilities(
        DialogueStateTracker.from_events(
            "casd", evts=conversation_events, slots=domain.slots
        ),
        domain,
        RegexInterpreter(),
    )
    # check that action_listen from general rule is overwritten by form action
    assert_predicted_action(prediction, domain, form_name)


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
            UserUttered(intent={"name": GREET_INTENT_NAME}),
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

    prediction = policy.predict_action_probabilities(
        DialogueStateTracker.from_events(
            "casd", evts=conversation_events, slots=domain.slots
        ),
        domain,
        RegexInterpreter(),
    )
    assert_predicted_action(prediction, domain, handle_rejection_action_name)

    # Check that RulePolicy triggers form again after handling unhappy path
    conversation_events.append(ActionExecuted(handle_rejection_action_name))
    prediction = policy.predict_action_probabilities(
        DialogueStateTracker.from_events(
            "casd", evts=conversation_events, slots=domain.slots
        ),
        domain,
        RegexInterpreter(),
    )
    assert_predicted_action(prediction, domain, form_name)


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
            ActionExecuted(ACTION_LISTEN_NAME),
            # in training stories there is either intent or text, never both
            UserUttered(intent={"name": GREET_INTENT_NAME}),
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

    prediction = policy.predict_action_probabilities(
        DialogueStateTracker.from_events(
            "casd", evts=conversation_events, slots=domain.slots
        ),
        domain,
        RegexInterpreter(),
    )
    assert_predicted_action(prediction, domain, UTTER_GREET_ACTION)

    # Check that RulePolicy doesn't trigger form or action_listen
    # after handling unhappy path
    conversation_events.append(ActionExecuted(handle_rejection_action_name))
    prediction = policy.predict_action_probabilities(
        DialogueStateTracker.from_events(
            "casd", evts=conversation_events, slots=domain.slots
        ),
        domain,
        RegexInterpreter(),
    )
    assert prediction.max_confidence == policy._core_fallback_threshold


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
            UserUttered(intent={"name": GREET_INTENT_NAME}),
            ActionExecuted(handle_rejection_action_name),
            # Next user utterance is an answer to the previous question
            # and shouldn't be validated by the form
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(intent={"name": GREET_INTENT_NAME}),
            ActionExecuted(form_name),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
        is_rule_tracker=True,
    )
    # unhappy rule is multi user turn rule, therefore remove restriction for policy
    policy = RulePolicy(restrict_rules=False)
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

    prediction = policy.predict_action_probabilities(
        DialogueStateTracker.from_events(
            "casd", evts=conversation_events, slots=domain.slots
        ),
        domain,
        RegexInterpreter(),
    )
    assert_predicted_action(prediction, domain, handle_rejection_action_name)

    # Check that RulePolicy predicts action_listen
    conversation_events.append(ActionExecuted(handle_rejection_action_name))
    prediction = policy.predict_action_probabilities(
        DialogueStateTracker.from_events(
            "casd", evts=conversation_events, slots=domain.slots
        ),
        domain,
        RegexInterpreter(),
    )
    assert_predicted_action(prediction, domain, ACTION_LISTEN_NAME)

    # Check that RulePolicy triggers form again after handling unhappy path
    conversation_events.append(ActionExecuted(ACTION_LISTEN_NAME))
    tracker = DialogueStateTracker.from_events(
        "casd", evts=conversation_events, slots=domain.slots
    )
    prediction = policy.predict_action_probabilities(
        tracker, domain, RegexInterpreter()
    )
    assert_predicted_action(prediction, domain, form_name)
    # check that RulePolicy entered unhappy path based on the training story
    assert prediction.events == [LoopInterrupted(True)]


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
            ActionExecuted(ACTION_LISTEN_NAME),
            # When a user says "hi", and the form is unhappy,
            # we want to run a specific action
            UserUttered(intent={"name": GREET_INTENT_NAME}),
            ActionExecuted(handle_rejection_action_name),
            ActionExecuted(ACTION_LISTEN_NAME),
            # Next user utterance is an answer to the previous question
            # and shouldn't be validated by the form
            UserUttered(intent={"name": GREET_INTENT_NAME}),
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
    prediction = policy.predict_action_probabilities(
        tracker, domain, RegexInterpreter()
    )
    # there is no rule for next action
    assert prediction.max_confidence == policy._core_fallback_threshold
    # check that RulePolicy entered unhappy path based on the training story
    assert prediction.events == [LoopInterrupted(True)]


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
    prediction = policy.predict_action_probabilities(
        DialogueStateTracker.from_events(
            "casd", evts=conversation_events, slots=domain.slots
        ),
        domain,
        RegexInterpreter(),
    )

    assert prediction.max_confidence == policy._core_fallback_threshold


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
    prediction = policy.predict_action_probabilities(
        DialogueStateTracker.from_events(
            "casd", evts=conversation_events, slots=domain.slots
        ),
        domain,
        RegexInterpreter(),
    )

    assert_predicted_action(prediction, domain, form_name)


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
    prediction = policy.predict_action_probabilities(
        DialogueStateTracker.from_events(
            "casd", evts=conversation_events, slots=domain.slots
        ),
        domain,
        RegexInterpreter(),
    )

    assert prediction.max_confidence == policy._core_fallback_threshold


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
    prediction = policy.predict_action_probabilities(
        form_conversation, domain, RegexInterpreter()
    )
    assert_predicted_action(prediction, domain, submit_action_name)


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
    policy.train([form_activation_rule, form_submit_rule], domain, RegexInterpreter())

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
    prediction = policy.predict_action_probabilities(
        form_conversation, domain, RegexInterpreter()
    )
    assert_predicted_action(prediction, domain, submit_action_name)


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
    prediction = trained_rule_policy.predict_action_probabilities(
        form_conversation, trained_rule_policy_domain, RegexInterpreter()
    )
    assert_predicted_action(prediction, trained_rule_policy_domain, "utter_stop")


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
            UserUttered(intent={"name": DEFAULT_NLU_FALLBACK_INTENT_NAME}),
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
            UserUttered(intent={"name": GREET_INTENT_NAME}),
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
    prediction = policy.predict_action_probabilities(
        tracker, domain, RegexInterpreter()
    )
    assert_predicted_action(prediction, domain, ACTION_DEFAULT_FALLBACK_NAME)

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

    prediction = policy.predict_action_probabilities(
        tracker, domain, RegexInterpreter()
    )
    assert_predicted_action(prediction, domain, UTTER_GREET_ACTION)


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
    prediction = policy.predict_action_probabilities(
        new_conversation, domain, RegexInterpreter()
    )

    assert_predicted_action(prediction, domain, expected_action_name)


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

    prediction = rule_policy.predict_action_probabilities(
        new_conversation, domain, RegexInterpreter()
    )

    assert_predicted_action(
        prediction, domain, expected_prediction, expected_confidence
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
    prediction = policy.predict_action_probabilities(
        new_conversation, domain, RegexInterpreter()
    )

    assert prediction.max_confidence == 0

from pathlib import Path
from typing import Text

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
    USER,
    PREVIOUS_ACTION,
    ACTIVE_LOOP,
    LOOP_NAME,
    RULE_ONLY_SLOTS,
    RULE_ONLY_LOOPS,
)
from rasa.shared.nlu.constants import TEXT, INTENT, ACTION_NAME
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import (
    ActionExecuted,
    UserUttered,
    ActiveLoop,
    SlotSet,
    ActionExecutionRejected,
    LoopInterrupted,
    FollowupAction,
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


def test_potential_contradiction_resolved_by_conversation_start():
    # Two rules that contradict each other except that one of them applies only at
    # conversation start -> ensure that this isn't flagged as a contradiction.

    utter_anti_greet_action = "utter_anti_greet"
    domain = Domain.from_yaml(
        f"""
        version: "2.0"
        intents:
        - {GREET_INTENT_NAME}
        actions:
        - {UTTER_GREET_ACTION}
        - {utter_anti_greet_action}
        """
    )
    policy = RulePolicy()
    greet_rule_at_conversation_start = TrackerWithCachedStates.from_events(
        "greet rule at conversation start",
        domain=domain,
        slots=domain.slots,
        evts=[
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(intent={"name": GREET_INTENT_NAME}),
            ActionExecuted(UTTER_GREET_ACTION),
        ],
        is_rule_tracker=True,
    )

    anti_greet_rule = TrackerWithCachedStates.from_events(
        "anti greet rule",
        domain=domain,
        slots=domain.slots,
        evts=[
            ActionExecuted(RULE_SNIPPET_ACTION_NAME),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(intent={"name": GREET_INTENT_NAME}),
            ActionExecuted(utter_anti_greet_action),
        ],
        is_rule_tracker=True,
    )

    # Contradicting rules abort training, hence policy training here needs to succeed
    # since there aren't contradicting rules in this case.
    policy.train(
        [greet_rule_at_conversation_start, anti_greet_rule], domain, RegexInterpreter()
    )


def test_potential_contradiction_resolved_by_conversation_start_when_slot_initial_value():
    # Two rules that contradict each other except that one of them applies only at
    # conversation start -> ensure that this isn't flagged as a contradiction.
    # Specifically, this checks that the conversation-start-checking logic doesn't
    # depend on initial rule tracker states being empty as these can be non-empty due to
    # initial slot values

    utter_anti_greet_action = "utter_anti_greet"
    some_slot = "slot1"
    some_slot_initial_value = "slot1value"
    domain = Domain.from_yaml(
        f"""
        version: "2.0"
        intents:
        - {GREET_INTENT_NAME}
        actions:
        - {UTTER_GREET_ACTION}
        - {utter_anti_greet_action}
        slots:
          {some_slot}:
            type: text
            initial_value: {some_slot_initial_value}
        """
    )
    policy = RulePolicy()
    greet_rule_at_conversation_start = TrackerWithCachedStates.from_events(
        "greet rule at conversation start",
        domain=domain,
        slots=domain.slots,
        evts=[
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(intent={"name": GREET_INTENT_NAME}),
            ActionExecuted(UTTER_GREET_ACTION),
        ],
        is_rule_tracker=True,
    )

    anti_greet_rule = TrackerWithCachedStates.from_events(
        "anti greet rule",
        domain=domain,
        slots=domain.slots,
        evts=[
            ActionExecuted(RULE_SNIPPET_ACTION_NAME),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(intent={"name": GREET_INTENT_NAME}),
            ActionExecuted(utter_anti_greet_action),
        ],
        is_rule_tracker=True,
    )

    # Policy training needs to succeed to confirm that no contradictions have been
    # detected
    policy.train(
        [greet_rule_at_conversation_start, anti_greet_rule], domain, RegexInterpreter()
    )

    # Check that the correct rule is applied when predicting next action in a story.
    conversation_events = [
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered(intent={"name": GREET_INTENT_NAME}),
    ]
    action_probabilities_1 = policy.predict_action_probabilities(
        DialogueStateTracker.from_events(
            "test conversation", evts=conversation_events, slots=domain.slots
        ),
        domain,
        RegexInterpreter(),
    )
    assert_predicted_action(action_probabilities_1, domain, UTTER_GREET_ACTION)


def test_potential_contradiction_resolved_by_conversation_start_when_slot_initial_value_explicit():
    # Two rules that contradict each other except that one of them applies only at
    # conversation start -> ensure that this isn't flagged as a contradiction.
    # Specifically, this checks that the conversation-start-checking logic doesn't
    # depend on whether or not the initial slot value is made explicit in the initial
    # state of the conversation tracker

    utter_anti_greet_action = "utter_anti_greet"
    some_slot = "slot1"
    some_slot_initial_value = "slot1value"
    domain = Domain.from_yaml(
        f"""
        version: "2.0"
        intents:
        - {GREET_INTENT_NAME}
        actions:
        - {UTTER_GREET_ACTION}
        - {utter_anti_greet_action}
        slots:
          {some_slot}:
            type: text
            initial_value: {some_slot_initial_value}
        """
    )
    policy = RulePolicy()
    greet_rule_at_conversation_start = TrackerWithCachedStates.from_events(
        "greet rule at conversation start",
        domain=domain,
        slots=domain.slots,
        evts=[
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(intent={"name": GREET_INTENT_NAME}),
            ActionExecuted(UTTER_GREET_ACTION),
        ],
        is_rule_tracker=True,
    )

    anti_greet_rule = TrackerWithCachedStates.from_events(
        "anti greet rule",
        domain=domain,
        slots=domain.slots,
        evts=[
            ActionExecuted(RULE_SNIPPET_ACTION_NAME),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(intent={"name": GREET_INTENT_NAME}),
            ActionExecuted(utter_anti_greet_action),
        ],
        is_rule_tracker=True,
    )

    # Policy training needs to succeed to confirm that no contradictions have been
    # detected
    policy.train(
        [greet_rule_at_conversation_start, anti_greet_rule], domain, RegexInterpreter()
    )

    conversation_events_with_initial_slot_explicit = [
        SlotSet(some_slot, some_slot_initial_value),
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered(intent={"name": GREET_INTENT_NAME}),
    ]
    action_probabilities_2 = policy.predict_action_probabilities(
        DialogueStateTracker.from_events(
            "test conversation with initial slot value explicitly set",
            evts=conversation_events_with_initial_slot_explicit,
            slots=domain.slots,
        ),
        domain,
        RegexInterpreter(),
    )
    assert_predicted_action(action_probabilities_2, domain, UTTER_GREET_ACTION)


def test_restrict_multiple_user_inputs_in_rules():
    domain = Domain.from_yaml(
        f"""
        version: "2.0"
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
        version: "2.0"
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
        version: "2.0"
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


def test_no_incomplete_rules_due_to_additional_slots_set():
    # Check that rules aren't automatically flagged as incomplete just because an action
    # doesn't set all the slots that are set in the same context in a different rule.
    # There may be slots that were set by other preceding actions (or by using
    # initial_value for a slot), and a rule shouldn't be marked as incomplete if some of
    # those other slots aren't set by the action in the rule.

    some_action = "some_action"
    some_slot = "some_slot"
    some_slot_value = "value1"
    some_other_slot = "some_other_slot"
    some_other_slot_value = "value2"
    domain = Domain.from_yaml(
        f"""
        version: "2.0"
        intents:
        - {GREET_INTENT_NAME}
        actions:
        - {some_action}
        slots:
          {some_slot}:
            type: text
          {some_other_slot}:
            type: text
        """
    )
    policy = RulePolicy()
    simple_rule = TrackerWithCachedStates.from_events(
        "simple rule with an action that sets 1 slot",
        domain=domain,
        slots=domain.slots,
        evts=[
            ActionExecuted(RULE_SNIPPET_ACTION_NAME),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(intent={"name": GREET_INTENT_NAME}),
            ActionExecuted(some_action),
            SlotSet(some_slot, some_slot_value),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
        is_rule_tracker=True,
    )
    simple_rule_with_slot_set = TrackerWithCachedStates.from_events(
        "simple rule with an additional slot set before it starts",
        domain=domain,
        slots=domain.slots,
        evts=[
            SlotSet(some_other_slot, some_other_slot_value),
            ActionExecuted(RULE_SNIPPET_ACTION_NAME),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(intent={"name": GREET_INTENT_NAME}),
            ActionExecuted(some_action),
            SlotSet(some_slot, some_slot_value),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
        is_rule_tracker=True,
    )

    # this should finish without raising any errors about incomplete rules
    policy.train([simple_rule, simple_rule_with_slot_set], domain, RegexInterpreter())


def test_incomplete_rules_due_to_loops():
    some_form = "some_form"
    domain = Domain.from_yaml(
        f"""
        version: "2.0"
        intents:
        - {GREET_INTENT_NAME}
        forms:
          {some_form}:
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
        version: "2.0"
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
        version: "2.0"
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


async def test_rule_policy_finetune(
    tmp_path: Path, trained_rule_policy: RulePolicy, trained_rule_policy_domain: Domain
):
    trained_rule_policy.persist(tmp_path)

    loaded_policy = RulePolicy.load(tmp_path, should_finetune=True)

    assert loaded_policy.finetune_mode

    new_rule = TrackerWithCachedStates.from_events(
        "stop story",
        domain=trained_rule_policy_domain,
        slots=trained_rule_policy_domain.slots,
        evts=[
            ActionExecuted(RULE_SNIPPET_ACTION_NAME),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(intent={"name": "stopp"}),
            ActionExecuted("utter_stop"),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
        is_rule_tracker=True,
    )

    original_data = await training.load_data(
        "examples/rules/data/rules.yml", trained_rule_policy_domain
    )

    loaded_policy.train(
        original_data + [new_rule], trained_rule_policy_domain, RegexInterpreter()
    )

    assert (
        len(loaded_policy.lookup["rules"])
        == len(trained_rule_policy.lookup["rules"]) + 1
    )
    assert (
        """[{"prev_action": {"action_name": "action_listen"}, "user": {"intent": "stopp"}}]"""
        in loaded_policy.lookup["rules"]
    )


async def test_rule_policy_contradicting_rule_finetune(
    tmp_path: Path, trained_rule_policy: RulePolicy, trained_rule_policy_domain: Domain
):
    trained_rule_policy.persist(tmp_path)

    loaded_policy = RulePolicy.load(tmp_path, should_finetune=True)

    assert loaded_policy.finetune_mode

    new_rule = TrackerWithCachedStates.from_events(
        "stop story",
        domain=trained_rule_policy_domain,
        slots=trained_rule_policy_domain.slots,
        evts=[
            ActionExecuted(RULE_SNIPPET_ACTION_NAME),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(intent={"name": "activate_q_form"}),
            ActionExecuted("utter_stop"),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
        is_rule_tracker=True,
    )

    original_data = await training.load_data(
        "examples/rules/data/rules.yml", trained_rule_policy_domain
    )

    with pytest.raises(InvalidRule) as execinfo:
        loaded_policy.train(
            original_data + [new_rule], trained_rule_policy_domain, RegexInterpreter()
        )
        assert all(
            name in execinfo.value.message for name in {"utter_stop", "stop story"}
        )


def test_faq_rule():
    domain = Domain.from_yaml(
        f"""
        version: "2.0"
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
    is_end_to_end_prediction: bool = False,
    is_no_user_prediction: bool = False,
) -> None:
    assert prediction.max_confidence == confidence
    index_of_predicted_action = prediction.max_confidence_index
    prediction_action_name = domain.action_names_or_texts[index_of_predicted_action]
    assert prediction_action_name == expected_action_name
    assert prediction.is_end_to_end_prediction == is_end_to_end_prediction
    assert prediction.is_no_user_prediction == is_no_user_prediction


async def test_predict_form_action_if_in_form():
    form_name = "some_form"

    domain = Domain.from_yaml(
        f"""
        version: "2.0"
        intents:
        - {GREET_INTENT_NAME}
        actions:
        - {UTTER_GREET_ACTION}
        - some-action
        slots:
          {REQUESTED_SLOT}:
            type: unfeaturized
        forms:
          {form_name}:
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
    assert_predicted_action(prediction, domain, form_name, is_no_user_prediction=True)


async def test_predict_loop_action_if_in_loop_but_there_is_e2e_rule():
    loop_name = "some_loop"

    domain = Domain.from_yaml(
        f"""
        version: "2.0"
        intents:
        - {GREET_INTENT_NAME}
        actions:
        - {UTTER_GREET_ACTION}
        - some-action
        slots:
          {REQUESTED_SLOT}:
            type: unfeaturized
        forms:
          {loop_name}:
        """
    )
    e2e_rule = TrackerWithCachedStates.from_events(
        "bla",
        domain=domain,
        evts=[
            ActionExecuted(RULE_SNIPPET_ACTION_NAME),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(text="haha"),
            ActionExecuted(UTTER_GREET_ACTION),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
        is_rule_tracker=True,
    )
    policy = RulePolicy()
    policy.train([e2e_rule], domain, RegexInterpreter())

    loop_conversation = DialogueStateTracker.from_events(
        "in a loop",
        evts=[
            # We are in an activate form
            ActionExecuted(loop_name),
            ActiveLoop(loop_name),
            SlotSet(REQUESTED_SLOT, "some value"),
            ActionExecuted(ACTION_LISTEN_NAME),
            # User sends message as response to a requested slot
            UserUttered("haha", {"name": GREET_INTENT_NAME}),
        ],
        slots=domain.slots,
    )

    # RulePolicy triggers form again
    prediction = policy.predict_action_probabilities(
        loop_conversation, domain, RegexInterpreter()
    )
    assert_predicted_action(prediction, domain, loop_name, is_no_user_prediction=True)


async def test_predict_form_action_if_multiple_turns():
    form_name = "some_form"
    other_intent = "bye"
    domain = Domain.from_yaml(
        f"""
        version: "2.0"
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
          {form_name}:
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
    assert_predicted_action(prediction, domain, form_name, is_no_user_prediction=True)


async def test_predict_slot_initial_value_not_required_in_rule():
    domain = Domain.from_yaml(
        """
intents:
- i1
actions:
- action1
slots:
  s_cat1:
    type: categorical
    values:
      - v1
      - v2
    initial_value: v1
"""
    )

    rule = DialogueStateTracker.from_events(
        "slot rule",
        evts=[
            ActionExecuted(RULE_SNIPPET_ACTION_NAME),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(intent={"name": "i1"}),
            ActionExecuted("action1"),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
        domain=domain,
        slots=domain.slots,
    )
    rule.is_rule_tracker = True

    policy = RulePolicy()
    policy.train([rule], domain, RegexInterpreter())

    form_conversation = DialogueStateTracker.from_events(
        "slot rule test",
        evts=[
            SlotSet("s_cat1", "v2"),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(intent={"name": "i1"}),
        ],
        domain=domain,
        slots=domain.slots,
    )

    prediction = policy.predict_action_probabilities(
        form_conversation, domain, RegexInterpreter()
    )
    assert_predicted_action(prediction, domain, "action1")


async def test_predict_slot_with_initial_slot_matches_rule():
    domain = Domain.from_yaml(
        """
intents:
- i1
actions:
- action1
slots:
  s_cat1:
    type: categorical
    values:
      - v1
      - v2
    initial_value: v1
"""
    )

    rule = DialogueStateTracker.from_events(
        "slot rule",
        evts=[
            ActionExecuted(RULE_SNIPPET_ACTION_NAME),
            SlotSet("s_cat1", "v1"),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(intent={"name": "i1"}),
            ActionExecuted("action1"),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
        domain=domain,
        slots=domain.slots,
    )
    rule.is_rule_tracker = True

    policy = RulePolicy()
    policy.train([rule], domain, RegexInterpreter())

    form_conversation = DialogueStateTracker.from_events(
        "slot rule test",
        evts=[ActionExecuted(ACTION_LISTEN_NAME), UserUttered(intent={"name": "i1"}),],
        domain=domain,
        slots=domain.slots,
    )

    prediction = policy.predict_action_probabilities(
        form_conversation, domain, RegexInterpreter()
    )
    assert_predicted_action(prediction, domain, "action1")


async def test_predict_action_listen_after_form():
    form_name = "some_form"

    domain = Domain.from_yaml(
        f"""
        version: "2.0"
        intents:
        - {GREET_INTENT_NAME}
        actions:
        - {UTTER_GREET_ACTION}
        - some-action
        slots:
          {REQUESTED_SLOT}:
            type: unfeaturized
        forms:
          {form_name}:
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
    assert_predicted_action(
        prediction, domain, ACTION_LISTEN_NAME, is_no_user_prediction=True
    )


async def test_dont_predict_form_if_already_finished():
    form_name = "some_form"

    domain = Domain.from_yaml(
        f"""
        version: "2.0"
        intents:
        - {GREET_INTENT_NAME}
        actions:
        - {UTTER_GREET_ACTION}
        - some-action
        slots:
          {REQUESTED_SLOT}:
            type: unfeaturized
        forms:
          {form_name}:
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
        version: "2.0"
        intents:
        - {GREET_INTENT_NAME}
        actions:
        - {UTTER_GREET_ACTION}
        - some-action
        slots:
          {REQUESTED_SLOT}:
            type: unfeaturized
        forms:
          {form_name}:
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
        version: "2.0"
        intents:
        - {GREET_INTENT_NAME}
        actions:
        - {UTTER_GREET_ACTION}
        - some-action
        slots:
          {REQUESTED_SLOT}:
            type: unfeaturized
        forms:
          {form_name}:
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
        version: "2.0"
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
          {form_name}:
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
        version: "2.0"
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
          {form_name}:
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
        version: "2.0"
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
          {form_name}:
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
        version: "2.0"
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
          {form_name}:
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
        version: "2.0"
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
          {form_name}:
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
        version: "2.0"
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
          {form_name}:
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
        version: "2.0"
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
          {form_name}:
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
        version: "2.0"
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
          {form_name}:
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
        version: "2.0"
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
          {form_name}:
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
        version: "2.0"
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
        TemplatedNaturalLanguageGenerator(domain.responses),
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
        version: "2.0"
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
    "intent_name", [USER_INTENT_RESTART, USER_INTENT_BACK, USER_INTENT_SESSION_START]
)
def test_e2e_beats_default_actions(intent_name: Text):
    domain = Domain.from_yaml(
        f"""
        version: "2.0"
        intents:
        - {GREET_INTENT_NAME}
        actions:
        - {UTTER_GREET_ACTION}
        """
    )

    e2e_rule = TrackerWithCachedStates.from_events(
        "bla",
        domain=domain,
        evts=[
            ActionExecuted(RULE_SNIPPET_ACTION_NAME),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(text="haha"),
            ActionExecuted(UTTER_GREET_ACTION),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
        is_rule_tracker=True,
    )

    policy = RulePolicy()
    policy.train([e2e_rule], domain, RegexInterpreter())

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
    assert_predicted_action(
        prediction, domain, UTTER_GREET_ACTION, is_end_to_end_prediction=True
    )


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
        version: "2.0"
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
        version: "2.0"
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


def test_hide_rule_turn():
    chitchat = "chitchat"
    action_chitchat = "action_chitchat"
    domain = Domain.from_yaml(
        f"""
        version: "2.0"
        intents:
        - {GREET_INTENT_NAME}
        - {chitchat}
        actions:
        - {UTTER_GREET_ACTION}
        - {action_chitchat}
        """
    )
    chitchat_story = TrackerWithCachedStates.from_events(
        "chitchat story",
        domain=domain,
        slots=domain.slots,
        evts=[
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(intent={"name": chitchat}),
            ActionExecuted(action_chitchat),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
    )
    policy = RulePolicy()
    policy.train(
        [GREET_RULE, chitchat_story], domain, RegexInterpreter(),
    )

    conversation_events = [
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered("haha", {"name": GREET_INTENT_NAME}),
    ]
    prediction = policy.predict_action_probabilities(
        DialogueStateTracker.from_events(
            "casd", evts=conversation_events, slots=domain.slots
        ),
        domain,
        RegexInterpreter(),
    )
    assert_predicted_action(prediction, domain, UTTER_GREET_ACTION)
    assert prediction.hide_rule_turn

    conversation_events += [
        ActionExecuted(UTTER_GREET_ACTION, hide_rule_turn=prediction.hide_rule_turn)
    ]
    prediction = policy.predict_action_probabilities(
        DialogueStateTracker.from_events(
            "casd", evts=conversation_events, slots=domain.slots
        ),
        domain,
        RegexInterpreter(),
    )
    assert_predicted_action(prediction, domain, ACTION_LISTEN_NAME)
    assert prediction.hide_rule_turn

    conversation_events += [
        ActionExecuted(ACTION_LISTEN_NAME, hide_rule_turn=prediction.hide_rule_turn),
        UserUttered("haha", {"name": chitchat}),
    ]
    tracker = DialogueStateTracker.from_events(
        "casd", evts=conversation_events, slots=domain.slots
    )
    states = tracker.past_states(domain, ignore_rule_only_turns=True)
    assert states == [
        {},
        {
            USER: {TEXT: "haha", INTENT: chitchat},
            PREVIOUS_ACTION: {ACTION_NAME: ACTION_LISTEN_NAME},
        },
    ]


def test_hide_rule_turn_with_slots():
    some_action = "some_action"
    some_other_action = "some_other_action"
    some_intent = "some_intent"
    some_other_intent = "some_other_intent"
    slot_which_is_only_in_rule = "slot_which_is_only_in_rule"
    some_slot_value = "value1"
    slot_which_is_also_in_story = "slot_which_is_also_in_story"
    some_other_slot_value = "value2"
    domain = Domain.from_yaml(
        f"""
        version: "2.0"
        intents:
        - {some_intent}
        - {some_other_intent}
        actions:
        - {some_action}
        - {some_other_action}
        slots:
          {slot_which_is_only_in_rule}:
            type: text
          {slot_which_is_also_in_story}:
            type: text
        """
    )

    simple_rule = TrackerWithCachedStates.from_events(
        "simple rule with an action that sets 1 slot",
        domain=domain,
        slots=domain.slots,
        evts=[
            ActionExecuted(RULE_SNIPPET_ACTION_NAME),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(intent={"name": some_intent}),
            ActionExecuted(some_action),
            SlotSet(slot_which_is_only_in_rule, some_slot_value),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
        is_rule_tracker=True,
    )
    simple_rule_with_slot_set = TrackerWithCachedStates.from_events(
        "simple rule with an additional slot set before it starts",
        domain=domain,
        slots=domain.slots,
        evts=[
            SlotSet(slot_which_is_also_in_story, some_other_slot_value),
            ActionExecuted(RULE_SNIPPET_ACTION_NAME),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(intent={"name": some_intent}),
            ActionExecuted(some_action),
            SlotSet(slot_which_is_only_in_rule, some_slot_value),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
        is_rule_tracker=True,
    )
    simple_story_with_other_slot_set = TrackerWithCachedStates.from_events(
        "simple rule with an additional slot set before it starts",
        domain=domain,
        slots=domain.slots,
        evts=[
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(intent={"name": some_other_intent}),
            ActionExecuted(some_other_action),
            SlotSet(slot_which_is_also_in_story, some_other_slot_value),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
    )

    policy = RulePolicy()
    policy.train(
        [simple_rule, simple_rule_with_slot_set, simple_story_with_other_slot_set],
        domain,
        RegexInterpreter(),
    )
    assert policy.lookup[RULE_ONLY_SLOTS] == [slot_which_is_only_in_rule]

    conversation_events = [
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered("haha", {"name": some_intent}),
    ]
    prediction = policy.predict_action_probabilities(
        DialogueStateTracker.from_events(
            "casd", evts=conversation_events, slots=domain.slots
        ),
        domain,
        RegexInterpreter(),
    )
    assert_predicted_action(prediction, domain, some_action)
    assert prediction.hide_rule_turn

    conversation_events += [
        ActionExecuted(some_action, hide_rule_turn=prediction.hide_rule_turn),
        SlotSet(slot_which_is_only_in_rule, some_slot_value),
    ]
    prediction = policy.predict_action_probabilities(
        DialogueStateTracker.from_events(
            "casd", evts=conversation_events, slots=domain.slots
        ),
        domain,
        RegexInterpreter(),
    )
    assert_predicted_action(prediction, domain, ACTION_LISTEN_NAME)
    assert prediction.hide_rule_turn

    conversation_events += [
        ActionExecuted(ACTION_LISTEN_NAME, hide_rule_turn=prediction.hide_rule_turn),
        UserUttered("haha", {"name": some_other_intent}),
    ]
    tracker = DialogueStateTracker.from_events(
        "casd", evts=conversation_events, slots=domain.slots
    )
    states = tracker.past_states(
        domain, ignore_rule_only_turns=True, rule_only_data=policy.get_rule_only_data()
    )
    assert states == [
        {},
        {
            USER: {TEXT: "haha", INTENT: some_other_intent},
            PREVIOUS_ACTION: {ACTION_NAME: ACTION_LISTEN_NAME},
        },
    ]


def test_hide_rule_turn_no_last_action_listen():
    action_after_chitchat = "action_after_chitchat"
    chitchat = "chitchat"
    action_chitchat = "action_chitchat"
    followup_on_chitchat = "followup_on_chitchat"
    domain = Domain.from_yaml(
        f"""
        version: "2.0"
        intents:
        - {chitchat}
        actions:
        - {action_chitchat}
        - {action_after_chitchat}
        slots:
          {followup_on_chitchat}:
            type: bool
        """
    )
    simple_rule_no_last_action_listen = TrackerWithCachedStates.from_events(
        "simple rule without action listen in the end",
        domain=domain,
        slots=domain.slots,
        evts=[
            ActionExecuted(RULE_SNIPPET_ACTION_NAME),
            ActionExecuted(action_chitchat),
            SlotSet(followup_on_chitchat, True),
            ActionExecuted(action_after_chitchat),
            ActionExecuted(RULE_SNIPPET_ACTION_NAME),
        ],
        is_rule_tracker=True,
    )
    chitchat_story = TrackerWithCachedStates.from_events(
        "chitchat story",
        domain=domain,
        slots=domain.slots,
        evts=[
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(intent={"name": chitchat}),
            ActionExecuted(action_chitchat),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
    )
    policy = RulePolicy()
    policy.train(
        [simple_rule_no_last_action_listen, chitchat_story], domain, RegexInterpreter()
    )
    assert policy.lookup[RULE_ONLY_SLOTS] == [followup_on_chitchat]

    conversation_events = [
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered(intent={"name": chitchat}),
        ActionExecuted(action_chitchat),
        SlotSet(followup_on_chitchat, True),
    ]
    prediction = policy.predict_action_probabilities(
        DialogueStateTracker.from_events(
            "casd", evts=conversation_events, slots=domain.slots
        ),
        domain,
        RegexInterpreter(),
    )
    assert_predicted_action(prediction, domain, action_after_chitchat)
    assert prediction.hide_rule_turn

    conversation_events += [
        ActionExecuted(action_after_chitchat, hide_rule_turn=prediction.hide_rule_turn)
    ]
    tracker = DialogueStateTracker.from_events(
        "casd", evts=conversation_events, slots=domain.slots
    )
    states = tracker.past_states(
        domain, ignore_rule_only_turns=True, rule_only_data=policy.get_rule_only_data()
    )
    assert states == [
        {},
        {USER: {INTENT: chitchat}, PREVIOUS_ACTION: {ACTION_NAME: ACTION_LISTEN_NAME}},
        {USER: {INTENT: chitchat}, PREVIOUS_ACTION: {ACTION_NAME: action_chitchat}},
    ]


def test_hide_rule_turn_with_loops():
    form_name = "some_form"
    another_form_name = "another_form"
    activate_form = "activate_form"
    activate_another_form = "activate_another_form"
    chitchat = "chitchat"
    action_chitchat = "action_chitchat"
    domain = Domain.from_yaml(
        f"""
        version: "2.0"
        intents:
        - {GREET_INTENT_NAME}
        - {activate_form}
        - {chitchat}
        - {activate_another_form}
        actions:
        - {UTTER_GREET_ACTION}
        - {action_chitchat}
        slots:
          {REQUESTED_SLOT}:
            type: unfeaturized
        forms:
          {form_name}:
          {another_form_name}:
        """
    )

    form_activation_rule = _form_activation_rule(domain, form_name, activate_form)

    another_form_activation_rule = _form_activation_rule(
        domain, another_form_name, activate_another_form
    )
    another_form_activation_story = another_form_activation_rule.copy()
    another_form_activation_story.is_rule_tracker = False

    chitchat_story = TrackerWithCachedStates.from_events(
        "chitchat story",
        domain=domain,
        slots=domain.slots,
        evts=[
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(intent={"name": chitchat}),
            ActionExecuted(action_chitchat),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
    )
    policy = RulePolicy()
    policy.train(
        [
            form_activation_rule,
            chitchat_story,
            another_form_activation_rule,
            another_form_activation_story,
        ],
        domain,
        RegexInterpreter(),
    )
    assert policy.lookup[RULE_ONLY_LOOPS] == [form_name]

    conversation_events = [
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered("haha", {"name": activate_form}),
    ]
    prediction = policy.predict_action_probabilities(
        DialogueStateTracker.from_events(
            "casd", evts=conversation_events, slots=domain.slots
        ),
        domain,
        RegexInterpreter(),
    )
    assert_predicted_action(prediction, domain, form_name)
    assert prediction.hide_rule_turn

    conversation_events += [
        ActionExecuted(form_name, hide_rule_turn=prediction.hide_rule_turn),
        ActiveLoop(form_name),
    ]
    prediction = policy.predict_action_probabilities(
        DialogueStateTracker.from_events(
            "casd", evts=conversation_events, slots=domain.slots
        ),
        domain,
        RegexInterpreter(),
    )
    assert_predicted_action(
        prediction, domain, ACTION_LISTEN_NAME, is_no_user_prediction=True
    )
    assert prediction.hide_rule_turn

    conversation_events += [
        ActionExecuted(ACTION_LISTEN_NAME, hide_rule_turn=prediction.hide_rule_turn),
        UserUttered("haha", {"name": chitchat}),
    ]
    tracker = DialogueStateTracker.from_events(
        "casd", evts=conversation_events, slots=domain.slots
    )
    states = tracker.past_states(
        domain, ignore_rule_only_turns=True, rule_only_data=policy.get_rule_only_data()
    )
    assert states == [
        {},
        {
            USER: {TEXT: "haha", INTENT: chitchat},
            PREVIOUS_ACTION: {ACTION_NAME: ACTION_LISTEN_NAME},
        },
    ]


def test_do_not_hide_rule_turn_with_loops_in_stories():
    form_name = "some_form"
    activate_form = "activate_form"
    domain = Domain.from_yaml(
        f"""
        version: "2.0"
        intents:
        - {activate_form}
        slots:
          {REQUESTED_SLOT}:
            type: unfeaturized
        forms:
          {form_name}:
        """
    )

    form_activation_rule = _form_activation_rule(domain, form_name, activate_form)
    form_activation_story = form_activation_rule.copy()
    form_activation_story.is_rule_tracker = False

    policy = RulePolicy()
    policy.train(
        [form_activation_rule, form_activation_story], domain, RegexInterpreter(),
    )
    assert policy.lookup[RULE_ONLY_LOOPS] == []

    conversation_events = [
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered("haha", {"name": activate_form}),
    ]
    prediction = policy.predict_action_probabilities(
        DialogueStateTracker.from_events(
            "casd", evts=conversation_events, slots=domain.slots
        ),
        domain,
        RegexInterpreter(),
    )
    assert_predicted_action(prediction, domain, form_name)
    assert not prediction.hide_rule_turn

    conversation_events += [
        ActionExecuted(form_name, hide_rule_turn=prediction.hide_rule_turn),
        ActiveLoop(form_name),
    ]
    prediction = policy.predict_action_probabilities(
        DialogueStateTracker.from_events(
            "casd", evts=conversation_events, slots=domain.slots
        ),
        domain,
        RegexInterpreter(),
    )
    assert_predicted_action(
        prediction, domain, ACTION_LISTEN_NAME, is_no_user_prediction=True
    )
    assert not prediction.hide_rule_turn


def test_hide_rule_turn_with_loops_as_followup_action():
    form_name = "some_form"
    activate_form = "activate_form"
    domain = Domain.from_yaml(
        f"""
        version: "2.0"
        intents:
        - {GREET_INTENT_NAME}
        - {activate_form}
        actions:
        - {UTTER_GREET_ACTION}
        slots:
          {REQUESTED_SLOT}:
            type: unfeaturized
        forms:
          {form_name}:
        """
    )

    form_activation_rule = _form_activation_rule(domain, form_name, activate_form)
    form_activation_story = form_activation_rule.copy()
    form_activation_story.is_rule_tracker = False

    policy = RulePolicy()
    policy.train(
        [form_activation_rule, GREET_RULE, form_activation_story],
        domain,
        RegexInterpreter(),
    )
    assert policy.lookup[RULE_ONLY_LOOPS] == []

    conversation_events = [
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered("haha", {"name": activate_form}),
    ]
    prediction = policy.predict_action_probabilities(
        DialogueStateTracker.from_events(
            "casd", evts=conversation_events, slots=domain.slots
        ),
        domain,
        RegexInterpreter(),
    )
    assert_predicted_action(prediction, domain, form_name)
    assert not prediction.hide_rule_turn

    conversation_events += [
        ActionExecuted(form_name, hide_rule_turn=prediction.hide_rule_turn),
        ActiveLoop(form_name),
    ]
    prediction = policy.predict_action_probabilities(
        DialogueStateTracker.from_events(
            "casd", evts=conversation_events, slots=domain.slots
        ),
        domain,
        RegexInterpreter(),
    )
    assert_predicted_action(
        prediction, domain, ACTION_LISTEN_NAME, is_no_user_prediction=True
    )
    assert not prediction.hide_rule_turn

    conversation_events += [
        ActionExecuted(ACTION_LISTEN_NAME, hide_rule_turn=prediction.hide_rule_turn),
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
    assert prediction.hide_rule_turn

    conversation_events += [
        ActionExecuted(UTTER_GREET_ACTION, hide_rule_turn=prediction.hide_rule_turn),
        FollowupAction(form_name),
        ActionExecuted(form_name),
    ]
    prediction = policy.predict_action_probabilities(
        DialogueStateTracker.from_events(
            "casd", evts=conversation_events, slots=domain.slots
        ),
        domain,
        RegexInterpreter(),
    )
    assert_predicted_action(
        prediction, domain, ACTION_LISTEN_NAME, is_no_user_prediction=True
    )
    tracker = DialogueStateTracker.from_events(
        "casd", evts=conversation_events, slots=domain.slots
    )
    states = tracker.past_states(domain, ignore_rule_only_turns=True)
    assert states == [
        {},
        {
            USER: {TEXT: "haha", INTENT: activate_form},
            PREVIOUS_ACTION: {ACTION_NAME: ACTION_LISTEN_NAME},
        },
        {
            USER: {TEXT: "haha", INTENT: activate_form},
            PREVIOUS_ACTION: {ACTION_NAME: form_name},
            ACTIVE_LOOP: {LOOP_NAME: form_name},
        },
    ]

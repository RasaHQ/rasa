import pytest

from rasa.shared.constants import RASA_PATTERN_HUMAN_HANDOFF
from rasa.shared.core.flows.steps import LinkFlowStep
from rasa.shared.core.flows.steps.constants import (
    START_STEP,
    END_STEP,
    CONTINUE_STEP_PREFIX,
)
from rasa.shared.core.flows.validation import (
    DuplicatedStepIdException,
    EmptyFlowException,
    EmptyStepSequenceException,
    MissingElseBranchException,
    NoLinkAllowedInCalledFlowException,
    PatternReferencedFlowException,
    ReferenceToPatternException,
    UnreachableFlowStepException,
    MissingNextLinkException,
    ReservedFlowStepIdException,
    NoNextAllowedForLinkException,
    UnresolvedFlowException,
    UnresolvedFlowStepIdException,
    DuplicateNLUTriggerException,
    SlotNamingException,
    FlowIdNamingException,
)
from rasa.shared.core.flows.yaml_flows_io import (
    flows_from_str,
    flows_from_str_including_defaults,
)


def test_validation_does_not_always_fail() -> None:
    valid_flows = """
            flows:
              empty_branch_flow:
                description: "A flow with an empty branch"
                steps:
                  - action: utter_greet
                    next:
                      - if: "status == logged_in"
                        then:
                          - action: utter_already_logged_in
                            next: "END"
                      - else:
                        - action: "utter_need_to_log_in"
                          next: "END"
            """
    flows_from_str(valid_flows)


def test_validation_fails_on_empty_steps() -> None:
    with pytest.raises(EmptyFlowException) as e:
        flows_from_str(
            """
        flows:
          abc:
            description: "A flow without steps"
            steps: []
        """
        )
    assert e.value.flow_id == "abc"


def test_validation_fails_on_empty_steps_for_branch() -> None:
    with pytest.raises(EmptyStepSequenceException) as e:
        flows_from_str(
            """
        flows:
          foo:
            description: "a test flow"
            steps:
              - action: utter_greet
          xyz:
            description: "A flow with an empty branch"
            steps:
              - action: utter_greet
                next:
                  - if: "status == logged_in"
                    then: []
                  - else:
                    - action: "utter_need_to_log_in"
                      next: "END"
        """
        )
    assert e.value.flow_id == "xyz"
    assert "greet" in e.value.step_id


def test_validation_fails_on_unreachable_step_inside_of_flow() -> None:
    with pytest.raises(UnreachableFlowStepException) as e:
        flows_from_str(
            """
        flows:
          abc:
            description: "A flow with an empty branch"
            steps:
              - action: utter_start
                next: the_end
              - action: utter_middle
              - id: the_end
                action: utter_end
        """
        )
    assert e.value.flow_id == "abc"
    assert "utter_middle" in e.value.step_id


def test_validation_fails_on_unreachable_step_at_the_end_of_flow() -> None:
    with pytest.raises(UnreachableFlowStepException) as e:
        flows_from_str(
            """
        flows:
          abc:
            description: "A flow with an empty branch"
            steps:
              - action: utter_start
              - action: utter_middle
                next: END
              - action: utter_end
        """
        )
    assert e.value.flow_id == "abc"
    assert "utter_end" in e.value.step_id


def test_validation_fails_on_unreachable_step_at_the_end_of_a_branch() -> None:
    with pytest.raises(UnreachableFlowStepException) as e:
        flows_from_str(
            """
        flows:
          abc:
            description: "A flow with an empty branch"
            steps:
              - action: utter_start
              - action: utter_middle
                next:
                  - if: "x == 10"
                    then:
                      - action: utter_middle_two
                        next: the_end
                      - action: utter_middle_forgotten
                        next: END
                  - else:
                      - action: utter_middle_three
                        next: the_end
              - id: the_end
                action: utter_end
        """
        )
    assert e.value.flow_id == "abc"
    assert "utter_middle_forgotten" in e.value.step_id


def test_validation_fails_on_missing_next_link() -> None:
    with pytest.raises(MissingNextLinkException) as e:
        flows_from_str(
            """
        flows:
          abc:
            description: "A flow with an empty branch"
            steps:
              - action: utter_start
              - action: utter_middle
                next:
                  - if: "true"
                    then:
                      - action: utter_middle_two
                        next: the_end
                  - else:
                      - action: utter_middle_impossible
              - id: the_end
                action: utter_end
        """
        )
    assert e.value.flow_id == "abc"
    assert "utter_middle_impossible" in e.value.step_id


def test_validation_fails_on_using_reserved_start_step_id() -> None:
    with pytest.raises(ReservedFlowStepIdException) as e:
        flows_from_str(
            f"""
        flows:
          abc:
            description: "A flow with an empty branch"
            steps:
              - id: {START_STEP}
                action: utter_start
              - action: utter_middle
              - action: utter_end
        """
        )
    assert e.value.flow_id == "abc"
    assert START_STEP in e.value.step_id


def test_validation_fails_on_using_reserved_end_step_id() -> None:
    with pytest.raises(ReservedFlowStepIdException) as e:
        flows_from_str(
            f"""
        flows:
          abc:
            description: "A flow with an empty branch"
            steps:
              - action: utter_start
              - action: utter_middle
              - id: {END_STEP}
                action: utter_end
        """
        )
    assert e.value.flow_id == "abc"
    assert END_STEP in e.value.step_id


def test_validation_fails_on_using_reserved_continuation_step_prefix() -> None:
    with pytest.raises(ReservedFlowStepIdException) as e:
        flows_from_str(
            f"""
        flows:
          abc:
            description: "A flow with an empty branch"
            steps:
              - action: utter_start
              - action: utter_middle
              - id: {CONTINUE_STEP_PREFIX}the_end
                action: utter_end
        """
        )
    assert e.value.flow_id == "abc"
    assert "the_end" in e.value.step_id


def test_validation_fails_on_missing_else_branch() -> None:
    with pytest.raises(MissingElseBranchException) as e:
        flows_from_str(
            """
        flows:
          abc:
            description: "A flow with an empty branch"
            steps:
              - action: utter_start
              - action: utter_middle
                next:
                  - if: "x == 10"
                    then:
                      - action: utter_middle_two
                        next: the_end
              - id: the_end
                action: utter_end
        """
        )
    assert e.value.flow_id == "abc"
    assert "utter_middle" in e.value.step_id


def test_validation_fails_on_link_step_with_next() -> None:
    with pytest.raises(NoNextAllowedForLinkException) as e:
        flows_from_str(
            """
        flows:
          abc:
            description: "A flow with an empty branch"
            steps:
              - action: utter_start
              - action: utter_middle
              - link: get_user_name
                next: utter_end    # needs explicit next as there is no default
              - id: utter_end
                action: utter_end
        """
        )
    assert e.value.flow_id == "abc"
    assert "get_user_name" in e.value.step_id


def test_validation_fails_on_unresolvable_next_step_id() -> None:
    bad_id = "TO_THE_UNIVERSE_AND_BEYOND"
    with pytest.raises(UnresolvedFlowStepIdException) as e:
        flows_from_str(
            f"""
        flows:
          abc:
            description: "A flow with an empty branch"
            steps:
              - action: utter_start
              - action: utter_middle
              - action: utter_end
                next: {bad_id}
        """
        )
    assert e.value.flow_id == "abc"
    assert bad_id in e.value.step_id
    assert "utter_end" in e.value.referenced_from_step_id


def test_validation_fails_on_unresolvable_next_step_id_in_branch() -> None:
    bad_id = "the_eend"
    with pytest.raises(UnresolvedFlowStepIdException) as e:
        flows_from_str(
            f"""
        flows:
          abc:
            description: "A flow with an empty branch"
            steps:
              - action: utter_start
              - action: utter_middle
                next:
                  - if: "x == 10"
                    then:
                      - action: utter_middle_two
                        next: {bad_id}
                  - else:
                      - action: utter_middle_three
                        next: END
              - id: the_end
                action: utter_end
        """
        )
    assert e.value.flow_id == "abc"
    assert bad_id in e.value.step_id
    assert "utter_middle_two" in e.value.referenced_from_step_id


def test_validation_fails_on_multiple_flows_with_same_nlu_triggers():
    flow_config = """
        flows:
          foo:
            description: test foo flow
            nlu_trigger:
              - intent: foo
            steps:
              - action: utter_welcome
          bar:
            description: test bar flow
            nlu_trigger:
              - intent: foo
            steps:
              - action: utter_welcome
        """

    with pytest.raises(DuplicateNLUTriggerException):
        flows_from_str(flow_config)


def test_validation_fails_for_a_called_flow_with_a_link():
    flow_config = """
        flows:
          foo:
            description: foo flow
            steps:
              - call: bar
          bar:
            description: bar flow
            steps:
              - link: baz
          baz:
            description: baz flow
            steps:
              - action: action_listen
        """

    with pytest.raises(NoLinkAllowedInCalledFlowException):
        flows_from_str(flow_config)


def test_validation_fails_for_a_called_flow_that_does_not_exist():
    flow_config = """
        flows:
          foo:
            description: foo flow
            steps:
              - call: bar
        """

    with pytest.raises(UnresolvedFlowException):
        flows_from_str(flow_config)


def test_validation_fails_for_a_linked_flow_that_does_not_exist():
    flow_config = """
        flows:
          foo:
            description: foo flow
            steps:
              - link: bar
        """

    with pytest.raises(UnresolvedFlowException):
        flows_from_str(flow_config)


def test_validation_fails_for_a_linked_pattern():
    flow_config = """
        flows:
          foo:
            description: foo flow
            steps:
              - link: pattern_correction
        """

    with pytest.raises(ReferenceToPatternException):
        flows_from_str_including_defaults(flow_config)


def test_validation_fails_for_a_called_pattern():
    flow_config = """
        flows:
          foo:
            description: foo flow
            steps:
              - call: pattern_correction
        """

    with pytest.raises(ReferenceToPatternException):
        flows_from_str_including_defaults(flow_config)


def test_validation_fails_for_pattern_with_a_link_step():
    flow_config = """
        flows:
          foo:
            description: foo flow
            steps:
              - action: action_listen

          pattern_correction:
            description: pattern correction
            steps:
              - link: foo
        """

    with pytest.raises(PatternReferencedFlowException):
        flows_from_str(flow_config)


def test_validation_fails_for_pattern_with_a_call_step():
    flow_config = """
        flows:
          foo:
            description: foo flow
            steps:
              - action: action_listen

          pattern_correction:
            description: pattern correction
            steps:
              - call: foo
        """

    with pytest.raises(PatternReferencedFlowException):
        flows_from_str(flow_config)


def test_validate_step_ids_are_unique_fails_for_duplicate_ids():
    flow_config = """
        flows:
          foo:
            description: foo flow
            steps:
              - id: "1"
                action: action_listen
              - id: "1"
                action: utter_greet
        """

    with pytest.raises(DuplicatedStepIdException):
        flows_from_str(flow_config)


def test_validation_fails_slot_name_does_not_adhere_to_pattern():
    flow_config = """
        flows:
          abc:
            description: test flow
            steps:
              - collect: $welcome
        """

    with pytest.raises(SlotNamingException):
        flows_from_str(flow_config)


@pytest.mark.parametrize("flow_id", ["abc def", "abcü", "abc+def", "/abc", "abc/def"])
def test_validation_fails_flow_id_does_not_adhere_to_pattern(flow_id: str):
    flow_config = f"""
        flows:
          {flow_id}:
            description: test flow
            steps:
              - action: welcome
        """

    with pytest.raises(FlowIdNamingException):
        flows_from_str(flow_config)


@pytest.mark.parametrize(
    "flow_id", ["abcdef", "abc-def", "abc_def", "_abc_def", "_abc", "01_abc"]
)
def test_validation_flow_id_passes_validation(flow_id: str):
    flow_config = f"""
        flows:
          {flow_id}:
            description: test flow
            steps:
              - action: welcome
        """

    flows = flows_from_str(flow_config)
    assert flows.underlying_flows[0].id == flow_id


def test_validation_linking_to_a_pattern_human_handoff():
    flow_config = f"""
        flows:
          test_flow:
            description: test flow
            steps:
              - action: welcome
              - link: {RASA_PATTERN_HUMAN_HANDOFF}
        """

    flows = flows_from_str_including_defaults(flow_config)
    assert isinstance(flows.underlying_flows[0].steps[1], LinkFlowStep)
    assert flows.underlying_flows[0].steps[1].link == RASA_PATTERN_HUMAN_HANDOFF


@pytest.mark.parametrize("linked_flow", ["pattern_chitchat", "pattern_internal_error"])
def test_validation_fails_pattern_linking_to_a_pattern(linked_flow: str):
    flow_config = f"""
        flows:
          pattern_test_pattern:
            description: test pattern
            steps:
              - action: welcome
              - link: {linked_flow}
        """

    with pytest.raises(ReferenceToPatternException):
        flows_from_str_including_defaults(flow_config)


def test_validation_pattern_linking_to_a_pattern_human_handoff():
    flow_config = f"""
        flows:
          pattern_test_pattern:
            description: test pattern
            steps:
              - action: welcome
              - link: {RASA_PATTERN_HUMAN_HANDOFF}
        """

    flows = flows_from_str_including_defaults(flow_config)
    assert isinstance(flows.underlying_flows[0].steps[1], LinkFlowStep)
    assert flows.underlying_flows[0].steps[1].link == RASA_PATTERN_HUMAN_HANDOFF

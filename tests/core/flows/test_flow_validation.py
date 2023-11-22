import pytest

from rasa.shared.core.flows.steps.constants import (
    START_STEP,
    END_STEP,
    CONTINUE_STEP_PREFIX,
)
from rasa.shared.core.flows.yaml_flows_io import flows_from_str
from rasa.shared.core.flows.validation import (
    EmptyFlowException,
    EmptyStepSequenceException,
    MissingElseBranchException,
    UnreachableFlowStepException,
    MissingNextLinkException,
    ReservedFlowStepIdException,
    NoNextAllowedForLinkException,
    UnresolvedFlowStepIdException,
    DuplicateNLUTriggerException,
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
            nlu_trigger:
              - intent: foo
            steps:
              - action: utter_welcome
          bar:
            nlu_trigger:
              - intent: foo
            steps:
              - action: utter_welcome
        """

    with pytest.raises(DuplicateNLUTriggerException):
        flows_from_str(flow_config)

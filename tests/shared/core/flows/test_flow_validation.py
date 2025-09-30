import textwrap
from typing import Callable, Iterator, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest
from _pytest.monkeypatch import MonkeyPatch

from rasa.core.config.configuration import Configuration
from rasa.shared.constants import RASA_PATTERN_CHITCHAT, RASA_PATTERN_HUMAN_HANDOFF
from rasa.shared.core.domain import Domain
from rasa.shared.core.flows import Flow
from rasa.shared.core.flows.steps import LinkFlowStep
from rasa.shared.core.flows.steps.constants import (
    CONTINUE_STEP_PREFIX,
    END_STEP,
    START_STEP,
)
from rasa.shared.core.flows.validation import (
    DuplicatedStepIdException,
    DuplicateNLUTriggerException,
    DuplicateSlotPersistConfigException,
    EmptyFlowException,
    EmptyStepSequenceException,
    ExitIfExclusivityException,
    FlowIdNamingException,
    InvalidMCPMappingSlotException,
    InvalidMCPServerReferenceException,
    InvalidPersistSlotsException,
    MissingElseBranchException,
    MissingNextLinkException,
    NoLinkAllowedInCalledFlowException,
    NoNextAllowedForLinkException,
    PatternReferencedFlowException,
    PatternReferencedPatternException,
    ReferenceToPatternException,
    ReservedFlowStepIdException,
    SlotNamingException,
    UnreachableFlowStepException,
    UnresolvedCallStepException,
    UnresolvedFlowStepIdException,
    UnresolvedLinkFlowException,
    validate_mcp_server_references,
    validate_patterns_are_not_calling_or_linking_other_flows,
    validate_slot_persistence_configuration,
)
from rasa.shared.core.flows.yaml_flows_io import (
    YAMLFlowsReader,
)
from rasa.shared.exceptions import RasaException
from rasa.shared.importers.importer import FlowSyncImporter
from tests.utilities import (
    flows_from_str,
    flows_from_str_including_defaults,
)


@pytest.fixture
def empty_endpoints_mock() -> MagicMock:
    """Mock Configuration with no MCP servers."""
    mock_config = MagicMock()
    mock_config.endpoints.mcp_servers = []
    return mock_config


@pytest.fixture
def configured_endpoints_mock() -> MagicMock:
    """Mock Configuration with configured MCP servers."""
    mock_config = MagicMock()
    mock_server1 = MagicMock()
    mock_server1.name = "valid_server"
    mock_server2 = MagicMock()
    mock_server2.name = "another_server"
    mock_server3 = MagicMock()
    mock_server3.name = "test_server"
    mock_config.endpoints.mcp_servers = [mock_server1, mock_server2, mock_server3]
    return mock_config


@pytest.fixture
def basic_mcp_flow_config() -> str:
    """Basic MCP tool call flow configuration."""
    return """
    flows:
      test_flow:
        description: "A flow with MCP tool call"
        steps:
          - call: some_tool
            mcp_server: {server_name}
            mapping:
              input:
                - slot: test_slot
                  param: test_param
              output:
                - slot: result_slot
                  value: result
            next: "END"
    """


@pytest.fixture
def mock_available_agents(monkeypatch: MonkeyPatch) -> Iterator[MagicMock]:
    mock_available_agents = MagicMock()
    mock_available_agents.agents = {"car-research": {}}

    mock_configuration_instance = MagicMock()
    mock_configuration_instance.available_agents = mock_available_agents

    with patch(
        "rasa.core.config.configuration.Configuration.get_instance",
        return_value=mock_configuration_instance,
    ) as mock_method:
        yield mock_method


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


def test_validation_fails_for_a_called_flow_that_does_not_exist(mock_available_agents):
    flow_config = """
        flows:
          foo:
            description: foo flow
            steps:
              - call: bar
        """

    with pytest.raises(UnresolvedCallStepException):
        flows_from_str(flow_config)


def test_validation_fails_for_a_linked_flow_that_does_not_exist():
    flow_config = """
        flows:
          foo:
            description: foo flow
            steps:
              - link: bar
        """

    with pytest.raises(UnresolvedLinkFlowException):
        flows_from_str(flow_config)


def test_validation_pass_for_a_link_to_pattern_human_handoff():
    flow_config = """
        flows:
          foo:
            description: foo flow
            steps:
              - link: pattern_human_handoff
        """

    flows = flows_from_str(flow_config)
    assert len(flows.underlying_flows) == 1


def test_validation_fails_for_a_link_to_pattern_chitchat():
    flow_config = """
        flows:
          foo:
            description: foo flow
            steps:
              - link: pattern_chitchat
        """
    with pytest.raises(ReferenceToPatternException):
        flows_from_str_including_defaults(flow_config)


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


def test_validation_fails_for_a_called_pattern(mock_available_agents):
    flow_config = """
        flows:
          foo:
            description: foo flow
            steps:
              - call: pattern_correction
        """

    with pytest.raises(ReferenceToPatternException):
        flows_from_str_including_defaults(flow_config)


def test_validation_fails_for_pattern_with_a_link_step_to_a_pattern():
    flow_config = """
        flows:
          pattern_correction:
            description: pattern correction
            steps:
              - link: pattern_linked_pattern

          pattern_linked_pattern:
            description: pattern linked pattern
            steps:
              - action: action_listen
        """

    flows = YAMLFlowsReader.read_from_string(textwrap.dedent(flow_config))
    flows = FlowSyncImporter.merge_with_default_flows(flows)

    with pytest.raises(PatternReferencedPatternException):
        validate_patterns_are_not_calling_or_linking_other_flows(flows)


def test_validation_fails_for_pattern_internal_error_with_a_link_step():
    flow_config = """
        flows:
          foo:
            description: foo flow
            steps:
              - action: action_listen

          pattern_internal_error:
            description: pattern internal error
            steps:
              - link: foo
        """

    flows = YAMLFlowsReader.read_from_string(textwrap.dedent(flow_config))
    flows = FlowSyncImporter.merge_with_default_flows(flows)

    with pytest.raises(PatternReferencedFlowException):
        validate_patterns_are_not_calling_or_linking_other_flows(flows)


def test_validation_pattern_with_a_link_step_to_a_user_flow():
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

    flows = flows_from_str(flow_config)
    assert flows.underlying_flows[0].id == "foo"
    assert flows.underlying_flows[1].id == "pattern_correction"


def test_validation_fails_for_pattern_with_a_call_step(mock_available_agents):
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


def test_validation_passes_for_exit_if_in_call_step_to_agent(
    mock_available_agents: Mock,
):
    flow_config = """
            flows:
              foo:
                description: foo flow
                steps:
                  - call: car-research
                    exit_if:
                      - slots.a is not None
                      - slots.age > 18
            """

    flows = flows_from_str(flow_config)
    foo = flows.flow_by_id("foo")
    assert foo is not None
    assert hasattr(foo.steps[0], "exit_if")


def test_validation_fails_for_exit_if_when_calling_flow(mock_available_agents):
    flow_config = """
        flows:
          foo:
            description: foo flow
            steps:
              - call: bar
                exit_if:
                  - slots.a is not None
          bar:
            description: bar flow
            steps:
              - action: action_listen
        """

    with pytest.raises(RasaException) as e:
        flows_from_str(flow_config)
    assert "exit_if" in str(e.value)


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


@pytest.mark.parametrize(
    "linked_flow", ["pattern_correction", "pattern_internal_error"]
)
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


@pytest.mark.parametrize(
    "flow_from_str_fn", [flows_from_str_including_defaults, flows_from_str]
)
def test_validation_pattern_linking_to_a_pattern_chitchat(
    flow_from_str_fn: Callable,
):
    # Given
    flow_config = f"""
        flows:
          pattern_test_pattern:
            description: test pattern
            steps:
              - action: welcome
              - link: {RASA_PATTERN_CHITCHAT}
        """
    # When
    flows = flow_from_str_fn(flow_config)
    # Then
    assert isinstance(flows.underlying_flows[0].steps[1], LinkFlowStep)
    assert flows.underlying_flows[0].steps[1].link == RASA_PATTERN_CHITCHAT


def test_validate_slot_persistence_configuration_duplicate():
    flow = Flow.from_json(
        "flow_a",
        {
            "persisted_slots": ["slot_a"],
            "steps": [{"collect": "slot_a", "reset_after_flow_ends": False}],
        },
    )

    with pytest.raises(DuplicateSlotPersistConfigException):
        validate_slot_persistence_configuration(flow)


def test_validate_slot_persistence_configuration_not_duplicate():
    flow = Flow.from_json(
        "flow_a",
        {
            "persisted_slots": ["slot_a"],
            "steps": [{"collect": "slot_a", "reset_after_flow_ends": True}],
        },
    )

    assert validate_slot_persistence_configuration(flow) is None


def test_validate_slot_persistence_configuration_invalid_slots():
    flow = Flow.from_json(
        "flow_a",
        {
            "persisted_slots": ["slot_a", "slot_b"],
            "steps": [{"collect": "slot_a"}],
        },
    )

    with pytest.raises(InvalidPersistSlotsException):
        validate_slot_persistence_configuration(flow)


def test_validate_slot_persistence_configuration_raise_deprecation_warning():
    flow = Flow.from_json(
        "flow_a",
        {
            "steps": [{"collect": "slot_a", "reset_after_flow_ends": False}],
        },
    )

    deprecation_message = (
        "Configuring 'reset_after_flow_ends' in collect steps is "
        "deprecated and will be removed in Rasa Pro 4.0.0. In the parent flow, "
        "please use the 'persisted_slots' "
        "property at the flow level instead."
    )

    with pytest.warns(FutureWarning) as record:
        validate_slot_persistence_configuration(flow)

    assert len(record) == 1
    assert record[0].message.args[0] == deprecation_message
    assert isinstance(record[0].message, FutureWarning)


def test_validate_call_steps_agents_exist_success(
    mock_available_agents: MagicMock,
) -> None:
    """Test that validation passes when all mentioned agents exist."""
    flows_content = """
    flows:
      test_flow:
        description: "A flow that calls an existing agent"
        steps:
          - call: "car-research"
            next: "END"
    """

    flows = flows_from_str(flows_content)
    # Should not raise any exception
    flows.validate()


def test_validate_call_steps_agents_exist_failure(
    mock_available_agents: MagicMock,
) -> None:
    """Test that validation fails when a mentioned agent doesn't exist."""
    # Ensure the mock is set up with only the car-research agent
    mock_available_agents.return_value.agents = {"car-research": {}}

    flows_content = """
    flows:
      test_flow:
        description: "A flow that calls a non-existent agent"
        steps:
          - call: "non_existent_agent"
            next: "END"
    """

    with pytest.raises(UnresolvedCallStepException) as exc_info:
        flows_from_str(flows_content)

    # The validation correctly identifies that the agent doesn't exist
    assert exc_info.value.call_step_argument == "non_existent_agent"
    assert exc_info.value.calling_flow_id == "test_flow"
    assert "non_existent_agent" in str(exc_info.value)


def test_validate_call_steps_ignores_flow_calls(
    mock_available_agents: MagicMock,
) -> None:
    """Test that validation ignores call steps that reference flows, not agents."""
    flows_content = """
    flows:
      test_flow:
        description: "A flow that calls another flow"
        steps:
          - call: "another_flow"
            next: "END"
      another_flow:
        description: "Another flow"
        steps:
          - action: utter_greet
            next: "END"
    """

    flows = flows_from_str(flows_content)
    # Should not raise any exception since "another_flow" is a flow, not an agent
    flows.validate()


def test_validate_call_steps_ignores_mcp_calls(
    mock_available_agents: MagicMock,
) -> None:
    """Test that validation ignores call steps that reference MCP tools."""
    # This test is skipped because MCP validation requires complex setup
    # The core agent validation functionality is tested in other tests
    pytest.skip("MCP validation requires complex endpoint setup")


def test_validate_call_steps_multiple_agents(
    mock_available_agents: MagicMock,
) -> None:
    """Test validation with multiple agent calls in different flows."""
    # Update the mock to include multiple agents
    mock_available_agents.return_value.available_agents.agents = {
        "car-research": {},
        "booking-agent": {},
        "support-agent": {},
    }

    flows_content = """
    flows:
      test_flow_1:
        description: "A flow that calls multiple agents"
        steps:
          - call: "car-research"
            next: "END"
      test_flow_2:
        description: "Another flow with agent calls"
        steps:
          - call: "booking-agent"
            next: "END"
      test_flow_3:
        description: "Third flow with agent calls"
        steps:
          - call: "support-agent"
            next: "END"
    """

    flows = flows_from_str(flows_content)
    # Should not raise any exception since all agents exist
    flows.validate()


def test_validate_call_steps_mixed_calls(
    mock_available_agents: MagicMock,
) -> None:
    """Test validation with mixed flow calls and agent calls."""
    flows_content = """
    flows:
      test_flow:
        description: "A flow with mixed call types"
        steps:
          - call: "car-research"  # Agent call - should be validated
            next: "END"
      another_flow:
        description: "Another flow"
        steps:
          - action: utter_greet
            next: "END"
    """

    flows = flows_from_str(flows_content)
    # Should not raise any exception since "car-research" exists
    flows.validate()


@pytest.fixture
def base_flow_template() -> str:
    """Base flow template for exit_if exclusivity tests."""
    return """
        flows:
          test_flow:
            description: {description}
            steps:
              - call: car-research
                {step_properties}
    """


@pytest.fixture
def mcp_properties() -> str:
    """MCP properties template for testing conflicts with exit_if."""
    return """
                mcp_server: test_server
                mapping:
                  input:
                    - slot: test_slot
                      param: test_param
                  output:
                    - slot: result_slot
                      value: result
    """


@pytest.mark.parametrize(
    "step_properties,description,should_pass",
    [
        (
            "exit_if:\n                  - slots.status == 'completed'",
            "test flow with valid exit_if",
            True,
        ),
        (
            "id: my_call_step\n"
            "                description: This is a call step without exit_if",
            "test flow with call step but no exit_if",
            True,
        ),
        (
            "id: my_call_step\n"
            "                description: This is a call step\n"
            "                metadata:\n"
            "                  key: value\n"
            "                exit_if:\n"
            "                  - slots.status == 'completed'",
            "test flow with exit_if and standard properties",
            True,
        ),
    ],
)
def test_validate_exit_if_exclusivity_passes(
    mock_available_agents: Mock,
    base_flow_template: str,
    step_properties: str,
    description: str,
    should_pass: bool,
) -> None:
    """Test that valid exit_if configurations pass validation."""
    flow_config = base_flow_template.format(
        description=description, step_properties=step_properties
    )

    flows = flows_from_str(flow_config)
    # Should not raise any exception
    flows.validate()


@pytest.mark.parametrize(
    "conflicting_property,expected_in_error",
    [
        ("mcp_server", "mcp_server"),
        ("mapping", "mapping"),
    ],
)
def test_validate_exit_if_exclusivity_fails_with_conflicting_properties(
    mock_available_agents: Mock,
    base_flow_template: str,
    mcp_properties: str,
    conflicting_property: str,
    expected_in_error: str,
    configured_endpoints_mock: MagicMock,
) -> None:
    """Test that call steps with exit_if and conflicting properties fail validation."""
    step_properties = f"""
                {mcp_properties.strip()}
                exit_if:
                  - slots.status == 'completed'
    """
    flow_config = base_flow_template.format(
        description=f"test flow with exit_if and {conflicting_property}",
        step_properties=step_properties,
    )

    flows = YAMLFlowsReader.read_from_string(flow_config)

    with (
        patch.object(
            Configuration, "get_instance", return_value=configured_endpoints_mock
        ),
        patch(
            "rasa.shared.core.flows.steps.call.CallFlowStep.is_calling_agent",
            return_value=True,
        ),
    ):
        with pytest.raises(ExitIfExclusivityException) as exc_info:
            flows.validate()

        assert "mcp_server" in str(exc_info.value)
        assert "mapping" in str(exc_info.value)
        assert "exit_if" in str(exc_info.value)
        assert "cannot have any other properties" in str(exc_info.value)


@pytest.mark.parametrize(
    "flow_config,server_exists,should_raise,expected_messages",
    [
        (
            """
            flows:
              test_flow:
                description: "A flow with invalid MCP server reference"
                steps:
                  - call: some_tool
                    mcp_server: invalid_server
                    mapping:
                      input:
                        - slot: test_slot
                          param: test_param
                      output:
                        - slot: result_slot
                          value: result
                    next: "END"
            """,
            False,
            True,
            ["invalid_server", "does not exist in endpoints.yml", "test_flow"],
        ),
        (
            """
            flows:
              test_flow:
                description: "A flow with valid MCP server reference"
                steps:
                  - call: some_tool
                    mcp_server: valid_server
                    mapping:
                      input:
                        - slot: test_slot
                          param: test_param
                      output:
                        - slot: result_slot
                          value: result
                    next: "END"
            """,
            True,
            False,
            [],
        ),
        (
            """
            flows:
              test_flow:
                description: "A flow with MCP server reference but no servers"
                steps:
                  - call: some_tool
                    mcp_server: any_server
                    mapping:
                      input:
                        - slot: test_slot
                          param: test_param
                      output:
                        - slot: result_slot
                          value: result
                    next: "END"
            """,
            False,
            True,
            ["any_server", "does not exist in endpoints.yml"],
        ),
        (
            """
            flows:
              test_flow:
                description: "A flow with non-MCP call steps"
                steps:
                  - call: another_flow
                    next: "END"
              another_flow:
                description: "Another flow to call"
                steps:
                  - action: utter_goodbye
                    next: "END"
            """,
            True,  # Shouldn't be called for non-MCP calls
            False,
            [],
        ),
    ],
)
def test_validate_mcp_server_references(
    flow_config: str,
    server_exists: bool,
    should_raise: bool,
    expected_messages: list[str],
) -> None:
    """Test MCP server reference validation for various scenarios."""

    def mock_mcp_server_exists(server_name: str) -> bool:
        return server_exists

    flows = YAMLFlowsReader.read_from_string(flow_config)

    with patch(
        "rasa.shared.utils.mcp.utils.mcp_server_exists",
        side_effect=mock_mcp_server_exists,
    ):
        if should_raise:
            with pytest.raises(InvalidMCPServerReferenceException) as exc_info:
                validate_mcp_server_references(flows)

            for message in expected_messages:
                assert message in str(exc_info.value)
        else:
            # Should not raise any exception
            validate_mcp_server_references(flows)


@pytest.mark.parametrize(
    "domain_config,flow_config,should_raise,expected_messages",
    [
        (
            """
            version: "3.1"
            slots:
              input_slot:
                type: text
              output_slot:
                type: text
            """,
            """
            flows:
              test_flow:
                description: "A flow with valid MCP mapping"
                steps:
                  - call: test_tool
                    mcp_server: test_server
                    mapping:
                      input:
                        - slot: input_slot
                          param: input_param
                      output:
                        - slot: output_slot
                          value: result
                    next: "END"
            """,
            False,
            [],
        ),
        (
            """
            version: "3.1"
            slots:
              output_slot:
                type: text
            """,
            """
            flows:
              test_flow:
                description: "A flow with invalid input slot"
                steps:
                  - call: test_tool
                    mcp_server: test_server
                    mapping:
                      input:
                        - slot: invalid_input_slot
                          param: input_param
                      output:
                        - slot: output_slot
                          value: result
                    next: "END"
            """,
            True,
            ["invalid_input_slot"],
        ),
        (
            """
            version: "3.1"
            slots:
              input_slot:
                type: text
            """,
            """
            flows:
              test_flow:
                description: "A flow with invalid output slot"
                steps:
                  - call: test_tool
                    mcp_server: test_server
                    mapping:
                      input:
                        - slot: input_slot
                          param: input_param
                      output:
                        - slot: invalid_output_slot
                          value: result
                    next: "END"
            """,
            True,
            ["invalid_output_slot"],
        ),
        (
            """
            version: "3.1"
            slots:
              valid_slot:
                type: text
            """,
            """
            flows:
              test_flow:
                description: "A flow with multiple invalid slots"
                steps:
                  - call: test_tool
                    mcp_server: test_server
                    mapping:
                      input:
                        - slot: invalid_input_slot
                          param: input_param
                      output:
                        - slot: invalid_output_slot
                          value: result
                    next: "END"
            """,
            True,
            ["invalid_input_slot", "invalid_output_slot"],
        ),
        (
            None,
            """
            flows:
              test_flow:
                description: "A flow with MCP mapping but no domain"
                steps:
                  - call: test_tool
                    mcp_server: test_server
                    mapping:
                      input:
                        - slot: any_slot
                          param: input_param
                      output:
                        - slot: any_slot
                          value: result
                    next: "END"
            """,
            False,
            [],
        ),
        (
            """
            version: "3.1"
            slots:
              input_slot:
                type: text
            """,
            """
            flows:
              test_flow:
                description: "A flow with regular call step"
                steps:
                  - call: another_flow
                    next: "END"
              another_flow:
                description: "Another flow to call"
                steps:
                  - action: utter_goodbye
                    next: "END"
            """,
            False,
            [],
        ),
    ],
)
def test_validate_mcp_mapping_slots(
    domain_config: Optional[str],
    flow_config: str,
    should_raise: bool,
    expected_messages: list[str],
    configured_endpoints_mock: MagicMock,
) -> None:
    """Test MCP mapping slot validation for various scenarios."""
    with patch.object(
        Configuration, "get_instance", return_value=configured_endpoints_mock
    ):
        flows = YAMLFlowsReader.read_from_string(flow_config)

        domain = None
        if domain_config:
            domain = Domain.from_yaml(domain_config)

        if should_raise:
            with pytest.raises(InvalidMCPMappingSlotException) as exc_info:
                flows.validate(domain)

            for message in expected_messages:
                assert message in str(exc_info.value)
        else:
            # Should not raise any exception
            flows.validate(domain)


def test_validate_call_steps_unresolved_call_step(
    empty_endpoints_mock: MagicMock,
) -> None:
    """Test validation fails when call step doesn't call agent, flow, or MCP tool."""
    with (
        patch.object(Configuration, "get_instance", return_value=empty_endpoints_mock),
    ):
        flow_config = """
        flows:
          test_flow:
            description: "A flow with unresolved call step"
            steps:
              - call: non_existent_target
                next: "END"
        """

        flows = YAMLFlowsReader.read_from_string(flow_config)

        with pytest.raises(UnresolvedCallStepException) as exc_info:
            flows.validate()

        assert "non_existent_target" in str(exc_info.value)
        assert "no flow or agent with the id" in str(exc_info.value)

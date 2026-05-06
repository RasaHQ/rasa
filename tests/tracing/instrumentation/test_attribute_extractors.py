"""Tests for attribute extraction functions in tracing."""

import json
from typing import Any, Callable, Dict, List
from unittest.mock import Mock, patch

import pytest

from rasa.agents.core.types import ProtocolType
from rasa.shared.constants import (
    DEPLOYMENT_CONFIG_KEY,
    EMBEDDINGS_CONFIG_KEY,
    LLM_CONFIG_KEY,
    MODEL_CONFIG_KEY,
    MODEL_GROUP_ID_CONFIG_KEY,
    MODELS_CONFIG_KEY,
    PROVIDER_CONFIG_KEY,
    ROUTER_CONFIG_KEY,
    TEMPERATURE_CONFIG_KEY,
    TIMEOUT_CONFIG_KEY,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import DialogueStackUpdated
from rasa.shared.core.flows.flow_step_links import FlowStepLinks
from rasa.shared.core.flows.steps.call import CallFlowStep
from rasa.tracing.constants import (
    AGENT_NAME_ATTRIBUTE_NAME,
    EXECUTION_CONTEXT_ATTRIBUTE_NAME,
    LLM_MODEL_ATTRIBUTE_NAME,
    PROTOCOL_TYPE_ATTRIBUTE_NAME,
)
from rasa.tracing.instrumentation.attribute_extractors import (
    extract_attrs_for_datetime_configuration,
    extract_attrs_for_enterprise_search_invoke_llm,
    extract_attrs_for_llm_based_command_generator,
    extract_attrs_for_mcp_agent_llm_call,
    extract_attrs_for_remove_duplicated_set_slots,
    extract_call_flow_step_attributes,
    extract_embedding_config,
    extract_llm_config,
)
from rasa.utils.endpoints import EndpointConfig


@pytest.fixture
def base_call_flow_step() -> CallFlowStep:
    """Base CallFlowStep fixture with common attributes."""
    return CallFlowStep(
        idx=1,
        custom_id="test_step",
        description="Test call step",
        call="test_target",
        flow_id="test_flow",
        metadata={},
        next=FlowStepLinks(links=[]),
    )


@pytest.mark.parametrize(
    ("step_config", "expected_attrs"),
    [
        # MCPTaskAgent case - tests exit_if extraction
        (
            {
                "call": "mcp_task_agent",
                "exit_if": ["slots.selected_appointment_slot is not null"],
            },
            {
                "step_type": "CallFlowStep",
                "call_target": "mcp_task_agent",
                "exit_if_conditions": json.dumps(
                    ["slots.selected_appointment_slot is not null"], sort_keys=True
                ),
                "absent_attrs": ["mcp_server", "mapping_config"],
            },
        ),
        # MCPOpenAgent/A2AAgent case - tests basic agent call
        (
            {
                "call": "mcp_open_or_a2a_agent",
            },
            {
                "step_type": "CallFlowStep",
                "call_target": "mcp_open_or_a2a_agent",
                "absent_attrs": ["mcp_server", "mapping_config", "exit_if_conditions"],
            },
        ),
        # MCP Tool case - tests mcp_server and mapping extraction
        (
            {
                "call": "mcp_tool",
                "mcp_server": "appointment-booking",
                "mapping": {
                    "input": [
                        {
                            "param": "appointment_slot",
                            "slot": "selected_appointment_slot",
                        }
                    ],
                    "output": [
                        {
                            "slot": "appointment_confirmed",
                            "result_key": (
                                "result.structuredContent.appointment_confirmed"
                            ),
                        }
                    ],
                },
            },
            {
                "step_type": "CallFlowStep",
                "call_target": "mcp_tool",
                "mcp_server": "appointment-booking",
                "mapping_config": json.dumps(
                    {
                        "input": [
                            {
                                "param": "appointment_slot",
                                "slot": "selected_appointment_slot",
                            }
                        ],
                        "output": [
                            {
                                "slot": "appointment_confirmed",
                                "result_key": (
                                    "result.structuredContent.appointment_confirmed"
                                ),
                            }
                        ],
                    },
                    sort_keys=True,
                ),
                "absent_attrs": ["exit_if_conditions"],
            },
        ),
    ],
)
def test_call_flow_step_attributes(
    base_call_flow_step: CallFlowStep,
    step_config: Dict[str, Any],
    expected_attrs: Dict[str, Any],
) -> None:
    """Test CallFlowStep attribute extraction for different agent types."""
    # Update the base step with the specific configuration
    for key, value in step_config.items():
        setattr(base_call_flow_step, key, value)

    attrs: Dict[str, Any] = extract_call_flow_step_attributes(base_call_flow_step)

    # Check expected attributes are present
    for attr_name, expected_value in expected_attrs.items():
        if attr_name == "absent_attrs":
            continue
        assert attrs[attr_name] == expected_value

    # Check expected absent attributes are not present
    absent_attrs: List[str] = expected_attrs.get("absent_attrs", [])
    for attr_name in absent_attrs:
        assert attr_name not in attrs


def test_mapping_config_serialization() -> None:
    """Test that mapping configuration is properly serialized to JSON."""
    step: CallFlowStep = CallFlowStep(
        idx=1,
        custom_id="call_mcp_tool",
        description="Call MCP tool",
        call="mcp_tool",
        mcp_server="appointment-booking",
        mapping={
            "input": [
                {"param": "appointment_slot", "slot": "selected_appointment_slot"}
            ],
            "output": [
                {
                    "slot": "appointment_confirmed",
                    "result_key": "result.structuredContent.appointment_confirmed",
                }
            ],
        },
        flow_id="test_flow",
        metadata={},
        next=FlowStepLinks(links=[]),
    )

    attrs: Dict[str, Any] = extract_call_flow_step_attributes(step)

    # Verify mapping config is properly serialized and contains expected structure
    mapping_config: Dict[str, Any] = json.loads(attrs["mapping_config"])
    assert mapping_config["input"][0]["param"] == "appointment_slot"
    assert mapping_config["input"][0]["slot"] == "selected_appointment_slot"
    assert mapping_config["output"][0]["slot"] == "appointment_confirmed"
    assert (
        mapping_config["output"][0]["result_key"]
        == "result.structuredContent.appointment_confirmed"
    )


@pytest.mark.parametrize(
    ("component_attrs", "expected_attrs"),
    [
        # Component with public attributes
        # LLM command generators, EnterpriseSearchPolicy
        (
            {"include_date_time": True, "timezone": "America/New_York"},
            {"include_date_time": "True", "timezone": "America/New_York"},
        ),
        # Component with public attributes, different values
        (
            {"include_date_time": False, "timezone": "Europe/London"},
            {"include_date_time": "False", "timezone": "Europe/London"},
        ),
        # Component with private attributes (MCPBaseAgent)
        (
            {"_include_date_time": True, "_timezone": "Asia/Tokyo"},
            {"include_date_time": "True", "timezone": "Asia/Tokyo"},
        ),
        # Component with only include_date_time
        (
            {"include_date_time": True},
            {"include_date_time": "True"},
        ),
        # Component with only timezone
        (
            {"timezone": "UTC"},
            {"timezone": "UTC"},
        ),
        # Component with no datetime attributes
        (
            {},
            {},
        ),
    ],
)
def test_extract_attrs_for_datetime_configuration(
    component_attrs: Dict[str, Any],
    expected_attrs: Dict[str, Any],
) -> None:
    """Test datetime configuration extraction from different component types."""
    component = type("Component", (), {})()
    for attr_name, attr_value in component_attrs.items():
        setattr(component, attr_name, attr_value)

    # When
    result = extract_attrs_for_datetime_configuration(component)

    # Then
    assert result == expected_attrs


def test_extract_attrs_for_datetime_configuration_prioritizes_public_attrs() -> None:
    """Test that public attributes are preferred over private attributes."""
    # Given
    component = Mock()
    component.include_date_time = True
    component.timezone = "UTC"
    component._include_date_time = False
    component._timezone = "America/New_York"

    # When
    result = extract_attrs_for_datetime_configuration(component)

    # Then - should use public attributes
    assert result == {"include_date_time": "True", "timezone": "UTC"}


def test_extract_attrs_for_llm_based_command_generator_includes_datetime_config() -> (
    None
):
    """Test that LLMCommandGenerator extractor includes datetime configuration."""
    from rasa.dialogue_understanding.generator import LLMCommandGenerator

    # Given
    component = Mock(spec=LLMCommandGenerator)
    component.include_date_time = True
    component.timezone = "Europe/Berlin"
    component.trace_prompt_tokens = False
    component.get_default_llm_config.return_value = {"model": "test-model"}

    with (
        patch(
            "rasa.tracing.instrumentation.attribute_extractors.extract_llm_config"
        ) as mock_extract_llm,
        patch(
            "rasa.tracing.instrumentation.attribute_extractors.extract_embedding_config"
        ) as mock_extract_embedding,
    ):
        mock_extract_llm.return_value = {
            "llm_model": "test-model",
            "llm_type": "openai",
        }
        mock_extract_embedding.return_value = {
            "embeddings_model": "test-embedding-model",
            "embeddings_type": "openai",
        }

        # When
        result = extract_attrs_for_llm_based_command_generator(component, "test prompt")

        # Then
        assert "include_date_time" in result
        assert "timezone" in result
        assert result["include_date_time"] == "True"
        assert result["timezone"] == "Europe/Berlin"


def test_extract_attrs_for_enterprise_search_invoke_llm_includes_datetime_config() -> (
    None
):
    """Test that EnterpriseSearchPolicy extractor includes datetime configuration."""
    from rasa.core.policies.enterprise_search_policy import EnterpriseSearchPolicy

    # Given
    component = Mock(spec=EnterpriseSearchPolicy)
    component.include_date_time = False
    component.timezone = "America/Los_Angeles"
    component.trace_prompt_tokens = False

    with (
        patch(
            "rasa.tracing.instrumentation.attribute_extractors.extract_llm_config"
        ) as mock_extract_llm,
        patch(
            "rasa.tracing.instrumentation.attribute_extractors.extract_embedding_config"
        ) as mock_extract_embedding,
    ):
        mock_extract_llm.return_value = {
            "llm_model": "test-model",
            "llm_type": "openai",
        }
        mock_extract_embedding.return_value = {
            "embeddings_model": "test-embedding-model",
            "embeddings_type": "openai",
        }

        # When
        result = extract_attrs_for_enterprise_search_invoke_llm(
            component, "test prompt"
        )

        # Then
        assert "include_date_time" in result
        assert "timezone" in result
        assert result["include_date_time"] == "False"
        assert result["timezone"] == "America/Los_Angeles"


def test_extract_attrs_for_mcp_agent_llm_call_includes_datetime_config() -> None:
    """Test that MCP agent extractor includes datetime configuration."""
    from rasa.agents.protocol.mcp.mcp_base_agent import MCPBaseAgent
    from rasa.agents.schemas import AgentInput

    # Given
    component = Mock(spec=MCPBaseAgent)
    component._name = "test-mcp-agent"
    component.protocol_type = ProtocolType.MCP_OPEN
    component._include_date_time = True
    component._timezone = "Asia/Singapore"
    # Set up llm_client as a mock with config attribute
    component.llm_client = Mock()
    component.llm_client.config = {"model": "test-model"}
    component.build_messages_for_llm_request.return_value = [
        {"role": "user", "content": "test"}
    ]

    agent_input = Mock(spec=AgentInput)

    with (
        patch(
            "rasa.tracing.instrumentation.attribute_extractors.extract_llm_config"
        ) as mock_extract_llm,
        patch(
            "rasa.tracing.instrumentation.attribute_extractors.extend_attributes_with_prompt_tokens_length_for_mcp_agent"
        ) as mock_extend_tokens,
    ):
        mock_extract_llm.return_value = {
            "llm_model": "test-model",
            "llm_type": "openai",
        }

        # The extend function should preserve the datetime attributes that were added
        def side_effect(self, attributes, messages):
            return attributes

        mock_extend_tokens.side_effect = side_effect

        # When
        result = extract_attrs_for_mcp_agent_llm_call(component, agent_input)

        # Then
        assert "include_date_time" in result
        assert "timezone" in result
        assert result["include_date_time"] == "True"
        assert result["timezone"] == "Asia/Singapore"
        assert result["prompt_messages_count"] == 1
        assert result[AGENT_NAME_ATTRIBUTE_NAME] == "test-mcp-agent"
        assert result[EXECUTION_CONTEXT_ATTRIBUTE_NAME] == "agent"
        assert result[PROTOCOL_TYPE_ATTRIBUTE_NAME] == str(ProtocolType.MCP_OPEN)


def test_extract_attrs_for_mcp_agent_llm_call_handles_router_client_config() -> None:
    """Test MCP extractor works when llm_client.config is router-shaped."""
    from rasa.agents.protocol.mcp.mcp_base_agent import MCPBaseAgent
    from rasa.agents.schemas import AgentInput

    component = Mock(spec=MCPBaseAgent)
    component._name = "test-mcp-agent"
    component.protocol_type = ProtocolType.MCP_OPEN
    component._include_date_time = False
    component._timezone = "UTC"
    component._llm_config = {
        "id": "test-model-group",
        "models": [{"provider": "openai", "model": "gpt-4"}],
        "router": {"routing_strategy": "simple-shuffle"},
    }
    component.get_default_llm_config.return_value = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.0,
    }

    # Router client config shape has no top-level `provider`.
    component.llm_client = Mock()
    component.llm_client.config = {
        "id": "test-model-group",
        "model_list": [
            {
                "model_name": "test-model-group",
                "litellm_params": {"model": "openai/gpt-4"},
            }
        ],
        "router": {"routing_strategy": "simple-shuffle"},
    }
    component.build_messages_for_llm_request.return_value = [
        {"role": "user", "content": "test"}
    ]

    agent_input = Mock(spec=AgentInput)

    with patch(
        "rasa.tracing.instrumentation.attribute_extractors.compute_prompt_tokens_length",
        return_value=3,
    ):
        result = extract_attrs_for_mcp_agent_llm_call(component, agent_input)

    assert result["llm_type"] == "openai"
    assert result["llm_model"] == "gpt-4"
    assert result["llm_model_group_id"] == "test-model-group"
    assert result["prompt_messages_count"] == 1
    assert result["len_prompt_tokens"] == "3"


@pytest.mark.parametrize(
    ("update_json", "expected_absent", "expected_present"),
    [
        # Dictionary value with corrected_slots - should be removed
        (
            '[{"op": "add", "path": "/2", "value": {"frame_id": "H70JZZK1", "flow_id": "pattern_correction", "step_id": "START", "corrected_slots": {"recipient": "john"}, "type": "pattern_correction"}}]',  # noqa: E501
            ["corrected_slots"],
            ["pattern_correction"],
        ),
        # String value - should be preserved without raising AttributeError
        (
            '[{"op": "replace", "path": "/0/answer", "value": "The answer to your question is 42."}]',  # noqa: E501
            [],
            ["The answer to your question is 42."],
        ),
    ],
)
def test_extract_attrs_for_remove_duplicated_set_slots(
    update_json: str,
    expected_absent: List[str],
    expected_present: List[str],
) -> None:
    """Test attribute extraction for remove_duplicated_set_slots.

    Tests that:
    - Dictionary values have corrected_slots removed
    - String values are handled gracefully without raising AttributeError
    """
    # Given
    events = [DialogueStackUpdated(update=update_json)]

    # When
    result = extract_attrs_for_remove_duplicated_set_slots(events)

    # Then
    assert "resulting_events" in result
    assert "module_name" in result
    assert result["module_name"] == "command_processor"

    for absent in expected_absent:
        assert absent not in result["resulting_events"]

    for present in expected_present:
        assert present in result["resulting_events"]


_FLAT_LLM_CONFIG: Dict[str, Any] = {
    PROVIDER_CONFIG_KEY: "openai",
    MODEL_CONFIG_KEY: "gpt-5.1-2025-11-13",
    TEMPERATURE_CONFIG_KEY: 0.7,
    TIMEOUT_CONFIG_KEY: 30,
}

_SINGLE_MODEL_GROUP_LLM_CONFIG: Dict[str, Any] = {
    MODEL_GROUP_ID_CONFIG_KEY: "my-group",
    MODELS_CONFIG_KEY: [
        {
            PROVIDER_CONFIG_KEY: "openai",
            MODEL_CONFIG_KEY: "gpt-5.1-2025-11-13",
            TEMPERATURE_CONFIG_KEY: 0.5,
            TIMEOUT_CONFIG_KEY: 60,
            DEPLOYMENT_CONFIG_KEY: "my-deployment",
        }
    ],
}

_ROUTER_GROUP_WITH_OPENAI_LLM_CONFIG: Dict[str, Any] = {
    MODEL_GROUP_ID_CONFIG_KEY: "router-group",
    MODELS_CONFIG_KEY: [
        {PROVIDER_CONFIG_KEY: "openai", MODEL_CONFIG_KEY: "gpt-5.1-2025-11-13"},
        {
            PROVIDER_CONFIG_KEY: "azure",
            DEPLOYMENT_CONFIG_KEY: "az-deploy",
            MODEL_CONFIG_KEY: "gpt-5.1-2025-11-13",
        },
    ],
    ROUTER_CONFIG_KEY: {"routing_strategy": "latency-based-routing"},
}

_ROUTER_GROUP_AZURE_ONLY_LLM_CONFIG: Dict[str, Any] = {
    MODEL_GROUP_ID_CONFIG_KEY: "router-group",
    MODELS_CONFIG_KEY: [
        {
            PROVIDER_CONFIG_KEY: "azure",
            DEPLOYMENT_CONFIG_KEY: "az-deploy-1",
            MODEL_CONFIG_KEY: "gpt-5.1-2025-11-13",
        },
        {
            PROVIDER_CONFIG_KEY: "azure",
            DEPLOYMENT_CONFIG_KEY: "az-deploy-2",
            MODEL_CONFIG_KEY: "gpt-5.1-2025-11-13",
        },
    ],
    ROUTER_CONFIG_KEY: {"routing_strategy": "latency-based-routing"},
}

_FLAT_EMBEDDINGS_CONFIG: Dict[str, Any] = {
    PROVIDER_CONFIG_KEY: "openai",
    MODEL_CONFIG_KEY: "text-embedding-ada-002",
}

_SINGLE_MODEL_GROUP_EMBEDDINGS_CONFIG: Dict[str, Any] = {
    MODEL_GROUP_ID_CONFIG_KEY: "embed-group",
    MODELS_CONFIG_KEY: [
        {PROVIDER_CONFIG_KEY: "openai", MODEL_CONFIG_KEY: "text-embedding-3-small"}
    ],
}

_ROUTER_GROUP_EMBEDDINGS_CONFIG: Dict[str, Any] = {
    MODEL_GROUP_ID_CONFIG_KEY: "embed-router-group",
    MODELS_CONFIG_KEY: [
        {PROVIDER_CONFIG_KEY: "openai", MODEL_CONFIG_KEY: "text-embedding-3-small"},
        {
            PROVIDER_CONFIG_KEY: "azure",
            DEPLOYMENT_CONFIG_KEY: "az-embed-deploy",
        },
    ],
    ROUTER_CONFIG_KEY: {"routing_strategy": "latency-based-routing"},
}

_ROUTER_GROUP_NON_OPENAI_EMBEDDINGS_CONFIG: Dict[str, Any] = {
    MODEL_GROUP_ID_CONFIG_KEY: "embed-router-group",
    MODELS_CONFIG_KEY: [
        {PROVIDER_CONFIG_KEY: "cohere", MODEL_CONFIG_KEY: "embed-v4.0"},
        {
            PROVIDER_CONFIG_KEY: "azure",
            DEPLOYMENT_CONFIG_KEY: "text-embedding-ada-002",
        },
    ],
    ROUTER_CONFIG_KEY: {"routing_strategy": "latency-based-routing"},
}

_DEFAULT_LLM_CONFIG: Dict[str, Any] = {
    PROVIDER_CONFIG_KEY: "openai",
    MODEL_CONFIG_KEY: "gpt-4o-mini",
}

_DEFAULT_EMBEDDINGS_CONFIG: Dict[str, Any] = {
    PROVIDER_CONFIG_KEY: "openai",
    MODEL_CONFIG_KEY: "text-embedding-ada-002",
}


# ---------------------------------------------------------------------------
# Component factories – return real instances with the given LLM/embedding
# config so that resolve_model_client_config and combine_custom_and_default_config
# are exercised end-to-end without any patching.
# ---------------------------------------------------------------------------


def _make_command_generator(llm_config: Dict[str, Any]) -> Any:
    from rasa.dialogue_understanding.generator.single_step.compact_llm_command_generator import (  # noqa: E501
        CompactLLMCommandGenerator,
    )

    return CompactLLMCommandGenerator.create(
        config={LLM_CONFIG_KEY: llm_config},
        resource=Mock(),
        model_storage=Mock(),
        execution_context=Mock(),
    )


def _make_rephraser(llm_config: Dict[str, Any]) -> Any:
    from rasa.core.nlg.contextual_response_rephraser import ContextualResponseRephraser

    return ContextualResponseRephraser(
        endpoint_config=EndpointConfig(url="http://localhost", llm=llm_config),
        domain=Domain.empty(),
    )


def _make_enterprise_search_policy(llm_config: Dict[str, Any]) -> Any:
    from rasa.core.policies.enterprise_search_policy import EnterpriseSearchPolicy

    return EnterpriseSearchPolicy(
        config={LLM_CONFIG_KEY: llm_config},
        model_storage=Mock(),
        resource=Mock(),
        execution_context=Mock(),
    )


def _make_command_generator_with_embeddings(embeddings_config: Dict[str, Any]) -> Any:
    from rasa.dialogue_understanding.generator.constants import FLOW_RETRIEVAL_KEY
    from rasa.dialogue_understanding.generator.single_step.compact_llm_command_generator import (  # noqa: E501
        CompactLLMCommandGenerator,
    )

    return CompactLLMCommandGenerator.create(
        config={FLOW_RETRIEVAL_KEY: {EMBEDDINGS_CONFIG_KEY: embeddings_config}},
        resource=Mock(),
        model_storage=Mock(),
        execution_context=Mock(),
    )


def _make_enterprise_search_policy_with_embeddings(
    embeddings_config: Dict[str, Any],
) -> Any:
    from rasa.core.policies.enterprise_search_policy import EnterpriseSearchPolicy

    return EnterpriseSearchPolicy(
        config={EMBEDDINGS_CONFIG_KEY: embeddings_config},
        model_storage=Mock(),
        resource=Mock(),
        execution_context=Mock(),
    )


_LLM_COMPONENT_FACTORIES: List[Callable[[Dict[str, Any]], Any]] = [
    pytest.param(_make_command_generator, id="CompactLLMCommandGenerator"),
    pytest.param(_make_rephraser, id="ContextualResponseRephraser"),
    pytest.param(_make_enterprise_search_policy, id="EnterpriseSearchPolicy"),
]

_EMBEDDINGS_COMPONENT_FACTORIES: List[Callable[[Dict[str, Any]], Any]] = [
    pytest.param(
        _make_command_generator_with_embeddings, id="CompactLLMCommandGenerator"
    ),
    pytest.param(
        _make_enterprise_search_policy_with_embeddings, id="EnterpriseSearchPolicy"
    ),
]


class TestExtractLlmConfig:
    """Tests for extract_llm_config parametrized across generative components."""

    @pytest.fixture(autouse=True)
    def _no_health_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Prevent ContextualResponseRephraser from calling the live LLM health check
        during __init__ so it can be tested with any config, including router groups."""
        monkeypatch.setattr(
            "rasa.shared.utils.health_check.llm_health_check_mixin.LLMHealthCheckMixin"
            ".perform_llm_health_check",
            lambda *args, **kwargs: None,
        )

    @pytest.fixture(params=_LLM_COMPONENT_FACTORIES)
    def make_component(
        self, request: pytest.FixtureRequest
    ) -> Callable[[Dict[str, Any]], Any]:
        return request.param

    def test_flat_config(self, make_component: Callable[[Dict[str, Any]], Any]) -> None:
        """Attributes are extracted directly from a flat (non-model-group) config."""
        component = make_component(_FLAT_LLM_CONFIG)
        result = extract_llm_config(component, _DEFAULT_LLM_CONFIG)

        assert result[LLM_MODEL_ATTRIBUTE_NAME] == "gpt-5.1-2025-11-13"
        assert result["llm_type"] == "openai"
        assert result["llm_temperature"] == "0.7"
        assert result["llm_request_timeout"] == "30"
        assert "llm_is_router_group" not in result

    def test_single_model_group(
        self, make_component: Callable[[Dict[str, Any]], Any]
    ) -> None:
        """Attributes are extracted from models[0] for a single-model group."""
        component = make_component(_SINGLE_MODEL_GROUP_LLM_CONFIG)
        result = extract_llm_config(component, _DEFAULT_LLM_CONFIG)

        assert result[LLM_MODEL_ATTRIBUTE_NAME] == "gpt-5.1-2025-11-13"
        assert result["llm_type"] == "openai"
        assert result["llm_model_group_id"] == "my-group"
        assert "llm_is_router_group" not in result
        assert result["llm_temperature"] == "0.5"
        assert result["llm_request_timeout"] == "60"
        assert result["llm_engine"] == "my-deployment"

    def test_router_group_prefers_openai_model(
        self, make_component: Callable[[Dict[str, Any]], Any]
    ) -> None:
        """Router groups use the OpenAI model for llm_type/llm_model (token counting)
        and set llm_is_router_group=True."""
        component = make_component(_ROUTER_GROUP_WITH_OPENAI_LLM_CONFIG)
        result = extract_llm_config(component, _DEFAULT_LLM_CONFIG)

        assert result["llm_model_group_id"] == "router-group"
        assert result["llm_is_router_group"] == "true"
        assert result[LLM_MODEL_ATTRIBUTE_NAME] == "gpt-5.1-2025-11-13"
        assert result["llm_type"] == "openai"
        assert "llm_engine" not in result

    def test_router_group_falls_back_to_first_model_when_no_openai(
        self, make_component: Callable[[Dict[str, Any]], Any]
    ) -> None:
        """Router groups without an OpenAI model fall back to the first model."""
        component = make_component(_ROUTER_GROUP_AZURE_ONLY_LLM_CONFIG)
        result = extract_llm_config(component, _DEFAULT_LLM_CONFIG)

        assert result["llm_model_group_id"] == "router-group"
        assert result["llm_is_router_group"] == "true"
        assert result[LLM_MODEL_ATTRIBUTE_NAME] == "gpt-5.1-2025-11-13"
        assert result["llm_type"] == "azure"
        assert result["llm_engine"] == "az-deploy-1"


class TestExtractEmbeddingConfig:
    """Tests for extract_embedding_config parametrized across generative components."""

    @pytest.fixture(params=_EMBEDDINGS_COMPONENT_FACTORIES)
    def make_component(
        self, request: pytest.FixtureRequest
    ) -> Callable[[Dict[str, Any]], Any]:
        return request.param

    def test_flat_config(self, make_component: Callable[[Dict[str, Any]], Any]) -> None:
        """Attributes are extracted directly from a flat (non-model-group) config."""
        component = make_component(_FLAT_EMBEDDINGS_CONFIG)
        result = extract_embedding_config(component, _DEFAULT_EMBEDDINGS_CONFIG)

        assert result["embeddings_model"] == "text-embedding-ada-002"
        assert result["embeddings_type"] == "openai"

    def test_single_model_group(
        self, make_component: Callable[[Dict[str, Any]], Any]
    ) -> None:
        """Attributes are extracted from models[0] for a single-model group."""
        component = make_component(_SINGLE_MODEL_GROUP_EMBEDDINGS_CONFIG)
        result = extract_embedding_config(component, _DEFAULT_EMBEDDINGS_CONFIG)

        assert result["embeddings_model"] == "text-embedding-3-small"
        assert result["embeddings_type"] == "openai"
        assert result["embeddings_model_group_id"] == "embed-group"

    def test_router_group_prefers_openai_model(
        self, make_component: Callable[[Dict[str, Any]], Any]
    ) -> None:
        """Router groups use the OpenAI model for token-counting compatibility."""
        component = make_component(_ROUTER_GROUP_EMBEDDINGS_CONFIG)
        result = extract_embedding_config(component, _DEFAULT_EMBEDDINGS_CONFIG)

        assert result["embeddings_model_group_id"] == "embed-router-group"
        assert result["embeddings_model"] == "text-embedding-3-small"
        assert result["embeddings_type"] == "openai"

    def test_router_group_falls_back_to_first_model_when_no_openai(
        self, make_component: Callable[[Dict[str, Any]], Any]
    ) -> None:
        """Router groups without an OpenAI model fall back to the first model."""
        component = make_component(_ROUTER_GROUP_NON_OPENAI_EMBEDDINGS_CONFIG)
        result = extract_embedding_config(component, _DEFAULT_EMBEDDINGS_CONFIG)
        print(result)

        assert result["embeddings_model_group_id"] == "embed-router-group"
        assert result["embeddings_model"] == "embed-v4.0"
        assert result["embeddings_type"] == "cohere"

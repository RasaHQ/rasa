import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Text
from unittest.mock import AsyncMock, Mock, patch

import pytest
from _pytest.tmpdir import TempPathFactory
from pytest import MonkeyPatch

import rasa.shared.utils.io
from rasa.dialogue_understanding.commands import (
    CancelFlowCommand,
    CannotHandleCommand,
    ChangeFlowCommand,
    ChitChatAnswerCommand,
    ClarifyCommand,
    Command,
    ErrorCommand,
    HumanHandoffCommand,
    KnowledgeAnswerCommand,
    SetSlotCommand,
    SkipQuestionCommand,
    StartFlowCommand,
)
from rasa.dialogue_understanding.generator.constants import (
    FLOW_RETRIEVAL_ACTIVE_KEY,
    FLOW_RETRIEVAL_KEY,
)
from rasa.dialogue_understanding.generator.multi_step.multi_step_llm_command_generator import (  # noqa: E501
    MULTI_STEP_LLM_COMMAND_GENERATOR_CONFIG_FILE,
    MultiStepLLMCommandGenerator,
)
from rasa.dialogue_understanding.patterns.cancel import (
    FLOW_PATTERN_CANCEL,
    CancelPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames import (
    DialogueStackFrame,
    UserFlowStackFrame,
)
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import FlowStackFrameType
from rasa.dialogue_understanding.utils import set_record_commands_and_prompts
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.constants import (
    EMBEDDINGS_CONFIG_KEY,
    LLM_CONFIG_KEY,
    MODEL_GROUP_CONFIG_KEY,
    OPENAI_API_KEY_ENV_VAR,
    RASA_PATTERN_CANNOT_HANDLE_NOT_SUPPORTED,
    ROUTE_TO_CALM_SLOT,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import BotUttered, UserUttered
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.slots import TextSlot
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import ProviderClientAPIException
from rasa.shared.nlu.constants import (
    KEY_COMPONENT_NAME,
    KEY_USER_PROMPT,
    PREDICTED_COMMANDS,
    PROMPTS,
)
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.providers.llm.llm_response import LLMResponse
from tests.utilities import (
    flows_from_str,
    flows_from_str_including_defaults,
)


class TestMultiStepLLMCommandGenerator:
    """Tests for the MultiStepLLMCommandGenerator."""

    @pytest.fixture
    def command_generator(self):
        """Create an MultiStepLLMCommandGenerator."""
        return MultiStepLLMCommandGenerator.create(
            config={}, resource=Mock(), model_storage=Mock(), execution_context=Mock()
        )

    @pytest.fixture
    def flows(self) -> FlowsList:
        """Create a FlowsList."""
        return flows_from_str(
            """
            flows:
              test_flow:
                name: a test flow
                description: some test flow
                steps:
                - id: first_step
                  action: action_listen
            """
        )

    @pytest.fixture(scope="session")
    def resource(self) -> Resource:
        return Resource(uuid.uuid4().hex)

    @pytest.fixture(scope="session")
    def model_storage(self, tmp_path_factory: TempPathFactory) -> ModelStorage:
        return LocalModelStorage(tmp_path_factory.mktemp(uuid.uuid4().hex))

    @pytest.fixture
    def tracker(self):
        """Create a Tracker."""
        return DialogueStateTracker.from_events("", [])

    async def test_llm_command_generator_init_custom_handle_flow(
        self,
        model_storage: ModelStorage,
        resource: Resource,
    ) -> None:
        # When
        generator = MultiStepLLMCommandGenerator(
            {
                "prompt_templates": {
                    "handle_flows": {
                        "file_path": "data/test_prompt_templates/test_prompt.jinja2",
                    }
                },
                FLOW_RETRIEVAL_KEY: {FLOW_RETRIEVAL_ACTIVE_KEY: False},
            },
            model_storage,
            resource,
        )
        # Then
        assert generator.handle_flows_prompt.startswith("This is a test prompt.")
        assert generator.fill_slots_prompt.startswith("{% if flow_active %}\nYour")
        assert generator.flow_retrieval is None

    async def test_llm_command_generator_init_custom_fill_slots(
        self,
        model_storage: ModelStorage,
        resource: Resource,
    ) -> None:
        # When
        generator = MultiStepLLMCommandGenerator(
            {
                "prompt_templates": {
                    "fill_slots": {
                        "file_path": "data/test_prompt_templates/test_prompt.jinja2"
                    }
                },
                FLOW_RETRIEVAL_KEY: {FLOW_RETRIEVAL_ACTIVE_KEY: False},
            },
            model_storage,
            resource,
        )
        # Then
        assert generator.fill_slots_prompt.startswith("This is a test prompt.")
        assert generator.handle_flows_prompt.startswith(
            "Your task is to analyze the current situation"
        )
        assert generator.flow_retrieval is None

    async def test_llm_command_generator_init_default(
        self,
        model_storage: ModelStorage,
        resource: Resource,
    ) -> None:
        # When
        generator = MultiStepLLMCommandGenerator({}, model_storage, resource)
        # Then
        assert generator.handle_flows_prompt.startswith(
            "Your task is to analyze the current situation"
        )
        assert generator.fill_slots_prompt.startswith("{% if flow_active %}\nYour")
        assert generator.flow_retrieval is not None

    @patch(
        "rasa.dialogue_understanding.generator.llm_based_command_generator.llm_factory"
    )
    async def test_predict_commands_for_starting_flows_calls_llm_factory_correctly(
        self,
        mock_llm_factory: Mock,
        command_generator: MultiStepLLMCommandGenerator,
        llm_response_object: LLMResponse,
    ):
        """Test predict_commands_for_handling_flows calls llm correctly."""
        # Given
        expected_llm_config = {
            "model": "gpt-4-0613",
            "provider": "openai",
            "timeout": 7,
            "temperature": 0.0,
            "max_tokens": 256,
        }

        mock_llm_client = AsyncMock()
        llm_response_object.choices = ["StartFlow(test_flow)"]
        mock_llm_client.acompletion.return_value = llm_response_object
        mock_llm_factory.return_value = mock_llm_client

        # When
        await command_generator._predict_commands_for_handling_flows(
            Message(),
            DialogueStateTracker.from_events(
                "test",
                evts=[UserUttered("Hello", {"name": "greet", "confidence": 1.0})],
            ),
            FlowsList(underlying_flows=[]),
            FlowsList(underlying_flows=[]),
        )

        # Then
        mock_llm_factory.assert_called_once_with(None, expected_llm_config)

    @patch(
        "rasa.dialogue_understanding.generator.llm_based_command_generator.llm_factory"
    )
    async def test_predict_commands_for_handling_flows_calls_llm_correctly(
        self,
        mock_llm_factory: Mock,
        command_generator: MultiStepLLMCommandGenerator,
        llm_response_object: LLMResponse,
    ):
        mock_llm = AsyncMock()
        mock_llm.acompletion.return_value = llm_response_object
        mock_llm_factory.return_value = mock_llm

        await command_generator._predict_commands_for_handling_flows(
            Message(),
            DialogueStateTracker.from_events(
                "test",
                evts=[UserUttered("Hello", {"name": "greet", "confidence": 1.0})],
            ),
            FlowsList(underlying_flows=[]),
            FlowsList(underlying_flows=[]),
        )
        # Then
        mock_llm.acompletion.assert_called_once()
        args, _ = mock_llm.acompletion.call_args
        assert args[0].startswith("Your task is to analyze the current")

    ### Test fingerprint
    async def test_llm_command_generator_fingerprint_addon_diff_in_prompt_template(
        self,
        model_storage: ModelStorage,
        tmp_path: Path,
        resource: Resource,
    ) -> None:
        prompt_dir = Path(tmp_path) / "prompt"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        prompt_file = prompt_dir / "fill_slots.jinja2"
        prompt_file.write_text("This is a test prompt")

        config = {"prompt_templates": {"fill_slots": {"file_path": str(prompt_file)}}}
        generator = MultiStepLLMCommandGenerator(config, model_storage, resource)
        fingerprint_1 = generator.fingerprint_addon(config)

        prompt_file.write_text("This is a test prompt. It has been changed.")
        fingerprint_2 = generator.fingerprint_addon(config)
        assert fingerprint_1 != fingerprint_2

    async def test_llm_command_generator_fingerprint_addon_diff_in_one_prompt_template(
        self,
        model_storage: ModelStorage,
        tmp_path: Path,
        resource: Resource,
    ) -> None:
        prompt_dir = Path(tmp_path) / "prompt"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        prompt_file_dynamic = prompt_dir / "fill_slots.jinja2"
        prompt_file_dynamic.write_text("This is a dynamic prompt")
        prompt_file_static = prompt_dir / "handle_flow.jinja2"
        prompt_file_static.write_text("This is a static prompt")

        config = {
            "prompt_templates": {
                "fill_slots": {"file_path": str(prompt_file_dynamic)},
                "handle_flow": {"file_path": str(prompt_file_static)},
            }
        }
        generator = MultiStepLLMCommandGenerator(config, model_storage, resource)
        fingerprint_1 = generator.fingerprint_addon(config)

        prompt_file_dynamic.write_text("This is a test prompt. It has been changed.")
        fingerprint_2 = generator.fingerprint_addon(config)
        assert fingerprint_1 != fingerprint_2

    async def test_llm_command_generator_fingerprint_addon_no_diff_in_prompt_template(
        self,
        model_storage: ModelStorage,
        tmp_path: Path,
        resource: Resource,
    ) -> None:
        prompt_dir = Path(tmp_path) / "prompt"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        prompt_file = prompt_dir / "fill_slots.jinja2"
        prompt_file.write_text("This is a test prompt")

        config = {"prompt_templates": {"fill_slots": {"file_path": str(prompt_file)}}}
        generator = MultiStepLLMCommandGenerator(config, model_storage, resource)

        fingerprint_1 = generator.fingerprint_addon(config)
        fingerprint_2 = generator.fingerprint_addon(config)
        assert fingerprint_1 is not None
        assert fingerprint_1 == fingerprint_2

    async def test_llm_command_generator_fingerprint_addon_default_prompt_template(
        self,
        model_storage: ModelStorage,
        resource: Resource,
    ) -> None:
        generator = MultiStepLLMCommandGenerator({}, model_storage, resource)
        fingerprint_1 = generator.fingerprint_addon({})
        fingerprint_2 = generator.fingerprint_addon({})
        assert fingerprint_1 is not None
        assert fingerprint_1 == fingerprint_2

    def test_load_with_default_prompt(
        self,
        model_storage: ModelStorage,
        flows: FlowsList,
        resource: Resource,
        monkeypatch: MonkeyPatch,
    ):
        # Given
        # Set an environment variable
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "my key")

        generator = MultiStepLLMCommandGenerator(
            {FLOW_RETRIEVAL_KEY: {FLOW_RETRIEVAL_ACTIVE_KEY: False}},
            model_storage,
            resource,
        )
        resource = generator.train(None, flows, Mock())
        # When
        loaded = MultiStepLLMCommandGenerator.load({}, model_storage, resource, Mock())
        # Then
        assert loaded.handle_flows_prompt.startswith(
            "Your task is to analyze the current situation"
        )
        assert loaded.fill_slots_prompt.startswith("{% if flow_active %}\nYour")

    async def test_llm_command_generator_load_prompt_from_model_storage(
        self, model_storage: ModelStorage, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        # Set an environment variable
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "my key")
        # Create and write prompt file.
        prompt_dir = Path(tmp_path) / "prompt"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        prompt_file = prompt_dir / "fill_slots.jinja2"
        prompt_file.write_text("This is a custom prompt")

        # Add the prompt file path to the config.
        config = {"prompt_templates": {"fill_slots": {"file_path": str(prompt_file)}}}

        # Persist the prompt file to the model storage.
        resource = Resource("llmcmdgen")
        generator = MultiStepLLMCommandGenerator(config, model_storage, resource)
        generator.persist()

        # Test loading the prompt from the model storage.
        # Case 1: No prompt in the config.
        loaded = MultiStepLLMCommandGenerator.load({}, model_storage, resource, Mock())
        assert loaded.fill_slots_prompt == "This is a custom prompt"
        assert loaded.config["prompt_templates"] == {}

        # Case 2: Specifying a invalid prompt path in the config.
        loaded = MultiStepLLMCommandGenerator.load(
            {"prompt_templates": {"fill_slots": {"file_path": "test_prompt.jinja2"}}},
            model_storage,
            resource,
            Mock(),
        )
        assert loaded.fill_slots_prompt == "This is a custom prompt"
        assert (
            loaded.config["prompt_templates"]["fill_slots"]["file_path"]
            == "test_prompt.jinja2"
        )

    @pytest.mark.parametrize(
        "input_action, expected_command",
        [
            (None, []),
            (
                "SetSlot(transfer_money_amount_of_money, )",
                [SetSlotCommand(name="transfer_money_amount_of_money", value=None)],
            ),
            (
                "SetSlot('transfer_money_amount_of_money', 'value')",
                [SetSlotCommand(name="transfer_money_amount_of_money", value="value")],
            ),
            ("SetSlot(flow_name, some_flow)", [StartFlowCommand(flow="some_flow")]),
            ("StartFlow(some_flow)", [StartFlowCommand(flow="some_flow")]),
            ("StartFlow('some_flow')", [StartFlowCommand(flow="some_flow")]),
            ('StartFlow("some_flow")', [StartFlowCommand(flow="some_flow")]),
            ("StartFlow(does_not_exist)", []),
            (
                "StartFlow(02_benefits_learning_days)",
                [StartFlowCommand(flow="02_benefits_learning_days")],
            ),
            ("CancelFlow()", [CancelFlowCommand()]),
            ("ChitChat()", [ChitChatAnswerCommand()]),
            ("SkipQuestion()", [SkipQuestionCommand()]),
            ("SearchAndReply()", [KnowledgeAnswerCommand()]),
            ("HumanHandoff()", [HumanHandoffCommand()]),
            (
                "Here is a list of commands:\nSetSlot(flow_name, some_flow)\n",
                [StartFlowCommand(flow="some_flow")],
            ),
            (
                """SetSlot(flow_name, some_flow)
                       SetSlot(transfer_money_amount_of_money,)""",
                [
                    StartFlowCommand(flow="some_flow"),
                    SetSlotCommand(name="transfer_money_amount_of_money", value=None),
                ],
            ),
            # Clarify of non-existent option is dropped
            ("Clarify(transfer_money)", []),
            # Clarify orders options
            (
                "Clarify(some_flow, 02_benefits_learning_days)",
                [ClarifyCommand(options=["02_benefits_learning_days", "some_flow"])],
            ),
            ("Clarify(some_flow)", [StartFlowCommand(flow="some_flow")]),
            ("Clarify(test_1, test_2, test_3, test_4, test_5, test_6)", []),
            (
                "Clarify(some_flow, test_2, test_3, test_4, test_5, test_6)",
                [StartFlowCommand(flow="some_flow")],
            ),
            (
                "Clarify(test_a, test_b, test_c, test_d, test_e)",
                [
                    ClarifyCommand(
                        options=["test_a", "test_b", "test_c", "test_d", "test_e"]
                    )
                ],
            ),
            (
                "Clarify('test_a', 'test_b', 'test_c', 'test_d', 'test_e')",
                [
                    ClarifyCommand(
                        options=["test_a", "test_b", "test_c", "test_d", "test_e"]
                    )
                ],
            ),
            (
                'Clarify("test_a", "test_b", "test_c", "test_d", "test_e")',
                [
                    ClarifyCommand(
                        options=["test_a", "test_b", "test_c", "test_d", "test_e"]
                    )
                ],
            ),
            ("ChangeFlow()", [ChangeFlowCommand()]),
            (
                "CannotHandle()",
                [CannotHandleCommand(RASA_PATTERN_CANNOT_HANDLE_NOT_SUPPORTED)],
            ),
        ],
    )
    def test_parse_commands_identifies_correct_command(
        self,
        input_action: Optional[str],
        expected_command: Command,
    ):
        """Test that parse_commands identifies the correct commands."""
        # When
        test_flows = flows_from_str(
            """
            flows:
              some_flow:
                description: some description
                steps:
                - id: first_step
                  collect: test_slot
              02_benefits_learning_days:
                description: some foo
                steps:
                - id: some_id
                  collect: some_slot
              test_a:
                description: some description
                steps:
                - id: first_step
                  collect: test_slot
              test_b:
                description: some description
                steps:
                - id: first_step
                  collect: test_slot
              test_c:
                description: some description
                steps:
                - id: first_step
                  collect: test_slot
              test_d:
                description: some description
                steps:
                - id: first_step
                  collect: test_slot
              test_e:
                description: some description
                steps:
                - id: first_step
                  collect: test_slot
            """
        )
        tracker = DialogueStateTracker.from_events(
            "test",
            evts=[UserUttered("Hello", {"name": "greet", "confidence": 1.0})],
        )
        parsed_commands = MultiStepLLMCommandGenerator.parse_commands(
            input_action, tracker, test_flows
        )
        # Then
        assert parsed_commands == expected_command

    def test_call_step_flow_and_slot_names_in_prepare_inputs(
        self,
        command_generator: MultiStepLLMCommandGenerator,
    ):
        """Test that template rendering receives the correct template string."""
        # Given
        test_message = Message.build(text="some message")
        test_slot = TextSlot(
            name="test_slot",
            mappings=[{}],
            initial_value=None,
            influence_conversation=False,
        )
        test_slot_2 = TextSlot(
            name="test_slot_2",
            mappings=[{}],
            initial_value=None,
            influence_conversation=False,
        )
        test_slot_3 = TextSlot(
            name="test_slot_3",
            mappings=[{}],
            initial_value=None,
            influence_conversation=False,
        )
        test_tracker = DialogueStateTracker.from_events(
            sender_id="test",
            evts=[UserUttered("Hello"), BotUttered("Hi")],
            slots=[test_slot, test_slot_2, test_slot_3],
        )
        stack = DialogueStack.from_dict(
            [
                {
                    "type": "flow",
                    "flow_id": "test_flow",
                    "step_id": "call_step",
                    "frame_id": "some-frame-id",
                },
                {
                    "type": "flow",
                    "frame_type": "call",
                    "frame_id": "some-other-id",
                    "step_id": "first_step",
                    "flow_id": "called_flow",
                },
            ]
        )
        test_flows = flows_from_str(
            """
            flows:
              test_flow:
                description: some description
                steps:
                - id: call_step
                  call: called_flow
                  next: "END"
              called_flow:
                if: False
                description: a flows that's called
                steps:
                - id: first_step
                  collect: test_slot
                - call: called_flow_level2
                  next: "END"
              called_flow_level2:
                description: called within another called flow
                steps:
                - collect: test_slot_2
                  next: "END"
              unrelated_flow:
                description: a flows that's not called
                steps:
                - collect: test_slot_3
            """
        )
        available_test_flows = test_flows.exclude_link_only_flows()
        test_tracker.update_stack(stack)

        # When
        variables_to_render = command_generator._prepare_inputs(
            message=test_message,
            tracker=test_tracker,
            available_flows=available_test_flows,
            all_flows=test_flows,
        )

        # Then
        # make sure non-startable flow isn't there
        flow_names_to_render = [
            flow["name"] for flow in variables_to_render["available_flows"]
        ]
        assert "called_flow" not in flow_names_to_render
        assert "called_flow_level2" in flow_names_to_render
        assert "test_flow" in flow_names_to_render
        assert "unrelated_flow" in flow_names_to_render

        # make sure it looks like we are in the calling flow
        assert "test_flow" == variables_to_render["current_flow"]

        # make sure the slot from the called flow is available in the template
        slot_names_to_render = [
            slot["name"] for slot in variables_to_render["flow_slots"]
        ]
        assert "test_slot" in slot_names_to_render
        assert "test_slot_2" in slot_names_to_render
        assert "test_slot_3" not in slot_names_to_render

    def test_call_step_flow_and_slot_names_in_prepare_inputs_for_single_flow(
        self,
        command_generator: MultiStepLLMCommandGenerator,
    ):
        """Test that template rendering receives the correct template string."""
        # Given
        test_message = Message.build(text="some message")
        test_slot = TextSlot(
            name="test_slot",
            mappings=[{}],
            initial_value=None,
            influence_conversation=False,
        )
        test_slot_2 = TextSlot(
            name="test_slot_2",
            mappings=[{}],
            initial_value=None,
            influence_conversation=False,
        )
        test_slot_3 = TextSlot(
            name="test_slot_3",
            mappings=[{}],
            initial_value=None,
            influence_conversation=False,
        )
        test_tracker = DialogueStateTracker.from_events(
            sender_id="test",
            evts=[UserUttered("Hello"), BotUttered("Hi")],
            slots=[test_slot, test_slot_2, test_slot_3],
        )
        stack = DialogueStack.from_dict(
            [
                {
                    "type": "flow",
                    "flow_id": "test_flow",
                    "step_id": "call_step",
                    "frame_id": "some-frame-id",
                },
                {
                    "type": "flow",
                    "frame_type": "call",
                    "frame_id": "some-other-id",
                    "step_id": "first_step",
                    "flow_id": "called_flow",
                },
            ]
        )
        test_flows = flows_from_str(
            """
            flows:
              test_flow:
                description: some description
                steps:
                - id: call_step
                  call: called_flow
                  next: "END"
              called_flow:
                if: False
                description: a flows that's called
                steps:
                - id: first_step
                  collect: test_slot
                - call: called_flow_level2
                  next: "END"
              called_flow_level2:
                description: called within another called flow
                steps:
                - collect: test_slot_2
                  next: "END"
              unrelated_flow:
                description: a flows that's not called
                steps:
                - collect: test_slot_3
            """
        )
        test_flows = [
            flow for flow in test_flows.underlying_flows if flow.id == "test_flow"
        ]
        test_flow = test_flows[0]
        test_tracker.update_stack(stack)

        # When
        variables_to_render = command_generator._prepare_inputs_for_single_flow(
            message=test_message,
            tracker=test_tracker,
            flow=test_flow,
        )

        # Then
        # make sure it looks like we are in the calling flow
        assert "test_flow" == variables_to_render["current_flow"]

        # make sure the slot from the called flow is available in the template
        slot_names_to_render = [
            slot["name"] for slot in variables_to_render["flow_slots"]
        ]
        assert "test_slot" in slot_names_to_render
        assert "test_slot_2" in slot_names_to_render
        assert "test_slot_3" not in slot_names_to_render

    @pytest.mark.parametrize(
        "input_commands, output_commands",
        [
            (
                [
                    StartFlowCommand("test_flow"),
                    SetSlotCommand("test_slot", "test_value"),
                ],
                [
                    StartFlowCommand("test_flow"),
                    SetSlotCommand("test_slot", "test_value"),
                ],
            ),
            (
                [
                    StartFlowCommand("test_flow"),
                    SetSlotCommand("test_slot", "test_value"),
                    CannotHandleCommand(RASA_PATTERN_CANNOT_HANDLE_NOT_SUPPORTED),
                ],
                [
                    StartFlowCommand("test_flow"),
                    SetSlotCommand("test_slot", "test_value"),
                ],
            ),
            (
                [
                    StartFlowCommand("test_flow"),
                    SetSlotCommand("test_slot", "test_value"),
                    CannotHandleCommand(RASA_PATTERN_CANNOT_HANDLE_NOT_SUPPORTED),
                    CannotHandleCommand(RASA_PATTERN_CANNOT_HANDLE_NOT_SUPPORTED),
                ],
                [
                    StartFlowCommand("test_flow"),
                    SetSlotCommand("test_slot", "test_value"),
                ],
            ),
            (
                [
                    CannotHandleCommand(RASA_PATTERN_CANNOT_HANDLE_NOT_SUPPORTED),
                    CannotHandleCommand(RASA_PATTERN_CANNOT_HANDLE_NOT_SUPPORTED),
                ],
                [CannotHandleCommand(RASA_PATTERN_CANNOT_HANDLE_NOT_SUPPORTED)],
            ),
            (
                [CannotHandleCommand(RASA_PATTERN_CANNOT_HANDLE_NOT_SUPPORTED)],
                [CannotHandleCommand(RASA_PATTERN_CANNOT_HANDLE_NOT_SUPPORTED)],
            ),
        ],
    )
    def test_clean_up_commands(
        self, input_commands: List[Command], output_commands: List[Command]
    ):
        # When
        result_commands = MultiStepLLMCommandGenerator._clean_up_commands(
            input_commands
        )
        # Then
        assert result_commands == output_commands

    @pytest.mark.parametrize(
        "stack_frames, available_flows_ids, current_flow_id,"
        "top_flow_is_pattern, top_user_flow_id",
        [
            (
                # Flow that can only be started via a specific link
                [
                    UserFlowStackFrame(
                        frame_id="a",
                        flow_id="link_initiated_flow",
                        frame_type=FlowStackFrameType.LINK,
                    ),
                ],
                ["flow_using_link", "calling_flow"],
                "link_initiated_flow",
                False,
                None,
            ),
            # Scenario with a calling flow and a called flow
            (
                [
                    UserFlowStackFrame(
                        frame_id="c",
                        flow_id="calling_flow",
                        frame_type=FlowStackFrameType.REGULAR,
                    ),
                    UserFlowStackFrame(
                        frame_id="b",
                        flow_id="called_flow",
                        frame_type=FlowStackFrameType.CALL,
                    ),
                ],
                ["flow_using_link", "calling_flow"],
                "calling_flow",
                False,
                None,
            ),
            # A frame that cannot initiate a flow
            (
                [CancelPatternFlowStackFrame()],
                ["flow_using_link", "calling_flow"],
                FLOW_PATTERN_CANCEL,
                True,
                None,
            ),
            (
                [
                    UserFlowStackFrame(
                        frame_id="d",
                        flow_id="calling_flow",
                        frame_type=FlowStackFrameType.REGULAR,
                    ),
                    CancelPatternFlowStackFrame(),
                ],
                ["flow_using_link", "calling_flow"],
                FLOW_PATTERN_CANCEL,
                True,
                "calling_flow",
            ),
        ],
    )
    def test_prepare_inputs_with_with_different_stack_frames(
        self,
        stack_frames: List[DialogueStackFrame],
        available_flows_ids: List[Text],
        current_flow_id: Text,
        top_flow_is_pattern: bool,
        top_user_flow_id: Text,
    ):
        # Given
        all_flows = flows_from_str_including_defaults(
            """
            flows:
                link_initiated_flow:
                    if: False
                    name: Link-Initiated Flow
                    description: This flow can only be started via a link.
                    steps:
                        - id: step_a
                          action: action_listen
                flow_using_link:
                    name: Flow Using Link
                    description: This flow uses the Link-Initiated Flow.
                    steps:
                        - id: step_b
                          link: link_initiated_flow
                called_flow:
                    if: False
                    name: called Flow
                    description: This is a called flow.
                    steps:
                        - id: step_c
                          action: action_listen
                calling_flow:
                    name: calling Flow
                    description: This flow calls another flow.
                    steps:
                        - id: step_d
                          call: called_flow
                          next: "END"
            """
        )
        test_available_flows = all_flows.user_flows.exclude_link_only_flows()

        mock_message = Mock()
        config = {"flow_retrieval": {"active": False}}
        command_generator = MultiStepLLMCommandGenerator.create(
            config=config,
            resource=Mock(),
            model_storage=Mock(),
            execution_context=Mock(),
        )

        # Initialize a tracker with the stack frames prepared for the test
        tracker = DialogueStateTracker.from_events(
            sender_id="test_user",
            evts=[],
        )
        stack = DialogueStack.empty()
        for frame in stack_frames:
            stack.push(frame)
        tracker.update_stack(stack)

        # When / Then
        # Validate input preparation with different stack configurations
        input_results = command_generator._prepare_inputs(
            message=mock_message,
            tracker=tracker,
            available_flows=test_available_flows,
            all_flows=all_flows,
        )

        assert [
            flow["name"] for flow in input_results["available_flows"]
        ] == available_flows_ids
        assert input_results["current_flow"] == current_flow_id
        assert input_results["top_flow_is_pattern"] == top_flow_is_pattern
        assert input_results["top_user_flow"] == top_user_flow_id

    @patch(
        "rasa.dialogue_understanding.generator"
        ".multi_step.multi_step_llm_command_generator.MultiStepLLMCommandGenerator"
        "._predict_commands_for_active_flow"
    )
    @patch(
        "rasa.dialogue_understanding.generator"
        ".multi_step.multi_step_llm_command_generator.MultiStepLLMCommandGenerator"
        "._predict_commands_for_handling_flows"
    )
    @patch(
        "rasa.dialogue_understanding.generator"
        ".multi_step.multi_step_llm_command_generator.MultiStepLLMCommandGenerator"
        "._predict_commands_for_newly_started_flows"
    )
    @patch(
        "rasa.dialogue_understanding.generator"
        ".multi_step.multi_step_llm_command_generator.MultiStepLLMCommandGenerator"
        ".filter_flows"
    )
    async def test_predict_commands_methods_called_with_correct_parameters(
        self,
        mock_filter_flows: Mock,
        mock_predict_new_flows: Mock,
        mock_predict_handling_flows: Mock,
        mock_predict_active_flows: Mock,
    ):
        # Given
        all_flows = flows_from_str_including_defaults(
            """
            flows:
                link_initiated_flow:
                    if: False
                    name: Link-Initiated Flow
                    description: This flow can only be started via a link.
                    steps:
                        - id: step_a
                          action: action_listen
                flow_using_link:
                    name: Flow Using Link
                    description: This flow uses the Link-Initiated Flow.
                    steps:
                        - id: step_b
                          link: link_initiated_flow
                called_flow:
                    if: False
                    name: Called Flow
                    description: |
                        This is a flow that is typically
                        triggered by another flow.
                    steps:
                        - id: step_c
                          action: action_listen
                calling_flow:
                    name: Calling Flow
                    description: This flow triggers the Called Flow.
                    steps:
                        - id: step_d
                          call: called_flow
                          next: "END"
            """
        )
        test_available_flows = all_flows.user_flows.exclude_link_only_flows()

        mock_message = Mock(spec=Message)
        mock_tracker = Mock(spec=DialogueStateTracker, has_active_flow=True)
        mock_domain = Mock(spec=Domain)
        command_generator = MultiStepLLMCommandGenerator.create(
            config={"flow_retrieval": {"active": False}},
            resource=Mock(),
            model_storage=Mock(),
            execution_context=Mock(),
        )

        # Mock behavior setup
        mock_filter_flows.return_value = test_available_flows
        mock_predict_active_flows.return_value = [ChangeFlowCommand()]
        mock_predict_handling_flows.return_value = [StartFlowCommand("calling_flow")]
        mock_predict_new_flows.return_value = [
            SetSlotCommand(name="test_slot", value="test_value")
        ]

        # When
        await command_generator.predict_commands(
            mock_message, all_flows, mock_tracker, domain=mock_domain
        )

        # Then
        mock_predict_active_flows.assert_called_once_with(
            mock_message,
            mock_tracker,
            available_flows=test_available_flows,
            all_flows=all_flows,
        )
        mock_predict_handling_flows.assert_called_once_with(
            mock_message,
            mock_tracker,
            available_flows=test_available_flows,
            all_flows=all_flows,
        )
        mock_predict_new_flows.assert_called_once_with(
            mock_message,
            mock_tracker,
            newly_started_flows=FlowsList([all_flows.flow_by_id("calling_flow")]),
            all_flows=all_flows,
        )

    @patch("rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval")
    def test_train_with_no_flows(
        self,
        mock_flow_retrieval: Mock,
        model_storage: ModelStorage,
    ):
        # Given
        resource = Resource("llmcmdgen")
        command_generator = MultiStepLLMCommandGenerator.create(
            config={"flow_retrieval": {"active": False}},
            resource=resource,
            model_storage=model_storage,
            execution_context=Mock(),
        )
        # When
        command_generator.train(Mock(), FlowsList(underlying_flows=[]), Mock())
        # Then
        assert mock_flow_retrieval.populate.call_count == 0

    def test_multi_step_llm_command_generator_persist_config(
        self,
        model_storage: LocalModelStorage,
        resource: Resource,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class MockAvailableEndpoints:
            @staticmethod
            def get_instance():
                return MockAvailableEndpoints()

            def __init__(self):
                self.model_groups = [
                    {
                        "id": "model_group_id",
                        "models": [{"provider": "openai", "model": "gpt-4"}],
                    }
                ]

        mock_endpoints = MockAvailableEndpoints()
        monkeypatch.setattr("rasa.shared.utils.llm.AvailableEndpoints", mock_endpoints)

        config = {LLM_CONFIG_KEY: {MODEL_GROUP_CONFIG_KEY: "model_group_id"}}
        generator = MultiStepLLMCommandGenerator(config, model_storage, resource)

        # Ensure the config is resolved
        assert generator.config[LLM_CONFIG_KEY] == {
            "id": "model_group_id",
            "models": [{"provider": "openai", "model": "gpt-4"}],
        }

        # Persist the generator
        generator.persist()

        # Check that the persisted config is equal to our config
        with model_storage.read_from(resource) as path:
            persisted_config = rasa.shared.utils.io.read_json_file(
                path / MULTI_STEP_LLM_COMMAND_GENERATOR_CONFIG_FILE
            )

        assert persisted_config[LLM_CONFIG_KEY] == {
            "id": "model_group_id",
            "models": [{"provider": "openai", "model": "gpt-4"}],
        }

    @pytest.mark.parametrize(
        "config, expected_llm_config, expected_flow_retrieval_embedding_config",
        [
            (
                {
                    LLM_CONFIG_KEY: {"provider": "openai", "model": "gpt-4"},
                    FLOW_RETRIEVAL_KEY: {FLOW_RETRIEVAL_ACTIVE_KEY: False},
                },
                {"provider": "openai", "model": "gpt-4"},
                None,
            ),
            (
                {
                    "user_input": {"max_characters": -1},
                    FLOW_RETRIEVAL_KEY: {FLOW_RETRIEVAL_ACTIVE_KEY: False},
                },
                None,
                None,
            ),
            (
                {
                    LLM_CONFIG_KEY: {MODEL_GROUP_CONFIG_KEY: "openai_gpt-4"},
                    FLOW_RETRIEVAL_KEY: {
                        EMBEDDINGS_CONFIG_KEY: {
                            MODEL_GROUP_CONFIG_KEY: "openai_embedding"
                        }
                    },
                },
                {
                    "id": "openai_gpt-4",
                    "models": [{"provider": "openai", "model": "gpt-4"}],
                },
                {
                    "id": "openai_embedding",
                    "models": [
                        {"model": "text-embedding-ada-002", "provider": "openai"}
                    ],
                },
            ),
            (
                {
                    LLM_CONFIG_KEY: {MODEL_GROUP_CONFIG_KEY: "openai_gpt-4"},
                    FLOW_RETRIEVAL_KEY: {
                        EMBEDDINGS_CONFIG_KEY: {
                            "provider": "openai",
                            "model": "text-embedding-ada-002",
                        }
                    },
                },
                {
                    "id": "openai_gpt-4",
                    "models": [{"provider": "openai", "model": "gpt-4"}],
                },
                {"provider": "openai", "model": "text-embedding-ada-002"},
            ),
        ],
    )
    def test_multi_step_llm_command_generator_init_with_different_llm_configs(
        self,
        config: Optional[Dict[str, Any]],
        expected_llm_config: Optional[Dict[str, Any]],
        expected_flow_retrieval_embedding_config: Optional[Dict[str, Any]],
        model_storage: ModelStorage,
        resource: Resource,
        monkeypatch,
    ) -> None:
        class MockAvailableEndpoints:
            @staticmethod
            def get_instance():
                return MockAvailableEndpoints()

            def __init__(self):
                self.model_groups = [
                    {
                        "id": "openai_gpt-4",
                        "models": [{"provider": "openai", "model": "gpt-4"}],
                    },
                    {
                        "id": "openai_embedding",
                        "models": [
                            {"provider": "openai", "model": "text-embedding-ada-002"}
                        ],
                    },
                ]

        mock_endpoints = MockAvailableEndpoints()
        monkeypatch.setattr("rasa.shared.utils.llm.AvailableEndpoints", mock_endpoints)

        generator = MultiStepLLMCommandGenerator(
            config,
            model_storage,
            resource,
        )
        assert generator.config[LLM_CONFIG_KEY] == expected_llm_config

        if expected_flow_retrieval_embedding_config is None:
            assert EMBEDDINGS_CONFIG_KEY not in generator.config[FLOW_RETRIEVAL_KEY]
        else:
            assert (
                generator.config[FLOW_RETRIEVAL_KEY][EMBEDDINGS_CONFIG_KEY]
                == expected_flow_retrieval_embedding_config
            )

    @pytest.mark.parametrize(
        "config_1, model_groups_1, config_2, model_groups_2, fingerprint_differs",
        [
            (
                {"user_input": {"max_characters": 100}},
                [],
                {"user_input": {"max_characters": 200}},
                [],
                False,
            ),
            (
                {LLM_CONFIG_KEY: {MODEL_GROUP_CONFIG_KEY: "openai_gpt"}},
                [
                    {
                        "id": "openai_gpt",
                        "models": [{"provider": "openai", "model": "gpt-4"}],
                    },
                ],
                {LLM_CONFIG_KEY: {MODEL_GROUP_CONFIG_KEY: "openai_gpt"}},
                [
                    {
                        "id": "openai_gpt",
                        "models": [{"provider": "openai", "model": "gpt-3.5-turbo"}],
                    },
                ],
                True,
            ),
            (
                {LLM_CONFIG_KEY: {MODEL_GROUP_CONFIG_KEY: "openai_gpt-1"}},
                [
                    {
                        "id": "openai_gpt-1",
                        "models": [{"provider": "openai", "model": "gpt-4"}],
                    },
                ],
                {LLM_CONFIG_KEY: {MODEL_GROUP_CONFIG_KEY: "openai_gpt-2"}},
                [
                    {
                        "id": "openai_gpt-2",
                        "models": [{"provider": "openai", "model": "gpt-3.5-turbo"}],
                    },
                ],
                True,
            ),
            (
                {
                    LLM_CONFIG_KEY: {MODEL_GROUP_CONFIG_KEY: "openai_gpt-4"},
                    FLOW_RETRIEVAL_KEY: {
                        EMBEDDINGS_CONFIG_KEY: {
                            MODEL_GROUP_CONFIG_KEY: "openai_embedding"
                        }
                    },
                },
                [
                    {
                        "id": "openai_gpt-4",
                        "models": [{"provider": "openai", "model": "gpt-4"}],
                    },
                    {
                        "id": "openai_embedding",
                        "models": [{"provider": "openai", "model": "embedding-model"}],
                    },
                ],
                {
                    LLM_CONFIG_KEY: {MODEL_GROUP_CONFIG_KEY: "openai_gpt-4"},
                    FLOW_RETRIEVAL_KEY: {
                        EMBEDDINGS_CONFIG_KEY: {
                            MODEL_GROUP_CONFIG_KEY: "openai_embedding_2"
                        }
                    },
                },
                [
                    {
                        "id": "openai_gpt-4",
                        "models": [{"provider": "openai", "model": "gpt-4"}],
                    },
                    {
                        "id": "openai_embedding_2",
                        "models": [
                            {"provider": "openai", "model": "different-embedding-model"}
                        ],
                    },
                ],
                True,
            ),
        ],
    )
    async def test_fingerprint_addon_with_different_model_configs(
        self,
        config_1: Dict[str, Any],
        model_groups_1: List[Dict[str, Any]],
        config_2: Dict[str, Any],
        model_groups_2: List[Dict[str, Any]],
        fingerprint_differs: bool,
        model_storage: ModelStorage,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        generator = MultiStepLLMCommandGenerator(
            {}, model_storage, Resource("llmcmdgen")
        )

        class MockAvailableEndpoints:
            @staticmethod
            def get_instance():
                return MockAvailableEndpoints()

            def __init__(self):
                self.model_groups = model_groups_1

        mock_endpoints_1 = MockAvailableEndpoints()
        monkeypatch.setattr(
            "rasa.shared.utils.llm.AvailableEndpoints", mock_endpoints_1
        )

        fingerprint_1 = generator.fingerprint_addon(config_1)

        class MockAvailableEndpoints:
            @staticmethod
            def get_instance():
                return MockAvailableEndpoints()

            def __init__(self):
                self.model_groups = model_groups_2

        mock_endpoints_2 = MockAvailableEndpoints()
        monkeypatch.setattr(
            "rasa.shared.utils.llm.AvailableEndpoints", mock_endpoints_2
        )

        fingerprint_2 = generator.fingerprint_addon(config_2)

        assert fingerprint_1 is not None
        assert fingerprint_2 is not None
        if fingerprint_differs:
            assert fingerprint_1 != fingerprint_2
        else:
            assert fingerprint_1 == fingerprint_2

    @patch(
        "rasa.dialogue_understanding.generator.llm_based_command_generator.llm_factory"
    )
    @patch(
        "rasa.dialogue_understanding.generator"
        ".multi_step.multi_step_llm_command_generator.MultiStepLLMCommandGenerator"
        ".filter_flows"
    )
    async def test_predict_commands_adds_commands_and_prompt_to_message_object(
        self,
        mock_filter_flows: Mock,
        mock_llm_factory: Mock,
        command_generator: MultiStepLLMCommandGenerator,
        flows: FlowsList,
        tracker: DialogueStateTracker,
        llm_response_object: LLMResponse,
    ):
        """Test that predict_commands sets the routing slot to True."""
        message = Message.build(text="start test_flow")

        # Given
        with set_record_commands_and_prompts():
            llm_mock = AsyncMock()
            llm_response_object.choices = ["StartFlow(test_flow)"]
            llm_mock.acompletion.return_value = llm_response_object
            mock_llm_factory.return_value = llm_mock

            mock_filter_flows.return_value = flows

            # When
            await command_generator.predict_commands(
                message,
                flows=flows,
                tracker=tracker,
            )

        # Then
        prompts = message.get(PROMPTS)
        assert prompts is not None
        assert prompts[0][KEY_COMPONENT_NAME] == MultiStepLLMCommandGenerator.__name__
        assert prompts[0][KEY_USER_PROMPT].startswith(
            "Your task is to analyze the current situation and to start and/or end "
            "business processes that we call flows"
        )
        assert message.get(PREDICTED_COMMANDS)[
            MultiStepLLMCommandGenerator.__name__
        ] == [{"command": "start flow", "flow": "test_flow"}]

    @patch(
        "rasa.dialogue_understanding.generator.llm_based_command_generator.llm_factory"
    )
    @patch(
        "rasa.dialogue_understanding.generator"
        ".multi_step.multi_step_llm_command_generator.MultiStepLLMCommandGenerator"
        ".filter_flows"
    )
    async def test_predict_commands_does_not_add_commands_and_prompt_by_default(
        self,
        mock_filter_flows: Mock,
        mock_llm_factory: Mock,
        command_generator: MultiStepLLMCommandGenerator,
        flows: FlowsList,
        tracker: DialogueStateTracker,
    ):
        """Test that predict_commands sets the routing slot to True."""
        message = Message.build(text="start test_flow")

        # Given
        llm_mock = AsyncMock()
        llm_mock.acompletion.return_value = AsyncMock(
            spec=LLMResponse, choices=["StartFlow(test_flow)"]
        )
        mock_llm_factory.return_value = llm_mock

        mock_filter_flows.return_value = flows

        # When
        await command_generator.predict_commands(
            message,
            flows=flows,
            tracker=tracker,
        )

        # Then
        assert message.get(PROMPTS) is None
        assert message.get(PREDICTED_COMMANDS) is None


class TestMultiStepLLMCommandGeneratorPredictCommandsErrorHandling:
    @pytest.fixture
    def multi_step_llm_command_generator(self) -> MultiStepLLMCommandGenerator:
        return MultiStepLLMCommandGenerator.create(
            config={"flow_retrieval": {"active": False}},
            resource=Mock(),
            model_storage=Mock(),
            execution_context=Mock(),
        )

    @pytest.fixture
    def test_flows_with_defaults(self):
        return flows_from_str_including_defaults(
            """
            flows:
                flow_a:
                    name: flow a
                    description: Some description.
                    steps:
                        - id: step_a
                          action: action_listen
                flow_b:
                    name: flow b
                    description: Yet another description.
                    steps:
                        - id: step_b
                          action: action_listen
            """
        )

    @pytest.fixture
    def mock_predict_commands_for_active_flow(self, monkeypatch) -> AsyncMock:
        mock_method = AsyncMock()
        monkeypatch.setattr(
            "rasa.dialogue_understanding.generator.multi_step"
            ".multi_step_llm_command_generator.MultiStepLLMCommandGenerator"
            "._predict_commands_for_active_flow",
            mock_method,
        )
        mock_method.return_value = [ChangeFlowCommand()]
        return mock_method

    @pytest.fixture
    def mock_predict_commands_for_handling_flows(self, monkeypatch) -> AsyncMock:
        mock_method = AsyncMock()
        monkeypatch.setattr(
            "rasa.dialogue_understanding.generator.multi_step"
            ".multi_step_llm_command_generator.MultiStepLLMCommandGenerator"
            "._predict_commands_for_handling_flows",
            mock_method,
        )
        mock_method.return_value = [StartFlowCommand("flow_a")]
        return mock_method

    @pytest.fixture
    def mock_predict_commands_for_newly_started_flows(self, monkeypatch) -> AsyncMock:
        mock_method = AsyncMock()
        monkeypatch.setattr(
            "rasa.dialogue_understanding.generator.multi_step"
            ".multi_step_llm_command_generator.MultiStepLLMCommandGenerator"
            "._predict_commands_for_newly_started_flows",
            mock_method,
        )
        mock_method.return_value = [
            SetSlotCommand(name="test_slot", value="test_value")
        ]
        return mock_method

    @pytest.fixture
    def mock_filter_flows(
        self, monkeypatch: MonkeyPatch, test_flows_with_defaults: FlowsList
    ) -> AsyncMock:
        mock_method = AsyncMock()
        monkeypatch.setattr(
            "rasa.dialogue_understanding.generator.multi_step"
            ".multi_step_llm_command_generator.MultiStepLLMCommandGenerator"
            ".filter_flows",
            mock_method,
        )
        mock_method.return_value = (
            test_flows_with_defaults.user_flows.exclude_link_only_flows()
        )
        return mock_method

    async def test_predict_commands_no_errors(
        self,
        multi_step_llm_command_generator: MultiStepLLMCommandGenerator,
        test_flows_with_defaults: FlowsList,
        mock_filter_flows: AsyncMock,
        mock_predict_commands_for_active_flow: AsyncMock,
        mock_predict_commands_for_handling_flows: AsyncMock,
        mock_predict_commands_for_newly_started_flows: AsyncMock,
    ):
        # Given
        filtered_flows = test_flows_with_defaults.user_flows.exclude_link_only_flows()
        mock_message = Mock(spec=Message)
        mock_tracker = Mock(spec=DialogueStateTracker, has_active_flow=True)
        mock_tracker.has_coexistence_routing_slot = True
        mock_domain = Mock(spec=Domain)

        # When
        predicted_commands = await multi_step_llm_command_generator.predict_commands(
            mock_message, filtered_flows, mock_tracker, domain=mock_domain
        )

        # Then
        assert set(predicted_commands) == {
            StartFlowCommand("flow_a"),
            SetSlotCommand(name="test_slot", value="test_value"),
            SetSlotCommand(name=ROUTE_TO_CALM_SLOT, value=True),
        }

    async def test_predict_commands_flow_retrieval_raises_an_exception(
        self,
        multi_step_llm_command_generator: MultiStepLLMCommandGenerator,
        test_flows_with_defaults: FlowsList,
        mock_filter_flows: AsyncMock,
        mock_predict_commands_for_active_flow: AsyncMock,
        mock_predict_commands_for_handling_flows: AsyncMock,
        mock_predict_commands_for_newly_started_flows: AsyncMock,
    ):
        # Given
        filtered_flows = test_flows_with_defaults.user_flows.exclude_link_only_flows()
        mock_message = Mock(spec=Message)
        mock_tracker = Mock(spec=DialogueStateTracker, has_active_flow=True)
        mock_domain = Mock(spec=Domain)

        mock_filter_flows.side_effect = ProviderClientAPIException(
            message="Something went wrong",
            original_exception=Exception("API exception"),
        )

        # When
        predicted_commands = await multi_step_llm_command_generator.predict_commands(
            mock_message, filtered_flows, mock_tracker, domain=mock_domain
        )

        # Then
        assert len(predicted_commands) == 2
        assert ErrorCommand() in predicted_commands
        assert SetSlotCommand(ROUTE_TO_CALM_SLOT, True) in predicted_commands

    async def test_predict_commands_for_active_flow_raises_an_exception(
        self,
        multi_step_llm_command_generator: MultiStepLLMCommandGenerator,
        test_flows_with_defaults: FlowsList,
        mock_filter_flows: AsyncMock,
        mock_predict_commands_for_active_flow: AsyncMock,
        mock_predict_commands_for_handling_flows: AsyncMock,
        mock_predict_commands_for_newly_started_flows: AsyncMock,
    ):
        # Given
        filtered_flows = test_flows_with_defaults.user_flows.exclude_link_only_flows()
        mock_message = Mock(spec=Message)
        mock_tracker = Mock(spec=DialogueStateTracker, has_active_flow=True)
        mock_domain = Mock(spec=Domain)

        mock_predict_commands_for_active_flow.side_effect = ProviderClientAPIException(
            message="Something went wrong",
            original_exception=Exception("API exception"),
        )

        # When
        predicted_commands = await multi_step_llm_command_generator.predict_commands(
            mock_message, filtered_flows, mock_tracker, domain=mock_domain
        )

        # Then
        assert len(predicted_commands) == 2
        assert ErrorCommand() in predicted_commands
        assert SetSlotCommand(ROUTE_TO_CALM_SLOT, True) in predicted_commands

    async def test_predict_commands_for_handling_flows_raises_an_exception(
        self,
        multi_step_llm_command_generator: MultiStepLLMCommandGenerator,
        test_flows_with_defaults: FlowsList,
        mock_filter_flows: AsyncMock,
        mock_predict_commands_for_active_flow: AsyncMock,
        mock_predict_commands_for_handling_flows: AsyncMock,
        mock_predict_commands_for_newly_started_flows: AsyncMock,
    ):
        # Given
        filtered_flows = test_flows_with_defaults.user_flows.exclude_link_only_flows()
        mock_message = Mock(spec=Message)
        mock_tracker = Mock(spec=DialogueStateTracker, has_active_flow=True)
        mock_domain = Mock(spec=Domain)

        mock_predict_commands_for_handling_flows.side_effect = (
            ProviderClientAPIException(
                message="Something went wrong",
                original_exception=Exception("API exception"),
            )
        )

        # When
        predicted_commands = await multi_step_llm_command_generator.predict_commands(
            mock_message, filtered_flows, mock_tracker, domain=mock_domain
        )

        # Then
        assert len(predicted_commands) == 2
        assert ErrorCommand() in predicted_commands
        assert SetSlotCommand(ROUTE_TO_CALM_SLOT, True) in predicted_commands

    async def test_predict_commands_for_newly_started_flows_raises_an_exception(
        self,
        multi_step_llm_command_generator: MultiStepLLMCommandGenerator,
        test_flows_with_defaults: FlowsList,
        mock_filter_flows: AsyncMock,
        mock_predict_commands_for_active_flow: AsyncMock,
        mock_predict_commands_for_handling_flows: AsyncMock,
        mock_predict_commands_for_newly_started_flows: AsyncMock,
    ):
        # Given
        filtered_flows = test_flows_with_defaults.user_flows.exclude_link_only_flows()
        mock_message = Mock(spec=Message)
        mock_tracker = Mock(spec=DialogueStateTracker, has_active_flow=True)
        mock_domain = Mock(spec=Domain)

        mock_predict_commands_for_newly_started_flows.side_effect = (
            ProviderClientAPIException(
                message="Something went wrong",
                original_exception=Exception("API exception"),
            )
        )

        # When
        predicted_commands = await multi_step_llm_command_generator.predict_commands(
            mock_message, filtered_flows, mock_tracker, domain=mock_domain
        )

        # Then
        assert len(predicted_commands) == 2
        assert ErrorCommand() in predicted_commands
        assert SetSlotCommand(ROUTE_TO_CALM_SLOT, True) in predicted_commands

    async def test_prediction_doesnt_fail_on_change_flow_command(
        self,
        multi_step_llm_command_generator: MultiStepLLMCommandGenerator,
        test_flows_with_defaults: FlowsList,
        mock_filter_flows: AsyncMock,
        mock_predict_commands_for_active_flow: AsyncMock,
        mock_predict_commands_for_handling_flows: AsyncMock,
        mock_predict_commands_for_newly_started_flows: AsyncMock,
    ):
        # Given
        filtered_flows = test_flows_with_defaults.user_flows.exclude_link_only_flows()
        mock_message = Mock(spec=Message)
        mock_tracker = Mock(spec=DialogueStateTracker, has_active_flow=True)
        mock_domain = Mock(spec=Domain)

        mock_predict_commands_for_newly_started_flows.return_value = [
            ChangeFlowCommand()
        ]

        # When
        try:
            await multi_step_llm_command_generator.predict_commands(
                mock_message, filtered_flows, mock_tracker, domain=mock_domain
            )
        except TypeError:
            pytest.fail("Unexpected TypeError")

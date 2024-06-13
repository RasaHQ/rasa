import uuid
from pathlib import Path
from typing import Optional, List
from unittest.mock import Mock, patch, AsyncMock

import pytest
from _pytest.tmpdir import TempPathFactory

from rasa.dialogue_understanding.commands import (
    Command,
    ErrorCommand,
    SetSlotCommand,
    CancelFlowCommand,
    StartFlowCommand,
    HumanHandoffCommand,
    ChitChatAnswerCommand,
    SkipQuestionCommand,
    KnowledgeAnswerCommand,
    ClarifyCommand,
    ChangeFlowCommand,
    CannotHandleCommand,
)
from rasa.dialogue_understanding.generator.multi_step_llm_command_generator import (
    MultiStepLLMCommandGenerator,
)
from rasa.dialogue_understanding.generator.constants import (
    FLOW_RETRIEVAL_KEY,
    FLOW_RETRIEVAL_ACTIVE_KEY,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.constants import RASA_PATTERN_CANNOT_HANDLE_NOT_SUPPORTED
from rasa.shared.core.events import BotUttered, UserUttered
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.slots import TextSlot
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.training_data.message import Message
from tests.utilities import flows_from_str


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

    async def test_llm_command_generator_init_custom_start_or_end_flow(
        self,
        model_storage: ModelStorage,
        resource: Resource,
    ) -> None:
        # When
        generator = MultiStepLLMCommandGenerator(
            {
                "prompts": {
                    "start_or_end_flow": {
                        "file_path": "data/test_prompt_templates/test_prompt.jinja2",
                    }
                },
                FLOW_RETRIEVAL_KEY: {FLOW_RETRIEVAL_ACTIVE_KEY: False},
            },
            model_storage,
            resource,
        )
        # Then
        assert generator.start_or_end_flows_prompt.startswith("This is a test prompt.")
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
                "prompts": {
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
        assert generator.start_or_end_flows_prompt.startswith(
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
        assert generator.start_or_end_flows_prompt.startswith(
            "Your task is to analyze the current situation"
        )
        assert generator.fill_slots_prompt.startswith("{% if flow_active %}\nYour")
        assert generator.flow_retrieval is not None

    async def test_predict_commands_for_starting_flows_calls_llm_factory_correctly(
        self,
        command_generator: MultiStepLLMCommandGenerator,
    ):
        """Test predict_commands_for_starting_and_ending_flows calls llm correctly."""
        # Given
        llm_config = {
            "_type": "openai",
            "request_timeout": 7,
            "temperature": 0.0,
            "model_name": "gpt-4",
            "max_tokens": 256,
        }

        # When
        with patch(
            "rasa.dialogue_understanding.generator.llm_based_command_generator.llm_factory",
            Mock(),
        ) as mock_llm_factory:
            await command_generator.predict_commands_for_starting_and_ending_flows(
                Message(),
                DialogueStateTracker.from_events(
                    "test",
                    evts=[UserUttered("Hello", {"name": "greet", "confidence": 1.0})],
                ),
                FlowsList(underlying_flows=[]),
            )
            # Then
            mock_llm_factory.assert_called_once_with(None, llm_config)

    async def test_predict_commands_for_starting_and_ending_flows_calls_llm_correctly(
        self,
        command_generator: MultiStepLLMCommandGenerator,
    ):
        """Test predict_commands_for_starting_and_ending_flows calls llm correctly."""
        with patch(
            "rasa.dialogue_understanding.generator.llm_based_command_generator.llm_factory",
            Mock(),
        ) as mock_llm_factory:
            llm_mock = Mock()
            predict_mock = AsyncMock()
            llm_mock.apredict = predict_mock
            mock_llm_factory.return_value = llm_mock
            llm_mock.apredict.return_value = "some value"
            # When
            await command_generator.predict_commands_for_starting_and_ending_flows(
                Message(),
                DialogueStateTracker.from_events(
                    "test",
                    evts=[UserUttered("Hello", {"name": "greet", "confidence": 1.0})],
                ),
                FlowsList(underlying_flows=[]),
            )
            # Then
            predict_mock.assert_called_once()
            args, _ = predict_mock.call_args
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

        config = {"prompts": {"fill_slots": {"file_path": str(prompt_file)}}}
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
        prompt_file_static = prompt_dir / "start_or_end_flow.jinja2"
        prompt_file_static.write_text("This is a static prompt")

        config = {
            "prompts": {
                "fill_slots": {"file_path": str(prompt_file_dynamic)},
                "start_or_end_flow": {"file_path": str(prompt_file_static)},
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

        config = {"prompts": {"fill_slots": {"file_path": str(prompt_file)}}}
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
    ):
        # Given
        generator = MultiStepLLMCommandGenerator(
            {FLOW_RETRIEVAL_KEY: {FLOW_RETRIEVAL_ACTIVE_KEY: False}},
            model_storage,
            resource,
        )
        resource = generator.train(None, flows, Mock())
        # When
        loaded = MultiStepLLMCommandGenerator.load({}, model_storage, resource, Mock())
        # Then
        assert loaded.start_or_end_flows_prompt.startswith(
            "Your task is to analyze the current situation"
        )
        assert loaded.fill_slots_prompt.startswith("{% if flow_active %}\nYour")

    async def test_llm_command_generator_load_prompt_from_model_storage(
        self,
        model_storage: ModelStorage,
        tmp_path: Path,
    ) -> None:
        # Create and write prompt file.
        prompt_dir = Path(tmp_path) / "prompt"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        prompt_file = prompt_dir / "fill_slots.jinja2"
        prompt_file.write_text("This is a custom prompt")

        # Add the prompt file path to the config.
        config = {"prompts": {"fill_slots": {"file_path": str(prompt_file)}}}

        # Persist the prompt file to the model storage.
        resource = Resource("llmcmdgen")
        generator = MultiStepLLMCommandGenerator(config, model_storage, resource)
        generator.persist()

        # Test loading the prompt from the model storage.
        # Case 1: No prompt in the config.
        loaded = MultiStepLLMCommandGenerator.load({}, model_storage, resource, Mock())
        assert loaded.fill_slots_prompt == "This is a custom prompt"
        assert loaded.config["prompts"] == {}

        # Case 2: Specifying a invalid prompt path in the config.
        loaded = MultiStepLLMCommandGenerator.load(
            {"prompts": {"fill_slots": {"file_path": "test_prompt.jinja2"}}},
            model_storage,
            resource,
            Mock(),
        )
        assert loaded.fill_slots_prompt == "This is a custom prompt"
        assert (
            loaded.config["prompts"]["fill_slots"]["file_path"] == "test_prompt.jinja2"
        )

    @pytest.mark.parametrize(
        "input_action, expected_command",
        [
            (None, [ErrorCommand()]),
            (
                "SetSlot(transfer_money_amount_of_money, )",
                [SetSlotCommand(name="transfer_money_amount_of_money", value=None)],
            ),
            ("SetSlot(flow_name, some_flow)", [StartFlowCommand(flow="some_flow")]),
            ("StartFlow(some_flow)", [StartFlowCommand(flow="some_flow")]),
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
            ("Clarify(some_flow, test_a, test_b, test_c, test_d, test_e)", []),
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
        with patch.object(
            MultiStepLLMCommandGenerator,
            "get_nullable_slot_value",
            Mock(return_value=None),
        ):
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
        startable_test_flows = test_flows.exclude_link_only_flows()
        test_tracker.update_stack(stack)

        # When
        variables_to_render = command_generator.prepare_inputs(
            message=test_message,
            tracker=test_tracker,
            flows=startable_test_flows,
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
        variables_to_render = command_generator.prepare_inputs_for_single_flow(
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

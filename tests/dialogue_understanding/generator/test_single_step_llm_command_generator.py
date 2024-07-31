import os.path
import uuid
from pathlib import Path
from typing import Optional, Dict, Text, Any, Set, List
from unittest.mock import Mock, patch, AsyncMock, MagicMock

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
    CannotHandleCommand,
)
from rasa.dialogue_understanding.generator.constants import (
    LLM_CONFIG_KEY,
    DEFAULT_LLM_CONFIG,
    FLOW_RETRIEVAL_KEY,
    FLOW_RETRIEVAL_ACTIVE_KEY,
)
from rasa.dialogue_understanding.generator.flow_retrieval import (
    FlowRetrieval,
    DEFAULT_EMBEDDINGS_CONFIG,
)
from rasa.dialogue_understanding.generator.single_step.single_step_llm_command_generator import (  # noqa: E501
    SingleStepLLMCommandGenerator,
    DEFAULT_COMMAND_PROMPT_TEMPLATE,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.llm_fine_tuning.annotation_module import set_preparing_fine_tuning_data
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.constants import ROUTE_TO_CALM_SLOT
from rasa.shared.core.events import BotUttered, SlotSet, UserUttered
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.slots import (
    BooleanSlot,
    TextSlot,
)
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import ProviderClientAPIException
from rasa.shared.nlu.constants import TEXT, LLM_PROMPT, LLM_COMMANDS
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.utils.llm import (
    DEFAULT_MAX_USER_INPUT_CHARACTERS,
)
from tests.utilities import flows_from_str

EXPECTED_PROMPT_PATH = "./tests/dialogue_understanding/generator/rendered_prompt.txt"
EXPECTED_RENDERED_FLOW_DESCRIPTION_PATH = (
    "./tests/dialogue_understanding/generator/rendered_flow.txt"
)


class TestSingleStepLLMCommandGenerator:
    """Tests for the SingleStepLLMCommandGenerator."""

    @pytest.fixture
    def command_generator(self):
        """Create an SingleStepLLMCommandGenerator."""
        return SingleStepLLMCommandGenerator.create(
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

    @pytest.fixture
    def tracker_with_routing_slot(self):
        """Create a Tracker."""
        return DialogueStateTracker.from_events(
            sender_id="",
            evts=[],
            slots=[
                BooleanSlot(ROUTE_TO_CALM_SLOT, mappings=[], initial_value=True),
            ],
        )

    async def test_deprecation_warning_with_prompt(self, model_storage):
        # Given
        resource = Resource("llmcmdgen")
        config = {"prompt": "data/test_prompt_templates/test_prompt.jinja2"}

        # When
        with patch(
            "rasa.dialogue_understanding.generator.single_step.single_step_llm_command_generator.structlogger.warning"
        ) as mock_warning:
            SingleStepLLMCommandGenerator(
                config,
                model_storage,
                resource,
            )
        mock_warning.assert_called_once_with(
            "single_step_llm_command_generator.init",
            event_info=(
                "The config parameter 'prompt' is deprecated "
                "and will be removed in Rasa 4.0.0. "
                "Please use the config parameter 'prompt_template' instead. "
            ),
        )

    async def test_prompt_template_handling(self, model_storage):
        # Given
        resource = Resource("llmcmdgen")
        expected_template = "data/test_prompt_templates/test_prompt.jinja2"
        config = {"prompt_template": expected_template}

        # When
        generator = SingleStepLLMCommandGenerator(
            config,
            model_storage,
            resource,
        )

        # Then
        assert generator.prompt_template.startswith("This is a test prompt.")

    async def test_default_template_when_no_prompt_template_provided(
        self, model_storage
    ):
        # Given
        resource = Resource("llmcmdgen")
        config = {}  # No prompt or prompt_template provided

        # When
        generator = SingleStepLLMCommandGenerator(
            config,
            model_storage,
            resource,
        )

        # Then
        assert generator.prompt_template == DEFAULT_COMMAND_PROMPT_TEMPLATE

    async def test_single_step_llm_command_generator_init_custom(
        self,
        model_storage: ModelStorage,
    ) -> None:
        # Given
        resource = Resource("llmcmdgen")
        # When
        generator = SingleStepLLMCommandGenerator(
            {
                "prompt": "data/test_prompt_templates/test_prompt.jinja2",
                FLOW_RETRIEVAL_KEY: {FLOW_RETRIEVAL_ACTIVE_KEY: False},
            },
            model_storage,
            resource,
        )
        # Then
        assert generator.prompt_template.startswith("This is a test prompt.")
        assert generator.flow_retrieval is None

    async def test_single_step_llm_command_generator_init_default(
        self,
        model_storage: ModelStorage,
    ) -> None:
        # When
        generator = SingleStepLLMCommandGenerator(
            {}, model_storage, Resource("llmcmdgen")
        )
        # Then
        assert generator.prompt_template.startswith(
            "Your task is to analyze the current conversation"
        )
        assert (
            generator.user_input_config.max_characters
            == DEFAULT_MAX_USER_INPUT_CHARACTERS
        )
        assert generator.flow_retrieval is not None

    @pytest.mark.parametrize(
        "config, expected_limit",
        [
            ({"user_input": {"max_characters": 100}}, 100),
            ({"user_input": {"max_characters": -1}}, -1),
            (
                {"user_input": {"max_characters": None}},
                DEFAULT_MAX_USER_INPUT_CHARACTERS,
            ),
            ({"user_input": None}, DEFAULT_MAX_USER_INPUT_CHARACTERS),
            ({"user_input": {}}, DEFAULT_MAX_USER_INPUT_CHARACTERS),
        ],
    )
    def test_single_step_llm_command_generator_init_with_message_length_limit(
        self,
        config: Dict[Text, Any],
        expected_limit: Optional[int],
        model_storage: ModelStorage,
        resource: Resource,
    ) -> None:
        generator = SingleStepLLMCommandGenerator(
            config,
            model_storage,
            resource,
        )
        assert generator.user_input_config.max_characters == expected_limit

    async def test_predict_commands_with_no_flows(
        self,
        command_generator: SingleStepLLMCommandGenerator,
        tracker: DialogueStateTracker,
    ):
        """Test that predict_commands returns an empty list when flows is None."""
        # Given
        empty_flows = FlowsList(underlying_flows=[])
        # When
        predicted_commands = await command_generator.predict_commands(
            Mock(), flows=empty_flows, tracker=tracker
        )
        # Then
        assert not predicted_commands

    async def test_predict_commands_with_no_tracker(
        self, command_generator: SingleStepLLMCommandGenerator
    ):
        """Test that predict_commands returns an empty list when tracker is None."""
        # When
        predicted_commands = await command_generator.predict_commands(
            Mock(), flows=Mock(), tracker=None
        )
        # Then
        assert not predicted_commands

    async def test_predict_commands_sets_routing_slot(
        self,
        command_generator: SingleStepLLMCommandGenerator,
        flows: FlowsList,
        tracker_with_routing_slot: DialogueStateTracker,
    ):
        """Test that predict_commands sets the routing slot to True."""
        # When
        with patch(
            "rasa.dialogue_understanding.generator.llm_based_command_generator.llm_factory",
            Mock(),
        ) as mock_llm_factory:
            llm_mock = Mock()
            apredict_mock = AsyncMock(return_value="StartFlow(test_flow)")
            llm_mock.apredict = apredict_mock
            mock_llm_factory.return_value = llm_mock
            predicted_commands = await command_generator.predict_commands(
                Message.build(text="start test_flow"),
                flows=flows,
                tracker=tracker_with_routing_slot,
            )

        # Then
        assert StartFlowCommand("test_flow") in predicted_commands
        assert SetSlotCommand(ROUTE_TO_CALM_SLOT, True) in predicted_commands

    async def test_predict_commands_does_not_set_llm_commands_and_prompt(
        self,
        command_generator: SingleStepLLMCommandGenerator,
        flows: FlowsList,
        tracker: DialogueStateTracker,
    ):
        """Test that predict_commands sets the routing slot to True."""
        message = Message.build(text="start test_flow")

        # When
        with patch(
            "rasa.dialogue_understanding.generator.llm_based_command_generator.llm_factory",
            Mock(),
        ) as mock_llm_factory:
            llm_mock = Mock()
            apredict_mock = AsyncMock(return_value="StartFlow(test_flow)")
            llm_mock.apredict = apredict_mock
            mock_llm_factory.return_value = llm_mock
            await command_generator.predict_commands(
                message,
                flows=flows,
                tracker=tracker,
            )

        # Then
        assert message.get(LLM_PROMPT) is None
        assert message.get(LLM_COMMANDS) is None

    async def test_predict_commands_sets_llm_commands_and_prompt(
        self,
        command_generator: SingleStepLLMCommandGenerator,
        flows: FlowsList,
        tracker: DialogueStateTracker,
    ):
        """Test that predict_commands sets the routing slot to True."""
        message = Message.build(text="start test_flow")

        # When
        with set_preparing_fine_tuning_data():
            with patch(
                "rasa.dialogue_understanding.generator.llm_based_command_generator.llm_factory",
                Mock(),
            ) as mock_llm_factory:
                llm_mock = Mock()
                apredict_mock = AsyncMock(return_value="StartFlow(test_flow)")
                llm_mock.apredict = apredict_mock
                mock_llm_factory.return_value = llm_mock
                await command_generator.predict_commands(
                    message,
                    flows=flows,
                    tracker=tracker,
                )

        # Then
        assert message.get(LLM_PROMPT) is not None
        assert message.get(LLM_PROMPT).startswith(
            "Your task is to analyze the current conversation context"
        )
        assert message.get(LLM_COMMANDS) == [
            {"command": "start flow", "flow": "test_flow"}
        ]

    @pytest.mark.parametrize(
        "flow_guard_value, expected_flow_ids",
        (
            (None, {"flow_regular"}),
            (False, {"flow_regular"}),
            (True, {"flow_regular", "flow_with_guard"}),
            ("false", {"flow_regular", "flow_with_guard"}),
            ("true", {"flow_regular", "flow_with_guard"}),
        ),
    )
    @patch(
        "rasa.dialogue_understanding.generator.single_step.single_step_llm_command_generator"
        ".SingleStepLLMCommandGenerator"
        ".render_template"
    )
    @patch(
        "rasa.dialogue_understanding.generator.single_step.single_step_llm_command_generator"
        ".SingleStepLLMCommandGenerator"
        ".invoke_llm"
    )
    async def test_predict_commands_calls_prompt_rendering_with_startable_flows_only(
        self,
        mock_generate_action_list_using_llm: Mock,
        mock_render_template: Mock,
        flow_guard_value: Any,
        expected_flow_ids: Set[Text],
        command_generator: SingleStepLLMCommandGenerator,
    ):
        # Given
        test_flows = flows_from_str(
            """
            flows:
                flow_with_guard:
                    if: slots.some_slot
                    name: flow with guard
                    description: description for flow a
                    steps:
                        - id: step_a
                          action: action_listen
                flow_link:
                    if: False
                    name: flow link
                    description: description for flow b
                    steps:
                        - id: step_b
                          action: action_listen
                flow_regular:
                    name: flow regular
                    description: description for flow b
                    steps:
                        - id: step_c
                          action: action_listen
            """
        )
        tracker = DialogueStateTracker.from_events(
            sender_id="test",
            evts=[SlotSet(key="some_slot", value=flow_guard_value)],
        )
        mock_message = Mock()
        mock_message.data = {TEXT: "some_message"}
        # regardless of flow retrieval we want to make sure we are calling the
        # prompt rendering only with startable flows.
        config = {"flow_retrieval": {"active": False}}
        command_generator = SingleStepLLMCommandGenerator.create(
            config=config,
            resource=Mock(),
            model_storage=Mock(),
            execution_context=Mock(),
        )
        # the return value doesn't matter
        mock_generate_action_list_using_llm.return_value = None

        # When
        await command_generator.predict_commands(
            message=mock_message, flows=test_flows, tracker=tracker
        )
        mock_render_template.assert_called_once()
        filtered_flows = mock_render_template.call_args.args[2]
        all_available_flows = mock_render_template.call_args.args[3]

        # Then
        assert filtered_flows.flow_ids == expected_flow_ids
        assert all_available_flows.flow_ids == test_flows.flow_ids

    @pytest.mark.parametrize(
        "llm_response, expected_commands",
        [
            (
                None,
                [ErrorCommand()],
            ),
            (
                "StartFlow(this_flow_does_not_exists)",
                [
                    CannotHandleCommand(),
                ],
            ),
            (
                "A random response from LLM",
                [
                    CannotHandleCommand(),
                ],
            ),
            (
                "SetSlot(flow_name, some_flow)",
                [
                    StartFlowCommand(flow="some_flow"),
                ],
            ),
        ],
    )
    @patch(
        "rasa.dialogue_understanding.generator.single_step.single_step_llm_command_generator."
        "SingleStepLLMCommandGenerator.invoke_llm"
    )
    @patch(
        "rasa.dialogue_understanding.generator.single_step.single_step_llm_command_generator."
        "SingleStepLLMCommandGenerator.render_template"
    )
    @patch(
        "rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval.filter_flows"
    )
    async def test_predict_commands(
        self,
        mock_flow_retrieval_filter_flows: Mock,
        mock_render_template: Mock,
        mock_generate_action_list_using_llm: Mock,
        llm_response: Text,
        expected_commands: List[Command],
        command_generator: SingleStepLLMCommandGenerator,
        tracker_with_routing_slot: DialogueStateTracker,
    ):
        # Given
        test_flows = flows_from_str(
            """
            flows:
              some_flow:
                description: some description
                steps:
                - id: first_step
                  collect: test_slot
            """
        )
        mock_render_template.return_value = "some_template"
        mock_generate_action_list_using_llm.return_value = llm_response
        mock_flow_retrieval_filter_flows.return_value = FlowsList(underlying_flows=[])
        mock_message = Mock()
        mock_message.data = {TEXT: "some_message"}
        # When
        predicted_commands = await command_generator.predict_commands(
            message=mock_message, flows=test_flows, tracker=tracker_with_routing_slot
        )
        # Then
        mock_flow_retrieval_filter_flows.assert_called_once()
        assert len(predicted_commands) == len(expected_commands) + 1
        for expected_command in expected_commands:
            assert expected_command in predicted_commands

        # route session must be present when there is a
        # tracker with routing slot
        assert SetSlotCommand(ROUTE_TO_CALM_SLOT, True) in predicted_commands

    @patch(
        "rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval.filter_flows"
    )
    async def test_predict_commands_and_flow_retrieval_api_error_throws_exception(
        self,
        mock_flow_retrieval_filter_flows: AsyncMock,
        command_generator: SingleStepLLMCommandGenerator,
        tracker_with_routing_slot: DialogueStateTracker,
    ) -> None:
        # Given
        test_flows = flows_from_str(
            """
            flows:
              some_flow:
                description: some description
                steps:
                - id: first_step
                  collect: test_slot
            """
        )
        mock_message = Mock()
        mock_message.data = {TEXT: "some_message"}
        mock_flow_retrieval_filter_flows.side_effect = ProviderClientAPIException(
            message="Test Exception", original_exception=Exception("API exception")
        )
        # When
        predicted_commands = await command_generator.predict_commands(
            message=mock_message,
            flows=test_flows,
            tracker=tracker_with_routing_slot,
        )

        # Then
        mock_flow_retrieval_filter_flows.assert_called_once()

        assert len(predicted_commands) == 2
        assert ErrorCommand() in predicted_commands
        assert SetSlotCommand(ROUTE_TO_CALM_SLOT, True) in predicted_commands

    def test_render_template(
        self,
        command_generator: SingleStepLLMCommandGenerator,
    ):
        """Test that render_template renders the correct template string."""
        # Given
        test_message = Message.build(text="some message")
        test_slot = TextSlot(
            name="test_slot",
            mappings=[{}],
            initial_value=None,
            influence_conversation=False,
        )
        test_tracker = DialogueStateTracker.from_events(
            sender_id="test",
            evts=[UserUttered("Hello"), BotUttered("Hi")],
            slots=[test_slot],
        )
        test_flows = flows_from_str(
            """
            flows:
              test_flow:
                description: some description
                steps:
                - id: first_step
                  collect: test_slot
            """
        )
        with open(EXPECTED_PROMPT_PATH, "r", encoding="unicode_escape") as f:
            expected_template = f.readlines()
        # When
        rendered_template = command_generator.render_template(
            message=test_message,
            tracker=test_tracker,
            startable_flows=test_flows,
            all_flows=test_flows,
        )
        # Then
        for rendered_line, expected_line in zip(
            rendered_template.splitlines(True), expected_template
        ):
            assert rendered_line == expected_line

    def test_render_template_call(
        self,
        command_generator: SingleStepLLMCommandGenerator,
    ):
        """Test that render_template renders the correct template string."""
        # Given
        test_message = Message.build(text="some message")
        test_slot = TextSlot(
            name="test_slot",
            mappings=[{}],
            initial_value=None,
            influence_conversation=False,
        )
        test_tracker = DialogueStateTracker.from_events(
            sender_id="test",
            evts=[UserUttered("Hello"), BotUttered("Hi")],
            slots=[test_slot],
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
              called_flow:
                if: False
                description: a flows that's called
                steps:
                - id: first_step
                  collect: test_slot
              test_flow:
                description: some description
                steps:
                - id: call_step
                  call: called_flow
            """
        )
        startable_test_flows = test_flows.exclude_link_only_flows()
        test_tracker.update_stack(stack)
        # When
        rendered_template = command_generator.render_template(
            message=test_message,
            tracker=test_tracker,
            startable_flows=startable_test_flows,
            all_flows=test_flows,
        )
        # Then
        # make sure non-startable flow isn't there
        assert "called_flow" not in rendered_template
        # make sure it looks like we are in the calling flow
        assert 'You are currently in the flow "test_flow".' in rendered_template
        # make sure the slot from the called flow is available in the template
        assert (
            'You have just asked the user for the slot "test_slot".'
            in rendered_template
        )

    @pytest.mark.parametrize(
        "input_action, expected_command",
        [
            (None, []),
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
            # Clarify with single option is converted to a StartFlowCommand
            ("Clarify(some_flow)", [StartFlowCommand(flow="some_flow")]),
            # Clarify with multiple but same options is converted to a StartFlowCommand
            (
                "Clarify(some_flow, some_flow, some_flow, some_flow)",
                [StartFlowCommand(flow="some_flow")],
            ),
            # Clarify with multiple but same options is converted to a StartFlowCommand
            (
                "Clarify(some_flow, some_flow)",
                [StartFlowCommand(flow="some_flow")],
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
            """
        )
        with patch.object(
            SingleStepLLMCommandGenerator,
            "get_nullable_slot_value",
            Mock(return_value=None),
        ):
            parsed_commands = SingleStepLLMCommandGenerator.parse_commands(
                input_action, Mock(), test_flows
            )
        # Then
        assert parsed_commands == expected_command

    async def test_llm_command_generator_fingerprint_addon_diff_in_prompt_template(
        self,
        model_storage: ModelStorage,
        tmp_path: Path,
    ) -> None:
        prompt_dir = Path(tmp_path) / "prompt"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        prompt_file = prompt_dir / "llm_command_generator_prompt.jinja2"
        prompt_file.write_text("This is a test prompt")

        config = {"prompt": str(prompt_file)}
        generator = SingleStepLLMCommandGenerator(
            config, model_storage, Resource("llmcmdgen")
        )
        fingerprint_1 = generator.fingerprint_addon(config)

        prompt_file.write_text("This is a test prompt. It has been changed.")
        fingerprint_2 = generator.fingerprint_addon(config)
        assert fingerprint_1 != fingerprint_2

    async def test_llm_command_generator_fingerprint_addon_no_diff_in_prompt_template(
        self,
        model_storage: ModelStorage,
        tmp_path: Path,
    ) -> None:
        prompt_dir = Path(tmp_path) / "prompt"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        prompt_file = prompt_dir / "llm_command_generator_prompt.jinja2"
        prompt_file.write_text("This is a test prompt")

        config = {"prompt": str(prompt_file)}
        generator = SingleStepLLMCommandGenerator(
            config, model_storage, Resource("llmcmdgen")
        )

        fingerprint_1 = generator.fingerprint_addon(config)
        fingerprint_2 = generator.fingerprint_addon(config)
        assert fingerprint_1 is not None
        assert fingerprint_1 == fingerprint_2

    async def test_llm_command_generator_fingerprint_addon_default_prompt_template(
        self,
        model_storage: ModelStorage,
    ) -> None:
        generator = SingleStepLLMCommandGenerator(
            {}, model_storage, Resource("llmcmdgen")
        )
        fingerprint_1 = generator.fingerprint_addon({})
        fingerprint_2 = generator.fingerprint_addon({})
        assert fingerprint_1 is not None
        assert fingerprint_1 == fingerprint_2

    def test_train_with_flow_retrieval_disabled(
        self,
        model_storage: ModelStorage,
        flows: FlowsList,
        resource: Resource,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Given
        generator = SingleStepLLMCommandGenerator(
            {FLOW_RETRIEVAL_KEY: {FLOW_RETRIEVAL_ACTIVE_KEY: False}},
            model_storage,
            resource,
        )
        # When
        generator.train(TrainingData(), flows, Mock())
        # Then
        assert generator.flow_retrieval is None

    @patch(
        "rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval.populate"
    )
    def test_train_with_flow_retrieval_enabled(
        self,
        mock_flow_search_populate: Mock,
        model_storage: ModelStorage,
        flows: FlowsList,
        resource: Resource,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Given
        generator = SingleStepLLMCommandGenerator(
            {},
            model_storage,
            resource,
        )
        domain = Mock()
        # When
        generator.train(TrainingData(), flows, domain)
        # Then
        mock_flow_search_populate.assert_called_once_with(flows, domain)

    @patch(
        "rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval.populate"
    )
    def test_train_with_flow_retrieval_enabled_and_api_error_throws_exception(
        self,
        mock_flow_search_populate: Mock,
        model_storage: ModelStorage,
        flows: FlowsList,
        resource: Resource,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Given
        generator = SingleStepLLMCommandGenerator(
            {},
            model_storage,
            resource,
        )
        domain = Mock()
        mock_flow_search_populate.side_effect = Exception("Test Exception")
        # When
        with pytest.raises(Exception) as exc_info:
            generator.train(TrainingData(), flows, domain)
        # Then
        assert "Test Exception" in str(exc_info.value), "Expected exception not raised"
        mock_flow_search_populate.assert_called_once_with(flows, domain)

    def test_load_with_flow_retrieval_disabled(
        self,
        model_storage: ModelStorage,
        flows: FlowsList,
        resource: Resource,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Given
        generator = SingleStepLLMCommandGenerator(
            {FLOW_RETRIEVAL_KEY: {FLOW_RETRIEVAL_ACTIVE_KEY: False}},
            model_storage,
            resource,
        )
        domain = Mock()
        domain.slots = []
        train_resource = generator.train(TrainingData(), flows, domain)
        # When
        loaded = SingleStepLLMCommandGenerator.load(
            generator.config,
            model_storage,
            train_resource,
            Mock(),
        )
        # Then
        assert loaded is not None
        assert loaded.flow_retrieval is None
        assert not loaded.config[FLOW_RETRIEVAL_KEY][FLOW_RETRIEVAL_ACTIVE_KEY]

    @patch(
        "rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval.populate"
    )
    @patch("rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval.load")
    def test_load_with_flow_retrieval_enabled(
        self,
        mock_flow_retrieval_load: Mock,
        mock_flow_retrieval_populate: Mock,
        model_storage: ModelStorage,
        flows: FlowsList,
        resource: Resource,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Given
        config = {FLOW_RETRIEVAL_KEY: FlowRetrieval.get_default_config()}
        generator = SingleStepLLMCommandGenerator(
            config,
            model_storage,
            resource,
        )
        domain = Mock()
        train_resource = generator.train(TrainingData(), flows, domain)
        # When
        loaded = SingleStepLLMCommandGenerator.load(
            generator.config,
            model_storage,
            train_resource,
            Mock(),
        )
        # Then
        mock_flow_retrieval_load.assert_called_once_with(
            config=config[FLOW_RETRIEVAL_KEY],
            model_storage=model_storage,
            resource=resource,
        )
        assert loaded is not None
        assert loaded.flow_retrieval is not None

    @patch(
        "rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval.populate"
    )
    @patch("rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval.load")
    def test_load_with_custom_prompt(
        self,
        mock_flow_retrieval_load: Mock,
        mock_flow_retrieval_populate: Mock,
        model_storage: ModelStorage,
    ):
        # Given
        resource = Resource("llmcmdgen")
        config = {
            "prompt": os.path.join(
                "data", "test_prompt_templates", "test_prompt.jinja2"
            )
        }
        generator = SingleStepLLMCommandGenerator(config, model_storage, resource)
        resource = generator.train(Mock(), FlowsList(underlying_flows=[]), Mock())
        # When
        loaded = SingleStepLLMCommandGenerator.load({}, model_storage, resource, Mock())
        # Then
        assert loaded.prompt_template.startswith("This is a test prompt.")

    @patch(
        "rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval.populate"
    )
    @patch("rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval.load")
    def test_load_with_default_prompt(
        self,
        mock_flow_retrieval_load: Mock,
        mock_flow_retrieval_populate: Mock,
        model_storage: ModelStorage,
    ):
        # Given
        resource = Resource("llmcmdgen")
        generator = SingleStepLLMCommandGenerator({}, model_storage, resource)
        resource = generator.train(Mock(), FlowsList(underlying_flows=[]), Mock())
        # When
        loaded = SingleStepLLMCommandGenerator.load({}, model_storage, resource, Mock())
        # Then
        assert loaded.prompt_template.startswith(
            "Your task is to analyze the current conversation"
        )

    async def test_single_step_llm_command_generator_load_prompt_from_model_storage(
        self,
        model_storage: ModelStorage,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "some key")
        # Create and write prompt file.
        prompt_dir = Path(tmp_path) / "prompt"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        prompt_file = prompt_dir / "llm_command_generator_prompt.jinja2"
        prompt_file.write_text("This is a custom prompt")

        # Add the prompt file path to the config.
        config = {"prompt": str(prompt_file)}

        # Persist the prompt file to the model storage.
        resource = Resource("llmcmdgen")
        generator = SingleStepLLMCommandGenerator(config, model_storage, resource)
        generator.persist()

        # Test loading the prompt from the model storage.
        # Case 1: No prompt in the config.
        loaded = SingleStepLLMCommandGenerator.load({}, model_storage, resource, Mock())
        assert loaded.prompt_template == "This is a custom prompt"
        assert loaded.config["prompt"] is None

        # Case 2: Specifying a invalid prompt path in the config.
        loaded = SingleStepLLMCommandGenerator.load(
            {"prompt": "test_prompt.jinja2"},
            model_storage,
            resource,
            Mock(),
        )
        assert loaded.prompt_template == "This is a custom prompt"
        assert loaded.config["prompt"] == "test_prompt.jinja2"

    @pytest.mark.parametrize(
        "config, expected_calls",
        [
            # Test default configurations
            (
                {
                    LLM_CONFIG_KEY: {"model_name": "default_model"},
                    "prompt_template": None,
                },
                {
                    "llm_model_name": "default_model",
                    "custom_prompt_used": False,
                    "flow_retrieval_enabled": True,
                    "flow_retrieval_embedding_model_name": DEFAULT_EMBEDDINGS_CONFIG[
                        "model"
                    ],
                },
            ),
            # Test custom prompt and disabled flow retrieval
            (
                {"prompt": "custom prompt", FLOW_RETRIEVAL_KEY: {"active": False}},
                {
                    "llm_model_name": DEFAULT_LLM_CONFIG["model_name"],
                    "custom_prompt_used": True,
                    "flow_retrieval_enabled": False,
                    "flow_retrieval_embedding_model_name": None,
                },
            ),
            # Test custom model and embedding model
            (
                {
                    LLM_CONFIG_KEY: {"model_name": "custom_model"},
                    FLOW_RETRIEVAL_KEY: {"embeddings": {"model": "custom_embedding"}},
                },
                {
                    "llm_model_name": "custom_model",
                    "custom_prompt_used": False,
                    "flow_retrieval_enabled": True,
                    "flow_retrieval_embedding_model_name": "custom_embedding",
                },
            ),
        ],
    )
    def test_track_method(self, config, expected_calls):
        # Mocking the tracking function
        with patch(
            "rasa.dialogue_understanding.generator.single_step.single_step_llm_command_generator"
            ".track_single_step_llm_command_generator_init"
        ) as mock_track:
            mock_model_storage = MagicMock()
            mock_resource = MagicMock()
            SingleStepLLMCommandGenerator(
                config=config, model_storage=mock_model_storage, resource=mock_resource
            )
            mock_track.assert_called_once_with(**expected_calls)

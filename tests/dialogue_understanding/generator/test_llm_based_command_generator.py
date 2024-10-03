import uuid
from typing import Dict, Any, Optional, Text, ClassVar, List
from unittest.mock import Mock, AsyncMock, patch

import pytest
from _pytest.tmpdir import TempPathFactory
from rasa.shared.constants import ROUTE_TO_CALM_SLOT, OPENAI_API_KEY_ENV_VAR
from rasa.shared.providers.llm.openai_llm_client import OpenAILLMClient
from structlog.testing import capture_logs
from pytest import MonkeyPatch
from rasa.dialogue_understanding.commands import (
    Command,
    ErrorCommand,
    SetSlotCommand,
    ChitChatAnswerCommand,
)
from rasa.dialogue_understanding.generator import (
    LLMBasedCommandGenerator,
    LLMCommandGenerator,
    SingleStepLLMCommandGenerator,
    MultiStepLLMCommandGenerator,
)
from rasa.dialogue_understanding.generator.constants import (
    FLOW_RETRIEVAL_KEY,
    FLOW_RETRIEVAL_ACTIVE_KEY,
)
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.events import BotUttered, SlotSet, UserUttered
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.flows.steps.collect import (
    SlotRejection,
    CollectInformationFlowStep,
)
from rasa.shared.core.slots import (
    Slot,
    TextSlot,
)
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import ProviderClientAPIException
from rasa.shared.nlu.constants import TEXT
from rasa.shared.nlu.training_data.message import Message
from tests.utilities import flows_from_str


class TestLLMBasedCommandGenerator:
    """Tests for the LLMBasedCommandGenerator."""

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

    # Fixture for the base class
    @pytest.fixture
    def base_class_fixture(self):
        class TestLLMBasedCommandGenerator(LLMBasedCommandGenerator):
            def __init__(
                self,
                config: Dict[str, Any],
                model_storage: ModelStorage,
                resource: Resource,
                prompt_template: Optional[Text] = None,
                **kwargs: Any,
            ) -> None:
                super().__init__(
                    config,
                    model_storage,
                    resource,
                    prompt_template=prompt_template,
                    **kwargs,
                )

            @staticmethod
            def get_default_config() -> dict:
                return {}

            @classmethod
            def load(
                cls,
                config: dict,
                model_storage: ModelStorage,
                resource: Resource,
                execution_context: ExecutionContext,
                **kwargs,
            ):
                return cls(config, model_storage, resource)

            def persist(self) -> None:
                pass

            async def predict_commands(
                self,
                message: Message,
                flows: FlowsList,
                tracker: DialogueStateTracker = None,
                **kwargs: Any,
            ):
                return []

            def parse_commands(cls, actions, tracker, flows):
                return []

            def fingerprint_addon(cls, config):
                return None

        return TestLLMBasedCommandGenerator

    # Fixture for the implementated classes
    @pytest.fixture
    def llm_command_generator_fixture(self, model_storage, resource):
        return LLMCommandGenerator.create(
            config={},
            model_storage=model_storage,
            resource=resource,
            execution_context=Mock(spec=ExecutionContext),
        )

    @pytest.fixture
    def single_step_llm_command_generator_fixture(self, model_storage, resource):
        return SingleStepLLMCommandGenerator.create(
            config={},
            model_storage=model_storage,
            resource=resource,
            execution_context=Mock(spec=ExecutionContext),
        )

    @pytest.fixture
    def multi_step_llm_command_generator_fixture(self, model_storage, resource):
        return MultiStepLLMCommandGenerator.create(
            config={},
            model_storage=model_storage,
            resource=resource,
            execution_context=Mock(spec=ExecutionContext),
        )

    ### Tests for abstract methods (i.e. based on implemented classes)
    # Parameterized fixture
    @pytest.fixture(params=["implementation_single_step", "implementation_multi_step"])
    def command_generator_fixture(
        self,
        request,
        single_step_llm_command_generator_fixture,
        multi_step_llm_command_generator_fixture,
    ):
        if request.param == "implementation_single_step":
            return single_step_llm_command_generator_fixture
        elif request.param == "implementation_multi_step":
            return multi_step_llm_command_generator_fixture
        else:
            raise ValueError("Unknown fixture type")

    @pytest.mark.asyncio
    async def test_predict_commands_no_flows(self, command_generator_fixture):
        generator = command_generator_fixture
        message = Mock()
        message.data = {TEXT: "some_message"}
        tracker = Mock(spec=DialogueStateTracker)
        flows = FlowsList(underlying_flows=[])

        commands = await generator.predict_commands(message, flows, tracker)

        assert commands == []

    @pytest.mark.asyncio
    async def test_predict_commands_no_tracker(self, command_generator_fixture):
        generator = command_generator_fixture
        message = Mock()
        message.data = {TEXT: "some_message"}
        flows = FlowsList(underlying_flows=[])

        commands = await generator.predict_commands(message, flows, None)

        assert commands == []

    @pytest.mark.parametrize(
        "input_action, expected_command",
        [
            (
                "SetSlot(test_slot, 1234)",
                [SetSlotCommand(name="test_slot", value="1234")],
            ),
            (
                "SetSlot(phone_number, (412) 555-1234)",
                [SetSlotCommand(name="phone_number", value="(412) 555-1234")],
            ),
        ],
    )
    def test_parse_commands_uses_correct_regex(
        self,
        input_action: Optional[str],
        expected_command: Command,
        command_generator_fixture,
    ):
        """Test that parse_commands uses the expected regex."""
        generator = command_generator_fixture
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
        parsed_commands = generator.parse_commands(input_action, Mock(), test_flows)
        # Then
        assert parsed_commands == expected_command

    ## Test flow retrieval
    @pytest.mark.parametrize(
        "flow_retrieval_active, expected_initialization",
        [
            (True, True),  # Flow retrieval enabled
            (False, False),  # Flow retrieval disabled
        ],
    )
    async def test_flow_retrieval_initialization(
        self,
        flow_retrieval_active: bool,
        expected_initialization: bool,
        command_generator_fixture,
        model_storage: ModelStorage,
        resource: Resource,
    ):
        # When
        config = {
            FLOW_RETRIEVAL_KEY: {FLOW_RETRIEVAL_ACTIVE_KEY: flow_retrieval_active},
        }
        generator_class = command_generator_fixture.__class__
        generator = generator_class.create(
            config=config,
            model_storage=model_storage,
            resource=resource,
            execution_context=Mock(spec=ExecutionContext),
        )
        # Then
        assert (generator.flow_retrieval is not None) == expected_initialization

    @pytest.mark.parametrize(
        "flow_retrieval_active, should_call_flow_retrieval",
        [
            (True, True),  # Flow retrieval enabled
            (False, False),  # Flow retrieval disabled
        ],
    )
    async def test_predict_commands_with_flow_retrieval(
        self,
        flow_retrieval_active: bool,
        should_call_flow_retrieval: bool,
        command_generator_fixture,
        model_storage: ModelStorage,
        resource: Resource,
        flows: FlowsList,
        monkeypatch: MonkeyPatch,
    ):
        # Given
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "my key")
        config = {
            FLOW_RETRIEVAL_KEY: {FLOW_RETRIEVAL_ACTIVE_KEY: flow_retrieval_active},
        }
        generator_class = command_generator_fixture.__class__
        generator = generator_class.create(
            config=config,
            model_storage=model_storage,
            resource=resource,
            execution_context=Mock(spec=ExecutionContext),
        )

        if should_call_flow_retrieval:
            # Mock the flow retrieval filter_flows method
            generator.flow_retrieval = Mock()
            generator.flow_retrieval.filter_flows = AsyncMock(return_value=flows)

        # When
        await generator.predict_commands(
            Message(),
            flows,
            DialogueStateTracker.from_events(
                "test",
                evts=[UserUttered("Hello", {"name": "greet", "confidence": 1.0})],
            ),
        )

        # Then
        if should_call_flow_retrieval:
            generator.flow_retrieval.filter_flows.assert_called_once()
        else:
            assert generator.flow_retrieval is None

    async def test_predict_commands_with_mocked_flow_retrieval(
        self,
        command_generator_fixture,
        model_storage: ModelStorage,
        resource: Resource,
        flows: FlowsList,
        monkeypatch: MonkeyPatch,
    ):
        # Given
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "my key")
        config = {
            FLOW_RETRIEVAL_KEY: {FLOW_RETRIEVAL_ACTIVE_KEY: True},
        }
        generator_class = command_generator_fixture.__class__
        generator = generator_class.create(
            config=config,
            model_storage=model_storage,
            resource=resource,
            execution_context=Mock(spec=ExecutionContext),
        )

        # Mock the flow retrieval filter_flows method to return
        # a specific subset of flows
        mock_filtered_flows = FlowsList(underlying_flows=[])
        generator.flow_retrieval = Mock()
        generator.flow_retrieval.filter_flows = AsyncMock(
            return_value=mock_filtered_flows
        )

        # When
        result = await generator.predict_commands(
            Message(),
            flows,
            DialogueStateTracker.from_events(
                "test",
                evts=[UserUttered("Hello", {"name": "greet", "confidence": 1.0})],
            ),
        )

        # Then
        generator.flow_retrieval.filter_flows.assert_called_once()
        assert len(result) == 1

    @patch(
        "rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval.filter_flows"
    )
    async def test_predict_commands_and_flow_retrieval_api_error_throws_exception(
        self,
        mock_flow_retrieval_filter_flows,
        command_generator_fixture,
        tracker: DialogueStateTracker,
        flows: FlowsList,
    ) -> None:
        generator = command_generator_fixture
        message = Mock()
        message.data = {TEXT: "some_message"}
        mock_flow_retrieval_filter_flows.side_effect = ProviderClientAPIException(
            message="Test Exception", original_exception=Exception("API exception")
        )

        predicted_commands = await generator.predict_commands(message, flows, tracker)

        mock_flow_retrieval_filter_flows.assert_called_once()

        assert len(predicted_commands) == 2
        assert ErrorCommand() in predicted_commands
        assert SetSlotCommand(ROUTE_TO_CALM_SLOT, True) in predicted_commands

    ### Tests for methods implemented in the base class
    # Parameterized fixture
    @pytest.fixture(
        params=[
            "base_class",
            "implementation_single_step",
            "implementation_multi_step",
        ]
    )
    def base_command_generator_fixture(
        self,
        request,
        base_class_fixture,
        single_step_llm_command_generator_fixture,
        multi_step_llm_command_generator_fixture,
        model_storage,
        resource,
    ):
        if request.param == "base_class":
            config = {
                FLOW_RETRIEVAL_KEY: {FLOW_RETRIEVAL_ACTIVE_KEY: True},
            }
            return base_class_fixture(
                config=config, model_storage=model_storage, resource=resource
            )
        if request.param == "implementation_single_step":
            return single_step_llm_command_generator_fixture
        elif request.param == "implementation_multi_step":
            return multi_step_llm_command_generator_fixture
        else:
            raise ValueError("Unknown fixture type")

    @patch(
        "rasa.dialogue_understanding.generator.llm_based_command_generator.llm_factory"
    )
    async def test_generate_action_list_calls_llm_factory_correctly(
        self,
        mock_llm_factory: Mock,
        base_command_generator_fixture,
    ):
        """Test that _generate_action_list calls llm correctly."""
        command_generator = base_command_generator_fixture

        # Given
        expected_llm_config = {
            "model": "gpt-4",
            "provider": "openai",
            "timeout": 7,
            "temperature": 0.0,
            "max_tokens": 256,
        }
        mock_llm_factory.return_value = AsyncMock(spec=OpenAILLMClient)

        # When
        await command_generator.invoke_llm("some prompt")

        # Then
        mock_llm_factory.assert_called_once_with(None, expected_llm_config)

    @patch(
        "rasa.dialogue_understanding.generator.llm_based_command_generator.llm_factory"
    )
    async def test_generate_action_list_calls_llm_correctly(
        self,
        mock_llm_factory: Mock,
        base_command_generator_fixture,
    ):
        """Test that _generate_action_list calls llm correctly."""
        # Given
        command_generator = base_command_generator_fixture
        llm_mock = Mock()
        predict_mock = AsyncMock()
        llm_mock.acompletion = predict_mock
        mock_llm_factory.return_value = llm_mock

        # When
        await command_generator.invoke_llm("some prompt")
        # Then
        predict_mock.assert_called_once_with("some prompt")

    @patch(
        "rasa.dialogue_understanding.generator.llm_based_command_generator.llm_factory"
    )
    async def test_generate_action_list_catches_llm_exception(
        self,
        mock_llm_factory: Mock,
        base_command_generator_fixture,
    ):
        """Test that _generate_action_list calls llm correctly."""
        command_generator = base_command_generator_fixture
        mock_llm = AsyncMock()
        mock_llm.acompletion = AsyncMock(side_effect=Exception("API exception"))
        mock_llm_factory.return_value = mock_llm

        # When
        with capture_logs() as logs:
            with pytest.raises(ProviderClientAPIException):
                await command_generator.invoke_llm("some prompt")

            # Then
            assert len(logs) == 1
            assert logs[0]["event"] == "llm_based_command_generator.llm.error"

    @pytest.mark.parametrize(
        "input_value, expected_output",
        [
            ("text", "text"),
            (" text ", "text"),
            ('"text"', "text"),
            ("'text'", "text"),
            ("' \"text' \"  ", "text"),
            ("", ""),
        ],
    )
    def test_clean_extracted_value(
        self, input_value: str, expected_output: str, base_command_generator_fixture
    ):
        """Test that clean_extracted_value removes
        the leading and trailing whitespaces.
        """
        command_generator = base_command_generator_fixture
        # When
        cleaned_value = command_generator.clean_extracted_value(input_value)
        # Then
        assert cleaned_value == expected_output

    @pytest.mark.parametrize(
        "input_value, expected_truthiness",
        [
            ("", False),
            (" ", False),
            ("none", False),
            ("some text", False),
            ("[missing information]", True),
            ("[missing]", True),
            ("None", True),
            ("undefined", True),
            ("null", True),
        ],
    )
    def test_is_none_value(
        self,
        input_value: str,
        expected_truthiness: bool,
        base_command_generator_fixture,
    ):
        """Test that is_none_value returns True when the value is None."""
        command_generator = base_command_generator_fixture
        assert command_generator.is_none_value(input_value) == expected_truthiness

    @pytest.mark.parametrize(
        "slot, slot_name, expected_output",
        [
            (TextSlot("test_slot", [], initial_value="hello"), "test_slot", "hello"),
            (TextSlot("test_slot", []), "some_other_slot", "undefined"),
        ],
    )
    def test_slot_value(
        self,
        slot: Slot,
        slot_name: str,
        expected_output: str,
        base_command_generator_fixture,
    ):
        """Test that slot_value returns the correct string."""
        command_generator = base_command_generator_fixture
        # Given
        tracker = DialogueStateTracker.from_events("test", evts=[], slots=[slot])
        # When
        slot_value = command_generator.get_slot_value(tracker, slot_name)

        assert slot_value == expected_output

    @pytest.fixture
    def collect_info_step(self) -> CollectInformationFlowStep:
        """Create a CollectInformationFlowStep."""
        return CollectInformationFlowStep(
            collect="test_slot",
            idx=0,
            ask_before_filling=True,
            utter="hello",
            collect_action="action_ask_hello",
            rejections=[SlotRejection("test_slot", "some rejection")],
            custom_id="collect",
            description="test_slot",
            metadata={},
            next="next_step",
        )

    def test_is_extractable_with_no_slot(
        self,
        base_command_generator_fixture,
        collect_info_step: CollectInformationFlowStep,
    ):
        """Test that is_extractable returns False
        when there are no slots to be filled.
        """
        command_generator = base_command_generator_fixture
        # Given
        tracker = DialogueStateTracker.from_events(sender_id="test", evts=[], slots=[])
        # When
        is_extractable = command_generator.is_extractable(collect_info_step, tracker)
        # Then
        assert not is_extractable

    def test_is_extractable_when_slot_can_be_filled_without_asking(
        self,
        base_command_generator_fixture,
    ):
        """Test that is_extractable returns True when
        collect_information slot can be filled.
        """
        command_generator = base_command_generator_fixture
        # Given
        tracker = DialogueStateTracker.from_events(
            sender_id="test", evts=[], slots=[TextSlot(name="test_slot", mappings=[])]
        )
        collect_info_step = CollectInformationFlowStep(
            collect="test_slot",
            ask_before_filling=False,
            utter="hello",
            collect_action="action_ask_hello",
            rejections=[SlotRejection("test_slot", "some rejection")],
            custom_id="collect_information",
            idx=0,
            description="test_slot",
            metadata={},
            next="next_step",
        )
        # When
        is_extractable = command_generator.is_extractable(collect_info_step, tracker)
        # Then
        assert is_extractable

    def test_is_extractable_when_slot_has_already_been_set(
        self,
        base_command_generator_fixture,
        collect_info_step: CollectInformationFlowStep,
    ):
        """Test that is_extractable returns True
        when collect_information can be filled.
        """
        command_generator = base_command_generator_fixture
        # Given
        slot = TextSlot(name="test_slot", mappings=[])
        tracker = DialogueStateTracker.from_events(
            sender_id="test", evts=[SlotSet("test_slot", "hello")], slots=[slot]
        )
        # When
        is_extractable = command_generator.is_extractable(collect_info_step, tracker)
        # Then
        assert is_extractable

    def test_is_extractable_with_current_step(
        self,
        base_command_generator_fixture,
        collect_info_step: CollectInformationFlowStep,
    ):
        """Test that is_extractable returns True when the current step is a collect
        information step and matches the information step.
        """
        command_generator = base_command_generator_fixture
        # Given
        tracker = DialogueStateTracker.from_events(
            sender_id="test",
            evts=[UserUttered("Hello"), BotUttered("Hi")],
            slots=[TextSlot(name="test_slot", mappings=[])],
        )
        # When
        is_extractable = command_generator.is_extractable(
            collect_info_step, tracker, current_step=collect_info_step
        )
        # Then
        assert is_extractable

    @pytest.mark.parametrize(
        "message, max_characters, expected_exceeds_limit",
        [
            ("Hello", 5, False),
            ("Hello! I'm a long message", 3, True),
            ("Hello! I'm a long message", -1, False),
        ],
    )
    def test_check_if_message_exceeds_limit(
        self,
        message: Text,
        max_characters: int,
        expected_exceeds_limit: bool,
        base_command_generator_fixture,
        model_storage: ModelStorage,
        resource: Resource,
    ):
        # Given
        config = {
            "user_input": {"max_characters": max_characters},
            FLOW_RETRIEVAL_KEY: {FLOW_RETRIEVAL_ACTIVE_KEY: True},
        }
        generator_class = base_command_generator_fixture.__class__
        generator = generator_class.create(
            config=config,
            model_storage=model_storage,
            resource=resource,
            execution_context=Mock(spec=ExecutionContext),
        )
        message = Message.build(text=message)
        # When
        exceeds_limit = generator.check_if_message_exceeds_limit(message)
        assert exceeds_limit == expected_exceeds_limit

    def test_import_rasa_generators_from_generator_module(
        self, model_storage, resource
    ):
        """Test that rasa generator modules can be imported
        without errors from generator module."""
        from rasa.dialogue_understanding.generator import (
            LLMCommandGenerator,
            SingleStepLLMCommandGenerator,
            MultiStepLLMCommandGenerator,
        )

        assert LLMCommandGenerator(
            config={},
            model_storage=model_storage,
            resource=resource,
            execution_context=Mock(spec=ExecutionContext),
        )
        assert SingleStepLLMCommandGenerator(
            config={},
            model_storage=model_storage,
            resource=resource,
            execution_context=Mock(spec=ExecutionContext),
        )
        assert MultiStepLLMCommandGenerator(
            config={},
            model_storage=model_storage,
            resource=resource,
            execution_context=Mock(spec=ExecutionContext),
        )

    def test_import_rasa_generators_directly(self, model_storage, resource):
        """Test that rasa generator modules can be imported
        without errors directly."""
        from rasa.dialogue_understanding.generator.llm_command_generator import (
            LLMCommandGenerator,
        )
        from rasa.dialogue_understanding.generator.multi_step.multi_step_llm_command_generator import (  # noqa: E501
            MultiStepLLMCommandGenerator,
        )
        from rasa.dialogue_understanding.generator.single_step.single_step_llm_command_generator import (  # noqa: E501
            SingleStepLLMCommandGenerator,
        )

        assert LLMCommandGenerator(
            config={},
            model_storage=model_storage,
            resource=resource,
            execution_context=Mock(spec=ExecutionContext),
        )
        assert SingleStepLLMCommandGenerator(
            config={},
            model_storage=model_storage,
            resource=resource,
            execution_context=Mock(spec=ExecutionContext),
        )
        assert MultiStepLLMCommandGenerator(
            config={},
            model_storage=model_storage,
            resource=resource,
            execution_context=Mock(spec=ExecutionContext),
        )

    base_classes: ClassVar[List[type]] = [
        LLMCommandGenerator,
        SingleStepLLMCommandGenerator,
        MultiStepLLMCommandGenerator,
    ]

    @pytest.mark.parametrize("base_class", base_classes)
    async def test_new_subclass_uses_own_predict_commands(
        self, base_class, flows, model_storage, resource
    ):
        """Test that if custom component has overriden the predict_commands
        method, it will be called and not the parent's."""

        class CustomCommandGenerator(base_class):
            async def predict_commands(
                self,
                message: Message,
                flows: FlowsList,
                tracker: DialogueStateTracker = None,
            ):
                return [ChitChatAnswerCommand()]

        message = Mock()
        message.data = {TEXT: "some_message"}
        tracker = Mock(spec=DialogueStateTracker)
        flows = FlowsList(underlying_flows=[])

        generator = CustomCommandGenerator(
            config={}, model_storage=model_storage, resource=resource
        )
        result = await generator.predict_commands(message, flows, tracker)

        assert result == [ChitChatAnswerCommand()]

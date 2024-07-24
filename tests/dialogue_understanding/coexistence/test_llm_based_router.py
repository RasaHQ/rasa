import uuid
from typing import List, Optional
from unittest.mock import Mock, patch, AsyncMock

import pytest
from _pytest.tmpdir import TempPathFactory
from structlog.testing import capture_logs

from rasa.dialogue_understanding.coexistence.constants import (
    CALM_ENTRY,
    STICKY,
)
from rasa.dialogue_understanding.coexistence.llm_based_router import (
    LLMBasedRouter,
    DEFAULT_LLM_CONFIG,
)
from rasa.dialogue_understanding.commands import Command, SetSlotCommand
from rasa.dialogue_understanding.commands.noop_command import NoopCommand
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.constants import ROUTE_TO_CALM_SLOT
from rasa.shared.core.slots import BooleanSlot
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import InvalidConfigException
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData

EXPECTED_PROMPT_PATH = "./tests/dialogue_understanding/coexistence/rendered_prompt.txt"


class TestLLMBasedRouter:
    @pytest.fixture
    def llm_based_router(self):
        """Create an LLMCommandGenerator."""
        return LLMBasedRouter.create(
            config={CALM_ENTRY: {STICKY: "handles transactions"}},
            resource=Mock(),
            model_storage=Mock(),
            execution_context=Mock(),
        )

    @pytest.fixture(scope="session")
    def resource(self) -> Resource:
        return Resource(uuid.uuid4().hex)

    @pytest.fixture(scope="session")
    def model_storage(self, tmp_path_factory: TempPathFactory) -> ModelStorage:
        return LocalModelStorage(tmp_path_factory.mktemp(uuid.uuid4().hex))

    def test_llm_based_router_prompt_init_custom(
        self, model_storage: ModelStorage, resource: Resource
    ) -> None:
        llm_based_router = LLMBasedRouter(
            {
                "prompt": "data/test_prompt_templates/test_prompt.jinja2",
                CALM_ENTRY: {STICKY: "handles transactions"},
            },
            model_storage,
            resource,
        )
        assert llm_based_router.prompt_template.startswith("This is a test prompt.")

        resource = llm_based_router.train(TrainingData())
        loaded = LLMBasedRouter.load(
            {CALM_ENTRY: {STICKY: "handles transactions"}},
            model_storage,
            resource,
            None,
        )
        assert loaded.prompt_template.startswith("This is a test prompt.")

    def test_llm_based_router_without_calm_capabilities(
        self, model_storage: ModelStorage, resource: Resource
    ) -> None:
        # the calm capabilities need to be defined
        with pytest.raises(ValueError):
            LLMBasedRouter(
                {},
                model_storage,
                resource,
            )

        with pytest.raises(ValueError):
            LLMBasedRouter(
                {CALM_ENTRY: {}},
                model_storage,
                resource,
            )

    async def test_llm_based_router_process_with_no_tracker(
        self, llm_based_router: LLMBasedRouter
    ) -> None:
        message = Message.build(text="some message")
        returned_messages = await llm_based_router.process([message], None)

        assert len(returned_messages) == 1
        assert returned_messages[0] == message

    @pytest.mark.parametrize(
        "answer, commands",
        [
            (None, [SetSlotCommand(ROUTE_TO_CALM_SLOT, False)]),
            ("A ", []),
            ("B ", [NoopCommand()]),
            ("C ", [SetSlotCommand(ROUTE_TO_CALM_SLOT, False)]),
            ("A", []),
            ("B", [NoopCommand()]),
            ("C", [SetSlotCommand(ROUTE_TO_CALM_SLOT, False)]),
            ("something else", [SetSlotCommand(ROUTE_TO_CALM_SLOT, False)]),
        ],
    )
    def test_llm_based_router_parse_answer(
        self,
        answer: Optional[str],
        commands: List[Command],
        llm_based_router: LLMBasedRouter,
    ) -> None:
        actual_commands = llm_based_router.parse_answer(answer)
        assert actual_commands == commands

    async def test_llm_based_router_predict_commands_without_routing_slot(
        self, llm_based_router: LLMBasedRouter
    ) -> None:
        message = Message.build(text="some message")
        tracker = DialogueStateTracker("sender_id", [])

        # the routing slot needs to be present in the tracker
        with pytest.raises(InvalidConfigException):
            await llm_based_router.predict_commands(message, tracker)

    @pytest.mark.parametrize(
        "initial_value, commands",
        [(False, [NoopCommand()]), (True, [])],
    )
    async def test_llm_based_router_predict_commands_with_routing_slot_already_set(
        self,
        initial_value: Optional[bool],
        commands: List[Command],
        llm_based_router: (LLMBasedRouter),
    ) -> None:
        message = Message.build(text="some message")
        tracker = DialogueStateTracker(
            "sender_id",
            [BooleanSlot(ROUTE_TO_CALM_SLOT, mappings=[], initial_value=initial_value)],
        )

        # When
        with patch(
            "rasa.dialogue_understanding.coexistence.llm_based_router.llm_factory",
            Mock(),
        ) as mock_llm_factory:
            llm_mock = Mock()
            apredict_mock = AsyncMock(return_value="StartFlow(test_flow)")
            llm_mock.apredict = apredict_mock
            mock_llm_factory.return_value = llm_mock
            actual_commands = await llm_based_router.predict_commands(message, tracker)

        apredict_mock.assert_not_called()
        assert actual_commands == commands

    async def test_llm_based_router_predict_commands_with_routing_slot_set_to_none(
        self,
        llm_based_router: (LLMBasedRouter),
    ) -> None:
        message = Message.build(text="some message")
        tracker = DialogueStateTracker(
            "sender_id",
            [BooleanSlot(ROUTE_TO_CALM_SLOT, mappings=[], initial_value=None)],
        )

        # When
        with patch(
            "rasa.dialogue_understanding.coexistence.llm_based_router.llm_factory",
            Mock(),
        ) as mock_llm_factory:
            llm_mock = Mock()
            apredict_mock = AsyncMock(return_value="StartFlow(test_flow)")
            llm_mock.apredict = apredict_mock
            mock_llm_factory.return_value = llm_mock
            await llm_based_router.predict_commands(message, tracker)

            mock_llm_factory.assert_called_once_with(None, DEFAULT_LLM_CONFIG)
            apredict_mock.assert_called_once()

    async def test_predict_commands_llm_error(self, llm_based_router: (LLMBasedRouter)):
        message = Message.build(text="some message")
        tracker = DialogueStateTracker(
            "sender_id",
            [BooleanSlot(ROUTE_TO_CALM_SLOT, mappings=[], initial_value=None)],
        )

        mock_llm = Mock(side_effect=Exception("some exception"))
        with patch(
            "rasa.dialogue_understanding.coexistence.llm_based_router.llm_factory",
            Mock(return_value=mock_llm),
        ):
            with capture_logs() as logs:
                await llm_based_router.predict_commands(message, tracker)
                # Then
                assert len(logs) == 5
                assert logs[0]["event"] == "llm_based_router.prompt_rendered"
                assert logs[1]["event"] == "llm_based_router.llm.error"
                assert logs[2]["event"] == "llm_based_router.llm_answer"
                assert (
                    logs[3]["event"] == "llm_based_router.parse_answer.invalid_answer"
                )
                assert logs[4]["event"] == "llm_based_router.final_commands"

    def test_render_template(
        self,
        llm_based_router: LLMBasedRouter,
    ):
        message = Message.build(text="some message")

        with open(EXPECTED_PROMPT_PATH, "r", encoding="unicode_escape") as f:
            expected_template = f.readlines()

        rendered_template = llm_based_router.render_template(message=message)

        for rendered_line, expected_line in zip(
            rendered_template.splitlines(True), expected_template
        ):
            assert rendered_line.strip() == expected_line.strip()

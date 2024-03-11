import uuid
from typing import List, Optional
from unittest.mock import Mock, patch

import pytest
from _pytest.tmpdir import TempPathFactory
from structlog.testing import capture_logs

from rasa.dialogue_understanding.commands import Command, SetSlotCommand
from rasa.dialogue_understanding.commands.noop_command import NoopCommand
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.dialogue_understanding.coexistence.coexistence_router import (
    CoexistenceRouter,
    CALM_CAPABILITIES,
    DEFAULT_LLM_CONFIG,
)
from rasa.shared.constants import ROUTE_TO_CALM_SLOT
from rasa.shared.core.slots import BooleanSlot
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import InvalidConfigException
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData

EXPECTED_PROMPT_PATH = "./tests/dialogue_understanding/coexistence/rendered_prompt.txt"


class TestCoexistenceRouter:
    @pytest.fixture
    def coexistence_router(self):
        """Create an LLMCommandGenerator."""
        return CoexistenceRouter.create(
            config={CALM_CAPABILITIES: "handles transactions"},
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

    def test_coexistence_router_prompt_init_custom(
        self, model_storage: ModelStorage, resource: Resource
    ) -> None:
        coexistence_router = CoexistenceRouter(
            {
                "prompt": "data/test_prompt_templates/test_prompt.jinja2",
                CALM_CAPABILITIES: "handles transactions",
            },
            model_storage,
            resource,
        )
        assert coexistence_router.prompt_template.startswith("This is a test prompt.")

        resource = coexistence_router.train(TrainingData())
        loaded = CoexistenceRouter.load(
            {CALM_CAPABILITIES: "handles transactions"}, model_storage, resource, None
        )
        assert loaded.prompt_template.startswith("This is a test prompt.")

    def test_coexistence_router_without_calm_capabilities(
        self, model_storage: ModelStorage, resource: Resource
    ) -> None:
        # the calm capabilities need to be defined
        with pytest.raises(ValueError):
            CoexistenceRouter(
                {},
                model_storage,
                resource,
            )

    def test_coexistence_router_process_with_no_tracker(
        self, coexistence_router: CoexistenceRouter
    ) -> None:
        message = Message.build(text="some message")
        returned_messages = coexistence_router.process([message], None)

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
    def test_coexistence_parse_answer(
        self,
        answer: Optional[str],
        commands: List[Command],
        coexistence_router: CoexistenceRouter,
    ) -> None:
        actual_commands = coexistence_router.parse_answer(answer)
        assert actual_commands == commands

    def test_coexistence_router_predict_commands_without_routing_slot(
        self, coexistence_router: CoexistenceRouter
    ) -> None:
        message = Message.build(text="some message")
        tracker = DialogueStateTracker("sender_id", [])

        # the routing slot needs to be present in the tracker
        with pytest.raises(InvalidConfigException):
            coexistence_router.predict_commands(message, tracker)

    @pytest.mark.parametrize(
        "initial_value, commands",
        [(False, [NoopCommand()]), (True, [])],
    )
    def test_coexistence_router_predict_commands_with_routing_slot_already_set(
        self,
        initial_value: Optional[bool],
        commands: List[Command],
        coexistence_router: (CoexistenceRouter),
    ) -> None:
        message = Message.build(text="some message")
        tracker = DialogueStateTracker(
            "sender_id",
            [BooleanSlot(ROUTE_TO_CALM_SLOT, mappings=[], initial_value=initial_value)],
        )

        # When
        with patch(
            "rasa.dialogue_understanding.coexistence.coexistence_router.llm_factory",
            Mock(),
        ) as mock_llm_factory:
            actual_commands = coexistence_router.predict_commands(message, tracker)

        mock_llm_factory.assert_not_called()
        assert actual_commands == commands

    def test_coexistence_router_predict_commands_with_routing_slot_set_to_none(
        self,
        coexistence_router: (CoexistenceRouter),
    ) -> None:
        message = Message.build(text="some message")
        tracker = DialogueStateTracker(
            "sender_id",
            [BooleanSlot(ROUTE_TO_CALM_SLOT, mappings=[], initial_value=None)],
        )

        # When
        with patch(
            "rasa.dialogue_understanding.coexistence.coexistence_router.llm_factory",
            Mock(),
        ) as mock_llm_factory:
            mock_llm_factory.return_value = Mock()
            coexistence_router.predict_commands(message, tracker)

            mock_llm_factory.assert_called_once_with(None, DEFAULT_LLM_CONFIG)
            mock_llm_factory.return_value.assert_called_once()

    def test_predict_commands_llm_error(self, coexistence_router: (CoexistenceRouter)):
        message = Message.build(text="some message")
        tracker = DialogueStateTracker(
            "sender_id",
            [BooleanSlot(ROUTE_TO_CALM_SLOT, mappings=[], initial_value=None)],
        )

        mock_llm = Mock(side_effect=Exception("some exception"))
        with (
            (
                patch(
                    "rasa.dialogue_understanding.coexistence.coexistence_router.llm_factory",
                    Mock(return_value=mock_llm),
                )
            )
        ):
            with capture_logs() as logs:
                coexistence_router.predict_commands(message, tracker)
                # Then
                assert len(logs) == 5
                assert logs[0]["event"] == "coexistence_router.prompt_rendered"
                assert logs[1]["event"] == "coexistence_router.llm.error"
                assert logs[2]["event"] == "coexistence_router.llm_answer"
                assert (
                    logs[3]["event"] == "coexistence_router.parse_answer.invalid_answer"
                )
                assert logs[4]["event"] == "coexistence_router.predicated_commands"

    def test_render_template(
        self,
        coexistence_router: CoexistenceRouter,
    ):
        message = Message.build(text="some message")

        with open(EXPECTED_PROMPT_PATH, "r", encoding="unicode_escape") as f:
            expected_template = f.readlines()

        rendered_template = coexistence_router.render_template(message=message)

        for rendered_line, expected_line in zip(
            rendered_template.splitlines(True), expected_template
        ):
            assert rendered_line.strip() == expected_line.strip()

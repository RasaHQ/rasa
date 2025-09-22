import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from _pytest.tmpdir import TempPathFactory
from pytest import MonkeyPatch
from structlog.testing import capture_logs

import rasa.shared.utils.io
from rasa.dialogue_understanding.coexistence.constants import (
    CALM_ENTRY,
    STICKY,
)
from rasa.dialogue_understanding.coexistence.llm_based_router import (
    DEFAULT_LLM_CONFIG,
    LLM_BASED_ROUTER_CONFIG_FILE_NAME,
    LLMBasedRouter,
)
from rasa.dialogue_understanding.commands import Command, SetSlotCommand
from rasa.dialogue_understanding.commands.noop_command import NoopCommand
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.constants import (
    LLM_CONFIG_KEY,
    MODEL_GROUP_CONFIG_KEY,
    OPENAI_API_KEY_ENV_VAR,
    PROMPT_CONFIG_KEY,
    PROMPT_TEMPLATE_CONFIG_KEY,
    ROUTE_TO_CALM_SLOT,
)
from rasa.shared.core.slots import BooleanSlot
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import InvalidConfigException
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.providers.llm.llm_response import LLMResponse

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
        self, model_storage: ModelStorage, resource: Resource, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "mock key in test llm_based_router")
        llm_based_router = LLMBasedRouter(
            {
                PROMPT_TEMPLATE_CONFIG_KEY: "data/test_prompt_templates/test_prompt.jinja2",  # noqa: E501
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

    def test_llm_based_router_prompt_template_init_custom(
        self, model_storage: ModelStorage, resource: Resource, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "mock key in test llm_based_router")
        llm_based_router = LLMBasedRouter(
            {
                PROMPT_TEMPLATE_CONFIG_KEY: "data/test_prompt_templates/test_prompt.jinja2",  # noqa: E501
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
    @patch("rasa.dialogue_understanding.coexistence.llm_based_router.llm_factory")
    async def test_llm_based_router_predict_commands_with_routing_slot_already_set(
        self,
        mock_llm_factory: Mock,
        initial_value: Optional[bool],
        commands: List[Command],
        llm_based_router: LLMBasedRouter,
    ) -> None:
        # Given
        message = Message.build(text="some message")
        tracker = DialogueStateTracker(
            "sender_id",
            [BooleanSlot(ROUTE_TO_CALM_SLOT, mappings=[], initial_value=initial_value)],
        )

        llm_mock = Mock()
        llm_mock.acompletion.return_value = AsyncMock(
            spec=LLMResponse, choices=["StartFlow(test_flow)"]
        )
        mock_llm_factory.return_value = llm_mock

        # When
        actual_commands = await llm_based_router.predict_commands(message, tracker)

        # Then
        llm_mock.acompletion.assert_not_called()
        assert actual_commands == commands

    @patch("rasa.dialogue_understanding.coexistence.llm_based_router.llm_factory")
    async def test_llm_based_router_predict_commands_with_routing_slot_set_to_none(
        self,
        mock_llm_factory: Mock,
        llm_based_router: LLMBasedRouter,
    ) -> None:
        # Given
        message = Message.build(text="some message")
        tracker = DialogueStateTracker(
            "sender_id",
            [BooleanSlot(ROUTE_TO_CALM_SLOT, mappings=[], initial_value=None)],
        )

        llm_mock = Mock()
        llm_mock.acompletion.return_value = AsyncMock(
            spec=LLMResponse, choices=["StartFlow(test_flow)"]
        )
        mock_llm_factory.return_value = llm_mock
        await llm_based_router.predict_commands(message, tracker)

        mock_llm_factory.assert_called_once_with(None, DEFAULT_LLM_CONFIG)
        llm_mock.acompletion.assert_called_once()

    @patch("rasa.dialogue_understanding.coexistence.llm_based_router.llm_factory")
    async def test_predict_commands_llm_error(
        self, mock_llm_factory: Mock, llm_based_router: LLMBasedRouter
    ):
        # Given
        message = Message.build(text="some message")
        tracker = DialogueStateTracker(
            "sender_id",
            [BooleanSlot(ROUTE_TO_CALM_SLOT, mappings=[], initial_value=None)],
        )

        mock_llm = AsyncMock()
        mock_llm.acompletion = AsyncMock(side_effect=Exception("some exception"))
        mock_llm_factory.return_value = mock_llm

        with capture_logs() as logs:
            await llm_based_router.predict_commands(message, tracker)
            # Then
            assert len(logs) == 5
            assert logs[0]["event"] == "llm_based_router.prompt_rendered"
            assert logs[1]["event"] == "llm_based_router.llm.error"
            assert logs[2]["event"] == "llm_based_router.llm_answer"
            assert logs[3]["event"] == "llm_based_router.parse_answer.invalid_answer"
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

    @pytest.mark.parametrize(
        "config, expected_llm_config",
        [
            (
                {
                    LLM_CONFIG_KEY: {"provider": "openai", "model": "gpt-4"},
                },
                {"provider": "openai", "model": "gpt-4"},
            ),
            (
                {
                    "user_input": {"max_characters": -1},
                },
                None,
            ),
            (
                {
                    LLM_CONFIG_KEY: {MODEL_GROUP_CONFIG_KEY: "openai_gpt-4"},
                },
                {
                    "id": "openai_gpt-4",
                    "models": [{"provider": "openai", "model": "gpt-4"}],
                },
            ),
        ],
    )
    def test_llm_based_router_init_with_different_llm_configs(
        self,
        config: Optional[Dict[str, Any]],
        expected_llm_config: Optional[Dict[str, Any]],
        model_storage: ModelStorage,
        resource: Resource,
        mock_available_endpoints: MagicMock,
        mock_configuration: MagicMock,
        monkeypatch,
    ) -> None:
        mock_available_endpoints.model_groups = [
            {
                "id": "openai_gpt-4",
                "models": [{"provider": "openai", "model": "gpt-4"}],
            },
            {
                "id": "openai_embedding",
                "models": [{"provider": "openai", "model": "text-embedding-3-large"}],
            },
        ]

        monkeypatch.setattr("rasa.shared.utils.llm.Configuration", mock_configuration)

        config[CALM_ENTRY] = {STICKY: "handles transactions"}

        generator = LLMBasedRouter(
            config,
            model_storage,
            resource,
        )
        assert generator.config[LLM_CONFIG_KEY] == expected_llm_config

    def test_llm_based_router_persist_config(
        self,
        model_storage: LocalModelStorage,
        resource: Resource,
        mock_available_endpoints: MagicMock,
        mock_configuration: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_available_endpoints.model_groups = [
            {
                "id": "model_group_id",
                "models": [{"provider": "openai", "model": "gpt-4"}],
            }
        ]

        monkeypatch.setattr("rasa.shared.utils.llm.Configuration", mock_configuration)

        config = {
            LLM_CONFIG_KEY: {MODEL_GROUP_CONFIG_KEY: "model_group_id"},
            CALM_ENTRY: {STICKY: "handles transactions"},
        }
        router = LLMBasedRouter(config, model_storage, resource)

        # Ensure the config is resolved
        assert router.config[LLM_CONFIG_KEY] == {
            "id": "model_group_id",
            "models": [{"provider": "openai", "model": "gpt-4"}],
        }

        # Persist the generator
        router.persist()

        # Check that the persisted config is equal to our config
        with model_storage.read_from(resource) as path:
            persisted_config = rasa.shared.utils.io.read_json_file(
                path / LLM_BASED_ROUTER_CONFIG_FILE_NAME
            )
        assert persisted_config[LLM_CONFIG_KEY] == {
            "id": "model_group_id",
            "models": [{"provider": "openai", "model": "gpt-4"}],
        }

    @pytest.mark.parametrize(
        "config_1, model_groups_1, config_2, model_groups_2, fingerprint_differs",
        [
            (
                {CALM_ENTRY: {STICKY: "handles transactions"}},
                [],
                {CALM_ENTRY: {STICKY: "handles transactions"}},
                [],
                False,
            ),
            (
                {
                    LLM_CONFIG_KEY: {MODEL_GROUP_CONFIG_KEY: "openai_gpt"},
                    CALM_ENTRY: {STICKY: "handles transactions"},
                },
                [
                    {
                        "id": "openai_gpt",
                        "models": [{"provider": "openai", "model": "gpt-4"}],
                    },
                ],
                {
                    LLM_CONFIG_KEY: {MODEL_GROUP_CONFIG_KEY: "openai_gpt"},
                    CALM_ENTRY: {STICKY: "handles transactions"},
                },
                [
                    {
                        "id": "openai_gpt",
                        "models": [
                            {"provider": "openai", "model": "gpt-4o-2024-11-20"}
                        ],
                    },
                ],
                True,
            ),
            (
                {
                    LLM_CONFIG_KEY: {MODEL_GROUP_CONFIG_KEY: "openai_gpt-1"},
                    CALM_ENTRY: {STICKY: "handles transactions"},
                },
                [
                    {
                        "id": "openai_gpt-1",
                        "models": [{"provider": "openai", "model": "gpt-4"}],
                    },
                ],
                {
                    LLM_CONFIG_KEY: {MODEL_GROUP_CONFIG_KEY: "openai_gpt-2"},
                    CALM_ENTRY: {STICKY: "handles transactions"},
                },
                [
                    {
                        "id": "openai_gpt-2",
                        "models": [
                            {"provider": "openai", "model": "gpt-4o-2024-11-20"}
                        ],
                    },
                ],
                True,
            ),
        ],
    )
    async def test_llm_based_router_fingerprint_addon_with_different_model_configs(
        self,
        config_1: Dict[str, Any],
        model_groups_1: List[Dict[str, Any]],
        config_2: Dict[str, Any],
        model_groups_2: List[Dict[str, Any]],
        fingerprint_differs: bool,
        model_storage: ModelStorage,
        mock_available_endpoints: MagicMock,
        mock_configuration: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        generator = LLMBasedRouter(
            {CALM_ENTRY: {STICKY: "handles transactions"}},
            model_storage,
            Resource("llmcmdgen"),
        )

        monkeypatch.setattr("rasa.shared.utils.llm.Configuration", mock_configuration)

        mock_available_endpoints.model_groups = model_groups_1
        fingerprint_1 = generator.fingerprint_addon(config_1)

        mock_available_endpoints.model_groups = model_groups_2
        fingerprint_2 = generator.fingerprint_addon(config_2)

        assert fingerprint_1 is not None
        assert fingerprint_2 is not None
        if fingerprint_differs:
            assert fingerprint_1 != fingerprint_2
        else:
            assert fingerprint_1 == fingerprint_2

    async def test_llm_based_router_fingerprint_addon_diff_in_prompt_template(
        self,
        model_storage: ModelStorage,
        tmp_path: Path,
    ) -> None:
        prompt_dir = Path(tmp_path) / "prompt"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        prompt_file = prompt_dir / "llm_based_router_prompt.jinja2"
        prompt_file.write_text("This is a test prompt")

        config = {
            PROMPT_TEMPLATE_CONFIG_KEY: str(prompt_file),
            CALM_ENTRY: {STICKY: "handles transactions"},
        }
        generator = LLMBasedRouter(config, model_storage, Resource("llmcmdgen"))
        fingerprint_1 = generator.fingerprint_addon(config)

        prompt_file.write_text("This is a test prompt. It has been changed.")
        fingerprint_2 = generator.fingerprint_addon(config)
        assert fingerprint_1 != fingerprint_2

    async def test_llm_based_router_fingerprint_addon_no_diff_in_prompt_template(
        self,
        model_storage: ModelStorage,
        tmp_path: Path,
    ) -> None:
        prompt_dir = Path(tmp_path) / "prompt"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        prompt_file = prompt_dir / "llm_command_generator_prompt.jinja2"
        prompt_file.write_text("This is a test prompt")

        config = {
            PROMPT_TEMPLATE_CONFIG_KEY: str(prompt_file),
            CALM_ENTRY: {STICKY: "handles transactions"},
        }
        generator = LLMBasedRouter(config, model_storage, Resource("llmcmdgen"))

        fingerprint_1 = generator.fingerprint_addon(config)
        fingerprint_2 = generator.fingerprint_addon(config)
        assert fingerprint_1 is not None
        assert fingerprint_1 == fingerprint_2

    async def test_llm_based_router_fingerprint_addon_default_values(
        self,
        model_storage: ModelStorage,
    ) -> None:
        generator = LLMBasedRouter(
            {CALM_ENTRY: {STICKY: "handles transactions"}},
            model_storage,
            Resource("llmcmdgen"),
        )
        fingerprint_1 = generator.fingerprint_addon({})
        fingerprint_2 = generator.fingerprint_addon({})
        assert fingerprint_1 is not None
        assert fingerprint_1 == fingerprint_2

    async def test_deprecation_warning_with_prompt(
        self, resource: Resource, model_storage: ModelStorage
    ):
        # When
        with patch("rasa.shared.utils.llm.structlogger.warning") as mock_warning:
            LLMBasedRouter(
                {
                    PROMPT_CONFIG_KEY: "data/test_prompt_templates/test_prompt.jinja2",
                    CALM_ENTRY: {STICKY: "handles transactions"},
                },
                model_storage,
                resource,
            )
        mock_warning.assert_any_call(
            "llm_based_router.init.deprecated_config_key",
            event_info=(
                "The config parameter 'prompt' is deprecated "
                "and will be removed in Rasa 4.0.0. "
                "Please use the config parameter 'prompt_template' instead. "
            ),
        )

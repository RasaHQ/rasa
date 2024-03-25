import uuid
from typing import Any, Dict
from unittest.mock import Mock

import pytest
from _pytest.tmpdir import TempPathFactory

from rasa.dialogue_understanding.coexistence.constants import (
    CALM_ENTRY,
    NLU_ENTRY,
    STICKY,
    NON_STICKY,
)
from rasa.dialogue_understanding.coexistence.intent_based_router import (
    IntentBasedRouter,
)
from rasa.dialogue_understanding.commands import SetSlotCommand
from rasa.dialogue_understanding.commands.noop_command import NoopCommand
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.constants import ROUTE_TO_CALM_SLOT
from rasa.shared.core.slots import BooleanSlot
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import InvalidConfigException
from rasa.shared.nlu.training_data.message import Message
from tests.utilities import flows_from_str


class TestIntentBasedRouter:
    @pytest.fixture
    def intent_based_router(self):
        """Create an LLMCommandGenerator."""
        return IntentBasedRouter.create(
            config={
                NLU_ENTRY: {STICKY: "nlu_sticky_a", NON_STICKY: "nlu_non_sticky_b"},
                CALM_ENTRY: {STICKY: "calm_sticky_c"},
            },
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

    @pytest.mark.parametrize(
        "config",
        [
            ({NLU_ENTRY: {STICKY: "a", NON_STICKY: "b"}}),
            ({CALM_ENTRY: {STICKY: "c"}}),
            ({NLU_ENTRY: {STICKY: "a"}, CALM_ENTRY: {STICKY: "c"}}),
            ({NLU_ENTRY: {NON_STICKY: "b"}, CALM_ENTRY: {STICKY: "c"}}),
            ({NLU_ENTRY: {STICKY: None, NON_STICKY: "b"}, CALM_ENTRY: {STICKY: "c"}}),
            ({NLU_ENTRY: {STICKY: "a", NON_STICKY: None}, CALM_ENTRY: {STICKY: "c"}}),
            ({NLU_ENTRY: {STICKY: "a", NON_STICKY: "b"}, CALM_ENTRY: {STICKY: None}}),
            ({}),
        ],
    )
    def test_intent_based_router_invalid_config(
        self, model_storage: ModelStorage, resource: Resource, config: Dict[str, Any]
    ) -> None:
        # the config must be defined.
        with pytest.raises(ValueError):
            IntentBasedRouter(config, model_storage, resource)

    @pytest.mark.parametrize(
        "config",
        [
            ({NLU_ENTRY: {STICKY: "a", NON_STICKY: "b"}, CALM_ENTRY: {STICKY: "c"}}),
            (
                {
                    NLU_ENTRY: {STICKY: ["a", "b", "c"], NON_STICKY: ["d", "e"]},
                    CALM_ENTRY: {STICKY: ["f"]},
                }
            ),
        ],
    )
    def test_intent_based_router_valid_config(
        self, model_storage: ModelStorage, resource: Resource, config: Dict[str, Any]
    ) -> None:
        intent_based_router = IntentBasedRouter(config, model_storage, resource)
        assert intent_based_router.config == config

    def test_intent_based_router_process_with_no_tracker(
        self, intent_based_router: IntentBasedRouter
    ) -> None:
        message = Message.build(text="some message")
        returned_messages = intent_based_router.process([message], None)

        assert len(returned_messages) == 1
        assert returned_messages[0] == message

    async def test_intent_based_router_process_with_no_calm_slot(
        self, intent_based_router: IntentBasedRouter
    ) -> None:
        message = Message.build(text="some message")
        tracker = DialogueStateTracker("sender_id", [])
        with pytest.raises(InvalidConfigException):
            await intent_based_router.predict_commands(message, None, tracker)

    async def test_intent_based_router_process_with_calm_slot_set_to_true(
        self, intent_based_router: IntentBasedRouter
    ) -> None:
        message = Message.build(text="some message")
        tracker = DialogueStateTracker(
            "sender_id",
            [BooleanSlot(ROUTE_TO_CALM_SLOT, mappings=[], initial_value=True)],
        )

        commands = await intent_based_router.predict_commands(message, None, tracker)
        assert commands == []

    async def test_intent_based_router_process_with_calm_slot_set_to_false(
        self, intent_based_router: IntentBasedRouter
    ) -> None:
        message = Message.build(text="some message")
        tracker = DialogueStateTracker(
            "sender_id",
            [BooleanSlot(ROUTE_TO_CALM_SLOT, mappings=[], initial_value=False)],
        )

        commands = await intent_based_router.predict_commands(message, None, tracker)
        assert commands == [NoopCommand()]

    async def test_intent_based_router_process_with_no_predicted_intent(
        self, intent_based_router: IntentBasedRouter
    ) -> None:
        message = Message.build(text="some message")
        tracker = DialogueStateTracker(
            "sender_id",
            [BooleanSlot(ROUTE_TO_CALM_SLOT, mappings=[], initial_value=None)],
        )

        commands = await intent_based_router.predict_commands(message, None, tracker)
        assert commands == [SetSlotCommand(ROUTE_TO_CALM_SLOT, False)]

    async def test_intent_based_router_process_nlu_entry_sticky(
        self, intent_based_router: IntentBasedRouter
    ) -> None:
        message = Message.build(text="some message")
        message.set("intent", {"name": "nlu_sticky_a", "confidence": 0.9})
        tracker = DialogueStateTracker(
            "sender_id",
            [BooleanSlot(ROUTE_TO_CALM_SLOT, mappings=[], initial_value=None)],
        )

        commands = await intent_based_router.predict_commands(message, None, tracker)
        assert commands == [SetSlotCommand(ROUTE_TO_CALM_SLOT, False)]

    async def test_intent_based_router_process_nlu_entry_non_sticky(
        self, intent_based_router: IntentBasedRouter
    ) -> None:
        message = Message.build(text="some message")
        message.set("intent", {"name": "nlu_non_sticky_b", "confidence": 0.9})
        tracker = DialogueStateTracker(
            "sender_id",
            [BooleanSlot(ROUTE_TO_CALM_SLOT, mappings=[], initial_value=None)],
        )

        commands = await intent_based_router.predict_commands(message, None, tracker)
        assert commands == [NoopCommand()]

    async def test_intent_based_router_process_calm_entry_sticky(
        self, intent_based_router: IntentBasedRouter
    ) -> None:
        message = Message.build(text="some message")
        message.set("intent", {"name": "calm_sticky_c", "confidence": 0.9})
        tracker = DialogueStateTracker(
            "sender_id",
            [BooleanSlot(ROUTE_TO_CALM_SLOT, mappings=[], initial_value=None)],
        )

        commands = await intent_based_router.predict_commands(message, None, tracker)
        assert commands == []

    async def test_intent_based_router_process_intent_in_nlu_trigger(
        self, intent_based_router: IntentBasedRouter
    ) -> None:
        message = Message.build(text="some message")
        message.set("intent", {"name": "foo", "confidence": 0.9})
        tracker = DialogueStateTracker(
            "sender_id",
            [BooleanSlot(ROUTE_TO_CALM_SLOT, mappings=[], initial_value=None)],
        )
        flows = flows_from_str(
            """
            flows:
              foo:
                description: flow foo
                nlu_trigger:
                  - intent: foo
                  - intent:
                      name: bar
                      confidence_threshold: 0.5
                  - intent:
                      name: foobar
                steps:
                  - action: utter_welcome
            """
        )

        commands = await intent_based_router.predict_commands(message, flows, tracker)
        assert commands == []

    async def test_intent_based_router_process_intent_not_satisfy_any_condition(
        self, intent_based_router: IntentBasedRouter
    ) -> None:
        message = Message.build(text="some message")
        message.set("intent", {"name": "some_intent", "confidence": 0.9})
        tracker = DialogueStateTracker(
            "sender_id",
            [BooleanSlot(ROUTE_TO_CALM_SLOT, mappings=[], initial_value=None)],
        )
        flows = flows_from_str(
            """
            flows:
              foo:
                description: flow foo
                nlu_trigger:
                  - intent: foo
                  - intent:
                      name: bar
                      confidence_threshold: 0.5
                  - intent:
                      name: foobar
                steps:
                  - action: utter_welcome
            """
        )

        commands = await intent_based_router.predict_commands(message, flows, tracker)
        assert commands == [SetSlotCommand(ROUTE_TO_CALM_SLOT, False)]

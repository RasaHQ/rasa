from pathlib import Path
from typing import Dict, Text, Any, Optional, List, Tuple

from rasa.architecture_prototype.persistence import LocalModelPersistor
from rasa.architecture_prototype.model import Model
import rasa.core.tracker_store
from rasa.core.channels import UserMessage, CollectingOutputChannel
from rasa.core.lock_store import LockStore
from rasa.core.nlg import NaturalLanguageGenerator
from rasa.core.policies.policy import PolicyPrediction
from rasa.core.processor import MessageProcessor
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import UserUttered
from rasa.shared.core.trackers import DialogueStateTracker
import rasa.core.actions.action
from rasa.utils.endpoints import EndpointConfig


class GraphProcessor(MessageProcessor):
    def __init__(
        self,
        domain: Domain,
        tracker_store: rasa.core.tracker_store.TrackerStore,
        lock_store: LockStore,
        generator: NaturalLanguageGenerator,
        action_endpoint: Optional[EndpointConfig],
        model: Model,
    ) -> None:
        super().__init__(
            domain=domain,
            tracker_store=tracker_store,
            lock_store=lock_store,
            generator=generator,
            action_endpoint=action_endpoint,
            interpreter=None,
            policy_ensemble=None,
        )
        self.model = model

    @classmethod
    def create(
        cls,
        model_path: Text,
        tracker_store: rasa.core.tracker_store.TrackerStore,
        lock_store: LockStore,
        generator: Optional[NaturalLanguageGenerator],
        action_endpoint: Optional[EndpointConfig],
    ) -> "GraphProcessor":
        model = Model.load(model_path, LocalModelPersistor(Path(model_path)))

        domain = model.get_domain()
        tracker_store.domain = domain
        if hasattr(generator, "responses"):
            generator.responses = domain.responses

        return GraphProcessor(
            domain, tracker_store, lock_store, generator, action_endpoint, model
        )

    async def handle_message(
        self, message: UserMessage
    ) -> Optional[List[Dict[Text, Any]]]:
        """Handle a single message with this processor."""
        tracker = await self.fetch_tracker_and_update_session(
            message.sender_id, message.output_channel, message.metadata
        )

        await self._predict_and_execute_next_action(
            message.output_channel, tracker, message
        )

        # save tracker state to continue conversation from this state
        self._save_tracker(tracker)

        if isinstance(message.output_channel, CollectingOutputChannel):
            return message.output_channel.messages

        return None

    def predict_next_action(
        self, tracker: DialogueStateTracker, message: Optional[UserMessage]
    ) -> Tuple[
        rasa.core.actions.action.Action, PolicyPrediction, Optional[UserUttered]
    ]:
        prediction, user_event = self.model.handle_message(tracker, message)

        action = rasa.core.actions.action.action_for_index(
            prediction.max_confidence_index, self.domain, self.action_endpoint
        )

        return action, prediction, user_event

    def is_core_ready(self) -> bool:
        return True

    def is_ready(self) -> bool:
        return True

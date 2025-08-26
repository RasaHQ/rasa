from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from rasa.builder.copilot.constants import ROLE_ASSISTANT, ROLE_USER
from rasa.shared.core.constants import DEFAULT_SLOT_NAMES
from rasa.shared.core.events import (
    ActionExecuted,
    BotUttered,
    FlowCompleted,
    FlowStarted,
    SlotSet,
    UserUttered,
)
from rasa.shared.core.trackers import DialogueStateTracker


class AssistantMessage(BaseModel):
    text: str


class UserMessage(BaseModel):
    text: str
    predicted_commands: List[str] = Field(default_factory=list)


class TrackerEvent(BaseModel):
    event: str
    data: Dict[str, Any] = Field(default_factory=dict)


class AssistantConversationTurn(BaseModel):
    user_message: Optional[UserMessage] = None
    assistant_messages: List[AssistantMessage] = Field(default_factory=list)
    context_events: List[TrackerEvent] = Field(default_factory=list)


class CurrentState(BaseModel):
    latest_message: Optional[str] = None
    active_flow: Optional[str] = None
    flow_stack: Optional[List[Dict[str, Any]]] = None
    slots: Optional[Dict[str, Any]] = None
    latest_action: Optional[str] = None
    followup_action: Optional[str] = None


class TrackerContext(BaseModel):
    conversation_turns: List[AssistantConversationTurn]
    current_state: CurrentState

    def to_openai_format(self) -> List[Dict[str, Any]]:
        """Convert the TrackerContext to a format suitable for OpenAI.

        Returns:
            A list of user and assistant messages in the format suitable for OpenAI.
        """
        messages: List[Dict[str, Any]] = []
        for turn in self.conversation_turns:
            if turn.user_message:
                messages.append({"role": ROLE_USER, "content": turn.user_message.text})
            for message in turn.assistant_messages:
                messages.append({"role": ROLE_ASSISTANT, "content": message.text})
        return messages

    @classmethod
    def from_tracker(
        cls, tracker: Optional[DialogueStateTracker], max_turns: int = 10
    ) -> Optional["TrackerContext"]:
        """Convert a tracker to a TrackerContext."""
        if not tracker or not tracker.events:
            return None

        conversation_turns = cls._build_conversation_turns(tracker)
        conversation_turns = conversation_turns[-max_turns:]
        current_state = cls._build_current_state(tracker)

        return cls(
            conversation_turns=conversation_turns,
            current_state=current_state,
        )

    @classmethod
    def _build_conversation_turns(
        cls, tracker: DialogueStateTracker
    ) -> List[AssistantConversationTurn]:
        """Build conversation turns from tracker events."""
        conversation_turns: List[AssistantConversationTurn] = []
        current_user_message: Optional[UserMessage] = None
        current_assistant_messages: List[AssistantMessage] = []
        current_context_events: List[TrackerEvent] = []

        for event in tracker.applied_events():
            if isinstance(event, UserUttered):
                # Save previous turn if exists and has content. A new turn starts with a
                # user message. However, since it's possible that "turn" started with an
                # assistant message, we save that turn without the user message.
                if (
                    current_user_message
                    or current_assistant_messages
                    or current_context_events
                ):
                    conversation_turns.append(
                        AssistantConversationTurn(
                            user_message=current_user_message,
                            assistant_messages=current_assistant_messages,
                            context_events=current_context_events,
                        )
                    )
                current_assistant_messages = []
                current_context_events = []

                # Start new turn

                # Fetch the predicted commands for the user message
                predicted_commands = (
                    [command.get("command") for command in event.commands]
                    if event.commands
                    else []
                )

                current_user_message = UserMessage(
                    text=event.text or "",
                    predicted_commands=predicted_commands,
                )

            # Assistant conversation turn can have multiple messages from the assistant
            elif isinstance(event, BotUttered):
                current_assistant_messages.append(
                    AssistantMessage(text=event.text or "")
                )

            # Handle non-user and non-assistant events. These are useful for more
            # adding context to the conversation turn.
            else:
                context_event = cls._process_tracker_event(event)
                if context_event:
                    current_context_events.append(context_event)

        # Add the final turn if there is one
        if current_user_message or current_assistant_messages:
            conversation_turns.append(
                AssistantConversationTurn(
                    user_message=current_user_message,
                    assistant_messages=current_assistant_messages,
                    context_events=current_context_events,
                )
            )

        return conversation_turns

    @staticmethod
    def _process_tracker_event(event: Any) -> Optional[TrackerEvent]:
        if isinstance(event, ActionExecuted):
            return TrackerEvent(
                event=ActionExecuted.type_name,
                data={"action_name": event.action_name, "confidence": event.confidence},
            )

        elif isinstance(event, SlotSet) and event.key not in DEFAULT_SLOT_NAMES:
            return TrackerEvent(
                event=SlotSet.type_name,
                data={"slot_name": event.key, "slot_value": event.value},
            )

        elif isinstance(event, FlowStarted):
            return TrackerEvent(
                event=FlowStarted.type_name,
                data={"flow_id": event.flow_id},
            )

        elif isinstance(event, FlowCompleted):
            return TrackerEvent(
                event=FlowCompleted.type_name,
                data={"flow_id": event.flow_id},
            )
        else:
            return None

    @classmethod
    def _build_current_state(cls, tracker: DialogueStateTracker) -> CurrentState:
        """Build the current state from the tracker."""
        latest_message = tracker.latest_message.text if tracker.latest_message else None
        latest_action = (
            tracker.latest_action.get("action_name") if tracker.latest_action else None
        )
        followup_action = tracker.followup_action
        flow_stack = tracker.stack.as_dict() if tracker.stack else None
        slots = cls._extract_non_default_slots(tracker)
        active_flow = tracker.active_flow

        return CurrentState(
            latest_message=latest_message,
            active_flow=active_flow,
            flow_stack=flow_stack,
            latest_action=latest_action,
            followup_action=followup_action,
            slots=slots,
        )

    @classmethod
    def _extract_non_default_slots(
        cls, tracker: DialogueStateTracker
    ) -> Optional[Dict[str, Any]]:
        """Extract non-default slot values from the tracker."""
        if not tracker.slots:
            return None

        return {
            k: str(v.value)
            for k, v in tracker.slots.items()
            if v is not None and k not in DEFAULT_SLOT_NAMES
        }

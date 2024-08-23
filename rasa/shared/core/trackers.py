import copy
import dataclasses
import itertools
import logging
import os
import time
from collections import deque
from enum import Enum
from typing import (
    Dict,
    Text,
    Any,
    Optional,
    Iterator,
    Generator,
    Type,
    TypeVar,
    List,
    Deque,
    Iterable,
    Union,
    FrozenSet,
    Tuple,
    TYPE_CHECKING,
    cast,
)

import rasa.shared.utils.io
from rasa.shared.constants import (
    ASSISTANT_ID_KEY,
    DEFAULT_SENDER_ID,
    ROUTE_TO_CALM_SLOT,
)
from rasa.shared.core import events
from rasa.shared.core.constants import (
    ACTION_LISTEN_NAME,
    LOOP_NAME,
    SHOULD_NOT_BE_SET,
    PREVIOUS_ACTION,
    ACTIVE_LOOP,
    ACTION_SESSION_START_NAME,
    FOLLOWUP_ACTION,
)
from rasa.shared.core.conversation import Dialogue
from rasa.shared.core.domain import Domain, State
from rasa.shared.core.events import (
    DialogueStackUpdated,
    RoutingSessionEnded,
    SlotSet,
    UserUttered,
    ActionExecuted,
    Event,
    Restarted,
    ActionReverted,
    UserUtteranceReverted,
    BotUttered,
    ActiveLoop,
    SessionStarted,
    ActionExecutionRejected,
    DefinePrevUserUtteredFeaturization,
    FlowStarted,
)
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.slots import AnySlot, Slot
from rasa.shared.nlu.constants import (
    ENTITY_ATTRIBUTE_VALUE,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_GROUP,
    ENTITY_ATTRIBUTE_ROLE,
    ACTION_TEXT,
    ACTION_NAME,
    ENTITIES,
    METADATA_MODEL_ID,
)

if TYPE_CHECKING:
    from rasa.shared.core.events import NLUPredictionData
    from rasa.shared.core.training_data.structures import Story
    from rasa.shared.core.training_data.story_writer.story_writer import StoryWriter
    from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack

    EventTypeAlias = TypeVar("EventTypeAlias", bound=Event)


@dataclasses.dataclass
class TrackerActiveLoop:
    """Dataclass for `DialogueStateTracker.active_loop`."""

    name: Optional[Text]
    is_interrupted: bool
    rejected: bool
    trigger_message: Optional[Dict]


logger = logging.getLogger(__name__)

# same as State but with Dict[...] substituted with FrozenSet[Tuple[...]]
FrozenState = FrozenSet[Tuple[Text, FrozenSet[Tuple[Text, Tuple[Union[float, Text]]]]]]


class EventVerbosity(Enum):
    """Filter on which events to include in tracker dumps."""

    # no events will be included
    NONE = 1

    # all events, that contribute to the trackers state are included
    # these are all you need to reconstruct the tracker state
    APPLIED = 2

    # include even more events, in this case everything that comes
    # after the most recent restart event. this will also include
    # utterances that got reverted and actions that got undone.
    AFTER_RESTART = 3

    # include every logged event
    ALL = 4


class AnySlotDict(dict):
    """A slot dictionary that pretends every slot exists, by creating slots on demand.

    This only uses the generic slot type! This means certain functionality wont work,
    e.g. properly featurizing the slot.
    """

    def __missing__(self, key: Text) -> Slot:
        value = self[key] = AnySlot(key, mappings=[])
        return value

    def __contains__(self, key: Any) -> bool:
        return True


class DialogueStateTracker:
    """Maintains the state of a conversation.

    The field max_event_history will only give you these last events,
    it can be set in the tracker_store.
    """

    @classmethod
    def from_dict(
        cls,
        sender_id: Text,
        events_as_dict: List[Dict[Text, Any]],
        slots: Optional[Iterable[Slot]] = None,
        max_event_history: Optional[int] = None,
    ) -> "DialogueStateTracker":
        """Create a tracker from dump.

        The dump should be an array of dumped events. When restoring
        the tracker, these events will be replayed to recreate the state.
        """
        evts = events.deserialise_events(events_as_dict)

        return cls.from_events(sender_id, evts, slots, max_event_history)

    @classmethod
    def from_events(
        cls,
        sender_id: Text,
        evts: List[Event],
        slots: Optional[Iterable[Slot]] = None,
        max_event_history: Optional[int] = None,
        sender_source: Optional[Text] = None,
        domain: Optional[Domain] = None,
    ) -> "DialogueStateTracker":
        """Creates tracker from existing events.

        Args:
            sender_id: The ID of the conversation.
            evts: Existing events which should be applied to the new tracker.
            slots: Slots which can be set.
            max_event_history: Maximum number of events which should be stored.
            sender_source: File source of the messages.
            domain: The current model domain.

        Returns:
            Instantiated tracker with its state updated according to the given
            events.
        """
        tracker = cls(sender_id, slots, max_event_history, sender_source)

        for e in evts:
            tracker.update(e, domain)

        return tracker

    def __init__(
        self,
        sender_id: Text,
        slots: Optional[Iterable[Slot]],
        max_event_history: Optional[int] = None,
        sender_source: Optional[Text] = None,
        is_rule_tracker: bool = False,
    ) -> None:
        """Initialize the tracker.

        A set of events can be stored externally, and we will run through all
        of them to get the current state. The tracker will represent all the
        information we captured while processing messages of the dialogue.
        """
        from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack

        # maximum number of events to store
        self._max_event_history = max_event_history
        # list of previously seen events
        self.events = self._create_events([])
        # id of the source of the messages
        self.sender_id = sender_id
        # slots that can be filled in this domain
        self.slots: Dict[str, Slot] = AnySlotDict()
        if slots is not None:
            self.slots = {slot.name: copy.copy(slot) for slot in slots}

        self._underlying_stack = DialogueStack.empty()
        # file source of the messages
        self.sender_source = sender_source
        # whether the tracker belongs to a rule-based data
        self.is_rule_tracker = is_rule_tracker

        ###
        # current state of the tracker - MUST be re-creatable by processing
        # all the events. This only defines the attributes, values are set in
        # `reset()`
        ###
        # if tracker is paused, no actions should be taken
        self._paused = False
        # A deterministically scheduled action to be executed next
        self.followup_action: Optional[Text] = ACTION_LISTEN_NAME
        self.latest_action: Optional[Dict[Text, Text]] = None
        # Stores the most recent message sent by the user
        self.latest_message: Optional[UserUttered] = None
        self.latest_bot_utterance: Optional[BotUttered] = None
        self._reset()
        self.active_loop: Optional[TrackerActiveLoop] = None

        # Optional model_id to add to all events.
        self.model_id: Optional[Text] = None
        self.assistant_id: Optional[Text] = None

    ###
    # Public tracker interface
    ###
    def current_state(
        self, event_verbosity: EventVerbosity = EventVerbosity.NONE
    ) -> Dict[Text, Any]:
        """Returns the current tracker state as an object."""
        events = self._events_for_verbosity(event_verbosity)
        events_as_dict = [e.as_dict() for e in events] if events is not None else None
        latest_event_time = None
        if len(self.events) > 0:
            latest_event_time = self.events[-1].timestamp

        return {
            "sender_id": self.sender_id,
            "slots": self.current_slot_values(),
            "latest_message": self._latest_message_data(),
            "latest_event_time": latest_event_time,
            FOLLOWUP_ACTION: self.followup_action,
            "paused": self.is_paused(),
            "stack": self.stack.as_dict(),
            "events": events_as_dict,
            "latest_input_channel": self.get_latest_input_channel(),
            ACTIVE_LOOP: (
                dataclasses.asdict(self.active_loop) if self.active_loop else {}
            ),
            "latest_action": self.latest_action,
            "latest_action_name": self.latest_action_name,
        }

    def _events_for_verbosity(
        self, event_verbosity: EventVerbosity
    ) -> Optional[List[Event]]:
        if event_verbosity == EventVerbosity.ALL:
            return list(self.events)
        if event_verbosity == EventVerbosity.AFTER_RESTART:
            return self.events_after_latest_restart()
        if event_verbosity == EventVerbosity.APPLIED:
            return self.applied_events()

        return None

    def _latest_message_data(self) -> Optional["NLUPredictionData"]:
        if not self.latest_message:
            return None

        parse_data_with_nlu_state = self.latest_message.parse_data.copy()
        # Combine entities predicted by NLU with entities predicted by policies so that
        # users can access them together via `latest_message` (e.g. in custom actions)
        parse_data_with_nlu_state[ENTITIES] = self.latest_message.entities  # type: ignore[literal-required]

        return parse_data_with_nlu_state

    @staticmethod
    def freeze_current_state(state: State) -> FrozenState:
        """Convert State dict into a hashable format FrozenState.

        Args:
            state: The state which should be converted

        Return:
            hashable form of the state of type `FrozenState`
        """
        return frozenset(
            {
                key: frozenset(values.items())
                if isinstance(values, Dict)
                else frozenset(values)
                for key, values in state.items()
            }.items()
        )

    def past_states(
        self,
        domain: Domain,
        omit_unset_slots: bool = False,
        ignore_rule_only_turns: bool = False,
        rule_only_data: Optional[Dict[Text, Any]] = None,
    ) -> List[State]:
        """Generates the past states of this tracker based on the history.

        Args:
            domain: The Domain.
            omit_unset_slots: If `True` do not include the initial values of slots.
            ignore_rule_only_turns: If True ignore dialogue turns that are present
                only in rules.
            rule_only_data: Slots and loops,
                which only occur in rules but not in stories.

        Returns:
            A list of states
        """
        return domain.states_for_tracker_history(
            self,
            omit_unset_slots=omit_unset_slots,
            ignore_rule_only_turns=ignore_rule_only_turns,
            rule_only_data=rule_only_data,
        )

    def change_loop_to(self, loop_name: Optional[Text]) -> None:
        """Set the currently active loop.

        Args:
            loop_name: The name of loop which should be marked as active.
        """
        if loop_name is not None:
            self.active_loop = TrackerActiveLoop(
                loop_name,
                False,
                False,
                self.latest_message.parse_data if self.latest_message else None,
            )
        else:
            self.active_loop = None

    def interrupt_loop(self, is_interrupted: bool) -> None:
        """Interrupt loop and mark that we entered an unhappy path in the conversation.

        Args:
            is_interrupted: `True` if the loop was run after an unhappy path.
        """
        if self.active_loop is not None:
            self.active_loop.is_interrupted = is_interrupted

    def reject_action(self, action_name: Text) -> None:
        """Notify active loop that it was rejected."""
        if self.active_loop is not None and action_name == self.active_loop_name:
            self.active_loop.rejected = True

    def set_latest_action(self, action: Dict[Text, Text]) -> None:
        """Sets latest action name or text.

        Resets loop validation and rejection parameters.

        Args:
            action: Serialized action event.
        """
        self.latest_action = action
        if self.active_loop is not None and self.active_loop_name:
            # reset form validation if some loop is active
            self.active_loop.is_interrupted = False

        if (
            self.active_loop is not None
            and action.get(ACTION_NAME) == self.active_loop_name
        ):
            # reset loop rejection if it was predicted again
            self.active_loop.rejected = False

    def current_slot_values(self) -> Dict[Text, Any]:
        """Return the currently set values of the slots."""
        return {key: slot.value for key, slot in self.slots.items()}

    def get_slot(self, key: Text) -> Optional[Any]:
        """Retrieves the value of a slot."""
        if key in self.slots:
            return self.slots[key].value
        else:
            logger.info(f"Tried to access non existent slot '{key}'")
            return None

    def create_stack_updated_events(
        self, updated_stack: "DialogueStack"
    ) -> List[Event]:
        """Creates events to update the stack to the given one."""
        patch = self._underlying_stack.create_stack_patch(updated_stack)

        # if there is no patch, this is a no-op
        if patch:
            return [DialogueStackUpdated(patch)]
        return []

    def update_stack(self, updated_stack: "DialogueStack") -> None:
        """Set's the updated stack on this tracker."""
        for event in self.create_stack_updated_events(updated_stack):
            self.update(event)

    def apply_stack_update(self, update: str) -> None:
        self._underlying_stack = self._underlying_stack.update_from_patch(update)

    def previous_stack_states(self) -> Generator["DialogueStack", None, None]:
        """Generates the previous stack states of this tracker."""
        from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack

        tracker = self.init_copy()
        previous_stack = DialogueStack.empty()
        yield previous_stack
        for event in self.applied_events():
            tracker.update(event)
            stack = tracker.stack
            if stack != previous_stack:
                previous_stack = stack
                yield stack

    @property
    def stack(self) -> "DialogueStack":
        """Returns the current stack as a copy.

        Important, modifying the returned stack does not modify the stack
        stored on the tracker.
        """
        return self._underlying_stack.copy()

    @property
    def active_flow(self) -> Optional[Text]:
        """Returns the name of the active flow."""
        current_context = self.stack.current_context()
        return current_context.get("flow_id")

    @property
    def current_step_id(self) -> Optional[str]:
        """Returns the id of the current step id."""
        current_context = self.stack.current_context()
        return current_context.get("step_id")

    @property
    def has_coexistence_routing_slot(self) -> bool:
        """Returns whether the coexistence routing slot is present."""
        if self.slots:
            return ROUTE_TO_CALM_SLOT in self.slots
        return False

    def has_bot_message_after_latest_user_message(self) -> bool:
        """Checks if there is a bot message after the most recent user message.

        Returns:
            `True` if there is an action after the most recent user message.
        """
        for event in reversed(self.applied_events()):
            if isinstance(event, BotUttered):
                return True
            elif isinstance(event, UserUttered):
                return False
        return False

    def has_action_after_latest_user_message(self) -> bool:
        """Check if there is an action after the most recent user message.

        Returns:
            `True` if there is an action after the most recent user message.
        """
        for event in reversed(self.applied_events()):
            if isinstance(event, ActionExecuted):
                return True
            elif isinstance(event, UserUttered):
                return False
        return False

    def get_latest_entity_values(
        self,
        entity_type: Text,
        entity_role: Optional[Text] = None,
        entity_group: Optional[Text] = None,
    ) -> Iterator[Text]:
        """Get entity values for latest message.

        Returns entity values found for the passed entity type and
        optional role and group in latest message.

        If you are only interested in the first entity of a given type use
        `next(tracker.get_latest_entity_values(`"`my_entity_name`"`), None)`.
        If no entity is found `None` is the default result.

        Args:
            entity_type: the entity type of interest
            entity_role: optional entity role of interest
            entity_group: optional entity group of interest

        Returns:
            Entity values.
        """
        if self.latest_message is None:
            return iter([])

        return (
            cast(Text, x[ENTITY_ATTRIBUTE_VALUE])
            for x in self.latest_message.entities
            if x.get(ENTITY_ATTRIBUTE_TYPE) == entity_type
            and x.get(ENTITY_ATTRIBUTE_GROUP) == entity_group
            and x.get(ENTITY_ATTRIBUTE_ROLE) == entity_role
        )

    def get_latest_input_channel(self) -> Optional[Text]:
        """Get the name of the input_channel of the latest UserUttered event."""
        for e in reversed(self.events):
            if isinstance(e, UserUttered):
                return e.input_channel
        return None

    def is_paused(self) -> bool:
        """State whether the tracker is currently paused."""
        return self._paused

    def idx_after_latest_restart(self) -> int:
        """Return the idx of the most recent restart in the list of events.

        If the conversation has not been restarted, ``0`` is returned.
        """
        for i, event in enumerate(reversed(self.events)):
            if isinstance(event, Restarted):
                return len(self.events) - i

        return 0

    def events_after_latest_restart(self) -> List[Event]:
        """Return a list of events after the most recent restart."""
        return list(self.events)[self.idx_after_latest_restart() :]

    def init_copy(self) -> "DialogueStateTracker":
        """Creates a new state tracker with the same initial values."""
        return DialogueStateTracker(
            self.sender_id or DEFAULT_SENDER_ID,
            self.slots.values(),
            self._max_event_history,
            is_rule_tracker=self.is_rule_tracker,
        )

    def generate_all_prior_trackers(
        self,
    ) -> Generator[Tuple["DialogueStateTracker", bool], None, None]:
        """Returns a generator of the previous trackers of this tracker.

        Returns:
            The tuple with the tracker before each action,
            and the boolean flag representing whether this action should be hidden
            in the dialogue history created for ML-based policies.
        """
        tracker = self.init_copy()

        for event in self.applied_events(True):
            if isinstance(event, ActionExecuted):
                yield tracker, event.hide_rule_turn

            tracker.update(event)

        yield tracker, False

    def applied_events(self, featurization_for_policies: bool = False) -> List[Event]:
        """Returns all actions that should be applied - w/o reverted events.

        Returns:
            The events applied to the tracker.
        """
        loop_names = [
            event.name
            for event in self.events
            if isinstance(event, ActiveLoop) and event.name
        ]

        applied_events: List[Event] = []

        for event in self.events:
            if isinstance(event, (Restarted, SessionStarted)):
                applied_events = []
            elif isinstance(event, RoutingSessionEnded) and featurization_for_policies:
                # remove all events but the set slots events for the slots that are
                # shared for coexistence.
                applied_events = [
                    e
                    for e in applied_events
                    if isinstance(e, SlotSet)
                    and (slot := self.slots.get(e.key)) is not None
                    and slot.shared_for_coexistence
                ]
            elif isinstance(event, ActionReverted):
                self._undo_till_previous(ActionExecuted, applied_events)
            elif isinstance(event, UserUtteranceReverted):
                # Seeing a user uttered event automatically implies there was
                # a listen event right before it, so we'll first rewind the
                # user utterance, then get the action right before it (also removes
                # the `action_listen` action right before it).
                self._undo_till_previous(UserUttered, applied_events)
                self._undo_till_previous(ActionExecuted, applied_events)
            elif (
                isinstance(event, ActionExecuted)
                and event.action_name in loop_names
                and not self._first_loop_execution_or_unhappy_path(
                    event.action_name, applied_events
                )
            ):
                self._undo_till_previous_loop_execution(
                    event.action_name, applied_events
                )
            else:
                applied_events.append(event)

        return applied_events

    @staticmethod
    def _undo_till_previous(event_type: Type[Event], done_events: List[Event]) -> None:
        """Removes events from `done_events`.

        Removes events from `done_events` until the first occurrence `event_type`
        is found which is also removed.
        """
        # list gets modified - hence we need to copy events!
        for e in reversed(done_events[:]):
            del done_events[-1]
            if isinstance(e, event_type):
                break

    def _first_loop_execution_or_unhappy_path(
        self, loop_action_name: Text, applied_events: List[Event]
    ) -> bool:
        next_action: Optional[Text] = None

        for event in reversed(applied_events):
            # Stop looking for a previous loop execution if there is a loop deactivation
            # event because it means that the current loop is running for the first
            # time and previous loop events belong to different loops.
            if isinstance(event, ActiveLoop) and event.name is None:
                return True

            if self._is_within_unhappy_path(loop_action_name, event, next_action):
                return True

            if isinstance(event, ActionExecuted):
                # We found a previous execution of the loop and we are not within an
                # unhappy path.
                if event.action_name == loop_action_name:
                    return False

                # Remember the action as we need that to check whether we might be
                # within an unhappy path.
                next_action = event.action_name

        return True

    @staticmethod
    def _is_within_unhappy_path(
        loop_action_name: Text, event: Event, next_action_in_the_future: Optional[Text]
    ) -> bool:
        # When actual users are talking to the action has to return an
        # `ActionExecutionRejected` in order to enter an unhappy path.
        loop_was_rejected_previously = (
            isinstance(event, ActionExecutionRejected)
            and event.action_name == loop_action_name
        )
        # During the policy training there are no `ActionExecutionRejected` events
        # which let us see whether we are within an unhappy path. Hence, we check if a
        # different action was executed instead of the loop after last user utterance.
        other_action_after_latest_user_utterance = (
            isinstance(event, UserUttered)
            and next_action_in_the_future is not None
            and next_action_in_the_future != loop_action_name
        )

        return loop_was_rejected_previously or other_action_after_latest_user_utterance

    @staticmethod
    def _undo_till_previous_loop_execution(
        loop_action_name: Text, done_events: List[Event]
    ) -> None:
        offset = 0
        for e in reversed(done_events[:]):
            if isinstance(e, ActionExecuted) and e.action_name == loop_action_name:
                break

            if isinstance(
                e, (ActionExecuted, UserUttered, DefinePrevUserUtteredFeaturization)
            ):
                del done_events[-1 - offset]
            else:
                # Remember events which aren't unfeaturized to get the index right
                offset += 1

    def replay_events(self) -> None:
        """Update the tracker based on a list of events."""
        applied_events = self.applied_events()
        for event in applied_events:
            event.apply_to(self)

    def recreate_from_dialogue(self, dialogue: Dialogue) -> None:
        """Use a serialised `Dialogue` to update the trackers state.

        This uses the state as is persisted in a ``TrackerStore``. If the
        tracker is blank before calling this method, the final state will be
        identical to the tracker from which the dialogue was created.
        """
        if not isinstance(dialogue, Dialogue):
            raise ValueError(
                f"story {dialogue} is not of type Dialogue. "
                f"Have you deserialized it?"
            )

        self._reset()
        self.events.extend(dialogue.events)
        self.replay_events()

    def copy(self) -> "DialogueStateTracker":
        """Creates a duplicate of this tracker."""
        return copy.deepcopy(self)

    def travel_back_in_time(self, target_time: float) -> "DialogueStateTracker":
        """Creates a new tracker with a state at a specific timestamp.

        A new tracker will be created and all events previous to the
        passed time stamp will be replayed. Events that occur exactly
        at the target time will be included.
        """
        tracker = self.init_copy()

        for event in self.events:
            if event.timestamp <= target_time:
                tracker.update(event)
            else:
                break

        return tracker  # yields the final state

    def as_dialogue(self) -> Dialogue:
        """Return a ``Dialogue`` object containing all of the turns.

        This can be serialised and later used to recover the state
        of this tracker exactly.
        """
        return Dialogue(self.sender_id, list(self.events))

    def update(self, event: Event, domain: Optional[Domain] = None) -> None:
        """Modify the state of the tracker according to an ``Event``."""
        if not isinstance(event, Event):  # pragma: no cover
            raise ValueError("event to log must be an instance of a subclass of Event.")

        if self.model_id and METADATA_MODEL_ID not in event.metadata:
            event.metadata = {**event.metadata, METADATA_MODEL_ID: self.model_id}

        if self.assistant_id and ASSISTANT_ID_KEY not in event.metadata:
            event.metadata = {**event.metadata, ASSISTANT_ID_KEY: self.assistant_id}

        self.events.append(event)
        event.apply_to(self)

    def update_with_events(
        self,
        new_events: List[Event],
        # TODO: remove domain argument - breaking change.
        domain: Optional[Domain] = None,
        override_timestamp: bool = True,
    ) -> None:
        """Adds multiple events to the tracker.

        Args:
            new_events: Events to apply.
            domain: The current model's domain. Not needed anymore.
              kept for backwards compatibility.
            override_timestamp: If `True` refresh all timestamps of the events. As the
                events are usually created at some earlier point, this makes sure that
                all new events come after any current tracker events.
        """
        for e in new_events:
            if override_timestamp:
                e.timestamp = time.time()
            self.update(e, domain)

    def as_story(self, include_source: bool = False) -> "Story":
        """Dump the tracker as a story in the Rasa Core story format.

        Returns the dumped tracker as a string.
        """
        from rasa.shared.core.training_data.structures import Story

        story_name = (
            f"{self.sender_id} ({self.sender_source})"
            if include_source
            else self.sender_id
        )
        return Story.from_events(list(self.events), story_name)

    def export_stories(
        self,
        writer: "StoryWriter",
        e2e: bool = False,
        include_source: bool = False,
        should_append_stories: bool = False,
    ) -> Text:
        """Dump the tracker as a story in the Rasa Core story format.

        Returns:
            The dumped tracker as a string.
        """
        story = self.as_story(include_source)
        return writer.dumps(
            story.story_steps, is_appendable=should_append_stories, is_test_story=e2e
        )

    def export_stories_to_file(self, export_path: Text = "debug_stories.yml") -> None:
        """Dump the tracker as a story to a file."""
        from rasa.shared.core.training_data.story_writer.yaml_story_writer import (
            YAMLStoryWriter,
        )

        append = os.path.exists(export_path)

        rasa.shared.utils.io.write_text_file(
            self.export_stories(YAMLStoryWriter(), should_append_stories=append) + "\n",
            export_path,
            append=append,
        )

    def get_last_event_for(
        self,
        event_type: Union[Type["EventTypeAlias"], Tuple[Type["EventTypeAlias"], ...]],
        action_names_to_exclude: Optional[List[Text]] = None,
        skip: int = 0,
        event_verbosity: EventVerbosity = EventVerbosity.APPLIED,
    ) -> Optional["EventTypeAlias"]:
        """Gets the last event of a given type which was actually applied.

        Args:
            event_type: The type of event you want to find.
            action_names_to_exclude: Events of type `ActionExecuted` which
                should be excluded from the results. Can be used to skip
                `action_listen` events.
            skip: Skips n possible results before return an event.
            event_verbosity: Which `EventVerbosity` should be used to search for events.

        Returns:
            event which matched the query or `None` if no event matched.
        """
        to_exclude = action_names_to_exclude or []

        def filter_function(e: Event) -> bool:
            has_instance = isinstance(e, event_type)
            excluded = isinstance(e, ActionExecuted) and e.action_name in to_exclude
            return has_instance and not excluded

        filtered = filter(
            filter_function, reversed(self._events_for_verbosity(event_verbosity) or [])
        )

        for i in range(skip):
            next(filtered, None)

        return next(filtered, None)

    def last_executed_action_has(self, name: Text, skip: int = 0) -> bool:
        """Returns whether last `ActionExecuted` event had a specific name.

        Args:
            name: Name of the event which should be matched.
            skip: Skips n possible results in between.

        Returns:
            `True` if last executed action had name `name`, otherwise `False`.
        """
        last: Optional[ActionExecuted] = self.get_last_event_for(
            ActionExecuted, action_names_to_exclude=[ACTION_LISTEN_NAME], skip=skip
        )
        return last is not None and last.action_name == name

    ###
    # Internal methods for the modification of the trackers state. Should
    # only be called by events, not directly. Rather update the tracker
    # with an event that in its ``apply_to`` method modifies the tracker.
    ###
    def _reset(self, is_coexistence_reset: bool = False) -> None:
        """Reset tracker to initial state - doesn't delete events though!."""
        from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack

        self._reset_slots(is_coexistence_reset)
        self._paused = False
        self.latest_action = {}
        self.latest_message = UserUttered.empty()
        self.latest_bot_utterance = BotUttered.empty()
        self.followup_action = ACTION_LISTEN_NAME
        self.active_loop = None
        self._underlying_stack = DialogueStack.empty()

    def _reset_slots(self, is_coexistence_reset: bool = False) -> None:
        """Set all the slots to their initial value."""
        for slot in self.slots.values():
            # skip slots with shared_for_coexistence during this reset
            if is_coexistence_reset and slot.shared_for_coexistence:
                continue
            slot.reset()

    def _set_slot(self, key: Text, value: Any) -> None:
        """Sets the value of a slot if that slot exists."""
        if key in self.slots:
            slot = self.slots[key]
            slot.value = value
        else:
            logger.error(
                f"Tried to set non existent slot '{key}'. Make sure you "
                f"added all your slots to your domain file."
            )

    def _create_events(self, evts: List[Event]) -> Deque[Event]:
        if evts and not isinstance(evts[0], Event):  # pragma: no cover
            raise ValueError("events, if given, must be a list of events")
        return deque(evts, self._max_event_history)

    def __eq__(self, other: Any) -> bool:
        if isinstance(self, type(other)):
            return other.events == self.events and self.sender_id == other.sender_id
        else:
            return False

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __repr__(self) -> Text:
        """Returns event as string for debugging."""
        return f"DialogueStateTracker(sender_id: {self.sender_id})"

    def __str__(self) -> Text:
        """Returns event as human-readable string."""
        return f"{self.__class__.__name__}({self.sender_id})"

    def trigger_followup_action(self, action: Text) -> None:
        """Triggers another action following the execution of the current."""
        self.followup_action = action

    def clear_followup_action(self) -> None:
        """Clears follow up action when it was executed."""
        self.followup_action = None

    @property
    def active_loop_name(self) -> Optional[Text]:
        """Get the name of the currently active loop.

        Returns: `None` if no active loop or the name of the currently active loop.
        """
        if not self.active_loop or self.active_loop.name == SHOULD_NOT_BE_SET:
            return None

        return self.active_loop.name

    @property
    def latest_action_name(self) -> Optional[Text]:
        """Get the name of the previously executed action or text of e2e action.

        Returns: name of the previously executed action or text of e2e action
        """
        if self.latest_action is None:
            return None

        return self.latest_action.get(ACTION_NAME) or self.latest_action.get(
            ACTION_TEXT
        )

    @property
    def is_active_loop_rejected(self) -> bool:
        """Return True if there is an active loop and it's rejected."""
        return self.active_loop is not None and self.active_loop.rejected

    @property
    def is_active_loop_interrupted(self) -> bool:
        """Return True if there is an active loop and it's interrupted."""
        return self.active_loop is not None and self.active_loop.is_interrupted

    def fingerprint(self) -> Text:
        """Returns a unique hash for the tracker which is stable across python runs.

        Returns:
            fingerprint of the tracker
        """
        data: Dict[Text, Any] = {"sender_id": self.sender_id}

        if self.slots:
            data.update(self.slots)

        if self.events:
            data["events"] = list(self.events)

        return rasa.shared.utils.io.get_dictionary_fingerprint(data)

    def get_previously_started_flows(
        self,
        flows: FlowsList,
        max_turns: Optional[int] = 20,
    ) -> FlowsList:
        """Retrieves a list of previously started flows.

        Returned flows have been started in the past within a given
        number of conversation turns.

        Args:
            flows: list of flows to check against for started flows.
            max_turns: the maximum number of turns to include in the transcript.

        Returns:
            List of flows that have been started within the specified number of turns
        """
        previously_started_flows = dict()
        turn_counter = 0

        # cycle through events in reverse order (newest events are appended at the end)
        for event in reversed(self.events):
            # check for FlowStarted event and append flow if it's in the flows lists
            if (
                isinstance(event, FlowStarted)
                and event.flow_id not in previously_started_flows
                and (flow := flows.flow_by_id(event.flow_id))
            ):
                previously_started_flows[event.flow_id] = flow

            # count turns only for user or bot utterances
            if isinstance(event, (UserUttered, BotUttered)):
                turn_counter += 1
                if max_turns is not None and turn_counter > max_turns:
                    break

        return FlowsList(underlying_flows=list(previously_started_flows.values()))

    def get_startable_flows(self, flows: FlowsList) -> FlowsList:
        """Retrieves a list of flows that can be started.

        Returned flows are startable given the current
        state (context and slot values) of the tracker.

        Args:
            flows: list of flows to check against for startable flows.

        Returns:
            List of flows that are startable within the current state of the tracker
        """
        # get the current context and slot values for the flow guard check
        context = self.stack.current_context()
        slots = self.slots
        return flows.get_startable_flows(context, slots)

    @property
    def has_active_user_flow(self) -> bool:
        from rasa.dialogue_understanding.stack.utils import top_user_flow_frame

        top_relevant_frame = top_user_flow_frame(self.stack)
        return bool(top_relevant_frame and top_relevant_frame.flow_id)

    def get_active_flows(self, flows: FlowsList) -> FlowsList:
        """Retrieve a list of all currently active flows.

        Args:
            flows: list of flows to check against for active flows.

        Returns:
            List of flows that are active within the current state of the tracker
        """
        from rasa.dialogue_understanding.stack.frames import UserFlowStackFrame
        from rasa.dialogue_understanding.stack.frames.flow_stack_frame import (
            FlowStackFrameType,
        )

        active_flows = []
        for frame in reversed(self.stack.frames):
            # The stack just contains the current active frames and we are just
            # interested in the user flow stack frames.
            if isinstance(frame, UserFlowStackFrame):
                active_flows.append(frame.flow(flows))
                if frame.frame_type != FlowStackFrameType.CALL:
                    # Iterate unitl we reach a frame that is not a call frame.
                    break

        return FlowsList(active_flows)


class TrackerEventDiffEngine:
    """Computes event difference of two trackers."""

    @staticmethod
    def event_difference(
        original: DialogueStateTracker, tracker: DialogueStateTracker
    ) -> List[Event]:
        """Find all events in the tracker not present in the original tracker.

        Args:
            original: Original tracker to compare against.
            tracker: Tracker containing events from the current conversation session.

        Returns:
            List of events from the new tracker which are not present
            in the original tracker.
        """
        offset = len(original.events) if original else 0
        events = tracker.events
        return list(itertools.islice(events, offset, len(events)))


def get_active_loop_name(
    state: State,
) -> Optional[Text]:
    """Get the name of current active loop.

    Args:
        state: The state from which the name of active loop should be extracted

    Return:
        the name of active loop or None
    """
    if (
        not state.get(ACTIVE_LOOP)
        or state[ACTIVE_LOOP].get(LOOP_NAME) == SHOULD_NOT_BE_SET
    ):
        return None

    # FIXME: better type annotation for `State` would require
    # a larger refactoring (e.g. switch to dataclass)
    return cast(Optional[Text], state[ACTIVE_LOOP].get(LOOP_NAME))


def is_prev_action_listen_in_state(state: State) -> bool:
    """Check if action_listen is the previous executed action.

    Args:
        state: The state for which the check should be performed

    Return:
        boolean value indicating whether action_listen is previous action
    """
    prev_action_name = state.get(PREVIOUS_ACTION, {}).get(ACTION_NAME)
    return prev_action_name == ACTION_LISTEN_NAME


def get_trackers_for_conversation_sessions(
    tracker: DialogueStateTracker,
) -> List[DialogueStateTracker]:
    """Generate trackers for `tracker` that are split by conversation sessions.

    Args:
        tracker: Instance of `DialogueStateTracker` to split.

    Returns:
        The trackers split by conversation sessions.
    """
    split_conversations = events.split_events(
        tracker.events,
        ActionExecuted,
        {"action_name": ACTION_SESSION_START_NAME},
        include_splitting_event=True,
    )

    return [
        DialogueStateTracker.from_events(
            tracker.sender_id,
            evts,
            tracker.slots.values(),
            sender_source=tracker.sender_source,
            max_event_history=tracker._max_event_history,
        )
        for evts in split_conversations
    ]

import copy
import logging
from functools import lru_cache
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Set,
    Text,
    Tuple,
    cast,
)

import rasa.core
import rasa.shared.utils.io
from rasa.core.actions.custom_action_executor import (
    CustomActionExecutor,
    NoEndpointCustomActionExecutor,
    RetryCustomActionExecutor,
)
from rasa.core.actions.direct_custom_actions_executor import DirectCustomActionExecutor
from rasa.core.actions.e2e_stub_custom_action_executor import (
    E2EStubCustomActionExecutor,
)
from rasa.core.actions.grpc_custom_action_executor import GRPCCustomActionExecutor
from rasa.core.actions.http_custom_action_executor import HTTPCustomActionExecutor
from rasa.core.constants import (
    UTTER_SOURCE_METADATA_KEY,
)
from rasa.core.policies.policy import PolicyPrediction
from rasa.core.utils import add_bot_utterance_metadata
from rasa.e2e_test.constants import KEY_STUB_CUSTOM_ACTIONS
from rasa.nlu.constants import (
    RESPONSE_SELECTOR_DEFAULT_INTENT,
    RESPONSE_SELECTOR_PREDICTION_KEY,
    RESPONSE_SELECTOR_PROPERTY_NAME,
    RESPONSE_SELECTOR_UTTER_ACTION_KEY,
)
from rasa.shared.constants import (
    DEFAULT_NLU_FALLBACK_INTENT_NAME,
    DOCS_BASE_URL,
    FLOW_PREFIX,
    ROUTE_TO_CALM_SLOT,
    UTTER_PREFIX,
)
from rasa.shared.core.constants import (
    ACTION_BACK_NAME,
    ACTION_DEACTIVATE_LOOP_NAME,
    ACTION_DEFAULT_ASK_AFFIRMATION_NAME,
    ACTION_DEFAULT_ASK_REPHRASE_NAME,
    ACTION_DEFAULT_FALLBACK_NAME,
    ACTION_EXTRACT_SLOTS,
    ACTION_LISTEN_NAME,
    ACTION_RESET_ROUTING,
    ACTION_RESTART_NAME,
    ACTION_REVERT_FALLBACK_EVENTS_NAME,
    ACTION_SEND_TEXT_NAME,
    ACTION_SESSION_START_NAME,
    ACTION_UNLIKELY_INTENT_NAME,
    ACTION_VALIDATE_SLOT_MAPPINGS,
    DEFAULT_SLOT_NAMES,
    KNOWLEDGE_BASE_SLOT_NAMES,
    MAPPING_TYPE,
    REQUESTED_SLOT,
    USER_INTENT_OUT_OF_SCOPE,
    SlotMappingType,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import (
    ActionExecuted,
    ActiveLoop,
    BotUttered,
    Event,
    Restarted,
    RoutingSessionEnded,
    SessionStarted,
    SlotSet,
    UserUtteranceReverted,
    UserUttered,
)
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.slot_mappings import (
    SlotFillingManager,
    extract_slot_value,
)
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import RasaException
from rasa.shared.nlu.constants import (
    INTENT_NAME_KEY,
    INTENT_RANKING_KEY,
)
from rasa.shared.utils.io import raise_warning
from rasa.shared.utils.schemas.events import EVENTS_SCHEMA
from rasa.utils.endpoints import ClientResponseError, EndpointConfig
from rasa.utils.url_tools import UrlSchema, get_url_schema

if TYPE_CHECKING:
    from rasa.core.channels.channel import OutputChannel
    from rasa.core.nlg import NaturalLanguageGenerator
    from rasa.shared.core.events import IntentPrediction


logger = logging.getLogger(__name__)


def default_actions(action_endpoint: Optional[EndpointConfig] = None) -> List["Action"]:
    """List default actions."""
    from rasa.core.actions.action_clean_stack import ActionCleanStack
    from rasa.core.actions.action_run_slot_rejections import ActionRunSlotRejections
    from rasa.core.actions.action_trigger_chitchat import ActionTriggerChitchat
    from rasa.core.actions.action_trigger_search import ActionTriggerSearch
    from rasa.core.actions.two_stage_fallback import TwoStageFallbackAction
    from rasa.dialogue_understanding.patterns.cancel import ActionCancelFlow
    from rasa.dialogue_understanding.patterns.clarify import ActionClarifyFlows
    from rasa.dialogue_understanding.patterns.correction import ActionCorrectFlowSlot

    return [
        ActionListen(),
        ActionRestart(),
        ActionSessionStart(),
        ActionDefaultFallback(),
        ActionDeactivateLoop(),
        ActionRevertFallbackEvents(),
        ActionDefaultAskAffirmation(),
        ActionDefaultAskRephrase(),
        TwoStageFallbackAction(action_endpoint),
        ActionUnlikelyIntent(),
        ActionSendText(),
        ActionBack(),
        ActionExtractSlots(action_endpoint),
        ActionCancelFlow(),
        ActionCorrectFlowSlot(),
        ActionClarifyFlows(),
        ActionRunSlotRejections(),
        ActionCleanStack(),
        ActionTriggerSearch(),
        ActionTriggerChitchat(),
        ActionResetRouting(),
    ]


def action_for_index(
    index: int,
    domain: Domain,
    action_endpoint: Optional[EndpointConfig],
) -> "Action":
    """Get an action based on its index in the list of available actions.

    Args:
        index: The index of the action. This is usually used by `Policy`s as they
            predict the action index instead of the name.
        domain: The `Domain` of the current model. The domain contains the actions
            provided by the user + the default actions.
        action_endpoint: Can be used to run `custom_actions`
            (e.g. using the `rasa-sdk`).

    Returns:
        The instantiated `Action` or `None` if no `Action` was found for the given
        index.
    """
    if domain.num_actions <= index or index < 0:
        raise IndexError(
            f"Cannot access action at index {index}. "
            f"Domain has {domain.num_actions} actions."
        )

    return action_for_name_or_text(
        domain.action_names_or_texts[index],
        domain,
        action_endpoint,
    )


def is_retrieval_action(action_name: Text, retrieval_intents: List[Text]) -> bool:
    """Check if an action name is a retrieval action.

    The name for a retrieval action has an extra `utter_` prefix added to
    the corresponding retrieval intent name.

    Args:
        action_name: Name of the action.
        retrieval_intents: List of retrieval intents defined in the NLU training data.

    Returns:
        `True` if the resolved intent name is present in the list of retrieval
        intents, `False` otherwise.
    """
    return (
        ActionRetrieveResponse.intent_name_from_action(action_name) in retrieval_intents
    )


def action_for_name_or_text(
    action_name_or_text: Text,
    domain: Domain,
    action_endpoint: Optional[EndpointConfig],
) -> "Action":
    """Retrieves an action by its name or by its text in case it's an end-to-end action.

    Args:
        action_name_or_text: The name of the action.
        domain: The current model domain.
        action_endpoint: The endpoint to execute custom actions.

    Raises:
        ActionNotFoundException: If action not in current domain.

    Returns:
        The instantiated action.
    """
    if action_name_or_text not in domain.action_names_or_texts:
        domain.raise_action_not_found_exception(action_name_or_text)

    defaults = {a.name(): a for a in default_actions(action_endpoint)}

    if (
        action_name_or_text in defaults
        and action_name_or_text not in domain.user_actions_and_forms
    ):
        return defaults[action_name_or_text]

    if action_name_or_text.startswith(UTTER_PREFIX) and is_retrieval_action(
        action_name_or_text, domain.retrieval_intents
    ):
        return ActionRetrieveResponse(action_name_or_text)

    if action_name_or_text in domain.action_texts:
        return ActionEndToEndResponse(action_name_or_text)

    if action_name_or_text.startswith(UTTER_PREFIX):
        return ActionBotResponse(action_name_or_text)

    is_form = action_name_or_text in domain.form_names
    # Users can override the form by defining an action with the same name as the form
    user_overrode_form_action = is_form and action_name_or_text in domain.user_actions
    if is_form and not user_overrode_form_action:
        from rasa.core.actions.forms import FormAction

        return FormAction(action_name_or_text, action_endpoint)

    if action_name_or_text.startswith(FLOW_PREFIX):
        from rasa.core.actions.action_trigger_flow import ActionTriggerFlow

        return ActionTriggerFlow(action_name_or_text)
    return RemoteAction(action_name_or_text, action_endpoint)


def create_bot_utterance(message: Dict[Text, Any]) -> BotUttered:
    """Create BotUttered event from message."""
    bot_message = BotUttered(
        text=message.pop("text", None),
        data={
            "elements": message.pop("elements", None),
            "quick_replies": message.pop("quick_replies", None),
            "buttons": message.pop("buttons", None),
            # for legacy / compatibility reasons we need to set the image
            # to be the attachment if there is no other attachment (the
            # `.get` is intentional - no `pop` as we still need the image`
            # property to set it in the following line)
            "attachment": message.pop("attachment", None) or message.get("image", None),
            "image": message.pop("image", None),
            "custom": message.pop("custom", None),
        },
        metadata=message,
    )
    return bot_message


class Action:
    """Next action to be taken in response to a dialogue state."""

    def name(self) -> Text:
        """Unique identifier of this simple action."""
        raise NotImplementedError

    async def run(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> List[Event]:
        """Execute the side effects of this action.

        Args:
            nlg: which ``nlg`` to use for response generation
            output_channel: ``output_channel`` to which to send the resulting message.
            tracker (DialogueStateTracker): the state tracker for the current
                user. You can access slot values using
                ``tracker.get_slot(slot_name)`` and the most recent user
                message is ``tracker.latest_message.text``.
            domain (Domain): the bot's domain
            metadata: Additional information for the action.

        Returns:
            A list of :class:`rasa.core.events.Event` instances
        """
        raise NotImplementedError

    def __str__(self) -> Text:
        """Returns text representation of form."""
        return f"{self.__class__.__name__}('{self.name()}')"

    def event_for_successful_execution(
        self, prediction: PolicyPrediction
    ) -> ActionExecuted:
        """Event which should be logged for the successful execution of this action.

        Args:
            prediction: Prediction which led to the execution of this event.

        Returns:
            Event which should be logged onto the tracker.
        """
        return ActionExecuted(
            self.name(),
            prediction.policy_name,
            prediction.max_confidence,
            hide_rule_turn=prediction.hide_rule_turn,
            metadata=prediction.action_metadata,
        )


class ActionBotResponse(Action):
    """An action which only effect is to utter a response when it is run."""

    def __init__(self, name: Text, silent_fail: Optional[bool] = False) -> None:
        """Creates action.

        Args:
            name: Name of the action.
            silent_fail: `True` if the action should fail silently in case no response
                was defined for this action.
        """
        self.utter_action = name
        self.silent_fail = silent_fail

    async def run(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> List[Event]:
        """Simple run implementation uttering a (hopefully defined) response."""
        kwargs = {
            "domain_responses": domain.responses,
        }

        message = await nlg.generate(
            self.utter_action,
            tracker,
            output_channel.name(),
            **kwargs,
        )
        if message is None:
            if not self.silent_fail:
                logger.error(
                    "Couldn't create message for response '{}'.".format(
                        self.utter_action
                    )
                )
            return []

        message.update(metadata or {})
        message = add_bot_utterance_metadata(
            message, self.utter_action, nlg, domain, tracker
        )
        return [create_bot_utterance(message)]

    def name(self) -> Text:
        """Returns action name."""
        return self.utter_action


class ActionEndToEndResponse(Action):
    """Action to utter end-to-end responses to the user."""

    def __init__(self, action_text: Text) -> None:
        """Creates action.

        Args:
            action_text: Text of end-to-end bot response.
        """
        self.action_text = action_text

    def name(self) -> Text:
        """Returns action name."""
        # In case of an end-to-end action there is no label (aka name) for the action.
        # We fake a name by returning the text which the bot sends back to the user.
        return self.action_text

    async def run(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> List[Event]:
        """Runs action (see parent class for full docstring)."""
        message = {"text": self.action_text}
        return [create_bot_utterance(message)]

    def event_for_successful_execution(
        self, prediction: PolicyPrediction
    ) -> ActionExecuted:
        """Event which should be logged for the successful execution of this action.

        Args:
            prediction: Prediction which led to the execution of this event.

        Returns:
            Event which should be logged onto the tracker.
        """
        return ActionExecuted(
            policy=prediction.policy_name,
            confidence=prediction.max_confidence,
            action_text=self.action_text,
            hide_rule_turn=prediction.hide_rule_turn,
            metadata=prediction.action_metadata,
        )


class ActionRetrieveResponse(ActionBotResponse):
    """An action which queries the Response Selector for the appropriate response."""

    def __init__(self, name: Text, silent_fail: Optional[bool] = False) -> None:
        """Creates action. See docstring of parent class."""
        super().__init__(name, silent_fail)
        self.action_name = name
        self.silent_fail = silent_fail

    @staticmethod
    def intent_name_from_action(action_name: Text) -> Text:
        """Resolve the name of the intent from the action name."""
        return action_name.split(UTTER_PREFIX)[1]

    def get_full_retrieval_name(
        self, tracker: "DialogueStateTracker"
    ) -> Optional[Text]:
        """Returns full retrieval name for the action.

        Extracts retrieval intent from response selector and
        returns complete action utterance name.

        Args:
            tracker: Tracker containing past conversation events.

        Returns:
            Full retrieval name of the action if the last user utterance
            contains a response selector output, `None` otherwise.
        """
        latest_message = tracker.latest_message

        if latest_message is None:
            return None

        if RESPONSE_SELECTOR_PROPERTY_NAME not in latest_message.parse_data:
            return None

        response_selector_properties = latest_message.parse_data[
            RESPONSE_SELECTOR_PROPERTY_NAME  # type: ignore[literal-required]
        ]

        if (
            self.intent_name_from_action(self.action_name)
            in response_selector_properties
        ):
            query_key = self.intent_name_from_action(self.action_name)
        elif RESPONSE_SELECTOR_DEFAULT_INTENT in response_selector_properties:
            query_key = RESPONSE_SELECTOR_DEFAULT_INTENT
        else:
            return None

        selected = response_selector_properties[query_key]
        full_retrieval_utter_action = selected[RESPONSE_SELECTOR_PREDICTION_KEY][
            RESPONSE_SELECTOR_UTTER_ACTION_KEY
        ]
        return full_retrieval_utter_action

    async def run(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> List[Event]:
        """Query the appropriate response and create a bot utterance with that."""
        latest_message = tracker.latest_message

        if latest_message is None:
            return []

        response_selector_properties = latest_message.parse_data[
            RESPONSE_SELECTOR_PROPERTY_NAME  # type: ignore[literal-required]
        ]

        if (
            self.intent_name_from_action(self.action_name)
            in response_selector_properties
        ):
            query_key = self.intent_name_from_action(self.action_name)
        elif RESPONSE_SELECTOR_DEFAULT_INTENT in response_selector_properties:
            query_key = RESPONSE_SELECTOR_DEFAULT_INTENT
        else:
            if not self.silent_fail:
                logger.error(
                    "Couldn't create message for response action '{}'.".format(
                        self.action_name
                    )
                )
            return []

        logger.debug(f"Picking response from selector of type {query_key}")
        selected = response_selector_properties[query_key]

        # Override utter action of ActionBotResponse
        # with the complete utter action retrieved from
        # the output of response selector.
        self.utter_action = selected[RESPONSE_SELECTOR_PREDICTION_KEY][
            RESPONSE_SELECTOR_UTTER_ACTION_KEY
        ]

        return await super().run(output_channel, nlg, tracker, domain)

    def name(self) -> Text:
        """Returns action name."""
        return self.action_name


class ActionBack(ActionBotResponse):
    """Revert the tracker state by two user utterances."""

    def name(self) -> Text:
        """Returns action back name."""
        return ACTION_BACK_NAME

    def __init__(self) -> None:
        """Initializes action back."""
        super().__init__("utter_back", silent_fail=True)

    async def run(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> List[Event]:
        """Runs action. Please see parent class for the full docstring."""
        # only utter the response if it is available
        evts = await super().run(output_channel, nlg, tracker, domain)

        return evts + [UserUtteranceReverted(), UserUtteranceReverted()]


class ActionListen(Action):
    """The first action in any turn - bot waits for a user message.

    The bot should stop taking further actions and wait for the user to say
    something.
    """

    def name(self) -> Text:
        """Returns action listen name."""
        return ACTION_LISTEN_NAME

    async def run(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> List[Event]:
        """Runs action. Please see parent class for the full docstring."""
        return []


class ActionResetRouting(Action):
    """Resets the tracker to its initial state.

    Utters the restart response if available.
    """

    def name(self) -> Text:
        """Returns action restart name."""
        return ACTION_RESET_ROUTING

    async def run(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> List[Event]:
        """Runs action. Please see parent class for the full docstring."""
        # SlotSet(ROUTE_TO_CALM_SLOT, None) is needed to ensure the routing slot
        # is really reset to None and not just reset to an initial value
        return [RoutingSessionEnded(), SlotSet(ROUTE_TO_CALM_SLOT, None)]


class ActionRestart(ActionBotResponse):
    """Resets the tracker to its initial state.

    Utters the restart response if available.
    """

    def name(self) -> Text:
        """Returns action restart name."""
        return ACTION_RESTART_NAME

    def __init__(self) -> None:
        """Initializes action restart."""
        super().__init__("utter_restart", silent_fail=True)

    async def run(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> List[Event]:
        """Runs action. Please see parent class for the full docstring."""
        # only utter the response if it is available
        evts = await super().run(output_channel, nlg, tracker, domain)

        return evts + [Restarted()]


class ActionSessionStart(Action):
    """Applies a conversation session start.

    Takes all `SlotSet` events from the previous session and applies them to the new
    session.
    """

    def name(self) -> Text:
        """Returns action start name."""
        return ACTION_SESSION_START_NAME

    @staticmethod
    def _slot_set_events_from_tracker(
        tracker: "DialogueStateTracker",
    ) -> List["SlotSet"]:
        """Fetch SlotSet events from tracker and carry over key, value and metadata."""
        return [
            SlotSet(key=event.key, value=event.value, metadata=event.metadata)
            for event in tracker.applied_events()
            if isinstance(event, SlotSet)
        ]

    async def run(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> List[Event]:
        """Runs action. Please see parent class for the full docstring."""
        _events: List[Event] = [SessionStarted()]

        if domain.session_config.carry_over_slots:
            _events.extend(self._slot_set_events_from_tracker(tracker))

        _events.append(ActionExecuted(ACTION_LISTEN_NAME))

        return _events


class ActionDefaultFallback(ActionBotResponse):
    """Executes the fallback action and goes back to the prev state of the dialogue."""

    def name(self) -> Text:
        """Returns action default fallback name."""
        return ACTION_DEFAULT_FALLBACK_NAME

    def __init__(self) -> None:
        """Initializes action default fallback."""
        super().__init__("utter_default", silent_fail=True)

    async def run(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> List[Event]:
        """Runs action. Please see parent class for the full docstring."""
        # only utter the response if it is available
        evts = await super().run(output_channel, nlg, tracker, domain)

        return evts + [UserUtteranceReverted()]


class ActionDeactivateLoop(Action):
    """Deactivates an active loop."""

    def name(self) -> Text:
        return ACTION_DEACTIVATE_LOOP_NAME

    async def run(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> List[Event]:
        """Runs action. Please see parent class for the full docstring."""
        return [ActiveLoop(None), SlotSet(REQUESTED_SLOT, None)]


class RemoteAction(Action):
    def __init__(
        self,
        name: Text,
        action_endpoint: Optional[EndpointConfig] = None,
    ) -> None:
        self._name = name
        self.action_endpoint = action_endpoint
        self.executor = self._create_executor()

    @lru_cache(maxsize=1)
    def _create_executor(self) -> CustomActionExecutor:
        """Creates an executor based on the action endpoint configuration.

        Returns:
            An instance of CustomActionExecutor.

        Raises:
            RasaException: If no valid action endpoint is configured.
        """
        if not self.action_endpoint:
            return NoEndpointCustomActionExecutor(self.name())

        if self.action_endpoint.kwargs.get(KEY_STUB_CUSTOM_ACTIONS):
            return E2EStubCustomActionExecutor(self.name(), self.action_endpoint)

        if self.action_endpoint.url and self.action_endpoint.actions_module:
            raise_warning(
                "Both 'actions_module' and 'url' are defined. "
                "As they are mutually exclusive and 'actions_module' "
                "is prioritized, actions will be executed by the assistant."
            )

        if self.action_endpoint and self.action_endpoint.actions_module:
            return DirectCustomActionExecutor(self.name(), self.action_endpoint)

        url_schema = get_url_schema(self.action_endpoint.url)

        if url_schema == UrlSchema.GRPC:
            return RetryCustomActionExecutor(
                GRPCCustomActionExecutor(self.name(), self.action_endpoint)
            )
        elif (
            url_schema == UrlSchema.HTTP
            or url_schema == UrlSchema.HTTPS
            or url_schema == UrlSchema.NOT_SPECIFIED
        ):
            return RetryCustomActionExecutor(
                HTTPCustomActionExecutor(self.name(), self.action_endpoint)
            )
        raise RasaException(
            f"Failed to create a custom action executor. "
            f"Please make sure to include an action endpoint configuration in your "
            f"endpoints configuration file. Make sure that for grpc, http and https "
            f"an url schema is set. "
            f"Found url '{self.action_endpoint.url}'."
        )

    @staticmethod
    def action_response_format_spec() -> Dict[Text, Any]:
        """Expected response schema for an Action endpoint.

        Used for validation of the response returned from the
        Action endpoint.
        """
        schema = {
            "type": "object",
            "properties": {
                "events": EVENTS_SCHEMA,
                "responses": {"type": "array", "items": {"type": "object"}},
            },
        }
        return schema

    def _validate_action_result(self, result: Dict[Text, Any]) -> bool:
        from jsonschema import ValidationError, validate

        try:
            validate(result, self.action_response_format_spec())
            return True
        except ValidationError as e:
            e.message += (
                f". Failed to validate Action server response from API, "
                f"make sure your response from the Action endpoint is valid. "
                f"For more information about the format visit "
                f"{DOCS_BASE_URL}/custom-actions"
            )
            raise e

    @staticmethod
    async def _utter_responses(
        responses: List[Dict[Text, Any]],
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        **kwargs: Any,
    ) -> List[BotUttered]:
        """Use the responses generated by the action endpoint and utter them."""
        bot_messages = []
        domain: Domain = kwargs.get("domain", None)
        action_name: str = kwargs.get("action_name", None)
        for response in responses:
            generated_response = response.pop("response", None)
            if generated_response is not None:
                draft = await nlg.generate(
                    generated_response, tracker, output_channel.name(), **response
                )
                if not draft:
                    continue
                draft = add_bot_utterance_metadata(
                    draft, generated_response, nlg, domain, tracker
                )
            else:
                draft = {UTTER_SOURCE_METADATA_KEY: action_name}

            buttons = response.pop("buttons", []) or []
            if buttons:
                draft.setdefault("buttons", [])
                draft["buttons"].extend(buttons)

            # Avoid overwriting `draft` values with empty values
            response = {k: v for k, v in response.items() if v}
            draft.update(response)
            bot_messages.append(create_bot_utterance(draft))

        return bot_messages

    async def run(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> List[Event]:
        """Runs action. Please see parent class for the full docstring."""
        response = await self.executor.run(
            domain=domain,
            tracker=tracker,
        )
        self._validate_action_result(response)

        events_json = response.get("events", [])
        responses = response.get("responses", [])
        bot_messages = await self._utter_responses(
            responses,
            output_channel,
            nlg,
            tracker,
            domain=domain,
            action_name=self.name(),
        )

        events = rasa.shared.core.events.deserialise_events(events_json)
        return cast(List[Event], bot_messages) + events

    def name(self) -> Text:
        return self._name


class ActionRevertFallbackEvents(Action):
    """Reverts events which were done during the `TwoStageFallbackPolicy`.

    This reverts user messages and bot utterances done during a fallback
    of the `TwoStageFallbackPolicy`. By doing so it is not necessary to
    write custom stories for the different paths, but only of the happy
    path. This is deprecated and can be removed once the
    `TwoStageFallbackPolicy` is removed.
    """

    def name(self) -> Text:
        return ACTION_REVERT_FALLBACK_EVENTS_NAME

    async def run(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> List[Event]:
        """Runs action. Please see parent class for the full docstring."""
        from rasa.core.policies.two_stage_fallback import has_user_rephrased

        # User rephrased
        if has_user_rephrased(tracker):
            return _revert_successful_rephrasing(tracker)
        # User affirmed
        elif has_user_affirmed(tracker):
            return _revert_affirmation_events(tracker)
        else:
            return []


class ActionUnlikelyIntent(Action):
    """An action that indicates that the intent predicted by NLU is unexpected.

    This action can be predicted by `UnexpecTEDIntentPolicy`.
    """

    def name(self) -> Text:
        """Returns the name of the action."""
        return ACTION_UNLIKELY_INTENT_NAME

    async def run(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> List[Event]:
        """Runs action. Please see parent class for the full docstring."""
        return []


def has_user_affirmed(tracker: "DialogueStateTracker") -> bool:
    """Indicates if the last executed action is `action_default_ask_affirmation`."""
    return tracker.last_executed_action_has(ACTION_DEFAULT_ASK_AFFIRMATION_NAME)


def _revert_affirmation_events(tracker: "DialogueStateTracker") -> List[Event]:
    revert_events = _revert_single_affirmation_events()

    # User affirms the rephrased intent
    rephrased_intent = tracker.last_executed_action_has(
        name=ACTION_DEFAULT_ASK_REPHRASE_NAME, skip=1
    )
    if rephrased_intent:
        revert_events += _revert_rephrasing_events()

    last_user_event = tracker.get_last_event_for(UserUttered)
    if not last_user_event:
        raise TypeError("Cannot find last event to revert to.")

    last_user_event = copy.deepcopy(last_user_event)
    # FIXME: better type annotation for `parse_data` would require
    # a larger refactoring (e.g. switch to dataclass)
    last_user_event.parse_data["intent"]["confidence"] = 1.0  # type: ignore[typeddict-item]

    return revert_events + [last_user_event]


def _revert_single_affirmation_events() -> List[Event]:
    return [
        UserUtteranceReverted(),  # revert affirmation and request
        # revert original intent (has to be re-added later)
        UserUtteranceReverted(),
        # add action listen intent
        ActionExecuted(action_name=ACTION_LISTEN_NAME),
    ]


def _revert_successful_rephrasing(tracker: "DialogueStateTracker") -> List[Event]:
    last_user_event = tracker.get_last_event_for(UserUttered)
    if not last_user_event:
        raise TypeError("Cannot find last event to revert to.")

    last_user_event = copy.deepcopy(last_user_event)
    return _revert_rephrasing_events() + [last_user_event]


def _revert_rephrasing_events() -> List[Event]:
    return [
        UserUtteranceReverted(),  # remove rephrasing
        # remove feedback and rephrase request
        UserUtteranceReverted(),
        # remove affirmation request and false intent
        UserUtteranceReverted(),
        # replace action with action listen
        ActionExecuted(action_name=ACTION_LISTEN_NAME),
    ]


class ActionDefaultAskAffirmation(Action):
    """Default implementation which asks the user to affirm his intent.

    It is suggested to overwrite this default action with a custom action
    to have more meaningful prompts for the affirmations. E.g. have a
    description of the intent instead of its identifier name.
    """

    def name(self) -> Text:
        return ACTION_DEFAULT_ASK_AFFIRMATION_NAME

    async def run(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> List[Event]:
        """Runs action. Please see parent class for the full docstring."""
        latest_message = tracker.latest_message
        if latest_message is None:
            raise TypeError(
                "Cannot find last user message for detecting fallback affirmation."
            )

        intent_to_affirm = latest_message.intent.get(INTENT_NAME_KEY)

        # FIXME: better type annotation for `parse_data` would require
        # a larger refactoring (e.g. switch to dataclass)
        intent_ranking = cast(
            List["IntentPrediction"],
            latest_message.parse_data.get(INTENT_RANKING_KEY) or [],
        )
        if (
            intent_to_affirm == DEFAULT_NLU_FALLBACK_INTENT_NAME
            and len(intent_ranking) > 1
        ):
            intent_to_affirm = intent_ranking[1][INTENT_NAME_KEY]  # type: ignore[literal-required]

        affirmation_message = f"Did you mean '{intent_to_affirm}'?"

        message = {
            "text": affirmation_message,
            "buttons": [
                {"title": "Yes", "payload": f"/{intent_to_affirm}"},
                {"title": "No", "payload": f"/{USER_INTENT_OUT_OF_SCOPE}"},
            ],
            "utter_action": self.name(),
        }

        return [create_bot_utterance(message)]


class ActionDefaultAskRephrase(ActionBotResponse):
    """Default implementation which asks the user to rephrase his intent."""

    def name(self) -> Text:
        """Returns action default ask rephrase name."""
        return ACTION_DEFAULT_ASK_REPHRASE_NAME

    def __init__(self) -> None:
        """Initializes action default ask rephrase."""
        super().__init__("utter_ask_rephrase", silent_fail=True)


class ActionSendText(Action):
    """Sends a text message to the output channel."""

    def name(self) -> Text:
        return ACTION_SEND_TEXT_NAME

    async def run(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> List[Event]:
        """Runs action. Please see parent class for the full docstring."""
        fallback = {"text": ""}
        metadata_copy = copy.deepcopy(metadata) if metadata else {}
        message = metadata_copy.get("message", fallback)
        return [create_bot_utterance(message)]


class ActionExtractSlots(Action):
    """Default action that runs after each user turn.

    Action is executed automatically in MessageProcessor.handle_message(...)
    before the next predicted action is run.

    Set slots to extracted values from user message
    according to assigned slot mappings.
    """

    def __init__(self, action_endpoint: Optional[EndpointConfig]) -> None:
        """Initializes default action extract slots."""
        self._action_endpoint = action_endpoint

    def name(self) -> Text:
        """Returns action_extract_slots name."""
        return ACTION_EXTRACT_SLOTS

    async def _run_custom_action(
        self,
        custom_action: Text,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
    ) -> List[Event]:
        slot_events: List[Event] = []
        remote_action = RemoteAction(custom_action, self._action_endpoint)
        disallowed_types = set()

        try:
            custom_events = await remote_action.run(
                output_channel, nlg, tracker, domain
            )
            for event in custom_events:
                if isinstance(event, SlotSet):
                    slot_events.append(event)
                elif isinstance(event, BotUttered):
                    slot_events.append(event)
                else:
                    disallowed_types.add(event.type_name)
        except (RasaException, ClientResponseError) as e:
            logger.warning(
                f"Failed to execute custom action '{custom_action}' "
                f"as a result of error '{e!s}'. The default action "
                f"'{self.name()}' failed to fill slots with custom "
                f"mappings."
            )

        for type_name in disallowed_types:
            logger.info(
                f"Running custom action '{custom_action}' has resulted "
                f"in an event of type '{type_name}'. This is "
                f"disallowed and the tracker will not be "
                f"updated with this event."
            )

        return slot_events

    async def _execute_custom_action(
        self,
        mapping: Dict[Text, Any],
        executed_custom_actions: Set[Text],
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
        calm_custom_action_names: Optional[Set[str]] = None,
    ) -> Tuple[List[Event], Set[Text]]:
        custom_action = mapping.get("action")

        if not custom_action or custom_action in executed_custom_actions:
            return [], executed_custom_actions

        if (
            calm_custom_action_names is not None
            and custom_action in calm_custom_action_names
        ):
            return [], executed_custom_actions

        slot_events = await self._run_custom_action(
            custom_action, output_channel, nlg, tracker, domain
        )

        executed_custom_actions.add(custom_action)

        return slot_events, executed_custom_actions

    async def _execute_validation_action(
        self,
        extraction_events: List[Event],
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
    ) -> List[Event]:
        slot_events: List[SlotSet] = [
            event for event in extraction_events if isinstance(event, SlotSet)
        ]

        slot_candidates = "\n".join([e.key for e in slot_events])
        logger.debug(f"Validating extracted slots: {slot_candidates}")

        if ACTION_VALIDATE_SLOT_MAPPINGS not in domain.user_actions:
            return cast(List[Event], slot_events)

        _tracker = DialogueStateTracker.from_events(
            tracker.sender_id,
            tracker.events_after_latest_restart() + cast(List[Event], slot_events),
            slots=domain.slots,
        )
        validate_events = await self._run_custom_action(
            ACTION_VALIDATE_SLOT_MAPPINGS, output_channel, nlg, _tracker, domain
        )
        validated_slot_names = [
            event.key for event in validate_events if isinstance(event, SlotSet)
        ]

        # If the custom action doesn't return a SlotSet event for an extracted slot
        # candidate we assume that it was valid. The custom action has to return a
        # SlotSet(slot_name, None) event to mark a Slot as invalid.
        return validate_events + [
            event for event in slot_events if event.key not in validated_slot_names
        ]

    async def run(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> List[Event]:
        """Runs action. Please see parent class for the full docstring."""
        slot_events: List[Event] = []
        executed_custom_actions: Set[Text] = set()

        user_slots = [
            slot
            for slot in domain.slots
            if slot.name not in DEFAULT_SLOT_NAMES | KNOWLEDGE_BASE_SLOT_NAMES
        ]

        calm_slot_names = set()
        calm_custom_action_names = None
        flows = None

        if metadata is not None:
            flows = metadata.get("all_flows")

        if flows is not None:
            flows = FlowsList.from_json(flows)
            calm_slot_names = flows.available_slot_names()
            calm_custom_action_names = flows.available_custom_actions()

        slot_filling_manager = SlotFillingManager(
            domain, tracker, action_endpoint=self._action_endpoint
        )

        for slot in user_slots:
            if slot.name in calm_slot_names:
                continue

            slot_value, is_extracted = extract_slot_value(slot, slot_filling_manager)
            if is_extracted:
                slot_events.append(SlotSet(slot.name, slot_value))

            for mapping in slot.mappings:
                mapping_type = SlotMappingType(mapping.get(MAPPING_TYPE))
                should_fill_custom_slot = mapping_type == SlotMappingType.CUSTOM

                if should_fill_custom_slot:
                    (
                        custom_evts,
                        executed_custom_actions,
                    ) = await self._execute_custom_action(
                        mapping,
                        executed_custom_actions,
                        output_channel,
                        nlg,
                        tracker,
                        domain,
                        calm_custom_action_names,
                    )
                    slot_events.extend(custom_evts)

        validated_events = await self._execute_validation_action(
            slot_events, output_channel, nlg, tracker, domain
        )
        return validated_events

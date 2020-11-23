import copy
import json
import logging
from typing import List, Text, Optional, Dict, Any, Set, TYPE_CHECKING

import aiohttp

import rasa.core

from rasa.shared.core import events
from rasa.core.constants import DEFAULT_REQUEST_TIMEOUT

from rasa.nlu.constants import (
    RESPONSE_SELECTOR_DEFAULT_INTENT,
    RESPONSE_SELECTOR_PROPERTY_NAME,
    RESPONSE_SELECTOR_PREDICTION_KEY,
    RESPONSE_SELECTOR_TEMPLATE_NAME_KEY,
)

from rasa.shared.constants import (
    DOCS_BASE_URL,
    DEFAULT_NLU_FALLBACK_INTENT_NAME,
    UTTER_PREFIX,
)
from rasa.shared.core.constants import (
    USER_INTENT_OUT_OF_SCOPE,
    ACTION_LISTEN_NAME,
    ACTION_RESTART_NAME,
    ACTION_SESSION_START_NAME,
    ACTION_DEFAULT_FALLBACK_NAME,
    ACTION_DEACTIVATE_LOOP_NAME,
    ACTION_REVERT_FALLBACK_EVENTS_NAME,
    ACTION_DEFAULT_ASK_AFFIRMATION_NAME,
    ACTION_DEFAULT_ASK_REPHRASE_NAME,
    ACTION_BACK_NAME,
    REQUESTED_SLOT,
)
from rasa.shared.exceptions import RasaException
from rasa.shared.nlu.constants import INTENT_NAME_KEY, INTENT_RANKING_KEY
from rasa.shared.core.events import (
    UserUtteranceReverted,
    UserUttered,
    ActionExecuted,
    Event,
    BotUttered,
    SlotSet,
    ActiveLoop,
    Restarted,
    SessionStarted,
)
from rasa.utils.endpoints import EndpointConfig, ClientResponseError
from rasa.shared.core.domain import Domain


if TYPE_CHECKING:
    from rasa.shared.core.trackers import DialogueStateTracker
    from rasa.core.nlg import NaturalLanguageGenerator
    from rasa.core.channels.channel import OutputChannel

logger = logging.getLogger(__name__)


def default_actions(action_endpoint: Optional[EndpointConfig] = None) -> List["Action"]:
    """List default actions."""
    from rasa.core.actions.two_stage_fallback import TwoStageFallbackAction

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
        ActionBack(),
    ]


def action_for_index(
    index: int, domain: Domain, action_endpoint: Optional[EndpointConfig]
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

    return action_for_name(domain.action_names[index], domain, action_endpoint)


def action_for_name(
    action_name: Text, domain: Domain, action_endpoint: Optional[EndpointConfig]
) -> "Action":
    """Create an `Action` object based on the name of the `Action`.

    Args:
        action_name: The name of the `Action`.
        domain: The `Domain` of the current model. The domain contains the actions
            provided by the user + the default actions.
        action_endpoint: Can be used to run `custom_actions`
            (e.g. using the `rasa-sdk`).

    Returns:
        The instantiated `Action` or `None` if no `Action` was found for the given
        index.
    """

    if action_name not in domain.action_names:
        domain.raise_action_not_found_exception(action_name)

    return action_from_name(action_name, domain, action_endpoint)


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


def action_from_name(
    name: Text, domain: Domain, action_endpoint: Optional[EndpointConfig]
) -> "Action":
    """Retrieves an action by its name.

    Args:
        name: The name of the action.
        domain: The current model domain.
        action_endpoint: The endpoint to execute custom actions.

    Returns:
        The instantiated action.
    """
    defaults = {a.name(): a for a in default_actions(action_endpoint)}

    if name in defaults and name not in domain.user_actions_and_forms:
        return defaults[name]

    if name.startswith(UTTER_PREFIX) and is_retrieval_action(
        name, domain.retrieval_intents
    ):
        return ActionRetrieveResponse(name)

    if name.startswith(UTTER_PREFIX):
        return ActionUtterTemplate(name)

    is_form = name in domain.form_names
    # Users can override the form by defining an action with the same name as the form
    user_overrode_form_action = is_form and name in domain.user_actions
    if is_form and not user_overrode_form_action:
        from rasa.core.actions.forms import FormAction

        return FormAction(name, action_endpoint)

    return RemoteAction(name, action_endpoint)


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
    ) -> List[Event]:
        """
        Execute the side effects of this action.

        Args:
            nlg: which ``nlg`` to use for response generation
            output_channel: ``output_channel`` to which to send the resulting message.
            tracker (DialogueStateTracker): the state tracker for the current
                user. You can access slot values using
                ``tracker.get_slot(slot_name)`` and the most recent user
                message is ``tracker.latest_message.text``.
            domain (Domain): the bot's domain
            metadata: dictionary that can be sent to action server with custom
            data.
        Returns:
            List[Event]: A list of :class:`rasa.core.events.Event` instances
        """

        raise NotImplementedError

    def __str__(self) -> Text:
        return "Action('{}')".format(self.name())


class ActionUtterTemplate(Action):
    """An action which only effect is to utter a template when it is run.

    Both, name and utter template, need to be specified using
    the `name` method."""

    def __init__(self, name: Text, silent_fail: Optional[bool] = False):
        self.template_name = name
        self.silent_fail = silent_fail

    async def run(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
    ) -> List[Event]:
        """Simple run implementation uttering a (hopefully defined) template."""

        message = await nlg.generate(self.template_name, tracker, output_channel.name())
        if message is None:
            if not self.silent_fail:
                logger.error(
                    "Couldn't create message for response '{}'."
                    "".format(self.template_name)
                )
            return []
        message["template_name"] = self.template_name

        return [create_bot_utterance(message)]

    def name(self) -> Text:
        return self.template_name

    def __str__(self) -> Text:
        return "ActionUtterTemplate('{}')".format(self.name())


class ActionRetrieveResponse(ActionUtterTemplate):
    """An action which queries the Response Selector for the appropriate response."""

    def __init__(self, name: Text, silent_fail: Optional[bool] = False):
        super().__init__(name, silent_fail)
        self.action_name = name
        self.silent_fail = silent_fail

    @staticmethod
    def intent_name_from_action(action_name: Text) -> Text:
        """Resolve the name of the intent from the action name."""
        return action_name.split(UTTER_PREFIX)[1]

    @staticmethod
    def action_name_from_intent(intent_name: Text) -> Text:
        """Resolve the action name from the name of the intent."""
        return f"{UTTER_PREFIX}{intent_name}"

    async def run(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
    ) -> List[Event]:
        """Query the appropriate response and create a bot utterance with that."""

        response_selector_properties = tracker.latest_message.parse_data[
            RESPONSE_SELECTOR_PROPERTY_NAME
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
                    "Couldn't create message for response action '{}'."
                    "".format(self.action_name)
                )
            return []

        logger.debug(f"Picking response from selector of type {query_key}")
        selected = response_selector_properties[query_key]

        # Override template name of ActionUtterTemplate
        # with the complete template name retrieved from
        # the output of response selector.
        self.template_name = selected[RESPONSE_SELECTOR_PREDICTION_KEY][
            RESPONSE_SELECTOR_TEMPLATE_NAME_KEY
        ]

        return await super().run(output_channel, nlg, tracker, domain)

    def name(self) -> Text:
        return self.action_name

    def __str__(self) -> Text:
        return "ActionRetrieveResponse('{}')".format(self.name())


class ActionBack(ActionUtterTemplate):
    """Revert the tracker state by two user utterances."""

    def name(self) -> Text:
        return ACTION_BACK_NAME

    def __init__(self) -> None:
        super().__init__("utter_back", silent_fail=True)

    async def run(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
    ) -> List[Event]:
        # only utter the response if it is available
        evts = await super().run(output_channel, nlg, tracker, domain)

        return evts + [UserUtteranceReverted(), UserUtteranceReverted()]


class ActionListen(Action):
    """The first action in any turn - bot waits for a user message.

    The bot should stop taking further actions and wait for the user to say
    something."""

    def name(self) -> Text:
        return ACTION_LISTEN_NAME

    async def run(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
    ) -> List[Event]:
        return []


class ActionRestart(ActionUtterTemplate):
    """Resets the tracker to its initial state.

    Utters the restart response if available."""

    def name(self) -> Text:
        return ACTION_RESTART_NAME

    def __init__(self) -> None:
        super().__init__("utter_restart", silent_fail=True)

    async def run(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
    ) -> List[Event]:
        # only utter the response if it is available
        evts = await super().run(output_channel, nlg, tracker, domain)

        return evts + [Restarted()]


class ActionSessionStart(Action):
    """Applies a conversation session start.

    Takes all `SlotSet` events from the previous session and applies them to the new
    session.
    """

    # Optional arbitrary metadata that can be passed to the SessionStarted event.
    metadata: Optional[Dict[Text, Any]] = None

    def name(self) -> Text:
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
    ) -> List[Event]:
        _events = [SessionStarted(metadata=self.metadata)]

        if domain.session_config.carry_over_slots:
            _events.extend(self._slot_set_events_from_tracker(tracker))

        _events.append(ActionExecuted(ACTION_LISTEN_NAME))

        return _events


class ActionDefaultFallback(ActionUtterTemplate):
    """Executes the fallback action and goes back to the previous state
    of the dialogue"""

    def name(self) -> Text:
        return ACTION_DEFAULT_FALLBACK_NAME

    def __init__(self) -> None:
        super().__init__("utter_default", silent_fail=True)

    async def run(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
    ) -> List[Event]:
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
    ) -> List[Event]:
        return [ActiveLoop(None), SlotSet(REQUESTED_SLOT, None)]


class RemoteAction(Action):
    def __init__(self, name: Text, action_endpoint: Optional[EndpointConfig]) -> None:

        self._name = name
        self.action_endpoint = action_endpoint

    def _action_call_format(
        self, tracker: "DialogueStateTracker", domain: "Domain"
    ) -> Dict[Text, Any]:
        """Create the request json send to the action server."""
        from rasa.shared.core.trackers import EventVerbosity

        tracker_state = tracker.current_state(EventVerbosity.ALL)

        return {
            "next_action": self._name,
            "sender_id": tracker.sender_id,
            "tracker": tracker_state,
            "domain": domain.as_dict(),
            "version": rasa.__version__,
        }

    @staticmethod
    def action_response_format_spec() -> Dict[Text, Any]:
        """Expected response schema for an Action endpoint.

        Used for validation of the response returned from the
        Action endpoint."""
        return {
            "type": "object",
            "properties": {
                "events": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"event": {"type": "string"}},
                    },
                },
                "responses": {"type": "array", "items": {"type": "object"}},
            },
        }

    def _validate_action_result(self, result: Dict[Text, Any]) -> bool:
        from jsonschema import validate
        from jsonschema import ValidationError

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
    ) -> List[BotUttered]:
        """Use the responses generated by the action endpoint and utter them."""

        bot_messages = []
        for response in responses:
            template = response.pop("template", None)
            if template:
                draft = await nlg.generate(
                    template, tracker, output_channel.name(), **response
                )
                if not draft:
                    continue
                draft["template_name"] = template
            else:
                draft = {}

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
    ) -> List[Event]:
        json_body = self._action_call_format(tracker, domain)
        if not self.action_endpoint:
            raise RasaException(
                f"Failed to execute custom action '{self.name()}' "
                f"because no endpoint is configured to run this "
                f"custom action. Please take a look at "
                f"the docs and set an endpoint configuration via the "
                f"--endpoints flag. "
                f"{DOCS_BASE_URL}/custom-actions"
            )

        try:
            logger.debug(
                "Calling action endpoint to run action '{}'.".format(self.name())
            )
            response = await self.action_endpoint.request(
                json=json_body, method="post", timeout=DEFAULT_REQUEST_TIMEOUT
            )

            self._validate_action_result(response)

            events_json = response.get("events", [])
            responses = response.get("responses", [])
            bot_messages = await self._utter_responses(
                responses, output_channel, nlg, tracker
            )

            evts = events.deserialise_events(events_json)
            return bot_messages + evts

        except ClientResponseError as e:
            if e.status == 400:
                response_data = json.loads(e.text)
                exception = ActionExecutionRejection(
                    response_data["action_name"], response_data.get("error")
                )
                logger.error(exception.message)
                raise exception
            else:
                raise RasaException("Failed to execute custom action.") from e

        except aiohttp.ClientConnectionError as e:
            logger.error(
                "Failed to run custom action '{}'. Couldn't connect "
                "to the server at '{}'. Is the server running? "
                "Error: {}".format(self.name(), self.action_endpoint.url, e)
            )
            raise RasaException("Failed to execute custom action.")

        except aiohttp.ClientError as e:
            # not all errors have a status attribute, but
            # helpful to log if they got it

            # noinspection PyUnresolvedReferences
            status = getattr(e, "status", None)
            raise RasaException(
                "Failed to run custom action '{}'. Action server "
                "responded with a non 200 status code of {}. "
                "Make sure your action server properly runs actions "
                "and returns a 200 once the action is executed. "
                "Error: {}".format(self.name(), status, e)
            )

    def name(self) -> Text:
        return self._name


class ActionExecutionRejection(RasaException):
    """Raising this exception will allow other policies
    to predict a different action"""

    def __init__(self, action_name: Text, message: Optional[Text] = None) -> None:
        self.action_name = action_name
        self.message = message or "Custom action '{}' rejected to run".format(
            action_name
        )
        super(ActionExecutionRejection, self).__init__()

    def __str__(self) -> Text:
        return self.message


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
    ) -> List[Event]:
        from rasa.core.policies.two_stage_fallback import has_user_rephrased

        # User rephrased
        if has_user_rephrased(tracker):
            return _revert_successful_rephrasing(tracker)
        # User affirmed
        elif has_user_affirmed(tracker):
            return _revert_affirmation_events(tracker)
        else:
            return []


def has_user_affirmed(tracker: "DialogueStateTracker") -> bool:
    return tracker.last_executed_action_has(ACTION_DEFAULT_ASK_AFFIRMATION_NAME)


def _revert_affirmation_events(tracker: "DialogueStateTracker") -> List[Event]:
    revert_events = _revert_single_affirmation_events()

    last_user_event = tracker.get_last_event_for(UserUttered)
    last_user_event = copy.deepcopy(last_user_event)
    last_user_event.parse_data["intent"]["confidence"] = 1.0

    # User affirms the rephrased intent
    rephrased_intent = tracker.last_executed_action_has(
        name=ACTION_DEFAULT_ASK_REPHRASE_NAME, skip=1
    )
    if rephrased_intent:
        revert_events += _revert_rephrasing_events()

    return revert_events + [last_user_event]


def _revert_single_affirmation_events() -> List[Event]:
    return [
        UserUtteranceReverted(),  # revert affirmation and request
        # revert original intent (has to be re-added later)
        UserUtteranceReverted(),
        # add action listen intent
        ActionExecuted(action_name=ACTION_LISTEN_NAME),
    ]


def _revert_successful_rephrasing(tracker) -> List[Event]:
    last_user_event = tracker.get_last_event_for(UserUttered)
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
    ) -> List[Event]:
        intent_to_affirm = tracker.latest_message.intent.get(INTENT_NAME_KEY)

        intent_ranking = tracker.latest_message.intent.get(INTENT_RANKING_KEY, [])
        if (
            intent_to_affirm == DEFAULT_NLU_FALLBACK_INTENT_NAME
            and len(intent_ranking) > 1
        ):
            intent_to_affirm = intent_ranking[1][INTENT_NAME_KEY]

        affirmation_message = f"Did you mean '{intent_to_affirm}'?"

        message = {
            "text": affirmation_message,
            "buttons": [
                {"title": "Yes", "payload": f"/{intent_to_affirm}"},
                {"title": "No", "payload": f"/{USER_INTENT_OUT_OF_SCOPE}"},
            ],
            "template_name": self.name(),
        }

        return [create_bot_utterance(message)]


class ActionDefaultAskRephrase(ActionUtterTemplate):
    """Default implementation which asks the user to rephrase his intent."""

    def name(self) -> Text:
        return ACTION_DEFAULT_ASK_REPHRASE_NAME

    def __init__(self) -> None:
        super().__init__("utter_ask_rephrase", silent_fail=True)

import copy
import json
import logging
import typing
from typing import List, Text, Optional, Dict, Any
import random

import aiohttp

import rasa.core
from rasa.constants import DOCS_BASE_URL, DEFAULT_NLU_FALLBACK_INTENT_NAME
from rasa.core import events
from rasa.core.constants import (
    DEFAULT_REQUEST_TIMEOUT,
    REQUESTED_SLOT,
    USER_INTENT_OUT_OF_SCOPE,
    UTTER_PREFIX,
    RESPOND_PREFIX,
)
from rasa.nlu.constants import (
    RESPONSE_SELECTOR_DEFAULT_INTENT,
    RESPONSE_SELECTOR_PROPERTY_NAME,
    RESPONSE_SELECTOR_RESPONSES_KEY,
    RESPONSE_SELECTOR_PREDICTION_KEY,
    INTENT_RANKING_KEY,
    INTENT_NAME_KEY,
    INTENT_RESPONSE_KEY,
)

from rasa.core.events import (
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

if typing.TYPE_CHECKING:
    from rasa.core.trackers import DialogueStateTracker
    from rasa.core.domain import Domain
    from rasa.core.nlg import NaturalLanguageGenerator
    from rasa.core.channels.channel import OutputChannel

logger = logging.getLogger(__name__)

ACTION_LISTEN_NAME = "action_listen"

ACTION_RESTART_NAME = "action_restart"

ACTION_SESSION_START_NAME = "action_session_start"

ACTION_DEFAULT_FALLBACK_NAME = "action_default_fallback"

ACTION_DEACTIVATE_FORM_NAME = "action_deactivate_form"

ACTION_REVERT_FALLBACK_EVENTS_NAME = "action_revert_fallback_events"

ACTION_DEFAULT_ASK_AFFIRMATION_NAME = "action_default_ask_affirmation"

ACTION_DEFAULT_ASK_REPHRASE_NAME = "action_default_ask_rephrase"

ACTION_BACK_NAME = "action_back"

RULE_SNIPPET_ACTION_NAME = "..."


def default_actions(action_endpoint: Optional[EndpointConfig] = None) -> List["Action"]:
    """List default actions."""
    from rasa.core.actions.two_stage_fallback import TwoStageFallbackAction

    return [
        ActionListen(),
        ActionRestart(),
        ActionSessionStart(),
        ActionDefaultFallback(),
        ActionDeactivateForm(),
        ActionRevertFallbackEvents(),
        ActionDefaultAskAffirmation(),
        ActionDefaultAskRephrase(),
        TwoStageFallbackAction(action_endpoint),
        ActionBack(),
    ]


def default_action_names() -> List[Text]:
    """List default action names."""
    return [a.name() for a in default_actions()] + [RULE_SNIPPET_ACTION_NAME]


def combine_user_with_default_actions(user_actions: List[Text]) -> List[Text]:
    # remove all user actions that overwrite default actions
    # this logic is a bit reversed, you'd think that we should remove
    # the action name from the default action names if the user overwrites
    # the action, but there are some locations in the code where we
    # implicitly assume that e.g. "action_listen" is always at location
    # 0 in this array. to keep it that way, we remove the duplicate
    # action names from the users list instead of the defaults
    defaults = default_action_names()
    unique_user_actions = [a for a in user_actions if a not in defaults]
    return defaults + unique_user_actions


def combine_with_templates(
    actions: List[Text], templates: Dict[Text, Any]
) -> List[Text]:
    """Combines actions with utter actions listed in responses section."""
    unique_template_names = [
        a for a in sorted(list(templates.keys())) if a not in actions
    ]
    return actions + unique_template_names


def action_from_name(
    name: Text,
    action_endpoint: Optional[EndpointConfig],
    user_actions: List[Text],
    should_use_form_action: bool = False,
) -> "Action":
    """Return an action instance for the name."""

    defaults = {a.name(): a for a in default_actions(action_endpoint)}

    if name in defaults and name not in user_actions:
        return defaults[name]
    elif name.startswith(UTTER_PREFIX):
        return ActionUtterTemplate(name)
    elif name.startswith(RESPOND_PREFIX):
        return ActionRetrieveResponse(name)
    elif should_use_form_action:
        from rasa.core.actions.forms import FormAction

        return FormAction(name, action_endpoint)
    else:
        return RemoteAction(name, action_endpoint)


def actions_from_names(
    action_names: List[Text],
    action_endpoint: Optional[EndpointConfig],
    user_actions: List[Text],
) -> List["Action"]:
    """Converts the names of actions into class instances."""

    return [
        action_from_name(name, action_endpoint, user_actions) for name in action_names
    ]


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


class ActionRetrieveResponse(Action):
    """An action which queries the Response Selector for the appropriate response."""

    def __init__(self, name: Text, silent_fail: Optional[bool] = False):
        self.action_name = name
        self.silent_fail = silent_fail

    def intent_name_from_action(self) -> Text:
        return self.action_name.split(RESPOND_PREFIX)[1]

    async def run(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
    ):
        """Query the appropriate response and create a bot utterance with that."""

        response_selector_properties = tracker.latest_message.parse_data[
            RESPONSE_SELECTOR_PROPERTY_NAME
        ]

        if self.intent_name_from_action() in response_selector_properties:
            query_key = self.intent_name_from_action()
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
        possible_messages = selected[RESPONSE_SELECTOR_PREDICTION_KEY][
            RESPONSE_SELECTOR_RESPONSES_KEY
        ]

        # Pick a random message from list of candidate messages.
        # This should ideally be done by the NLG class but that's not
        # possible until the domain has all the response templates of the response selector.
        picked_message_idx = random.randint(0, len(possible_messages) - 1)
        picked_message = copy.deepcopy(possible_messages[picked_message_idx])

        picked_message["template_name"] = selected[RESPONSE_SELECTOR_PREDICTION_KEY][
            INTENT_RESPONSE_KEY
        ]

        return [create_bot_utterance(picked_message)]

    def name(self) -> Text:
        return self.action_name

    def __str__(self) -> Text:
        return "ActionRetrieveResponse('{}')".format(self.name())


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


class ActionDeactivateForm(Action):
    """Deactivates a form"""

    def name(self) -> Text:
        return ACTION_DEACTIVATE_FORM_NAME

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
        from rasa.core.trackers import EventVerbosity

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
                ". Failed to validate Action server response from API, "
                "make sure your response from the Action endpoint is valid. "
                "For more information about the format visit "
                "{}/core/actions/".format(DOCS_BASE_URL)
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
            logger.error(
                "The model predicted the custom action '{}', "
                "but you didn't configure an endpoint to "
                "run this custom action. Please take a look at "
                "the docs and set an endpoint configuration via the "
                "--endpoints flag. "
                "{}/core/actions"
                "".format(self.name(), DOCS_BASE_URL)
            )
            raise Exception("Failed to execute custom action.")

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
                raise Exception("Failed to execute custom action.") from e

        except aiohttp.ClientConnectionError as e:
            logger.error(
                "Failed to run custom action '{}'. Couldn't connect "
                "to the server at '{}'. Is the server running? "
                "Error: {}".format(self.name(), self.action_endpoint.url, e)
            )
            raise Exception("Failed to execute custom action.")

        except aiohttp.ClientError as e:
            # not all errors have a status attribute, but
            # helpful to log if they got it

            # noinspection PyUnresolvedReferences
            status = getattr(e, "status", None)
            logger.error(
                "Failed to run custom action '{}'. Action server "
                "responded with a non 200 status code of {}. "
                "Make sure your action server properly runs actions "
                "and returns a 200 once the action is executed. "
                "Error: {}".format(self.name(), status, e)
            )
            raise Exception("Failed to execute custom action.")

    def name(self) -> Text:
        return self._name


class ActionExecutionRejection(Exception):
    """Raising this exception will allow other policies
        to predict a different action"""

    def __init__(self, action_name: Text, message: Optional[Text] = None) -> None:
        self.action_name = action_name
        self.message = message or "Custom action '{}' rejected to run".format(
            action_name
        )

    def __str__(self) -> Text:
        return self.message


class ActionRevertFallbackEvents(Action):
    """Reverts events which were done during the `TwoStageFallbackPolicy`.

       This reverts user messages and bot utterances done during a fallback
       of the `TwoStageFallbackPolicy`. By doing so it is not necessary to
       write custom stories for the different paths, but only of the happy
       path.
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

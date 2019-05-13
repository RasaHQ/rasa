import copy
import json
import logging
import typing
from typing import List, Text, Optional, Dict, Any

import aiohttp

import rasa.core
from rasa.core import events
from rasa.core.constants import (
    DOCS_BASE_URL,
    DEFAULT_REQUEST_TIMEOUT,
    REQUESTED_SLOT,
    USER_INTENT_OUT_OF_SCOPE,
)
from rasa.core.events import (
    UserUtteranceReverted,
    UserUttered,
    ActionExecuted,
    Event,
    BotUttered,
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

ACTION_DEFAULT_FALLBACK_NAME = "action_default_fallback"

ACTION_DEACTIVATE_FORM_NAME = "action_deactivate_form"

ACTION_REVERT_FALLBACK_EVENTS_NAME = "action_revert_fallback_events"

ACTION_DEFAULT_ASK_AFFIRMATION_NAME = "action_default_ask_affirmation"

ACTION_DEFAULT_ASK_REPHRASE_NAME = "action_default_ask_rephrase"

ACTION_BACK_NAME = "action_back"

UTTER_PREFIX = "utter_"


def default_actions() -> List["Action"]:
    """List default actions."""
    return [
        ActionListen(),
        ActionRestart(),
        ActionDefaultFallback(),
        ActionDeactivateForm(),
        ActionRevertFallbackEvents(),
        ActionDefaultAskAffirmation(),
        ActionDefaultAskRephrase(),
        ActionBack(),
    ]


def default_action_names() -> List[Text]:
    """List default action names."""
    return [a.name() for a in default_actions()]


def combine_user_with_default_actions(user_actions):
    # remove all user actions that overwrite default actions
    # this logic is a bit reversed, you'd think that we should remove
    # the action name from the default action names if the user overwrites
    # the action, but there are some locations in the code where we
    # implicitly assume that e.g. "action_listen" is always at location
    # 0 in this array. to keep it that way, we remove the duplicate
    # action names from the users list instead of the defaults
    unique_user_actions = [a for a in user_actions if a not in default_action_names()]
    return default_action_names() + unique_user_actions


def ensure_action_name_uniqueness(action_names: List[Text]) -> None:
    """Check and raise an exception if there are two actions with same name."""

    unique_action_names = set()  # used to collect unique action names
    for a in action_names:
        if a in unique_action_names:
            raise ValueError(
                "Action names are not unique! Found two actions with name"
                " '{}'. Either rename or remove one of them.".format(a)
            )
        else:
            unique_action_names.add(a)


def action_from_name(
    name: Text, action_endpoint: Optional[EndpointConfig], user_actions: List[Text]
) -> "Action":
    """Return an action instance for the name."""

    defaults = {a.name(): a for a in default_actions()}

    if name in defaults and name not in user_actions:
        return defaults.get(name)
    elif name.startswith(UTTER_PREFIX):
        return ActionUtterTemplate(name)
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
        text=message.pop("text"),
        data={
            "elements": message.pop("elements", None),
            "quick_replies": message.pop("quick_replies", None),
            "buttons": message.pop("buttons", None),
            # for legacy / compatibility reasons we need to set the image to be the attachment if there
            # is no other attachment (the `.get` is intentional - no `pop` as we still need the image`
            # property to set it in the following line)
            "attachment": message.pop("attachment", None) or message.get("image", None),
            "image": message.pop("image", None),
            "custom": message.pop("custom", None),
        },
        metadata=message,
    )

    logger.debug("Bot utterance '{}'".format(bot_message))
    return bot_message


class Action(object):
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
    ) -> List["Event"]:
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

    def __init__(self, name, silent_fail: Optional[bool] = False):
        self.template_name = name
        self.silent_fail = silent_fail

    async def run(self, output_channel, nlg, tracker, domain):
        """Simple run implementation uttering a (hopefully defined) template."""

        message = await nlg.generate(self.template_name, tracker, output_channel.name())
        if message is None:
            if not self.silent_fail:
                logger.error(
                    "Couldn't create message for template '{}'."
                    "".format(self.template_name)
                )
            return []

        return [create_bot_utterance(message)]

    def name(self) -> Text:
        return self.template_name

    def __str__(self) -> Text:
        return "ActionUtterTemplate('{}')".format(self.name())


class ActionBack(ActionUtterTemplate):
    """Revert the tracker state by two user utterances."""

    def name(self) -> Text:
        return ACTION_BACK_NAME

    def __init__(self):
        super(ActionBack, self).__init__("utter_back", silent_fail=True)

    async def run(self, output_channel, nlg, tracker, domain):
        # only utter the template if it is available
        events = await super(ActionBack, self).run(output_channel, nlg, tracker, domain)

        return events + [UserUtteranceReverted(), UserUtteranceReverted()]


class ActionListen(Action):
    """The first action in any turn - bot waits for a user message.

    The bot should stop taking further actions and wait for the user to say
    something."""

    def name(self) -> Text:
        return ACTION_LISTEN_NAME

    async def run(self, output_channel, nlg, tracker, domain):
        return []


class ActionRestart(ActionUtterTemplate):
    """Resets the tracker to its initial state.

    Utters the restart template if available."""

    def name(self) -> Text:
        return ACTION_RESTART_NAME

    def __init__(self):
        super(ActionRestart, self).__init__("utter_restart", silent_fail=True)

    async def run(self, output_channel, nlg, tracker, domain):
        from rasa.core.events import Restarted

        # only utter the template if it is available
        events = await super(ActionRestart, self).run(
            output_channel, nlg, tracker, domain
        )

        return events + [Restarted()]


class ActionDefaultFallback(ActionUtterTemplate):
    """Executes the fallback action and goes back to the previous state
    of the dialogue"""

    def name(self) -> Text:
        return ACTION_DEFAULT_FALLBACK_NAME

    def __init__(self):
        super(ActionDefaultFallback, self).__init__("utter_default", silent_fail=True)

    async def run(self, output_channel, nlg, tracker, domain):
        from rasa.core.events import UserUtteranceReverted

        # only utter the template if it is available
        events = await super(ActionDefaultFallback, self).run(
            output_channel, nlg, tracker, domain
        )

        return events + [UserUtteranceReverted()]


class ActionDeactivateForm(Action):
    """Deactivates a form"""

    def name(self) -> Text:
        return ACTION_DEACTIVATE_FORM_NAME

    async def run(self, output_channel, nlg, tracker, domain):
        from rasa.core.events import Form, SlotSet

        return [Form(None), SlotSet(REQUESTED_SLOT, None)]


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
    def action_response_format_spec():
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

    def _validate_action_result(self, result):
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
                "{}/customactions/".format(DOCS_BASE_URL)
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
            if "template" in response:
                kwargs = response.copy()
                del kwargs["template"]
                draft = await nlg.generate(
                    response["template"], tracker, output_channel.name(), **kwargs
                )
                if not draft:
                    continue

                del response["template"]
            else:
                draft = {}

            if "buttons" in response:
                if "buttons" not in draft:
                    draft["buttons"] = []
                draft["buttons"].extend(response["buttons"])
                del response["buttons"]

            draft.update(response)

            bot_messages.append(create_bot_utterance(draft))
        return bot_messages

    async def run(self, output_channel, nlg, tracker, domain):
        json_body = self._action_call_format(tracker, domain)

        if not self.action_endpoint:
            raise Exception(
                "The model predicted the custom action '{}' "
                "but you didn't configure an endpoint to "
                "run this custom action. Please take a look at "
                "the docs and set an endpoint configuration. "
                "{}/customactions/"
                "".format(self.name(), DOCS_BASE_URL)
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
                logger.debug(exception.message)
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

    def __init__(self, action_name, message=None):
        self.action_name = action_name
        self.message = message or "Custom action '{}' rejected to run".format(
            action_name
        )

    def __str__(self):
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

        revert_events = []

        # User rephrased
        if has_user_rephrased(tracker):
            revert_events = _revert_successful_rephrasing(tracker)
        # User affirmed
        elif has_user_affirmed(tracker):
            revert_events = _revert_affirmation_events(tracker)

        return revert_events


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
        intent_to_affirm = tracker.latest_message.intent.get("name")
        affirmation_message = "Did you mean '{}'?".format(intent_to_affirm)

        message = {
            "text": affirmation_message,
            "buttons": [
                {"title": "Yes", "payload": "/{}".format(intent_to_affirm)},
                {"title": "No", "payload": "/{}".format(USER_INTENT_OUT_OF_SCOPE)},
            ],
        }

        return [create_bot_utterance(message)]


class ActionDefaultAskRephrase(ActionUtterTemplate):
    """Default implementation which asks the user to rephrase his intent."""

    def name(self) -> Text:
        return ACTION_DEFAULT_ASK_REPHRASE_NAME

    def __init__(self):
        super(ActionDefaultAskRephrase, self).__init__(
            "utter_ask_rephrase", silent_fail=True
        )

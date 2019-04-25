import asyncio
import logging
import os
import tempfile
import textwrap
import uuid
from functools import partial
from multiprocessing import Process
from typing import Any, Callable, Dict, List, Optional, Text, Tuple, Union

import numpy as np
from aiohttp import ClientError
from colorclass import Color
from sanic import Sanic, response
from sanic.exceptions import NotFound
from terminaltables import AsciiTable, SingleTable

import questionary
import rasa.cli.utils
from questionary import Choice, Form, Question
from rasa.cli import utils as cliutils
from rasa.core import constants, events, run, train, utils
from rasa.core.actions.action import ACTION_LISTEN_NAME, default_action_names
from rasa.core.channels import UserMessage
from rasa.core.channels.channel import button_to_string, element_to_string
from rasa.core.constants import (
    DEFAULT_SERVER_FORMAT,
    DEFAULT_SERVER_PORT,
    DEFAULT_SERVER_URL,
    REQUESTED_SLOT,
)
from rasa.core.domain import Domain
from rasa.core.events import ActionExecuted, BotUttered, Event, Restarted, UserUttered
from rasa.core.interpreter import INTENT_MESSAGE_PREFIX, NaturalLanguageInterpreter
from rasa.core.trackers import EventVerbosity
from rasa.core.training import visualization
from rasa.core.training.structures import Story
from rasa.core.training.visualization import (
    VISUALIZATION_TEMPLATE_PATH,
    visualize_neighborhood,
)
from rasa.core.utils import AvailableEndpoints
from rasa.utils.endpoints import EndpointConfig

# noinspection PyProtectedMember
from rasa.nlu.training_data.loading import _guess_format, load_data
from rasa.nlu.training_data.message import Message

# WARNING: This command line UI is using an external library
# communicating with the shell - these functions are hard to test
# automatically. If you change anything in here, please make sure to
# run the interactive learning and check if your part of the "ui"
# still works.

logger = logging.getLogger(__name__)

MAX_VISUAL_HISTORY = 3

PATHS = {
    "stories": "data/stories.md",
    "nlu": "data/nlu.md",
    "backup": "data/nlu_interactive.md",
    "domain": "domain.yml",
}

# choose other intent, making sure this doesn't clash with an existing intent
OTHER_INTENT = uuid.uuid4().hex
OTHER_ACTION = uuid.uuid4().hex
NEW_ACTION = uuid.uuid4().hex

NEW_TEMPLATES = {}


class RestartConversation(Exception):
    """Exception used to break out the flow and restart the conversation."""

    pass


class ForkTracker(Exception):
    """Exception used to break out the flow and fork at a previous step.

    The tracker will be reset to the selected point in the past and the
    conversation will continue from there."""

    pass


class UndoLastStep(Exception):
    """Exception used to break out the flow and undo the last step.

    The last step is either the most recent user message or the most
    recent action run by the bot."""

    pass


class Abort(Exception):
    """Exception used to abort the interactive learning and exit."""

    pass


async def send_message(
    endpoint: EndpointConfig,
    sender_id: Text,
    message: Text,
    parse_data: Optional[Dict[Text, Any]] = None,
) -> Dict[Text, Any]:
    """Send a user message to a conversation."""

    payload = {
        "sender": UserUttered.type_name,
        "message": message,
        "parse_data": parse_data,
    }

    return await endpoint.request(
        json=payload,
        method="post",
        subpath="/conversations/{}/messages".format(sender_id),
    )


async def request_prediction(
    endpoint: EndpointConfig, sender_id: Text
) -> Dict[Text, Any]:
    """Request the next action prediction from core."""

    return await endpoint.request(
        method="post", subpath="/conversations/{}/predict".format(sender_id)
    )


async def retrieve_domain(endpoint: EndpointConfig) -> Dict[Text, Any]:
    """Retrieve the domain from core."""

    return await endpoint.request(
        method="get", subpath="/domain", headers={"Accept": "application/json"}
    )


async def retrieve_status(endpoint: EndpointConfig) -> Dict[Text, Any]:
    """Retrieve the status from core."""

    return await endpoint.request(method="get", subpath="/status")


async def retrieve_tracker(
    endpoint: EndpointConfig,
    sender_id: Text,
    verbosity: EventVerbosity = EventVerbosity.ALL,
) -> Dict[Text, Any]:
    """Retrieve a tracker from core."""

    path = "/conversations/{}/tracker?include_events={}".format(
        sender_id, verbosity.name
    )
    return await endpoint.request(
        method="get", subpath=path, headers={"Accept": "application/json"}
    )


async def send_action(
    endpoint: EndpointConfig,
    sender_id: Text,
    action_name: Text,
    policy: Optional[Text] = None,
    confidence: Optional[float] = None,
    is_new_action: bool = False,
) -> Dict[Text, Any]:
    """Log an action to a conversation."""

    payload = ActionExecuted(action_name, policy, confidence).as_dict()

    subpath = "/conversations/{}/execute".format(sender_id)

    try:
        return await endpoint.request(json=payload, method="post", subpath=subpath)
    except ClientError:
        if is_new_action:
            if action_name in NEW_TEMPLATES:
                warning_questions = questionary.confirm(
                    "WARNING: You have created a new action: '{0}', "
                    "with matching template: '{1}'. "
                    "This action will not return its message in this session, "
                    "but the new utterance will be saved to your domain file "
                    "when you exit and save this session. "
                    "You do not need to do anything further. "
                    "".format(action_name, [*NEW_TEMPLATES[action_name]][0])
                )
                await _ask_questions(warning_questions, sender_id, endpoint)
            else:
                warning_questions = questionary.confirm(
                    "WARNING: You have created a new action: '{}', "
                    "which was not successfully executed. "
                    "If this action does not return any events, "
                    "you do not need to do anything. "
                    "If this is a custom action which returns events, "
                    "you are recommended to implement this action "
                    "in your action server and try again."
                    "".format(action_name)
                )
                await _ask_questions(warning_questions, sender_id, endpoint)

            payload = ActionExecuted(action_name).as_dict()
            return await send_event(endpoint, sender_id, payload)
        else:
            logger.error("failed to execute action!")
            raise


async def send_event(
    endpoint: EndpointConfig, sender_id: Text, evt: Dict[Text, Any]
) -> Dict[Text, Any]:
    """Log an event to a conversation."""

    subpath = "/conversations/{}/tracker/events".format(sender_id)

    return await endpoint.request(json=evt, method="post", subpath=subpath)


async def replace_events(
    endpoint: EndpointConfig, sender_id: Text, evts: List[Dict[Text, Any]]
) -> Dict[Text, Any]:
    """Replace all the events of a conversation with the provided ones."""

    subpath = "/conversations/{}/tracker/events".format(sender_id)

    return await endpoint.request(json=evts, method="put", subpath=subpath)


async def send_finetune(
    endpoint: EndpointConfig, evts: List[Dict[Text, Any]]
) -> Dict[Text, Any]:
    """Finetune a core model on the provided additional training samples."""

    return await endpoint.request(json=evts, method="post", subpath="/finetune")


def format_bot_output(message: Dict[Text, Any]) -> Text:
    """Format a bot response to be displayed in the history table."""

    # First, add text to output
    output = message.get("text") or ""

    # Then, append all additional items
    data = message.get("data", {})
    if not data:
        return output

    if data.get("image"):
        output += "\nImage: " + data.get("image")

    if data.get("attachment"):
        output += "\nAttachment: " + data.get("attachment")

    if data.get("buttons"):
        output += "\nButtons:"
        for idx, button in enumerate(data.get("buttons")):
            button_str = button_to_string(button, idx)
            output += "\n" + button_str

    if data.get("elements"):
        output += "\nElements:"
        for idx, element in enumerate(data.get("elements")):
            element_str = element_to_string(element, idx)
            output += "\n" + element_str

    if data.get("quick_replies"):
        output += "\nQuick replies:"
        for idx, element in enumerate(data.get("quick_replies")):
            element_str = element_to_string(element, idx)
            output += "\n" + element_str
    return output


def latest_user_message(evts: List[Dict[Text, Any]]) -> Optional[Dict[Text, Any]]:
    """Return most recent user message."""

    for i, e in enumerate(reversed(evts)):
        if e.get("event") == UserUttered.type_name:
            return e
    return None


def all_events_before_latest_user_msg(
    evts: List[Dict[Text, Any]]
) -> List[Dict[Text, Any]]:
    """Return all events that happened before the most recent user message."""

    for i, e in enumerate(reversed(evts)):
        if e.get("event") == UserUttered.type_name:
            return evts[: -(i + 1)]
    return evts


async def _ask_questions(
    questions: Union[Form, Question],
    sender_id: Text,
    endpoint: EndpointConfig,
    is_abort: Callable[[Dict[Text, Any]], bool] = lambda x: False,
) -> Any:
    """Ask the user a question, if Ctrl-C is pressed provide user with menu."""

    should_retry = True
    answers = {}

    while should_retry:
        answers = questions.ask()
        if answers is None or is_abort(answers):
            should_retry = await _ask_if_quit(sender_id, endpoint)
        else:
            should_retry = False
    return answers


def _selection_choices_from_intent_prediction(
    predictions: List[Dict[Text, Any]]
) -> List[Dict[Text, Text]]:
    """"Given a list of ML predictions create a UI choice list."""

    sorted_intents = sorted(predictions, key=lambda k: (-k["confidence"], k["name"]))

    choices = []
    for p in sorted_intents:
        name_with_confidence = "{:03.2f} {:40}".format(
            p.get("confidence"), p.get("name")
        )
        choice = {"name": name_with_confidence, "value": p.get("name")}
        choices.append(choice)

    return choices


async def _request_free_text_intent(sender_id: Text, endpoint: EndpointConfig) -> Text:
    question = questionary.text("Please type the intent name:")
    return await _ask_questions(question, sender_id, endpoint)


async def _request_free_text_action(sender_id: Text, endpoint: EndpointConfig) -> Text:
    question = questionary.text("Please type the action name:")
    return await _ask_questions(question, sender_id, endpoint)


async def _request_free_text_utterance(
    sender_id: Text, endpoint: EndpointConfig, action: Text
) -> Text:
    question = questionary.text(
        "Please type the message for your new utter_template '{}':".format(action)
    )
    return await _ask_questions(question, sender_id, endpoint)


async def _request_selection_from_intent_list(
    intent_list: List[Dict[Text, Text]], sender_id: Text, endpoint: EndpointConfig
) -> Text:
    question = questionary.select("What intent is it?", choices=intent_list)
    return await _ask_questions(question, sender_id, endpoint)


async def _request_fork_point_from_list(
    forks: List[Dict[Text, Text]], sender_id: Text, endpoint: EndpointConfig
) -> Text:
    question = questionary.select(
        "Before which user message do you want to fork?", choices=forks
    )
    return await _ask_questions(question, sender_id, endpoint)


async def _request_fork_from_user(
    sender_id, endpoint
) -> Optional[List[Dict[Text, Any]]]:
    """Take in a conversation and ask at which point to fork the conversation.

    Returns the list of events that should be kept. Forking means, the
    conversation will be reset and continued from this previous point."""

    tracker = await retrieve_tracker(endpoint, sender_id, EventVerbosity.AFTER_RESTART)

    choices = []
    for i, e in enumerate(tracker.get("events", [])):
        if e.get("event") == UserUttered.type_name:
            choices.append({"name": e.get("text"), "value": i})

    fork_idx = await _request_fork_point_from_list(
        list(reversed(choices)), sender_id, endpoint
    )

    if fork_idx is not None:
        return tracker.get("events", [])[: int(fork_idx)]
    else:
        return None


async def _request_intent_from_user(
    latest_message, intents, sender_id, endpoint
) -> Dict[Text, Any]:
    """Take in latest message and ask which intent it should have been.

    Returns the intent dict that has been selected by the user."""

    predictions = latest_message.get("parse_data", {}).get("intent_ranking", [])

    predicted_intents = {p["name"] for p in predictions}

    for i in intents:
        if i not in predicted_intents:
            predictions.append({"name": i, "confidence": 0.0})

    # convert intents to ui list and add <other> as a free text alternative
    choices = [
        {"name": "<create_new_intent>", "value": OTHER_INTENT}
    ] + _selection_choices_from_intent_prediction(predictions)

    intent_name = await _request_selection_from_intent_list(
        choices, sender_id, endpoint
    )

    if intent_name == OTHER_INTENT:
        intent_name = await _request_free_text_intent(sender_id, endpoint)
        return {"name": intent_name, "confidence": 1.0}
    # returns the selected intent with the original probability value
    return next((x for x in predictions if x["name"] == intent_name), None)


async def _print_history(sender_id: Text, endpoint: EndpointConfig) -> None:
    """Print information about the conversation for the user."""

    tracker_dump = await retrieve_tracker(
        endpoint, sender_id, EventVerbosity.AFTER_RESTART
    )
    evts = tracker_dump.get("events", [])

    table = _chat_history_table(evts)
    slot_strs = _slot_history(tracker_dump)

    print ("------")
    print ("Chat History\n")
    print (table)

    if slot_strs:
        print ("\n")
        print ("Current slots: \n\t{}\n".format(", ".join(slot_strs)))

    print ("------")


def _chat_history_table(evts: List[Dict[Text, Any]]) -> Text:
    """Create a table containing bot and user messages.

    Also includes additional information, like any events and
    prediction probabilities."""

    def wrap(txt, max_width):
        return "\n".join(textwrap.wrap(txt, max_width, replace_whitespace=False))

    def colored(txt, color):
        return "{" + color + "}" + txt + "{/" + color + "}"

    def format_user_msg(user_evt, max_width):
        _parsed = user_evt.get("parse_data", {})
        _intent = _parsed.get("intent", {}).get("name")
        _confidence = _parsed.get("intent", {}).get("confidence", 1.0)
        _md = _as_md_message(_parsed)

        _lines = [
            colored(wrap(_md, max_width), "hired"),
            "intent: {} {:03.2f}".format(_intent, _confidence),
        ]
        return "\n".join(_lines)

    def bot_width(_table: AsciiTable) -> int:
        return _table.column_max_width(1)

    def user_width(_table: AsciiTable) -> int:
        return _table.column_max_width(3)

    def add_bot_cell(data, cell):
        data.append([len(data), Color(cell), "", ""])

    def add_user_cell(data, cell):
        data.append([len(data), "", "", Color(cell)])

    # prints the historical interactions between the bot and the user,
    # to help with correctly identifying the action
    table_data = [
        [
            "#  ",
            Color(colored("Bot      ", "autoblue")),
            "  ",
            Color(colored("You       ", "hired")),
        ]
    ]

    table = SingleTable(table_data, "Chat History")

    bot_column = []
    for idx, evt in enumerate(evts):
        if evt.get("event") == ActionExecuted.type_name:
            bot_column.append(colored(evt["name"], "autocyan"))
            if evt["confidence"] is not None:
                bot_column[-1] += colored(
                    " {:03.2f}".format(evt["confidence"]), "autowhite"
                )

        elif evt.get("event") == UserUttered.type_name:
            if bot_column:
                text = "\n".join(bot_column)
                add_bot_cell(table_data, text)
                bot_column = []

            msg = format_user_msg(evt, user_width(table))
            add_user_cell(table_data, msg)

        elif evt.get("event") == BotUttered.type_name:
            wrapped = wrap(format_bot_output(evt), bot_width(table))
            bot_column.append(colored(wrapped, "autoblue"))

        else:
            e = Event.from_parameters(evt)
            if e.as_story_string():
                bot_column.append(wrap(e.as_story_string(), bot_width(table)))

    if bot_column:
        text = "\n".join(bot_column)
        add_bot_cell(table_data, text)

    table.inner_heading_row_border = False
    table.inner_row_border = True
    table.inner_column_border = False
    table.outer_border = False
    table.justify_columns = {0: "left", 1: "left", 2: "center", 3: "right"}

    return table.table


def _slot_history(tracker_dump: Dict[Text, Any]) -> List[Text]:
    """Create an array of slot representations to be displayed."""

    slot_strs = []
    for k, s in tracker_dump.get("slots").items():
        colored_value = cliutils.wrap_with_color(
            str(s), color=rasa.cli.utils.bcolors.WARNING
        )
        slot_strs.append("{}: {}".format(k, colored_value))
    return slot_strs


async def _write_data_to_file(sender_id: Text, endpoint: EndpointConfig):
    """Write stories and nlu data to file."""

    story_path, nlu_path, domain_path = _request_export_info()

    tracker = await retrieve_tracker(endpoint, sender_id)
    evts = tracker.get("events", [])

    await _write_stories_to_file(story_path, evts)
    await _write_nlu_to_file(nlu_path, evts)
    await _write_domain_to_file(domain_path, evts, endpoint)

    logger.info("Successfully wrote stories and NLU data")


async def _ask_if_quit(sender_id: Text, endpoint: EndpointConfig) -> bool:
    """Display the exit menu.

    Return `True` if the previous question should be retried."""

    answer = questionary.select(
        message="Do you want to stop?",
        choices=[
            Choice("Continue", "continue"),
            Choice("Undo Last", "undo"),
            Choice("Fork", "fork"),
            Choice("Start Fresh", "restart"),
            Choice("Export & Quit", "quit"),
        ],
    ).ask()

    if not answer or answer == "quit":
        # this is also the default answer if the user presses Ctrl-C
        await _write_data_to_file(sender_id, endpoint)
        raise Abort()
    elif answer == "continue":
        # in this case we will just return, and the original
        # question will get asked again
        return True
    elif answer == "undo":
        raise UndoLastStep()
    elif answer == "fork":
        raise ForkTracker()
    elif answer == "restart":
        raise RestartConversation()


async def _request_action_from_user(
    predictions: List[Dict[Text, Any]], sender_id: Text, endpoint: EndpointConfig
) -> (Text, bool):
    """Ask the user to correct an action prediction."""

    await _print_history(sender_id, endpoint)

    choices = [
        {
            "name": "{:03.2f} {:40}".format(a.get("score"), a.get("action")),
            "value": a.get("action"),
        }
        for a in predictions
    ]

    tracker = await retrieve_tracker(endpoint, sender_id)
    events = tracker.get("events", [])

    session_actions_all = [a["name"] for a in _collect_actions(events)]
    session_actions_unique = list(set(session_actions_all))
    old_actions = [action["value"] for action in choices]
    new_actions = [
        {"name": action, "value": OTHER_ACTION + action}
        for action in session_actions_unique
        if action not in old_actions
    ]
    choices = (
        [{"name": "<create new action>", "value": NEW_ACTION}] + new_actions + choices
    )
    question = questionary.select("What is the next action of the bot?", choices)

    action_name = await _ask_questions(question, sender_id, endpoint)
    is_new_action = action_name == NEW_ACTION

    if is_new_action:
        # create new action
        action_name = await _request_free_text_action(sender_id, endpoint)
        if "utter_" in action_name:
            utter_message = await _request_free_text_utterance(
                sender_id, endpoint, action_name
            )
            NEW_TEMPLATES[action_name] = {utter_message: ""}

    elif action_name[:32] == OTHER_ACTION:
        # action was newly created in the session, but not this turn
        is_new_action = True
        action_name = action_name[32:]

    print ("Thanks! The bot will now run {}.\n".format(action_name))
    return action_name, is_new_action


def _request_export_info() -> Tuple[Text, Text, Text]:
    """Request file path and export stories & nlu data to that path"""

    # export training data and quit
    questions = questionary.form(
        export_stories=questionary.text(
            message="Export stories to (if file exists, this "
            "will append the stories)",
            default=PATHS["stories"],
        ),
        export_nlu=questionary.text(
            message="Export NLU data to (if file exists, this will "
            "merge learned data with previous training examples)",
            default=PATHS["nlu"],
        ),
        export_domain=questionary.text(
            message="Export domain file to (if file exists, this "
            "will be overwritten)",
            default=PATHS["domain"],
        ),
    )

    answers = questions.ask()
    if not answers:
        raise Abort()

    return (answers["export_stories"], answers["export_nlu"], answers["export_domain"])


def _split_conversation_at_restarts(
    evts: List[Dict[Text, Any]]
) -> List[List[Dict[Text, Any]]]:
    """Split a conversation at restart events.

    Returns an array of event lists, without the restart events."""

    sub_conversations = []
    current = []
    for e in evts:
        if e.get("event") == "restart":
            if current:
                sub_conversations.append(current)
            current = []
        else:
            current.append(e)

    if current:
        sub_conversations.append(current)

    return sub_conversations


def _collect_messages(evts: List[Dict[Text, Any]]) -> List[Message]:
    """Collect the message text and parsed data from the UserMessage events
    into a list"""
    from rasa.nlu.extractors.duckling_http_extractor import DucklingHTTPExtractor
    from rasa.nlu.extractors.mitie_entity_extractor import MitieEntityExtractor
    from rasa.nlu.extractors.spacy_entity_extractor import SpacyEntityExtractor

    msgs = []

    for evt in evts:
        if evt.get("event") == UserUttered.type_name:
            data = evt.get("parse_data")

            for entity in data.get("entities", []):

                excluded_extractors = [
                    DucklingHTTPExtractor.__name__,
                    SpacyEntityExtractor.__name__,
                    MitieEntityExtractor.__name__,
                ]
                logger.debug(
                    "Exclude entity marking of following extractors"
                    " {} when writing nlu data "
                    "to file.".format(excluded_extractors)
                )

                if entity.get("extractor") in excluded_extractors:
                    data["entities"].remove(entity)

            msg = Message.build(data["text"], data["intent"]["name"], data["entities"])
            msgs.append(msg)

    return msgs


def _collect_actions(evts: List[Dict[Text, Any]]) -> List[Dict[Text, Any]]:
    """Collect all the `ActionExecuted` events into a list."""

    return [evt for evt in evts if evt.get("event") == ActionExecuted.type_name]


async def _write_stories_to_file(
    export_story_path: Text, evts: List[Dict[Text, Any]]
) -> None:
    """Write the conversation of the sender_id to the file paths."""

    sub_conversations = _split_conversation_at_restarts(evts)

    with open(export_story_path, "a", encoding="utf-8") as f:
        for conversation in sub_conversations:
            parsed_events = events.deserialise_events(conversation)
            s = Story.from_events(parsed_events)
            f.write(s.as_story_string(flat=True) + "\n")


async def _write_nlu_to_file(
    export_nlu_path: Text, evts: List[Dict[Text, Any]]
) -> None:
    """Write the nlu data of the sender_id to the file paths."""
    from rasa.nlu.training_data import TrainingData

    msgs = _collect_messages(evts)

    # noinspection PyBroadException
    try:
        previous_examples = load_data(export_nlu_path)
    except Exception as e:
        logger.exception("An exception occurred while trying to load the NLU data.")

        export_nlu_path = questionary.text(
            message="Could not load existing NLU data, please "
            "specify where to store NLU data learned in "
            "this session (this will overwrite any "
            "existing file). {}".format(str(e)),
            default=PATHS["backup"],
        ).ask()

        if export_nlu_path is None:
            return

        previous_examples = TrainingData()

    nlu_data = previous_examples.merge(TrainingData(msgs))

    # need to guess the format of the file before opening it to avoid a read
    # in a write
    if _guess_format(export_nlu_path) in {"md", "unk"}:
        fformat = "md"
    else:
        fformat = "json"

    with open(export_nlu_path, "w", encoding="utf-8") as f:
        if fformat == "md":
            f.write(nlu_data.as_markdown())
        else:
            f.write(nlu_data.as_json())


def _entities_from_messages(messages):
    """Return all entities that occur in atleast one of the messages."""
    return list({e["entity"] for m in messages for e in m.data.get("entities", [])})


def _intents_from_messages(messages):
    """Return all intents that occur in at least one of the messages."""

    # set of distinct intents
    intents = {m.data["intent"] for m in messages if "intent" in m.data}

    return [{i: {"use_entities": True}} for i in intents]


async def _write_domain_to_file(
    domain_path: Text, evts: List[Dict[Text, Any]], endpoint: EndpointConfig
) -> None:
    """Write an updated domain file to the file path."""

    domain = await retrieve_domain(endpoint)
    old_domain = Domain.from_dict(domain)

    messages = _collect_messages(evts)
    actions = _collect_actions(evts)
    templates = NEW_TEMPLATES

    # TODO for now there is no way to distinguish between action and form
    intent_properties = Domain.collect_intent_properties(
        _intents_from_messages(messages)
    )

    collected_actions = list(
        {e["name"] for e in actions if e["name"] not in default_action_names()}
    )

    new_domain = Domain(
        intent_properties=intent_properties,
        entities=_entities_from_messages(messages),
        slots=[],
        templates=templates,
        action_names=collected_actions,
        form_names=[],
    )

    old_domain.merge(new_domain).persist_clean(domain_path)


async def _predict_till_next_listen(
    endpoint: EndpointConfig,
    sender_id: Text,
    finetune: bool,
    sender_ids: List[Text],
    plot_file: Optional[Text],
) -> None:
    """Predict and validate actions until we need to wait for a user msg."""

    listen = False
    while not listen:
        result = await request_prediction(endpoint, sender_id)
        predictions = result.get("scores")
        probabilities = [prediction["score"] for prediction in predictions]
        pred_out = int(np.argmax(probabilities))
        action_name = predictions[pred_out].get("action")
        policy = result.get("policy")
        confidence = result.get("confidence")

        await _print_history(sender_id, endpoint)
        await _plot_trackers(
            sender_ids, plot_file, endpoint, unconfirmed=[ActionExecuted(action_name)]
        )

        listen = await _validate_action(
            action_name,
            policy,
            confidence,
            predictions,
            endpoint,
            sender_id,
            finetune=finetune,
        )

        await _plot_trackers(sender_ids, plot_file, endpoint)


async def _correct_wrong_nlu(
    corrected_nlu: Dict[Text, Any],
    evts: List[Dict[Text, Any]],
    endpoint: EndpointConfig,
    sender_id: Text,
) -> None:
    """A wrong NLU prediction got corrected, update core's tracker."""

    latest_message = latest_user_message(evts)
    corrected_events = all_events_before_latest_user_msg(evts)

    latest_message["parse_data"] = corrected_nlu

    await replace_events(endpoint, sender_id, corrected_events)

    await send_message(
        endpoint,
        sender_id,
        latest_message.get("text"),
        latest_message.get("parse_data"),
    )


async def _correct_wrong_action(
    corrected_action: Text,
    endpoint: EndpointConfig,
    sender_id: Text,
    finetune: bool = False,
    is_new_action: bool = False,
) -> None:
    """A wrong action prediction got corrected, update core's tracker."""

    result = await send_action(
        endpoint, sender_id, corrected_action, is_new_action=is_new_action
    )

    if finetune:
        await send_finetune(endpoint, result.get("tracker", {}).get("events", []))


def _form_is_rejected(action_name, tracker):
    """Check if the form got rejected with the most recent action name."""
    return (
        tracker.get("active_form", {}).get("name")
        and action_name != tracker["active_form"]["name"]
        and action_name != ACTION_LISTEN_NAME
    )


def _form_is_restored(action_name, tracker):
    """Check whether the form is called again after it was rejected."""
    return (
        tracker.get("active_form", {}).get("rejected")
        and tracker.get("latest_action_name") == ACTION_LISTEN_NAME
        and action_name == tracker.get("active_form", {}).get("name")
    )


async def _confirm_form_validation(action_name, tracker, endpoint, sender_id):
    """Ask a user whether an input for a form should be validated.

    Previous to this call, the active form was chosen after it was rejected."""

    requested_slot = tracker.get("slots", {}).get(REQUESTED_SLOT)

    validation_questions = questionary.confirm(
        "Should '{}' validate user input to fill "
        "the slot '{}'?".format(action_name, requested_slot)
    )
    validate_input = await _ask_questions(validation_questions, sender_id, endpoint)

    if not validate_input:
        # notify form action to skip validation
        await send_event(
            endpoint, sender_id, {"event": "form_validation", "validate": False}
        )

    elif not tracker.get("active_form", {}).get("validate"):
        # handle contradiction with learned behaviour
        warning_question = questionary.confirm(
            "ERROR: FormPolicy predicted no form validation "
            "based on previous training stories. "
            "Make sure to remove contradictory stories "
            "from training data. "
            "Otherwise predicting no form validation "
            "will not work as expected."
        )

        await _ask_questions(warning_question, sender_id, endpoint)
        # notify form action to validate an input
        await send_event(
            endpoint, sender_id, {"event": "form_validation", "validate": True}
        )


async def _validate_action(
    action_name: Text,
    policy: Text,
    confidence: float,
    predictions: List[Dict[Text, Any]],
    endpoint: EndpointConfig,
    sender_id: Text,
    finetune: bool = False,
) -> bool:
    """Query the user to validate if an action prediction is correct.

    Returns `True` if the prediction is correct, `False` otherwise."""

    question = questionary.confirm(
        "The bot wants to run '{}', correct?".format(action_name)
    )

    is_correct = await _ask_questions(question, sender_id, endpoint)

    if not is_correct:
        action_name, is_new_action = await _request_action_from_user(
            predictions, sender_id, endpoint
        )
    else:
        is_new_action = False

    tracker = await retrieve_tracker(endpoint, sender_id, EventVerbosity.AFTER_RESTART)

    if _form_is_rejected(action_name, tracker):
        # notify the tracker that form was rejected
        await send_event(
            endpoint,
            sender_id,
            {
                "event": "action_execution_rejected",
                "name": tracker["active_form"]["name"],
            },
        )

    elif _form_is_restored(action_name, tracker):
        await _confirm_form_validation(action_name, tracker, endpoint, sender_id)

    if not is_correct:
        await _correct_wrong_action(
            action_name,
            endpoint,
            sender_id,
            finetune=finetune,
            is_new_action=is_new_action,
        )
    else:
        await send_action(endpoint, sender_id, action_name, policy, confidence)

    return action_name == ACTION_LISTEN_NAME


def _as_md_message(parse_data: Dict[Text, Any]) -> Text:
    """Display the parse data of a message in markdown format."""
    from rasa.nlu.training_data.formats import MarkdownWriter

    if parse_data.get("text", "").startswith(INTENT_MESSAGE_PREFIX):
        return parse_data.get("text")

    if not parse_data.get("entities"):
        parse_data["entities"] = []
    # noinspection PyProtectedMember
    return MarkdownWriter()._generate_message_md(parse_data)


def _validate_user_regex(latest_message: Dict[Text, Any], intents: List[Text]) -> bool:
    """Validate if a users message input is correct.

    This assumes the user entered an intent directly, e.g. using
    `/greet`. Return `True` if the intent is a known one."""

    parse_data = latest_message.get("parse_data", {})
    intent = parse_data.get("intent", {}).get("name")

    if intent in intents:
        return True
    else:
        return False


async def _validate_user_text(
    latest_message: Dict[Text, Any], endpoint: EndpointConfig, sender_id: Text
) -> bool:
    """Validate a user message input as free text.

    This assumes the user message is a text message (so NOT `/greet`)."""

    parse_data = latest_message.get("parse_data", {})
    text = _as_md_message(parse_data)
    intent = parse_data.get("intent", {}).get("name")
    entities = parse_data.get("entities", [])
    if entities:
        message = (
            "Is the intent '{}' correct for '{}' and are "
            "all entities labeled correctly?".format(intent, text)
        )
    else:
        message = (
            "Your NLU model classified '{}' with intent '{}'"
            " and there are no entities, is this correct?".format(text, intent)
        )

    if intent is None:
        print ("The NLU classification for '{}' returned '{}'".format(text, intent))
        return False
    else:
        question = questionary.confirm(message)

        return await _ask_questions(question, sender_id, endpoint)


async def _validate_nlu(
    intents: List[Text], endpoint: EndpointConfig, sender_id: Text
) -> None:
    """Validate if a user message, either text or intent is correct.

    If the prediction of the latest user message is incorrect,
    the tracker will be corrected with the correct intent / entities."""

    tracker = await retrieve_tracker(endpoint, sender_id, EventVerbosity.AFTER_RESTART)

    latest_message = latest_user_message(tracker.get("events", []))

    if latest_message.get("text").startswith(INTENT_MESSAGE_PREFIX):
        valid = _validate_user_regex(latest_message, intents)
    else:
        valid = await _validate_user_text(latest_message, endpoint, sender_id)

    if not valid:
        corrected_intent = await _request_intent_from_user(
            latest_message, intents, sender_id, endpoint
        )
        evts = tracker.get("events", [])

        entities = await _correct_entities(latest_message, endpoint, sender_id)
        corrected_nlu = {
            "intent": corrected_intent,
            "entities": entities,
            "text": latest_message.get("text"),
        }

        await _correct_wrong_nlu(corrected_nlu, evts, endpoint, sender_id)


async def _correct_entities(
    latest_message: Dict[Text, Any], endpoint: EndpointConfig, sender_id: Text
) -> List[Dict[Text, Any]]:
    """Validate the entities of a user message.

    Returns the corrected entities"""
    from rasa.nlu.training_data.formats import MarkdownReader

    parse_original = latest_message.get("parse_data", {})
    entity_str = _as_md_message(parse_original)
    question = questionary.text(
        "Please mark the entities using [value](type) notation", default=entity_str
    )

    annotation = await _ask_questions(question, sender_id, endpoint)
    # noinspection PyProtectedMember
    parse_annotated = MarkdownReader()._parse_training_example(annotation)

    corrected_entities = _merge_annotated_and_original_entities(
        parse_annotated, parse_original
    )

    return corrected_entities


def _merge_annotated_and_original_entities(parse_annotated, parse_original):
    # overwrite entities which have already been
    # annotated in the original annotation to preserve
    # additional entity parser information
    entities = parse_annotated.get("entities", [])[:]
    for i, entity in enumerate(entities):
        for original_entity in parse_original.get("entities", []):
            if _is_same_entity_annotation(entity, original_entity):
                entities[i] = original_entity
                break
    return entities


def _is_same_entity_annotation(entity, other):
    return entity["value"] == other["value"] and entity["entity"] == other["entity"]


async def _enter_user_message(sender_id: Text, endpoint: EndpointConfig) -> None:
    """Request a new message from the user."""

    question = questionary.text("Your input ->")

    message = await _ask_questions(question, sender_id, endpoint, lambda a: not a)

    if message == (INTENT_MESSAGE_PREFIX + constants.USER_INTENT_RESTART):
        raise RestartConversation()

    await send_message(endpoint, sender_id, message)


async def is_listening_for_message(sender_id: Text, endpoint: EndpointConfig) -> bool:
    """Check if the conversation is in need for a user message."""

    tracker = await retrieve_tracker(endpoint, sender_id, EventVerbosity.APPLIED)

    for i, e in enumerate(reversed(tracker.get("events", []))):
        if e.get("event") == UserUttered.type_name:
            return False
        elif e.get("event") == ActionExecuted.type_name:
            return e.get("name") == ACTION_LISTEN_NAME
    return False


async def _undo_latest(sender_id: Text, endpoint: EndpointConfig) -> None:
    """Undo either the latest bot action or user message, whatever is last."""

    tracker = await retrieve_tracker(endpoint, sender_id, EventVerbosity.ALL)

    cutoff_index = None
    for i, e in enumerate(reversed(tracker.get("events", []))):
        if e.get("event") in {ActionExecuted.type_name, UserUttered.type_name}:
            cutoff_index = i
            break
        elif e.get("event") == Restarted.type_name:
            break

    if cutoff_index is not None:
        events_to_keep = tracker["events"][: -(cutoff_index + 1)]

        # reset the events of the conversation to the events before
        # the most recent bot or user event
        await replace_events(endpoint, sender_id, events_to_keep)


async def _fetch_events(
    sender_ids: List[Union[Text, List[Event]]], endpoint: EndpointConfig
) -> List[List[Event]]:
    """Retrieve all event trackers from the endpoint for all sender ids."""

    event_sequences = []
    for sender_id in sender_ids:
        if isinstance(sender_id, str):
            tracker = await retrieve_tracker(endpoint, sender_id)
            evts = tracker.get("events", [])

            for conversation in _split_conversation_at_restarts(evts):
                parsed_events = events.deserialise_events(conversation)
                event_sequences.append(parsed_events)
        else:
            event_sequences.append(sender_id)
    return event_sequences


async def _plot_trackers(
    sender_ids: List[Union[Text, List[Event]]],
    output_file: Optional[Text],
    endpoint: EndpointConfig,
    unconfirmed: Optional[List[Event]] = None,
):
    """Create a plot of the trackers of the passed sender ids.

    This assumes that the last sender id is the conversation we are currently
    working on. If there are events that are not part of this active tracker
    yet, they can be passed as part of `unconfirmed`. They will be appended
    to the currently active conversation."""

    if not output_file or not sender_ids:
        # if there is no output file provided, we are going to skip plotting
        # same happens if there are no sender ids
        return None

    event_sequences = await _fetch_events(sender_ids, endpoint)

    if unconfirmed:
        event_sequences[-1].extend(unconfirmed)

    graph = await visualize_neighborhood(
        event_sequences[-1], event_sequences, output_file=None, max_history=2
    )

    from networkx.drawing.nx_pydot import write_dot

    write_dot(graph, output_file)


def _print_help(skip_visualization: bool) -> None:
    """Print some initial help message for the user."""

    if not skip_visualization:
        visualization_url = DEFAULT_SERVER_FORMAT.format(DEFAULT_SERVER_PORT + 1)
        visualization_help = "Visualisation at {}/visualization.html.".format(
            visualization_url
        )
    else:
        visualization_help = ""

    rasa.cli.utils.print_success(
        "Bot loaded. {}\n"
        "Type a message and press enter "
        "(press 'Ctr-c' to exit). "
        "".format(visualization_help)
    )


async def record_messages(
    endpoint: EndpointConfig,
    sender_id: Text = UserMessage.DEFAULT_SENDER_ID,
    max_message_limit: Optional[int] = None,
    finetune: bool = False,
    stories: Optional[Text] = None,
    skip_visualization: bool = False,
):
    """Read messages from the command line and print bot responses."""

    from rasa.core import training

    try:
        _print_help(skip_visualization)

        try:
            domain = await retrieve_domain(endpoint)
        except ClientError:
            logger.exception(
                "Failed to connect to Rasa Core server at '{}'. "
                "Is the server running?".format(endpoint.url)
            )
            return

        trackers = await training.load_data(
            stories,
            Domain.from_dict(domain),
            augmentation_factor=0,
            use_story_concatenation=False,
        )

        intents = [next(iter(i)) for i in (domain.get("intents") or [])]

        num_messages = 0
        sender_ids = [t.events for t in trackers] + [sender_id]

        if not skip_visualization:
            plot_file = "story_graph.dot"
            await _plot_trackers(sender_ids, plot_file, endpoint)
        else:
            plot_file = None

        while not utils.is_limit_reached(num_messages, max_message_limit):
            try:
                if await is_listening_for_message(sender_id, endpoint):
                    await _enter_user_message(sender_id, endpoint)
                    await _validate_nlu(intents, endpoint, sender_id)
                await _predict_till_next_listen(
                    endpoint, sender_id, finetune, sender_ids, plot_file
                )

                num_messages += 1
            except RestartConversation:
                await send_event(endpoint, sender_id, Restarted().as_dict())

                await send_event(
                    endpoint, sender_id, ActionExecuted(ACTION_LISTEN_NAME).as_dict()
                )

                logger.info("Restarted conversation, starting a new one.")
            except UndoLastStep:
                await _undo_latest(sender_id, endpoint)
                await _print_history(sender_id, endpoint)
            except ForkTracker:
                await _print_history(sender_id, endpoint)

                evts_fork = await _request_fork_from_user(sender_id, endpoint)

                await send_event(endpoint, sender_id, Restarted().as_dict())

                if evts_fork:
                    for evt in evts_fork:
                        await send_event(endpoint, sender_id, evt)
                logger.info("Restarted conversation at fork.")

                await _print_history(sender_id, endpoint)
                await _plot_trackers(sender_ids, plot_file, endpoint)

    except Abort:
        return
    except Exception:
        logger.exception("An exception occurred while recording messages.")
        raise


def _serve_application(app, stories, finetune, skip_visualization):
    """Start a core server and attach the interactive learning IO."""

    endpoint = EndpointConfig(url=DEFAULT_SERVER_URL)

    async def run_interactive_io(running_app: Sanic):
        """Small wrapper to shut down the server once cmd io is done."""

        await record_messages(
            endpoint=endpoint,
            stories=stories,
            finetune=finetune,
            skip_visualization=skip_visualization,
            sender_id=uuid.uuid4().hex,
        )

        logger.info("Killing Sanic server now.")

        running_app.stop()  # kill the sanic server

    app.add_task(run_interactive_io)
    app.run(host="0.0.0.0", port=DEFAULT_SERVER_PORT, access_log=True)

    return app


def start_visualization(image_path: Text = None) -> None:
    """Add routes to serve the conversation visualization files."""

    app = Sanic(__name__)

    # noinspection PyUnusedLocal
    @app.exception(NotFound)
    async def ignore_404s(request, exception):
        return response.text("Not found", status=404)

    # noinspection PyUnusedLocal
    @app.route(VISUALIZATION_TEMPLATE_PATH, methods=["GET"])
    def visualisation_html(request):
        return response.file(visualization.visualization_html_path())

    # noinspection PyUnusedLocal
    @app.route("/visualization.dot", methods=["GET"])
    def visualisation_png(request):
        try:
            headers = {"Cache-Control": "no-cache"}
            return response.file(os.path.abspath(image_path), headers=headers)
        except FileNotFoundError:
            return response.text("", 404)

    app.run(host="0.0.0.0", port=DEFAULT_SERVER_PORT + 1, access_log=False)


# noinspection PyUnusedLocal
async def train_agent_on_start(args, endpoints, additional_arguments, app, loop):
    _interpreter = NaturalLanguageInterpreter.create(args.get("nlu"), endpoints.nlu)

    model_directory = args.get("out", tempfile.mkdtemp(suffix="_core_model"))

    _agent = await train(
        args.get("domain"),
        args.get("stories"),
        model_directory,
        _interpreter,
        endpoints,
        args.get("dump_stories"),
        args.get("config")[0],
        None,
        additional_arguments,
    )
    app.agent = _agent


async def wait_til_server_is_running(endpoint, max_retries=30, sleep_between_retries=1):
    """Try to reach the server, retry a couple of times and sleep in between."""

    while max_retries:
        try:
            r = await retrieve_status(endpoint)
            logger.info("Reached core: {}".format(r))
            if not r.get("is_ready"):
                # server did not finish loading the agent yet
                # in this case, we need to wait till the model trained
                # so we might be sleeping for a while...
                await asyncio.sleep(sleep_between_retries)
                continue
            else:
                # server is ready to go
                return True
        except ClientError:
            max_retries -= 1
            if max_retries:
                await asyncio.sleep(sleep_between_retries)

    return False


def run_interactive_learning(
    stories: Text = None,
    finetune: bool = False,
    skip_visualization: bool = False,
    server_args: Dict[Text, Any] = None,
    additional_arguments: Dict[Text, Any] = None,
):
    """Start the interactive learning with the model of the agent."""

    server_args = server_args or {}

    if server_args.get("nlu_data"):
        PATHS["nlu"] = server_args["nlu_data"]

    if server_args.get("stories"):
        PATHS["stories"] = server_args["stories"]

    if server_args.get("domain"):
        PATHS["domain"] = server_args["domain"]

    if not skip_visualization:
        p = Process(target=start_visualization, args=("story_graph.dot",))
        p.deamon = True
        p.start()
    else:
        p = None

    app = run.configure_app(enable_api=True)
    endpoints = AvailableEndpoints.read_endpoints(server_args.get("endpoints"))

    # before_server_start handlers make sure the agent is loaded before the
    # interactive learning IO starts
    if server_args.get("core"):
        app.register_listener(
            partial(
                run.load_agent_on_start,
                server_args.get("core"),
                endpoints,
                server_args.get("nlu"),
            ),
            "before_server_start",
        )
    else:
        app.register_listener(
            partial(train_agent_on_start, server_args, endpoints, additional_arguments),
            "before_server_start",
        )

    _serve_application(app, stories, finetune, skip_visualization)

    if not skip_visualization:
        p.terminate()
        p.join()

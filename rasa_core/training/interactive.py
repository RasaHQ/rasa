import sys

import io
import logging
import numpy as np
import os
import requests
import textwrap
import uuid
from PyInquirer import prompt
from colorclass import Color
from flask import Flask, send_file, abort
from gevent.pywsgi import WSGIServer
from terminaltables import SingleTable, AsciiTable
from threading import Thread
from typing import Any, Text, Dict, List, Optional, Callable, Union, Tuple

from rasa_core import utils, server, events, constants
from rasa_core.actions.action import ACTION_LISTEN_NAME, default_action_names
from rasa_core.agent import Agent
from rasa_core.channels import UserMessage
from rasa_core.channels.channel import button_to_string, element_to_string
from rasa_core.constants import (
    DEFAULT_SERVER_PORT, DEFAULT_SERVER_URL, REQUESTED_SLOT)
from rasa_core.domain import Domain
from rasa_core.events import (
    Event, ActionExecuted, UserUttered, Restarted,
    BotUttered)
from rasa_core.interpreter import INTENT_MESSAGE_PREFIX
from rasa_core.trackers import EventVerbosity
from rasa_core.training import visualization
from rasa_core.training.structures import Story
from rasa_core.training.visualization import (
    visualize_neighborhood, VISUALIZATION_TEMPLATE_PATH)
from rasa_core.utils import EndpointConfig
from rasa_nlu.training_data import TrainingData
from rasa_nlu.training_data.formats import MarkdownWriter, MarkdownReader
# noinspection PyProtectedMember
from rasa_nlu.training_data.loading import load_data, _guess_format
from rasa_nlu.training_data.message import Message

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

# WARNING: This command line UI is using an external library
# communicating with the shell - these functions are hard to test
# automatically. If you change anything in here, please make sure to
# run the interactive learning and check if your part of the "ui"
# still works.

logger = logging.getLogger(__name__)

MAX_VISUAL_HISTORY = 3

PATHS = {"stories": "data/stories.md",
         "nlu": "data/nlu.md",
         "backup": "data/nlu_interactive.md",
         "domain": "domain.yml"}

# choose other intent, making sure this doesn't clash with an existing intent
OTHER_INTENT = uuid.uuid4().hex
OTHER_ACTION = uuid.uuid4().hex


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


def _response_as_json(response: requests.Response) -> Dict[Text, Any]:
    """Convert a HTTP response to json, raise exception if response failed."""

    response.raise_for_status()

    if response.encoding is None:
        response.encoding = 'utf-8'

    return response.json()


def send_message(
    endpoint: EndpointConfig,
    sender_id: Text,
    message: Text,
    parse_data: Optional[Dict[Text, Any]] = None
) -> Dict[Text, Any]:
    """Send a user message to a conversation."""

    payload = {
        "sender": UserUttered.type_name,
        "message": message,
        "parse_data": parse_data
    }

    r = endpoint.request(json=payload,
                         method="post",
                         subpath="/conversations/{}/messages"
                                 "".format(sender_id))

    return _response_as_json(r)


def request_prediction(
    endpoint: EndpointConfig,
    sender_id: Text
) -> Dict[Text, Any]:
    """Request the next action prediction from core."""

    r = endpoint.request(method="post",
                         subpath="/conversations/{}/predict".format(sender_id))

    return _response_as_json(r)


def retrieve_domain(endpoint: EndpointConfig) -> Dict[Text, Any]:
    """Retrieve the domain from core."""

    r = endpoint.request(method="get",
                         subpath="/domain",
                         headers={"Accept": "application/json"})

    return _response_as_json(r)


def retrieve_tracker(
    endpoint: EndpointConfig,
    sender_id: Text,
    verbosity: EventVerbosity = EventVerbosity.ALL
) -> Dict[Text, Any]:
    """Retrieve a tracker from core."""

    path = "/conversations/{}/tracker?include_events={}".format(
        sender_id, verbosity.name)
    r = endpoint.request(method="get",
                         subpath=path,
                         headers={"Accept": "application/json"})

    return _response_as_json(r)


def send_action(
    endpoint: EndpointConfig,
    sender_id: Text,
    action_name: Text,
    policy: Optional[Text] = None,
    confidence: Optional[float] = None,
    is_new_action: bool = False
) -> Dict[Text, Any]:
    """Log an action to a conversation."""

    payload = ActionExecuted(action_name, policy, confidence).as_dict()

    subpath = "/conversations/{}/execute".format(sender_id)

    try:
        r = endpoint.request(json=payload,
                             method="post",
                             subpath=subpath)
        return _response_as_json(r)
    except requests.exceptions.HTTPError:
        if is_new_action:
            warning_questions = [{
                "name": "warning",
                "type": "confirm",
                "message": "WARNING: You have created a new action: '{}', "
                           "which was not successfully executed. "
                           "If this action does not return any events, "
                           "you do not need to do anything. "
                           "If this is a custom action which returns events, "
                           "you are recommended to implement this action "
                           "in your action server and try again."
                           "".format(action_name)
            }]
            _ask_questions(warning_questions, sender_id, endpoint)

            payload = ActionExecuted(action_name).as_dict()

            return send_event(endpoint, sender_id, payload)
        else:
            logger.error("failed to execute action!")
            raise


def send_event(
    endpoint: EndpointConfig,
    sender_id: Text,
    evt: Dict[Text, Any]
) -> Dict[Text, Any]:
    """Log an event to a conversation."""

    subpath = "/conversations/{}/tracker/events".format(sender_id)

    r = endpoint.request(json=evt,
                         method="post",
                         subpath=subpath)

    return _response_as_json(r)


def replace_events(
    endpoint: EndpointConfig,
    sender_id: Text,
    evts: List[Dict[Text, Any]]
) -> Dict[Text, Any]:
    """Replace all the events of a conversation with the provided ones."""

    subpath = "/conversations/{}/tracker/events".format(sender_id)

    r = endpoint.request(json=evts,
                         method="put",
                         subpath=subpath)

    return _response_as_json(r)


def send_finetune(
    endpoint: EndpointConfig,
    evts: List[Dict[Text, Any]]
) -> Dict[Text, Any]:
    """Finetune a core model on the provided additional training samples."""

    r = endpoint.request(json=evts,
                         method="post",
                         subpath="/finetune")

    return _response_as_json(r)


def format_bot_output(
    message: Dict[Text, Any]
) -> Text:
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
    return output


def latest_user_message(
    evts: List[Dict[Text, Any]]
) -> Optional[Dict[Text, Any]]:
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
            return evts[:-(i + 1)]
    return evts


def _ask_questions(
    questions: List[Dict[Text, Any]],
    sender_id: Text,
    endpoint: EndpointConfig,
    is_abort: Callable[[Dict[Text, Any]], bool] = lambda x: False
) -> Dict[Text, Any]:
    """Ask the user a question, if Ctrl-C is pressed provide user with menu."""

    should_retry = True
    answers = {}

    while should_retry:
        answers = prompt(questions)
        if not answers or is_abort(answers):
            should_retry = _ask_if_quit(sender_id, endpoint)
        else:
            should_retry = False
    return answers


def _selection_choices_from_intent_prediction(
    predictions: List[Dict[Text, Any]]
) -> List[Dict[Text, Text]]:
    """"Given a list of ML predictions create a UI choice list."""

    sorted_intents = sorted(predictions,
                            key=lambda k: (-k['confidence'], k['name']))

    choices = []
    for p in sorted_intents:
        name_with_confidence = "{:03.2f} {:40}".format(p.get("confidence"),
                                                       p.get("name"))
        choice = {
            "name": name_with_confidence,
            "value": p.get("name")
        }
        choices.append(choice)

    return choices


def _request_free_text_intent(
    sender_id: Text,
    endpoint: EndpointConfig
) -> Text:
    questions = [
        {
            "type": "input",
            "name": "intent",
            "message": "Please type the intent name",
        }
    ]
    answers = _ask_questions(questions, sender_id, endpoint)
    return answers["intent"]


def _request_free_text_action(
    sender_id: Text,
    endpoint: EndpointConfig
) -> Text:
    questions = [
        {
            "type": "input",
            "name": "action",
            "message": "Please type the action name",
        }
    ]
    answers = _ask_questions(questions, sender_id, endpoint)
    return answers["action"]


def _request_selection_from_intent_list(
    intent_list: List[Dict[Text, Text]],
    sender_id: Text,
    endpoint: EndpointConfig
) -> Text:
    questions = [
        {
            "type": "list",
            "name": "intent",
            "message": "What intent is it?",
            "choices": intent_list
        }
    ]
    return _ask_questions(questions, sender_id, endpoint)["intent"]


def _request_fork_point_from_list(
    forks: List[Dict[Text, Text]],
    sender_id: Text,
    endpoint: EndpointConfig
) -> Text:
    questions = [
        {
            "type": "list",
            "name": "fork",
            "message": "Before which user message do you want to fork?",
            "choices": forks
        }
    ]
    return _ask_questions(questions, sender_id, endpoint)["fork"]


def _request_fork_from_user(
    sender_id,
    endpoint
) -> Optional[List[Dict[Text, Any]]]:
    """Take in a conversation and ask at which point to fork the conversation.

    Returns the list of events that should be kept. Forking means, the
    conversation will be reset and continued from this previous point."""

    tracker = retrieve_tracker(endpoint, sender_id,
                               EventVerbosity.AFTER_RESTART)

    choices = []
    for i, e in enumerate(tracker.get("events", [])):
        if e.get("event") == UserUttered.type_name:
            choices.append({"name": e.get("text"), "value": i})

    fork_idx = _request_fork_point_from_list(list(reversed(choices)),
                                             sender_id,
                                             endpoint)

    if fork_idx is not None:
        return tracker.get("events", [])[:int(fork_idx)]
    else:
        return None


def _request_intent_from_user(
    latest_message,
    intents,
    sender_id,
    endpoint
) -> Dict[Text, Any]:
    """Take in latest message and ask which intent it should have been.

    Returns the intent dict that has been selected by the user."""

    predictions = latest_message.get("parse_data",
                                     {}).get("intent_ranking", [])

    predicted_intents = {p["name"] for p in predictions}

    for i in intents:
        if i not in predicted_intents:
            predictions.append({"name": i, "confidence": 0.0})

    # convert intents to ui list and add <other> as a free text alternative
    choices = ([{"name": "<create_new_intent>", "value": OTHER_INTENT}] +
               _selection_choices_from_intent_prediction(predictions))

    intent_name = _request_selection_from_intent_list(choices,
                                                      sender_id,
                                                      endpoint)

    if intent_name == OTHER_INTENT:
        intent_name = _request_free_text_intent(sender_id, endpoint)
        return {"name": intent_name, "confidence": 1.0}
    # returns the selected intent with the original probability value
    return next((x for x in predictions if x["name"] == intent_name), None)


def _print_history(sender_id: Text, endpoint: EndpointConfig) -> None:
    """Print information about the conversation for the user."""

    tracker_dump = retrieve_tracker(endpoint, sender_id,
                                    EventVerbosity.AFTER_RESTART)
    evts = tracker_dump.get("events", [])

    table = _chat_history_table(evts)
    slot_strs = _slot_history(tracker_dump)

    print("------")
    print("Chat History\n")
    print(table)

    if slot_strs:
        print("\n")
        print("Current slots: \n\t{}\n".format(", ".join(slot_strs)))

    print("------")


def _chat_history_table(evts: List[Dict[Text, Any]]) -> Text:
    """Create a table containing bot and user messages.

    Also includes additional information, like any events and
    prediction probabilities."""

    def wrap(txt, max_width):
        return "\n".join(textwrap.wrap(txt, max_width,
                                       replace_whitespace=False))

    def colored(txt, color):
        return "{" + color + "}" + txt + "{/" + color + "}"

    def format_user_msg(user_evt, max_width):
        _parsed = user_evt.get('parse_data', {})
        _intent = _parsed.get('intent', {}).get("name")
        _confidence = _parsed.get('intent', {}).get("confidence", 1.0)
        _md = _as_md_message(_parsed)

        _lines = [
            colored(wrap(_md, max_width), "hired"),
            "intent: {} {:03.2f}".format(_intent, _confidence)
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
        ["#  ",
         Color(colored('Bot      ', 'autoblue')),
         "  ",
         Color(colored('You       ', 'hired'))],
    ]

    table = SingleTable(table_data, 'Chat History')

    bot_column = []
    for idx, evt in enumerate(evts):
        if evt.get("event") == ActionExecuted.type_name:
            bot_column.append(colored(evt['name'], 'autocyan'))
            if evt['confidence'] is not None:
                bot_column[-1] += (
                    colored(" {:03.2f}".format(evt['confidence']), 'autowhite'))

        elif evt.get("event") == UserUttered.type_name:
            if bot_column:
                text = "\n".join(bot_column)
                add_bot_cell(table_data, text)
                bot_column = []

            msg = format_user_msg(evt, user_width(table))
            add_user_cell(table_data, msg)

        elif evt.get("event") == BotUttered.type_name:
            wrapped = wrap(format_bot_output(evt), bot_width(table))
            bot_column.append(colored(wrapped, 'autoblue'))

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
    table.justify_columns = {0: 'left', 1: 'left', 2: 'center', 3: 'right'}

    return table.table


def _slot_history(tracker_dump: Dict[Text, Any]) -> List[Text]:
    """Create an array of slot representations to be displayed."""

    slot_strs = []
    for k, s in tracker_dump.get("slots").items():
        colored_value = utils.wrap_with_color(str(s),
                                              utils.bcolors.WARNING)
        slot_strs.append("{}: {}".format(k, colored_value))
    return slot_strs


def _ask_if_quit(sender_id: Text, endpoint: EndpointConfig) -> bool:
    """Display the exit menu.

    Return `True` if the previous question should be retried."""

    questions = [{
        "name": "abort",
        "type": "list",
        "message": "Do you want to stop?",
        "choices": [
            {
                "name": "Continue",
                "value": "continue",
            },
            {
                "name": "Undo Last",
                "value": "undo",
            },
            {
                "name": "Fork",
                "value": "fork",
            },
            {
                "name": "Start Fresh",
                "value": "restart",
            },
            {
                "name": "Export & Quit",
                "value": "quit",
            },
        ]
    }]
    answers = prompt(questions)

    if not answers or answers["abort"] == "quit":
        # this is also the default answer if the user presses Ctrl-C
        story_path, nlu_path, domain_path = _request_export_info()

        tracker = retrieve_tracker(endpoint, sender_id)
        evts = tracker.get("events", [])

        _write_stories_to_file(story_path, evts)
        _write_nlu_to_file(nlu_path, evts)
        _write_domain_to_file(domain_path, evts, endpoint)

        logger.info("Successfully wrote stories and NLU data")
        sys.exit()
    elif answers["abort"] == "continue":
        # in this case we will just return, and the original
        # question will get asked again
        return True
    elif answers["abort"] == "undo":
        raise UndoLastStep()
    elif answers["abort"] == "fork":
        raise ForkTracker()
    elif answers["abort"] == "restart":
        raise RestartConversation()


def _request_action_from_user(
    predictions: List[Dict[Text, Any]],
    sender_id: Text, endpoint: EndpointConfig
) -> (Text, bool):
    """Ask the user to correct an action prediction."""

    _print_history(sender_id, endpoint)

    sorted_actions = sorted(predictions,
                            key=lambda k: (-k['score'], k['action']))

    choices = [{"name": "{:03.2f} {:40}".format(a.get("score"),
                                                a.get("action")),
                "value": a.get("action")}
               for a in sorted_actions]
    choices = [{"name": "<create new action>", "value": OTHER_ACTION}] + choices
    questions = [{
        "name": "action",
        "type": "list",
        "message": "What is the next action of the bot?",
        "choices": choices
    }]
    answers = _ask_questions(questions, sender_id, endpoint)
    action_name = answers["action"]
    is_new_action = action_name == OTHER_ACTION

    if is_new_action:
        action_name = _request_free_text_action(sender_id, endpoint)
    print("Thanks! The bot will now run {}.\n".format(action_name))
    return action_name, is_new_action


def _request_export_info() -> Tuple[Text, Text, Text]:
    """Request file path and export stories & nlu data to that path"""

    def validate_path(path):
        try:
            with io.open(path, "a", encoding="utf-8"):
                return True
        except Exception as e:
            return "Failed to open file. {}".format(e)

    # export training data and quit
    questions = [{
        "name": "export stories",
        "type": "input",
        "message": "Export stories to (if file exists, this "
                   "will append the stories)",
        "default": PATHS["stories"],
        "validate": validate_path
    }, {
        "name": "export nlu",
        "type": "input",
        "message": "Export NLU data to (if file exists, this "
                   "will merge learned data with previous training examples)",
        "default": PATHS["nlu"],
        "validate": validate_path
    }, {
        "name": "export domain",
        "type": "input",
        "message": "Export domain file to (if file exists, this "
                   "will be overwritten)",
        "default": PATHS["domain"],
        "validate": validate_path}]

    answers = prompt(questions)
    if not answers:
        sys.exit()

    return (answers["export stories"],
            answers["export nlu"],
            answers["export domain"])


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

    msgs = []

    for evt in evts:
        if evt.get("event") == UserUttered.type_name:
            data = evt.get("parse_data")
            msg = Message.build(data["text"], data["intent"]["name"],
                                data["entities"])
            msgs.append(msg)

    return msgs


def _collect_actions(evts: List[Dict[Text, Any]]) -> List[Dict[Text, Any]]:
    """Collect all the `ActionExecuted` events into a list."""

    return [evt
            for evt in evts
            if evt.get("event") == ActionExecuted.type_name]


def _write_stories_to_file(
    export_story_path: Text,
    evts: List[Dict[Text, Any]]
) -> None:
    """Write the conversation of the sender_id to the file paths."""

    sub_conversations = _split_conversation_at_restarts(evts)

    with io.open(export_story_path, 'a', encoding="utf-8") as f:
        for conversation in sub_conversations:
            parsed_events = events.deserialise_events(conversation)
            s = Story.from_events(parsed_events)
            f.write(s.as_story_string(flat=True) + "\n")


def _write_nlu_to_file(
    export_nlu_path: Text,
    evts: List[Dict[Text, Any]]
) -> None:
    """Write the nlu data of the sender_id to the file paths."""

    msgs = _collect_messages(evts)

    # noinspection PyBroadException
    try:
        previous_examples = load_data(export_nlu_path)

    except Exception:
        questions = [{"name": "export nlu",
                      "type": "input",
                      "message": "Could not load existing NLU data, please "
                                 "specify where to store NLU data learned in "
                                 "this session (this will overwrite any "
                                 "existing file)",
                      "default": PATHS["backup"]}]

        answers = prompt(questions)
        export_nlu_path = answers["export nlu"]
        previous_examples = TrainingData()

    nlu_data = previous_examples.merge(TrainingData(msgs))

    with io.open(export_nlu_path, 'w', encoding="utf-8") as f:
        if _guess_format(export_nlu_path) in {"md", "unk"}:
            f.write(nlu_data.as_markdown())
        else:
            f.write(nlu_data.as_json())


def _entities_from_messages(messages):
    """Return all entities that occur in atleast one of the messages."""
    return list({e["entity"]
                 for m in messages
                 for e in m.data.get("entities", [])})


def _intents_from_messages(messages):
    """Return all intents that occur in at least one of the messages."""

    # set of distinct intents
    intents = {m.data["intent"]
               for m in messages
               if "intent" in m.data}

    return [{i: {"use_entities": True}} for i in intents]


def _write_domain_to_file(
    domain_path: Text,
    evts: List[Dict[Text, Any]],
    endpoint: EndpointConfig
) -> None:
    """Write an updated domain file to the file path."""

    domain = retrieve_domain(endpoint)
    old_domain = Domain.from_dict(domain)

    messages = _collect_messages(evts)
    actions = _collect_actions(evts)

    domain_dict = dict.fromkeys(domain.keys(), [])

    # TODO for now there is no way to distinguish between action and form
    domain_dict["forms"] = []
    domain_dict["intents"] = _intents_from_messages(messages)
    domain_dict["entities"] = _entities_from_messages(messages)
    # do not automatically add default actions to the domain dict
    domain_dict["actions"] = list({e["name"]
                                   for e in actions
                                   if e["name"] not in default_action_names()})

    new_domain = Domain.from_dict(domain_dict)

    old_domain.merge(new_domain).persist_clean(domain_path)


def _predict_till_next_listen(endpoint: EndpointConfig,
                              sender_id: Text,
                              finetune: bool,
                              sender_ids: List[Text],
                              plot_file: Optional[Text]
                              ) -> None:
    """Predict and validate actions until we need to wait for a user msg."""

    listen = False
    while not listen:
        response = request_prediction(endpoint, sender_id)
        predictions = response.get("scores")
        probabilities = [prediction["score"] for prediction in predictions]
        pred_out = int(np.argmax(probabilities))
        action_name = predictions[pred_out].get("action")
        policy = response.get("policy")
        confidence = response.get("confidence")

        _print_history(sender_id, endpoint)
        _plot_trackers(sender_ids, plot_file, endpoint,
                       unconfirmed=[ActionExecuted(action_name)])

        listen = _validate_action(action_name, policy, confidence,
                                  predictions, endpoint, sender_id,
                                  finetune=finetune)

        _plot_trackers(sender_ids, plot_file, endpoint)


def _correct_wrong_nlu(corrected_nlu: Dict[Text, Any],
                       evts: List[Dict[Text, Any]],
                       endpoint: EndpointConfig,
                       sender_id: Text
                       ) -> None:
    """A wrong NLU prediction got corrected, update core's tracker."""

    latest_message = latest_user_message(evts)
    corrected_events = all_events_before_latest_user_msg(evts)

    latest_message["parse_data"] = corrected_nlu

    replace_events(endpoint, sender_id, corrected_events)

    send_message(endpoint, sender_id, latest_message.get("text"),
                 latest_message.get("parse_data"))


def _correct_wrong_action(corrected_action: Text,
                          endpoint: EndpointConfig,
                          sender_id: Text,
                          finetune: bool = False,
                          is_new_action: bool = False
                          ) -> None:
    """A wrong action prediction got corrected, update core's tracker."""

    response = send_action(endpoint,
                           sender_id,
                           corrected_action,
                           is_new_action=is_new_action)

    if finetune:
        send_finetune(endpoint,
                      response.get("tracker", {}).get("events", []))


def _form_is_rejected(action_name, tracker):
    """Check if the form got rejected with the most recent action name."""
    return (tracker.get('active_form', {}).get('name') and
            action_name != tracker['active_form']['name'] and
            action_name != ACTION_LISTEN_NAME)


def _form_is_restored(action_name, tracker):
    """Check whether the form is called again after it was rejected."""
    return (tracker.get('active_form', {}).get('rejected') and
            tracker.get('latest_action_name') == ACTION_LISTEN_NAME and
            action_name == tracker.get('active_form', {}).get('name'))


def _confirm_form_validation(action_name, tracker, endpoint, sender_id):
    """Ask a user whether an input for a form should be validated.

    Previous to this call, the active form was chosen after it was rejected."""

    requested_slot = tracker.get("slots", {}).get(REQUESTED_SLOT)

    validation_questions = [{
        "name": "validation",
        "type": "confirm",
        "message": "Should '{}' validate user input to fill "
                   "the slot '{}'?".format(action_name, requested_slot)
    }]
    form_answers = _ask_questions(validation_questions, sender_id, endpoint)

    if not form_answers["validation"]:
        # notify form action to skip validation
        send_event(endpoint, sender_id,
                   {"event": "form_validation", "validate": False})

    elif not tracker.get('active_form', {}).get('validate'):
        # handle contradiction with learned behaviour
        warning_questions = [{
            "name": "warning",
            "type": "confirm",
            "message": "ERROR: FormPolicy predicted no form validation "
                       "based on previous training stories. "
                       "Make sure to remove contradictory stories "
                       "from training data. "
                       "Otherwise predicting no form validation "
                       "will not work as expected."
        }]
        _ask_questions(warning_questions, sender_id, endpoint)
        # notify form action to validate an input
        send_event(endpoint, sender_id,
                   {"event": "form_validation", "validate": True})


def _validate_action(action_name: Text,
                     policy: Text,
                     confidence: float,
                     predictions: List[Dict[Text, Any]],
                     endpoint: EndpointConfig,
                     sender_id: Text,
                     finetune: bool = False
                     ) -> bool:
    """Query the user to validate if an action prediction is correct.

    Returns `True` if the prediction is correct, `False` otherwise."""

    q = "The bot wants to run '{}', correct?".format(action_name)
    questions = [
        {
            "type": "confirm",
            "name": "action",
            "message": q,
        }
    ]
    answers = _ask_questions(questions, sender_id, endpoint)

    if not answers["action"]:
        action_name, is_new_action = _request_action_from_user(
            predictions, sender_id, endpoint)
    else:
        is_new_action = False

    tracker = retrieve_tracker(endpoint, sender_id,
                               EventVerbosity.AFTER_RESTART)

    if _form_is_rejected(action_name, tracker):
        # notify the tracker that form was rejected
        send_event(endpoint, sender_id,
                   {"event": "action_execution_rejected",
                    "name": tracker['active_form']['name']})

    elif _form_is_restored(action_name, tracker):
        _confirm_form_validation(action_name, tracker, endpoint, sender_id)

    if not answers["action"]:
        _correct_wrong_action(action_name, endpoint, sender_id,
                              finetune=finetune,
                              is_new_action=is_new_action)
    else:
        send_action(endpoint, sender_id, action_name, policy, confidence)

    return action_name == ACTION_LISTEN_NAME


def _as_md_message(parse_data: Dict[Text, Any]) -> Text:
    """Display the parse data of a message in markdown format."""

    if parse_data.get("text", "").startswith(INTENT_MESSAGE_PREFIX):
        return parse_data.get("text")

    if not parse_data.get("entities"):
        parse_data["entities"] = []
    # noinspection PyProtectedMember
    return MarkdownWriter()._generate_message_md(parse_data)


def _validate_user_regex(latest_message: Dict[Text, Any],
                         intents: List[Text]) -> bool:
    """Validate if a users message input is correct.

    This assumes the user entered an intent directly, e.g. using
    `/greet`. Return `True` if the intent is a known one."""

    parse_data = latest_message.get("parse_data", {})
    intent = parse_data.get("intent", {}).get("name")

    if intent in intents:
        return True
    else:
        return False


def _validate_user_text(latest_message: Dict[Text, Any],
                        endpoint: EndpointConfig, sender_id: Text) -> bool:
    """Validate a user message input as free text.

    This assumes the user message is a text message (so NOT `/greet`)."""

    parse_data = latest_message.get("parse_data", {})
    entities = _as_md_message(parse_data)
    intent = parse_data.get("intent", {}).get("name")

    q = ("Is the NLU classification for '{}' with intent "
         "'{}' correct?".format(entities, intent))

    questions = [
        {
            "type": "confirm",
            "name": "nlu",
            "message": q,
        }
    ]
    answers = _ask_questions(questions, sender_id, endpoint)
    return answers["nlu"]


def _validate_nlu(intents: List[Text],
                  endpoint: EndpointConfig,
                  sender_id: Text) -> None:
    """Validate if a user message, either text or intent is correct.

    If the prediction of the latest user message is incorrect,
    the tracker will be corrected with the correct intent / entities."""

    tracker = retrieve_tracker(endpoint, sender_id,
                               EventVerbosity.AFTER_RESTART)

    latest_message = latest_user_message(tracker.get("events", []))

    if latest_message.get("text").startswith(INTENT_MESSAGE_PREFIX):
        valid = _validate_user_regex(latest_message, intents)
    else:
        valid = _validate_user_text(latest_message, endpoint, sender_id)

    if not valid:
        corrected_intent = _request_intent_from_user(latest_message, intents,
                                                     sender_id, endpoint)
        evts = tracker.get("events", [])

        entities = _correct_entities(latest_message, endpoint, sender_id)
        corrected_nlu = {
            "intent": corrected_intent,
            "entities": entities,
            "text": latest_message.get("text")
        }

        _correct_wrong_nlu(corrected_nlu, evts, endpoint, sender_id)


def _correct_entities(latest_message: Dict[Text, Any],
                      endpoint: EndpointConfig,
                      sender_id: Text) -> Dict[Text, Any]:
    """Validate the entities of a user message.

    Returns the corrected entities"""

    q = "Please mark the entities using [value](type) notation"
    entity_str = _as_md_message(latest_message.get("parse_data", {}))
    questions = [
        {
            "type": "input",
            "name": "annotation",
            "default": entity_str,
            "message": q,
        }
    ]
    answers = _ask_questions(questions, sender_id, endpoint)
    # noinspection PyProtectedMember
    parsed = MarkdownReader()._parse_training_example(answers["annotation"])
    return parsed.get("entities", [])


def _enter_user_message(sender_id: Text,
                        endpoint: EndpointConfig) -> None:
    """Request a new message from the user."""

    questions = [{
        "name": "message",
        "type": "input",
        "message": "Next user input (Ctr-c to abort):"
    }]

    answers = _ask_questions(questions, sender_id, endpoint,
                             lambda a: not a["message"])

    if answers["message"] == constants.USER_INTENT_RESTART:
        raise RestartConversation()

    send_message(endpoint, sender_id, answers["message"])


def is_listening_for_message(sender_id: Text,
                             endpoint: EndpointConfig) -> bool:
    """Check if the conversation is in need for a user message."""

    tracker = retrieve_tracker(endpoint, sender_id, EventVerbosity.APPLIED)

    for i, e in enumerate(reversed(tracker.get("events", []))):
        if e.get("event") == UserUttered.type_name:
            return False
        elif e.get("event") == ActionExecuted.type_name:
            return e.get("name") == ACTION_LISTEN_NAME
    return False


def _undo_latest(sender_id: Text,
                 endpoint: EndpointConfig) -> None:
    """Undo either the latest bot action or user message, whatever is last."""

    tracker = retrieve_tracker(endpoint, sender_id, EventVerbosity.ALL)

    cutoff_index = None
    for i, e in enumerate(reversed(tracker.get("events", []))):
        if e.get("event") in {ActionExecuted.type_name, UserUttered.type_name}:
            cutoff_index = i
            break
        elif e.get("event") == Restarted.type_name:
            break

    if cutoff_index is not None:
        events_to_keep = tracker["events"][:-(cutoff_index + 1)]

        # reset the events of the conversation to the events before
        # the most recent bot or user event
        replace_events(endpoint, sender_id, events_to_keep)


def _fetch_events(sender_ids: List[Union[Text, List[Event]]],
                  endpoint: EndpointConfig
                  ) -> List[List[Event]]:
    """Retrieve all event trackers from the endpoint for all sender ids."""

    event_sequences = []
    for sender_id in sender_ids:
        if isinstance(sender_id, str):
            tracker = retrieve_tracker(endpoint, sender_id)
            evts = tracker.get("events", [])

            for conversation in _split_conversation_at_restarts(evts):
                parsed_events = events.deserialise_events(conversation)
                event_sequences.append(parsed_events)
        else:
            event_sequences.append(sender_id)
    return event_sequences


def _plot_trackers(sender_ids: List[Union[Text, List[Event]]],
                   output_file: Optional[Text],
                   endpoint: EndpointConfig,
                   unconfirmed: Optional[List[Event]] = None
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

    event_sequences = _fetch_events(sender_ids, endpoint)

    if unconfirmed:
        event_sequences[-1].extend(unconfirmed)

    graph = visualize_neighborhood(event_sequences[-1],
                                   event_sequences,
                                   output_file=None,
                                   max_history=2)

    from networkx.drawing.nx_pydot import write_dot
    write_dot(graph, output_file)


def _print_help(skip_visualization: bool) -> None:
    """Print some initial help message for the user."""

    if not skip_visualization:
        visualization_help = "Visualisation at {}/visualization.html." \
                             "".format(DEFAULT_SERVER_URL)
    else:
        visualization_help = ""

    utils.print_color("Bot loaded. {}\n"
                      "Type a message and press enter "
                      "(press 'Ctr-c' to exit). "
                      "".format(visualization_help), utils.bcolors.OKGREEN)


def record_messages(endpoint: EndpointConfig,
                    sender_id: Text = UserMessage.DEFAULT_SENDER_ID,
                    max_message_limit: Optional[int] = None,
                    on_finish: Optional[Callable[[], None]] = None,
                    finetune: bool = False,
                    stories: Optional[Text] = None,
                    skip_visualization: bool = False
                    ):
    """Read messages from the command line and print bot responses."""

    from rasa_core import training

    try:
        _print_help(skip_visualization)

        try:
            domain = retrieve_domain(endpoint)
        except requests.exceptions.ConnectionError:
            logger.exception("Failed to connect to rasa core server at '{}'. "
                             "Is the server running?".format(endpoint.url))
            return

        trackers = training.load_data(stories, Domain.from_dict(domain),
                                      augmentation_factor=0,
                                      use_story_concatenation=False,
                                      )

        intents = [next(iter(i)) for i in (domain.get("intents") or [])]

        num_messages = 0
        sender_ids = [t.events for t in trackers] + [sender_id]

        if not skip_visualization:
            plot_file = "story_graph.dot"
            _plot_trackers(sender_ids, plot_file, endpoint)
        else:
            plot_file = None

        while not utils.is_limit_reached(num_messages, max_message_limit):
            try:
                if is_listening_for_message(sender_id, endpoint):
                    _enter_user_message(sender_id, endpoint)
                    _validate_nlu(intents, endpoint, sender_id)
                _predict_till_next_listen(endpoint, sender_id,
                                          finetune, sender_ids, plot_file)

                num_messages += 1
            except RestartConversation:
                send_event(endpoint, sender_id,
                           Restarted().as_dict())

                send_event(endpoint, sender_id,
                           ActionExecuted(ACTION_LISTEN_NAME).as_dict())

                logger.info("Restarted conversation, starting a new one.")
            except UndoLastStep:
                _undo_latest(sender_id, endpoint)
                _print_history(sender_id, endpoint)
            except ForkTracker:
                _print_history(sender_id, endpoint)

                evts = _request_fork_from_user(sender_id, endpoint)
                sender_id = uuid.uuid4().hex

                if evts is not None:
                    replace_events(endpoint, sender_id, evts)
                    sender_ids.append(sender_id)
                    _print_history(sender_id, endpoint)
                    _plot_trackers(sender_ids, plot_file, endpoint)

    except Exception:
        logger.exception("An exception occurred while recording messages.")
        raise
    finally:
        if on_finish:
            on_finish()


def _start_interactive_learning_io(endpoint: EndpointConfig,
                                   stories: Text,
                                   on_finish: Callable[[], None],
                                   finetune: bool = False,
                                   skip_visualization: bool = False) -> None:
    """Start the interactive learning message recording in a separate thread.
    """
    p = Thread(target=record_messages,
               kwargs={
                   "endpoint": endpoint,
                   "on_finish": on_finish,
                   "stories": stories,
                   "finetune": finetune,
                   "skip_visualization": skip_visualization,
                   "sender_id": uuid.uuid4().hex})
    p.setDaemon(True)
    p.start()


def _serve_application(app: Flask, stories: Text,
                       finetune: bool = False,
                       serve_forever: bool = True,
                       skip_visualization: bool = False) -> WSGIServer:
    """Start a core server and attach the interactive learning IO."""

    if not skip_visualization:
        _add_visualization_routes(app, "story_graph.dot")

    http_server = WSGIServer(('0.0.0.0', DEFAULT_SERVER_PORT), app, log=None)
    logger.info("Rasa Core server is up and running on "
                "{}".format(DEFAULT_SERVER_URL))
    http_server.start()

    endpoint = EndpointConfig(url=DEFAULT_SERVER_URL)
    _start_interactive_learning_io(endpoint, stories,
                                   http_server.stop,
                                   finetune=finetune,
                                   skip_visualization=skip_visualization)

    if serve_forever:
        try:
            http_server.serve_forever()
        except Exception as exc:
            logger.exception(exc)

    return http_server


def _add_visualization_routes(app: Flask, image_path: Text = None) -> None:
    """Add routes to serve the conversation visualization files."""

    @app.route(VISUALIZATION_TEMPLATE_PATH, methods=["GET"])
    def visualisation_html():
        return send_file(visualization.visualization_html_path())

    @app.route("/visualization.dot", methods=["GET"])
    def visualisation_png():
        try:
            response = send_file(os.path.abspath(image_path))
            response.headers['Cache-Control'] = "no-cache"
            return response
        except FileNotFoundError:
            abort(404)


def run_interactive_learning(agent: Agent,
                             stories: Text = None,
                             finetune: bool = False,
                             serve_forever: bool = True,
                             skip_visualization: bool = False) -> WSGIServer:
    """Start the interactive learning with the model of the agent."""

    app = server.create_app(agent)

    return _serve_application(app, stories, finetune,
                              serve_forever, skip_visualization)

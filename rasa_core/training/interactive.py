from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys

import io
import logging
import numpy as np
import requests
import textwrap
import uuid
from PyInquirer import prompt
from colorclass import Color
from flask import Flask
from gevent.pywsgi import WSGIServer
from terminaltables import SingleTable, AsciiTable
from threading import Thread
from typing import Any, Text, Dict, List, Optional, Callable

from rasa_core import utils, server, events
from rasa_core.actions.action import ACTION_LISTEN_NAME
from rasa_core.agent import Agent
from rasa_core.channels import UserMessage
from rasa_core.channels.channel import button_to_string
from rasa_core.constants import DEFAULT_SERVER_PORT, DEFAULT_SERVER_URL
from rasa_core.events import Event
from rasa_core.interpreter import INTENT_MESSAGE_PREFIX
from rasa_core.trackers import EventVerbosity
from rasa_core.training.structures import Story
from rasa_core.utils import EndpointConfig
from rasa_nlu.training_data.formats import MarkdownWriter, MarkdownReader
from rasa_nlu.training_data.loading import load_data, _guess_format
from rasa_nlu.training_data.message import Message
from rasa_nlu.training_data import TrainingData

logger = logging.getLogger(__name__)

MAX_VISUAL_HISTORY = 3

PATHS = {"stories": "data/stories.md",
         "nlu": "data/nlu.md",
         "backup": "data/nlu_interactive.md"}

# choose other intent, making sure this doesn't clash with an existing intent
OTHER_INTENT = uuid.uuid4().hex


class RestartConversation(Exception):
    """Exception used to break out the flow and restart the conversation."""
    pass


class UndoLastStep(Exception):
    """Exception used to break out the flow and undo the last step.

    The last step is either the most recent user message or the most
    recent action run by the bot."""
    pass


def _response_as_json(response):
    # type: (requests.Response) -> Dict[Text, Any]
    """Convert a HTTP response to json, raise exception if response failed."""

    response.raise_for_status()

    if response.encoding is None:
        response.encoding = 'utf-8'

    return response.json()


def send_message(endpoint,  # type: EndpointConfig
                 sender_id,  # type: Text
                 message,  # type: Text
                 parse_data=None  # type: Optional[Dict[Text, Any]]
                 ):
    # type: (...) -> Dict[Text, Any]
    """Send a user message to a conversation."""

    payload = {
        "sender": "user",
        "message": message,
        "parse_data": parse_data
    }

    r = endpoint.request(json=payload,
                         method="post",
                         subpath="/conversations/{}/messages".format(sender_id))

    return _response_as_json(r)


def request_prediction(endpoint, sender_id):
    # type: (EndpointConfig, Text) -> Dict[Text, Any]
    """Request the next action prediction from core."""

    r = endpoint.request(method="post",
                         subpath="/conversations/{}/predict".format(sender_id))

    return _response_as_json(r)


def retrieve_domain(endpoint):
    # type: (EndpointConfig) -> Dict[Text, Any]
    """Retrieve the domain from core."""

    r = endpoint.request(method="get",
                         subpath="/domain",
                         headers={"Accept": "application/json"})

    return _response_as_json(r)


def retrieve_tracker(endpoint, sender_id, verbosity=EventVerbosity.ALL):
    # type: (EndpointConfig, Text, EventVerbosity) -> Dict[Text, Any]
    """Retrieve a tracker from core."""

    path = "/conversations/{}/tracker?include_events={}".format(
            sender_id, verbosity.name)
    r = endpoint.request(method="get",
                         subpath=path,
                         headers={"Accept": "application/json"})

    return _response_as_json(r)


def send_action(endpoint, sender_id, action_name):
    # type: (EndpointConfig, Text, Text) -> Dict[Text, Any]
    """Log an action to a conversation."""

    payload = {"action": action_name}
    subpath = "/conversations/{}/execute".format(sender_id)

    r = endpoint.request(json=payload,
                         method="post",
                         subpath=subpath)

    return _response_as_json(r)


def send_event(endpoint, sender_id, evt):
    # type: (EndpointConfig, Text, Dict[Text, Any]) -> Dict[Text, Any]
    """Log an event to a conversation."""

    subpath = "/conversations/{}/tracker/events".format(sender_id)

    r = endpoint.request(json=evt,
                         method="post",
                         subpath=subpath)

    return _response_as_json(r)


def replace_events(endpoint, sender_id, evts):
    # type: (EndpointConfig, Text, List[Dict[Text, Any]]) -> Dict[Text, Any]
    """Replace all the events of a conversation with the provided ones."""

    subpath = "/conversations/{}/tracker/events".format(sender_id)

    r = endpoint.request(json=evts,
                         method="put",
                         subpath=subpath)

    return _response_as_json(r)


def send_finetune(endpoint, evts):
    # type: (EndpointConfig, List[Dict[Text, Any]]) -> Dict[Text, Any]
    """Finetune a core model on the provided additional training samples."""

    r = endpoint.request(json=evts,
                         method="post",
                         subpath="/finetune")

    return _response_as_json(r)


def format_bot_output(message):
    # type: (Dict[Text, Any]) -> Text
    """Format a bot response to be displayed in the history table."""

    if "text" in message:
        output = message.get("text")
    else:
        output = ""

    # Append all additional items
    data = message.get("data", {})
    if data.get("image"):
        output += "\nImage: " + data.get("image")

    if data.get("attachment"):
        output += "\nAttachment: " + data.get("attachment")

    if data.get("buttons"):
        for idx, button in enumerate(data.get("buttons")):
            button_str = button_to_string(button, idx)
            output += "\n" + button_str
    return output


def latest_user_message(evts):
    # type: (List[Dict[Text, Any]]) -> Optional[Dict[Text, Any]]
    """Return most recent user message."""

    for i, e in enumerate(reversed(evts)):
        if e.get("event") == "user":
            return e
    return None


def all_events_before_latest_user_msg(evts):
    # type: (List[Dict[Text, Any]]) -> List[Dict[Text, Any]]
    """Return all events that happened before the most recent user message."""

    for i, e in enumerate(reversed(evts)):
        if e.get("event") == "user":
            return evts[:-(i + 1)]
    return evts


def _ask_questions(
        questions,  # type: List[Dict[Text, Any]]
        sender_id,  # type: Text
        endpoint,  # type: EndpointConfig
        is_abort=None  # type: Optional[Callable[[Dict[Text, Text]], bool]]
):
    # type: (...) -> Dict[Text, Any]
    """Ask the user a question, if Ctrl-C is pressed provide user with menu."""

    should_retry = True
    answers = {}

    while should_retry:
        answers = prompt(questions)
        if not answers or (is_abort and is_abort(answers)):
            should_retry = _ask_if_quit(sender_id, endpoint)
        else:
            should_retry = False
    return answers


def _selection_choices_from_intent_prediction(predictions):
    # type: (List[Dict[Text, Any]]) -> List[Dict[Text, Text]]
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


def _request_free_text_intent(sender_id, endpoint):
    # type: (Text, EndpointConfig) -> Text
    questions = [
        {
            "type": "input",
            "name": "intent",
            "message": "Please type the intent name",
        }
    ]
    answers = _ask_questions(questions, sender_id, endpoint)
    return answers["intent"]


def _request_selection_from_intent_list(intent_list, sender_id, endpoint):
    # type: (List[Dict[Text, Text]], Text, EndpointConfig) -> Text
    questions = [
        {
            "type": "list",
            "name": "intent",
            "message": "What intent is it?",
            "choices": intent_list
        }
    ]
    return _ask_questions(questions, sender_id, endpoint)["intent"]


def _request_intent_from_user(latest_message,
                              intents,
                              sender_id,
                              endpoint
                              ):
    # type: (...) -> Dict[Text, Any]
    """Take in latest message and ask which intent it should have been.

    Returns the intent dict that has been selected by the user."""

    predictions = latest_message.get("parse_data", {}).get("intent_ranking", [])

    predicted_intents = {p["name"] for p in predictions}

    for i in intents:
        if i not in predicted_intents:
            predictions.append({"name": i, "confidence": 0.0})

    # convert intents to ui list and add <other> as a free text alternative
    choices = (_selection_choices_from_intent_prediction(predictions) +
               [{"name": "     <other>", "value": OTHER_INTENT}])

    intent_name = _request_selection_from_intent_list(choices,
                                                      sender_id,
                                                      endpoint)

    if intent_name == OTHER_INTENT:
        intent_name = _request_free_text_intent(sender_id, endpoint)

    # returns the selected intent with the original probability value
    return next((x for x in predictions if x["name"] == intent_name), None)


def _print_history(sender_id, endpoint):
    # type: (Text, EndpointConfig) -> None
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


def _chat_history_table(evts):
    # type: (List[Dict[Text, Any]]) -> Text
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

    def bot_width(_table):
        # type: (AsciiTable) -> int
        return _table.column_max_width(1)

    def user_width(_table):
        # type: (AsciiTable) -> int
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
        if evt.get("event") == "action":
            bot_column.append(colored(evt['name'], 'autocyan'))
            if evt['confidence'] is not None:
                bot_column[-1] += (colored(" {:03.2f}".format(evt['confidence']), 'autowhite'))

        elif evt.get("event") == 'user':
            if bot_column:
                text = "\n".join(bot_column)
                add_bot_cell(table_data, text)
                bot_column = []

            msg = format_user_msg(evt, user_width(table))
            add_user_cell(table_data, msg)

        elif evt.get("event") == "bot":
            wrapped = wrap(format_bot_output(evt), bot_width(table))
            bot_column.append(colored(wrapped, 'autoblue'))

        elif evt.get("event") != "bot":
            e = Event.from_parameters(evt)
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


def _slot_history(tracker_dump):
    # type: (Dict[Text, Any]) -> List[Text]
    """Create an array of slot representations to be displayed."""

    slot_strs = []
    for k, s in tracker_dump.get("slots").items():
        colored_value = utils.wrap_with_color(str(s),
                                              utils.bcolors.WARNING)
        slot_strs.append("{}: {}".format(k, colored_value))
    return slot_strs


def _ask_if_quit(sender_id, endpoint):
    # type: (Text, EndpointConfig) -> bool
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
        story_path, nlu_path = _request_export_info()

        tracker = retrieve_tracker(endpoint, sender_id)
        evts = tracker.get("events", [])

        _write_stories_to_file(story_path, evts)
        _write_nlu_to_file(nlu_path, evts)

        logger.info("Successfully wrote stories and NLU data")
        sys.exit()
    elif answers["abort"] == "continue":
        # in this case we will just return, and the original
        # question will get asked again
        return True
    elif answers["abort"] == "undo":
        raise UndoLastStep()
    elif answers["abort"] == "restart":
        raise RestartConversation()


def _request_action_from_user(predictions, sender_id, endpoint):
    # type: (List[Dict[Text, Any]],Text, EndpointConfig) -> Text
    """Ask the user to correct an action prediction."""

    _print_history(sender_id, endpoint)

    sorted_actions = sorted(predictions,
                            key=lambda k: (-k['score'], k['action']))

    choices = [{"name": "{:03.2f} {:40}".format(a.get("score"),
                                                a.get("action")),
                "value": a.get("action")}
               for a in sorted_actions]

    questions = [{
        "name": "action",
        "type": "list",
        "message": "What is the next action of the bot?",
        "choices": choices
    }]
    answers = _ask_questions(questions, sender_id, endpoint)
    action_name = answers["action"]
    print("Thanks! The bot will now run {}.\n".format(action_name))
    return action_name


def _request_export_info():
    # type: () -> (Text, Text)
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
    }, {"name": "export nlu",
        "type": "input",
        "message": "Export NLU data to (if file exists, this "
                   "will merge learned data with previous training examples)",
        "default": PATHS["nlu"],
        "validate": validate_path}]

    answers = prompt(questions)
    if not answers:
        sys.exit()

    return answers["export stories"], answers["export nlu"]


def _split_conversation_at_restarts(evts):
    # type: (List[Dict[Text, Any]]) -> List[List[Dict[Text, Any]]]
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


def _collect_messages(evts):
    # type: (List[Dict[Text, Any]]) -> List[Dict[Text, Any]]
    """Collect the message text and parsed data from the UserMessage events into a list"""

    msgs = []

    for evt in evts:
        if evt.get("event") == "user":
            data = evt.get("parse_data")
            msg = Message.build(data["text"], data["intent"]["name"], data["entities"])
            msgs.append(msg)

    return msgs


def _write_stories_to_file(export_story_path, evts):
    # type: (Text, List[Dict[Text, Any]]) -> None
    """Write the conversation of the sender_id to the file paths."""

    sub_conversations = _split_conversation_at_restarts(evts)

    with io.open(export_story_path, 'a', encoding="utf-8") as f:
        for conversation in sub_conversations:
            parsed_events = events.deserialise_events(conversation)
            s = Story.from_events(parsed_events)
            f.write(s.as_story_string(flat=True) + "\n")


def _write_nlu_to_file(export_nlu_path, evts):
    # type: (Text, List[Dict[Text, Any]]) -> None
    """Write the nlu data of the sender_id to the file paths."""

    msgs = _collect_messages(evts)

    try:
        previous_examples = load_data(export_nlu_path)

    except:
        questions = [{"name": "export nlu",
                     "type": "input",
                     "message": "Could not load existing NLU data, please specify where to store NLU data "
                                "learned in this session (this will overwrite any existing file)",
                     "default": PATHS["backup"]}]

        answers = prompt(questions)
        export_nlu_path = answers["export nlu"]
        previous_examples = TrainingData()

    nlu_data = previous_examples.merge(TrainingData(msgs))

    with io.open(export_nlu_path, 'w', encoding="utf-8") as f:
        if _guess_format(export_nlu_path) in ["md", "unk"]:
            f.write(nlu_data.as_markdown())
        else:
            f.write(nlu_data.as_json())


def _predict_till_next_listen(endpoint,  # type: EndpointConfig
                              sender_id,  # type: Text
                              finetune  # type: bool
                              ):
    # type: (...) -> None
    """Predict and validate actions until we need to wait for a user msg."""

    listen = False
    while not listen:
        response = request_prediction(endpoint, sender_id)
        predictions = response.get("scores")

        probabilities = [prediction["score"] for prediction in predictions]
        pred_out = int(np.argmax(probabilities))

        action_name = predictions[pred_out].get("action")

        _print_history(sender_id, endpoint)
        listen = _validate_action(action_name, predictions,
                                  endpoint, sender_id, finetune=finetune)


def _correct_wrong_nlu(corrected_nlu,  # type: Dict[Text, Any]
                       evts,  # type: List[Dict[Text, Any]]
                       endpoint,  # type: EndpointConfig
                       sender_id  # type: Text
                       ):
    # type: (...) -> None
    """A wrong NLU prediction got corrected, update core's tracker."""

    latest_message = latest_user_message(evts)
    corrected_events = all_events_before_latest_user_msg(evts)

    latest_message["parse_data"] = corrected_nlu

    replace_events(endpoint, sender_id, corrected_events)

    send_message(endpoint, sender_id, latest_message.get("text"),
                 latest_message.get("parse_data"))


def _correct_wrong_action(corrected_action,  # type: Text
                          endpoint,  # type: EndpointConfig
                          sender_id,  # type: Text
                          finetune=False  # type: bool
                          ):
    # type: (...) -> None
    """A wrong action prediction got corrected, update core's tracker."""

    response = send_action(endpoint,
                           sender_id,
                           corrected_action)

    if finetune:
        send_finetune(endpoint,
                      response.get("tracker", {}).get("events", []))


def _validate_action(action_name,  # type: Text
                     predictions,  # type: List[Dict[Text, Any]]
                     endpoint,  # type: EndpointConfig
                     sender_id,  # type: Text
                     finetune=False  # type: bool
                     ):
    # type: (...) -> bool
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
        corrected_action = _request_action_from_user(predictions, sender_id,
                                                     endpoint)
        _correct_wrong_action(corrected_action, endpoint, sender_id,
                              finetune=finetune)
        return corrected_action == ACTION_LISTEN_NAME
    else:
        send_action(endpoint, sender_id, action_name)
        return action_name == ACTION_LISTEN_NAME


def _as_md_message(parse_data):
    # type: (Dict[Text, Any]) -> Text
    """Display the parse data of a message in markdown format."""

    if parse_data.get("text", "").startswith(INTENT_MESSAGE_PREFIX):
        return parse_data.get("text")

    if not parse_data.get("entities"):
        parse_data["entities"] = []
    # noinspection PyProtectedMember
    return MarkdownWriter()._generate_message_md(parse_data)


def _validate_user_regex(latest_message, intents):
    # type: (Dict[Text, Any], List[Text]) -> bool
    """Validate if a users message input is correct.

    This assumes the user entered an intent directly, e.g. using
    `/greet`. Return `True` if the intent is a known one."""

    parse_data = latest_message.get("parse_data", {})
    intent = parse_data.get("intent", {}).get("name")

    if intent in intents:
        return True
    else:
        return False


def _validate_user_text(latest_message, endpoint, sender_id):
    # type: (Dict[Text, Any], EndpointConfig, Text) -> bool
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


def _validate_nlu(intents, endpoint, sender_id):
    # type: (List[Text], EndpointConfig, Text) -> None
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


def _correct_entities(latest_message, endpoint, sender_id):
    # type: (Dict[Text, Any], EndpointConfig, Text) -> Dict[Text, Any]
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


def _enter_user_message(sender_id, endpoint, exit_text):
    # type: (Text, EndpointConfig, Text) -> None
    """Request a new message from the user."""

    questions = [{
        "name": "message",
        "type": "input",
        "message": "Next user input:"
    }]

    answers = _ask_questions(
        questions, sender_id, endpoint,
        is_abort=lambda a: a["message"] == exit_text)

    send_message(endpoint, sender_id, answers["message"])


def is_listening_for_message(sender_id, endpoint):
    # type: (Text, EndpointConfig) -> bool
    """Check if the conversation is in need for a user message."""

    tracker = retrieve_tracker(endpoint, sender_id, EventVerbosity.APPLIED)

    for i, e in enumerate(reversed(tracker.get("events", []))):
        if e.get("event") == "user":
            return False
        elif e.get("event") == "action":
            return e.get("name") == ACTION_LISTEN_NAME
    return False


def _undo_latest(sender_id, endpoint):
    # type: (Text, EndpointConfig) -> None
    """Undo either the latest bot action or user message, whatever is last."""

    tracker = retrieve_tracker(endpoint, sender_id, EventVerbosity.ALL)

    cutoff_index = None
    for i, e in enumerate(reversed(tracker.get("events", []))):
        if e.get("event") in {"user", "action"}:
            cutoff_index = i
            break
        elif e.get("event") == "restart":
            break

    if cutoff_index is not None:
        events_to_keep = tracker["events"][:-(cutoff_index + 1)]

        # reset the events of the conversation to the events before
        # the most recent bot or user event
        replace_events(endpoint, sender_id, events_to_keep)


def record_messages(endpoint,  # type: EndpointConfig
                    sender_id=UserMessage.DEFAULT_SENDER_ID,  # type: Text
                    max_message_limit=None,  # type: Optional[int]
                    on_finish=None,  # type: Optional[Callable[[], None]]
                    finetune=False  # type: bool
                    ):
    """Read messages from the command line and print bot responses."""

    try:
        exit_text = INTENT_MESSAGE_PREFIX + 'stop'

        utils.print_color("Bot loaded. Type a message and press enter "
                          "(use '{}' to exit). ".format(exit_text),
                          utils.bcolors.OKGREEN)

        try:
            domain = retrieve_domain(endpoint)
        except requests.exceptions.ConnectionError:
            logger.exception("Failed to connect to rasa core server at '{}'. "
                             "Is the server running?".format(endpoint.url))
            return

        intents = [next(iter(i)) for i in (domain.get("intents") or [])]

        num_messages = 0
        while not utils.is_limit_reached(num_messages, max_message_limit):
            try:
                if is_listening_for_message(sender_id, endpoint):
                    _enter_user_message(sender_id, endpoint, exit_text)
                    _validate_nlu(intents, endpoint, sender_id)
                _predict_till_next_listen(endpoint, sender_id,
                                          finetune=finetune)
                num_messages += 1
            except RestartConversation:
                send_event(endpoint, sender_id, {"event": "restart"})
                send_event(endpoint, sender_id, {"event": "action",
                                                 "name": ACTION_LISTEN_NAME})

                logger.info("Restarted conversation, starting a new one.")
            except UndoLastStep:
                _undo_latest(sender_id, endpoint)
                _print_history(sender_id, endpoint)

    except Exception:
        logger.exception("An exception occurred while recording messages.")
        raise
    finally:
        if on_finish:
            on_finish()


def _start_interactive_learning_io(endpoint, on_finish, finetune=False):
    # type: (EndpointConfig, Callable[[], None], bool) -> None
    """Start the interactive learning message recording in a separate thread."""

    p = Thread(target=record_messages,
               kwargs={
                   "endpoint": endpoint,
                   "on_finish": on_finish,
                   "finetune": finetune})
    p.setDaemon(True)
    p.start()


def _serve_application(app, finetune=False, serve_forever=True):
    # type: (Flask, bool, bool) -> WSGIServer
    """Start a core server and attach the interactive learning IO."""

    http_server = WSGIServer(('0.0.0.0', DEFAULT_SERVER_PORT), app)
    logger.info("Rasa Core server is up and running on "
                "{}".format(DEFAULT_SERVER_URL))
    http_server.start()

    endpoint = EndpointConfig(url=DEFAULT_SERVER_URL)
    _start_interactive_learning_io(endpoint, http_server.stop, finetune=finetune)

    if serve_forever:
        try:
            http_server.serve_forever()
        except Exception as exc:
            logger.exception(exc)

    return http_server


def run_interactive_learning(agent, finetune=False, serve_forever=True):
    # type: (Agent, bool, bool) -> WSGIServer
    """Start the interactive learning with the model of the agent."""

    app = server.create_app(agent)

    return _serve_application(app, finetune, serve_forever)

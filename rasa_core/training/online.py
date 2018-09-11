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
from gevent.pywsgi import WSGIServer
from terminaltables import SingleTable
from threading import Thread
from typing import Any, Text, Dict, List, Optional, Tuple

from rasa_core import utils, server, events
from rasa_core.actions.action import ACTION_LISTEN_NAME
from rasa_core.channels import UserMessage
from rasa_core.channels.channel import button_to_string
from rasa_core.constants import DEFAULT_SERVER_PORT, DEFAULT_SERVER_URL
from rasa_core.events import Event
from rasa_core.interpreter import INTENT_MESSAGE_PREFIX
from rasa_core.trackers import EventVerbosity
from rasa_core.training.structures import Story
from rasa_core.utils import EndpointConfig
from rasa_nlu.training_data.formats import MarkdownWriter, MarkdownReader

logger = logging.getLogger(__name__)

MAX_VISUAL_HISTORY = 3

DEFAULT_FILE_EXPORT_PATH = "stories.md"

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


def send_message(endpoint,  # type: EndpointConfig
                 sender_id,  # type: Text
                 message,  # type: Text
                 parse_data=None  # type: Optional[Dict[Text, Any]]
                 ):
    # type: (...) -> Dict[Text, Any]

    payload = {
        "sender": "user",
        "text": message,
        "parse_data": parse_data
    }

    r = endpoint.request(json=payload,
                         method="post",
                         subpath="/conversations/{}/messages".format(sender_id))

    r.raise_for_status()

    if r.encoding is None:
        r.encoding = 'utf-8'

    return r.json()


def request_prediction(endpoint, sender_id):
    # type: (EndpointConfig, Text) -> Dict[Text, Any]

    r = endpoint.request(method="post",
                         subpath="/conversations/{}/predict".format(sender_id))

    r.raise_for_status()

    if r.encoding is None:
        r.encoding = 'utf-8'

    return r.json()


def retrieve_domain(endpoint):
    r = endpoint.request(method="get",
                         subpath="/domain",
                         headers={"Accept": "application/json"})

    r.raise_for_status()

    if r.encoding is None:
        r.encoding = 'utf-8'

    return r.json()


def retrieve_tracker(endpoint, sender_id, verbosity=EventVerbosity.ALL):
    path = "/conversations/{}/tracker?events={}".format(
            sender_id, verbosity.name)
    r = endpoint.request(method="get",
                         subpath=path,
                         headers={"Accept": "application/json"})

    r.raise_for_status()

    if r.encoding is None:
        r.encoding = 'utf-8'

    return r.json()


def send_action(endpoint, sender_id, action_name):
    # type: (EndpointConfig, Text, Text) -> Dict[Text, Any]
    payload = {"action": action_name}
    subpath = "/conversations/{}/execute".format(sender_id)

    r = endpoint.request(json=payload,
                         method="post",
                         subpath=subpath)

    r.raise_for_status()

    if r.encoding is None:
        r.encoding = 'utf-8'

    return r.json()


def send_event(endpoint, sender_id, evt):
    # type: (EndpointConfig, Text, Dict[Text, Any]) -> Dict[Text, Any]
    subpath = "/conversations/{}/tracker/events".format(sender_id)

    r = endpoint.request(json=evt,
                         method="post",
                         subpath=subpath)

    r.raise_for_status()

    if r.encoding is None:
        r.encoding = 'utf-8'

    return r.json()


def replace_events(endpoint, sender_id, evts):
    # type: (EndpointConfig, Text, List[Dict[Text, Any]]) -> Dict[Text, Any]
    subpath = "/conversations/{}/tracker/events".format(sender_id)

    r = endpoint.request(json=evts,
                         method="put",
                         subpath=subpath)

    r.raise_for_status()

    if r.encoding is None:
        r.encoding = 'utf-8'

    return r.json()


def send_finetune(endpoint, evts):
    # type: (EndpointConfig, List[Dict[Text, Any]]) -> None

    r = endpoint.request(json=evts,
                         method="post",
                         subpath="/finetune")

    r.raise_for_status()

    if r.encoding is None:
        r.encoding = 'utf-8'


def revert_latest_message(
        evts  # type: List[Dict[Text, Any]]
):
    # type: (...) -> Tuple[Optional[Dict[Text, Any]], List[Dict[Text, Any]]]

    for i, e in enumerate(reversed(evts)):
        if e.get("event") == "user":
            return e, evts[:-(i + 1)]
    return None, evts


def format_bot_output(message):
    if "text" in message:
        output = message.get("text")
    else:
        output = ""

    # Append all additional items
    data = message.get("data")
    if data.get("image"):
        output += "\nImage: " + data.get("image")

    if data.get("attachment"):
        output += "\nAttachment: " + data.get("attachment")

    if data.get("buttons"):
        for idx, button in enumerate(data.get("buttons")):
            button_str = button_to_string(button, idx)
            output += "\n" + button_str
    return output


def _ask_questions(questions, sender_id, endpoint, is_abort=None):
    should_retry = True
    answers = {}

    while should_retry:
        answers = prompt(questions)
        if not answers or (is_abort and is_abort(answers)):
            should_retry = _ask_if_quit(sender_id, endpoint)
        else:
            should_retry = False
    return answers


def _request_intent_from_user(latest_message, intents, sender_id, endpoint):
    # take in some argument and ask which intent it should have been

    predictions = latest_message.get("parse_data", {}).get("intent_ranking", [])

    predicted_intents = {p["name"] for p in predictions}

    for i in intents:
        if i not in predicted_intents:
            predictions.append({"name": i, "confidence": 0})

    sorted_intents = sorted(predictions,
                            key=lambda k: (-k['confidence'], k['name']))

    choices = [
        {"name": "{:03.2f} {:40}".format(p.get("confidence"),
                                         p.get("name")),
         "value": p.get("name")}
        for p in sorted_intents]

    other = {"name": "     <other>", "value": OTHER_INTENT}
    questions = [
        {
            "type": "list",
            "name": "intent",
            "message": "What intent is it?",
            "choices": choices + [other]
        }
    ]
    answers = _ask_questions(questions, sender_id, endpoint)

    if answers["intent"] == OTHER_INTENT:
        questions = [
            {
                "type": "input",
                "name": "intent",
                "message": "Please type the intent name",
            }
        ]
        answers = _ask_questions(questions, sender_id, endpoint)
        intent_name = answers["intent"]
    else:
        intent_name = answers["intent"]

    return {'name': intent_name, 'confidence': 1.0}


def _print_history(sender_id, endpoint):
    def wrap(text, max_width):
        return "\n".join(textwrap.wrap(text, max_width,
                                       replace_whitespace=False))

    tracker_dump = retrieve_tracker(endpoint, sender_id,
                                    EventVerbosity.AFTER_RESTART)
    # prints the historical interactions between the bot and the user,
    # to help with correctly identifying the action
    tr_json = []
    n_history = 0
    for tr in reversed(tracker_dump.get("events", [])):
        tr_json.append(tr)
        if tr.get("event") == "action":
            n_history += 1

    tr_json = reversed(tr_json)

    table_data = [
        ["#  ",
         Color('{autoblue}Bot{/autoblue}      '),
         "  ",
         Color('{hired}You{/hired}       ')],
    ]

    table = SingleTable(table_data, 'Chat History')

    bot_column = ""
    for idx, evt in enumerate(tr_json):
        if evt.get("event") == "action":
            if bot_column:
                bot_column += "\n"
            bot_column += "{autocyan}" + evt['name'] + "{/autocyan}"
        elif evt.get("event") == 'user':
            if bot_column:
                table_data.append([len(table_data), Color(bot_column), "", ""])
                bot_column = ""
            parsed = evt.get('parse_data', {})
            intent = parsed.get('intent', {}).get("name")
            md = _md_message(parsed)
            msg = "{hired}" + wrap(md, table.column_max_width(3)) + "{/hired}\n"
            msg += ("intent: " + intent + "\n")
            table_data.append([len(table_data), "", "", Color(msg)])
        elif evt.get("event") == "bot":
            if bot_column:
                bot_column += "\n"
            wrapped = wrap(format_bot_output(evt), table.column_max_width(1))
            bot_column += ("{autoblue}" + wrapped + "{/autoblue}")
        elif evt.get("event") != "bot":
            e = Event.from_parameters(evt)
            if bot_column:
                bot_column += "\n"
            bot_column += wrap(e.as_story_string(), table.column_max_width(1))

    if bot_column:
        table_data.append([len(table_data), Color(bot_column), "", ""])

    table.inner_heading_row_border = False
    table.inner_row_border = True
    table.inner_column_border = False
    table.outer_border = False
    table.justify_columns = {0: 'left', 1: 'left', 2: 'center', 3: 'right'}

    print("------")
    print("Chat History\n")
    print(table.table)

    slot_strs = []
    for k, s in tracker_dump.get("slots").items():
        colored_value = utils.wrap_with_color(str(s),
                                              utils.bcolors.WARNING)
        slot_strs.append("{}: {}".format(k, colored_value))
    if slot_strs:
        print("Current slots: {}\n".format(", ".join(slot_strs)))

    print("------")


def _ask_if_quit(sender_id, endpoint):
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
        _export_stories(sender_id, endpoint)
        sys.exit()
    elif answers["abort"] == "continue":
        return True
    elif answers["abort"] == "restart":
        raise RestartConversation()
    elif answers["abort"] == "undo":
        raise UndoLastStep()
    else:
        logger.warning("Invalid selection. Answer: {}".format(answers))


def _request_action_from_user(predictions, sender_id, endpoint):
    # given the intent and the text
    # what is the correct action?
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


def _export_stories(sender_id, endpoint):
    def validate_path(path):
        try:
            with io.open(path, "a"):
                return True
        except Exception as e:
            return "Failed to open file. {}".format(e)

    # export current stories and quit
    questions = [{
        "name": "export",
        "type": "input",
        "message": "Export stories to (if file exists, this "
                   "will append the stories)",
        "default": DEFAULT_FILE_EXPORT_PATH,
        "validate": validate_path
    }]
    answers = prompt(questions)
    if not answers:
        # TODO: write to temp file, just to be save
        sys.exit()

    export_file_path = answers["export"]
    _write_stories_to_file(export_file_path, sender_id, endpoint)
    logger.info("Successfully wrote stories to {}.".format(export_file_path))


def _write_stories_to_file(export_file_path, sender_id, endpoint):
    tracker = retrieve_tracker(endpoint, sender_id)
    evts = tracker.get("events", [])
    split_events = []
    current = []
    for e in evts:
        if e.get("event") == "restart":
            if current:
                split_events.append(current)
            current = []
        else:
            current.append(e)

    if current:
        split_events.append(current)

    with io.open(export_file_path, 'a') as f:
        for dialogue_events in split_events:
            parsed_events = events.deserialise_events(dialogue_events)
            s = Story.from_events(parsed_events)
            f.write(s.as_story_string(flat=True) + "\n")


def _predict_till_next_listen(endpoint,  # type: EndpointConfig
                              sender_id,  # type: Text
                              finetune  # type: Boolean
                              ):
    # type: (...) -> None
    # given a state, predict next action via asking a human

    listen = False
    while not listen:
        response = request_prediction(endpoint, sender_id)
        predictions = response.get("scores")

        probabilities = [prediction["score"] for prediction in predictions]
        pred_out = int(np.argmax(probabilities))

        action_name = predictions[pred_out].get("action")

        listen = _validate_action(action_name, predictions,
                                  endpoint, sender_id, finetune=finetune)
        _print_history(sender_id, endpoint)


def _correct_wrong_nlu(corrected_nlu, evts, endpoint, sender_id):
    latest_message, corrected_events = revert_latest_message(evts)

    latest_message["parse_data"] = corrected_nlu

    replace_events(endpoint, sender_id, corrected_events)

    send_message(endpoint, sender_id, latest_message.get("text"),
                 latest_message.get("parse_data"))


def _correct_wrong_action(corrected_action, endpoint, sender_id, finetune=False):
    response = send_action(endpoint,
                           sender_id,
                           corrected_action)

    if finetune:
        send_finetune(endpoint,
                      response.get("tracker", {}).get("events", []))


def _validate_action(action_name, predictions, endpoint, sender_id, finetune=False):
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
        _correct_wrong_action(corrected_action, endpoint, sender_id, finetune=finetune)
        return corrected_action == ACTION_LISTEN_NAME
    else:
        send_action(endpoint, sender_id, action_name)
        return action_name == ACTION_LISTEN_NAME


def _md_message(parse_data):
    if parse_data.get("text", "").startswith(INTENT_MESSAGE_PREFIX):
        return parse_data.get("text")

    if not parse_data.get("entities"):
        parse_data["entities"] = []
    # noinspection PyProtectedMember
    return MarkdownWriter()._generate_message_md(parse_data)


def _validate_user_regex(latest_message, intents):
    parse_data = latest_message.get("parse_data", {})
    intent = parse_data.get("intent", {}).get("name")

    if intent in intents:
        return True
    else:
        return False


def _validate_user_text(latest_message, endpoint, sender_id):
    parse_data = latest_message.get("parse_data", {})
    entities = _md_message(parse_data)
    intent = parse_data.get("intent", {}).get("name")
    q = "Is the NLU classification for '{}' with intent '{}' correct?".format(
            entities, intent)
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
    tracker = retrieve_tracker(endpoint, sender_id,
                               EventVerbosity.AFTER_RESTART)

    latest_message, _ = revert_latest_message(tracker.get("events", []))

    if latest_message.get("text").startswith(INTENT_MESSAGE_PREFIX):
        valid = _validate_user_regex(latest_message, intents)
    else:
        valid = _validate_user_text(latest_message, endpoint, sender_id)

    if not valid:
        corrected_intent = _request_intent_from_user(latest_message, intents,
                                                     sender_id, endpoint)
        evts = tracker.get("events", [])

        entities = _validate_entities(latest_message, endpoint, sender_id)
        corrected_nlu = {
            "intent": corrected_intent,
            "entities": entities,
            "text": latest_message.get("text")
        }

        _correct_wrong_nlu(corrected_nlu, evts, endpoint, sender_id)


def _validate_entities(latest_message, endpoint, sender_id):
    q = "Please mark the entities using [value](type) notation"
    entity_str = _md_message(latest_message.get("parse_data", {}))
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
    questions = [{
        "name": "message",
        "type": "input",
        "message": "Next user input:"
    }]

    answers = _ask_questions(
            questions, sender_id, endpoint,
            is_abort=lambda a: a["message"] == exit_text)

    tracker = send_message(endpoint, sender_id, answers["message"])
    return tracker


def is_listening_for_message(sender_id, endpoint):
    tracker = retrieve_tracker(endpoint, sender_id, EventVerbosity.APPLIED)

    for i, e in enumerate(reversed(tracker.get("events", []))):
        if e.get("event") == "user":
            return False
        elif e.get("event") == "action":
            return e.get("name") == ACTION_LISTEN_NAME
    return False


def revert_tracker(sender_id, endpoint):
    # type: (...) -> None

    tracker = retrieve_tracker(endpoint, sender_id, EventVerbosity.ALL)

    cutoff_index = None
    last_event_type = None
    for i, e in enumerate(reversed(tracker.get("events", []))):
        if e.get("event") in {"user", "action"}:
            cutoff_index = i
            last_event_type = e.get("event")
            break
        elif e.get("event") == "restart":
            break

    if cutoff_index is not None:
        events_to_keep = tracker["events"][:-(cutoff_index + 1)]
        replace_events(endpoint, sender_id, events_to_keep)

        return last_event_type
    else:
        return None


def record_messages(endpoint,
                    sender_id=UserMessage.DEFAULT_SENDER_ID,
                    max_message_limit=None,
                    on_finish=None,
                    finetune=False):
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
                _predict_till_next_listen(endpoint, sender_id, finetune=finetune)
                num_messages += 1
            except RestartConversation:
                send_event(endpoint, sender_id, {"event": "restart"})
                send_event(endpoint, sender_id, {"event": "action",
                                                 "name": ACTION_LISTEN_NAME})

                logger.info("Restarted conversation, starting a new one.")
            except UndoLastStep:
                revert_tracker(sender_id, endpoint)
                _print_history(sender_id, endpoint)

    except Exception:
        logger.exception("An exception occurred while recording messages.")
        raise
    finally:
        if on_finish:
            on_finish()


def start_online_learning_io(endpoint, on_finish, finetune=False):
    p = Thread(target=record_messages,
               kwargs={
                   "endpoint": endpoint,
                   "on_finish": on_finish,
                   "finetune": finetune})
    p.start()


def serve_agent(agent, finetune=False, serve_forever=True):
    app = server.create_app(agent)

    return serve_application(app, finetune, serve_forever)


def serve_application(app, finetune=False, serve_forever=True):
    http_server = WSGIServer(('0.0.0.0', DEFAULT_SERVER_PORT), app)
    logger.info("Rasa Core server is up and running on "
                "{}".format(DEFAULT_SERVER_URL))
    http_server.start()

    endpoint = EndpointConfig(url=DEFAULT_SERVER_URL)
    start_online_learning_io(endpoint, http_server.stop, finetune=finetune)

    if serve_forever:
        try:
            http_server.serve_forever()
        except Exception as exc:
            logger.exception(exc)

    return http_server

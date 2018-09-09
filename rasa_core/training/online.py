from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
import logging
import numpy as np
import requests
from gevent.pywsgi import WSGIServer
from threading import Thread
from typing import Any, Text, Dict, List, Optional, Tuple

from rasa_core import utils, server, events
from rasa_core.actions.action import ACTION_LISTEN_NAME
from rasa_core.channels import UserMessage, console
from rasa_core.constants import DEFAULT_SERVER_PORT, DEFAULT_SERVER_URL
from rasa_core.interpreter import INTENT_MESSAGE_PREFIX
from rasa_core.training.structures import Story
from rasa_core.utils import EndpointConfig

logger = logging.getLogger(__name__)

MAX_VISUAL_HISTORY = 3

DEFAULT_FILE_EXPORT_PATH = "stories.md"


def _request_intent_from_user(tracker_dump, intents):
    # take in some argument and ask which intent it should have been
    # save the intent to a json like file
    latest_message, _ = revert_latest_message(tracker_dump.get("events", []))
    colored_user_msg = utils.wrap_with_color(latest_message.get("text"),
                                             utils.bcolors.OKGREEN)
    print("------\n")
    print("Message:\n")
    print(latest_message.get("text"))
    print("User said:\t {}".format(colored_user_msg))
    print("What intent is this?\t")
    for idx, intent in enumerate(intents):
        print('\t{}\t{}'.format(idx, intent))

    out = int(utils.request_input(
            utils.str_range_list(0, len(intents))))

    return {'name': intents[out], 'confidence': 1.0}


def _print_history(tracker_dump):
    # prints the historical interactions between the bot and the user,
    # to help with correctly identifying the action
    latest_listen_flag = False
    tr_json = []
    n_history = 0
    for tr in reversed(tracker_dump.get("events", [])):
        tr_json.append(tr)
        if tr.get("event") == "action":
            n_history += 1

    tr_json = reversed(tr_json)

    print("------")
    print("Chat history:\n")

    for idx, evt in enumerate(tr_json):
        if evt.get("event") == "action":
            print("\tbot did:\t{}\n".format(evt['name']))
            latest_listen_flag = False
        if evt.get("event") == 'user':
            parsed = evt.get('parse_data', {})
            print("\tuser said:\t{}\n".format(
                    utils.wrap_with_color(evt.get("text"),
                                          utils.bcolors.OKGREEN)))
            print("\t\t whose intent is:\t{}\n".format(
                    parsed.get('intent')))
            for entity in parsed.get('entities'):
                print("\twith {}:\t{}\n".format(entity['entity'],
                                                entity['value']))
            latest_listen_flag = True
    slot_strs = []
    for k, s in tracker_dump.get("slots").items():
        colored_value = utils.wrap_with_color(str(s),
                                              utils.bcolors.WARNING)
        slot_strs.append("{}: {}".format(k, colored_value))
    print("we currently have slots: {}\n".format(", ".join(slot_strs)))

    print("------")
    return latest_listen_flag


def _request_action_from_user(predictions, tracker_dump):
    # given the intent and the text
    # what is the correct action?
    _print_history(tracker_dump)
    print("what is the next action for the bot?\n")

    for idx, p in enumerate(predictions):
        print("{:>10}{:>40}    {:03.2f}".format(idx,
                                                p.get("action"),
                                                p.get("score")))

    out = int(utils.request_input(
            utils.str_range_list(0, len(predictions))))
    print("thanks! The bot will now "
          "[{}]\n -----------".format(predictions[out].get("action")))
    return out


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


def send_events(endpoint, sender_id, evts):
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


def _export_stories(tracker):
    # export current stories and quit
    file_prompt = ("File to export to (if file exists, this "
                   "will append the stories) "
                   "[{}]: ").format(DEFAULT_FILE_EXPORT_PATH)
    export_file_path = utils.request_input(prompt=file_prompt)

    if not export_file_path:
        export_file_path = DEFAULT_FILE_EXPORT_PATH

    parsed_events = events.deserialise_events(tracker.get("events", []))

    s = Story.from_events(parsed_events)

    with io.open(export_file_path, 'a') as f:
        f.write(s.as_story_string(flat=True) + "\n")


def predict_till_next_listen(endpoint,  # type: EndpointConfig
                             intents,  # type:  List[Text]
                             sender_id  # type: Text
                             ):
    # type: (...) -> bool
    # given a state, predict next action via asking a human

    listen = False
    while not listen:
        response = request_prediction(endpoint, sender_id)
        tracker_dump = response.get("tracker")
        predictions = response.get("scores")

        probabilities = [prediction["score"] for prediction in predictions]
        pred_out = int(np.argmax(probabilities))
        latest_action_was_listen = _print_history(tracker_dump)

        action_name = predictions[pred_out].get("action")
        colored_name = utils.wrap_with_color(action_name,
                                             utils.bcolors.OKBLUE)
        if latest_action_was_listen:
            print("The bot wants to [{}] due to the intent. "
                  "Is this correct?\n".format(colored_name))

            user_input = utils.request_input(
                    ["1", "2", "3", "0"],
                    "\t1.\tYes\n" +
                    "\t2.\tNo, intent is right but the action is wrong\n" +
                    "\t3.\tThe intent is wrong\n" +
                    "\t0.\tExport current conversations as stories and quit\n")
        else:
            print("The bot wants to [{}]. "
                  "Is this correct?\n".format(colored_name))
            user_input = utils.request_input(
                    ["1", "2", "0"],
                    "\t1.\tYes.\n" +
                    "\t2.\tNo, the action is wrong.\n" +
                    "\t0.\tExport current conversations as stories and quit\n")

        if user_input == "1":
            # max prob prediction was correct
            response = send_action(endpoint, sender_id, action_name)

            for response in response.get("messages", []):
                console.print_bot_output(response)

            listen = action_name == ACTION_LISTEN_NAME

        elif user_input == "2":
            # max prob prediction was false, new action required
            y = _request_action_from_user(predictions, tracker_dump)

            new_action_name = predictions[y].get("action")
            listen = new_action_name == ACTION_LISTEN_NAME

            response = send_action(endpoint,
                                   sender_id,
                                   new_action_name)

            send_finetune(endpoint,
                          response.get("tracker", {}).get("events", []))

            for response in response.get("messages", []):
                console.print_bot_output(response)

        elif user_input == "3":
            # intent wrong and maybe action wrong
            intent = _request_intent_from_user(tracker_dump, intents)
            evts = tracker_dump.get("events", [])
            latest_message, corrected_events = revert_latest_message(evts)

            latest_message.get("parse_data")["intent"] = intent

            send_events(endpoint, sender_id, corrected_events)

            send_message(endpoint, sender_id, latest_message.get("text"),
                         latest_message.get("parse_data"))

        elif user_input == "0":
            _export_stories(response.get("tracker"))
            return True
        else:
            raise Exception(
                    "Incorrect user input received '{}'".format(user_input))
    return False


def record_messages(endpoint,
                    sender_id=UserMessage.DEFAULT_SENDER_ID,
                    max_message_limit=None,
                    on_finish=None,
                    get_next_message=None):
    """Read messages from the command line and print bot responses."""

    try:
        exit_text = INTENT_MESSAGE_PREFIX + 'stop'

        if not get_next_message:
            # default to reading messages from the command line
            get_next_message = console.get_cmd_input

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
        finished = False
        while (not finished and
               not utils.is_limit_reached(num_messages, max_message_limit)):
            print("Next user input:")
            text = get_next_message()
            if text == exit_text:
                break

            send_message(endpoint, sender_id, text)
            finished = predict_till_next_listen(endpoint,
                                                intents,
                                                sender_id)

            num_messages += 1
    except Exception:
        logger.exception("An exception occurred while recording messages.")
        raise
    finally:
        if on_finish:
            on_finish()


def start_online_learning_io(endpoint, on_finish, get_next_message=None):
    p = Thread(target=record_messages,
               kwargs={
                   "endpoint": endpoint,
                   "on_finish": on_finish,
                   "get_next_message": get_next_message})
    p.start()


def serve_agent(agent, serve_forever=True, get_next_message=None):
    app = server.create_app(agent)

    return serve_application(app, serve_forever, get_next_message)


def serve_application(app, serve_forever=True, get_next_message=None):
    http_server = WSGIServer(('0.0.0.0', DEFAULT_SERVER_PORT), app)
    logger.info("Rasa Core server is up and running on "
                "{}".format(DEFAULT_SERVER_URL))
    http_server.start()

    endpoint = EndpointConfig(url=DEFAULT_SERVER_URL)
    start_online_learning_io(endpoint, http_server.stop, get_next_message)

    if serve_forever:
        try:
            http_server.serve_forever()
        except Exception as exc:
            logger.exception(exc)

    return http_server

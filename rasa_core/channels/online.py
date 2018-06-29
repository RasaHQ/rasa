from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from typing import Any, Text, Dict, List

from rasa_core import utils
from rasa_core.actions.action import ACTION_LISTEN_NAME
from rasa_core.channels import UserMessage, console
from rasa_core.events import ActionExecuted, UserUtteranceReverted, UserUttered
from rasa_core.interpreter import INTENT_MESSAGE_PREFIX
from rasa_core.policies.online_trainer import TrainingFinishedException
from rasa_core.utils import EndpointConfig

MAX_VISUAL_HISTORY = 3


def _request_intent(tracker_dump, intents):
    # take in some argument and ask which intent it should have been
    # save the intent to a json like file
    latest_message = extract_latest_message(tracker_dump)
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
    json_example = {
        'text': latest_message.get("text"),
        'intent': intents[out]
    }
    intent_name = intents[out]
    return {'name': intent_name, 'confidence': 1.0}


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
            print("\tbot did:\t{}\n".format(evt['action']))
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


def _request_action(predictions, tracker_dump):
    # given the intent and the text (NOT IMPLEMENTED)
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


def send_message_receive_prediction(endpoint, sender_id, message):
    # type: (EndpointConfig, Text, Text) -> Dict[Text, Any]
    payload = {
        "sender": sender_id,
        "message": message
    }

    r = endpoint.request(payload,
                         method="post",
                         subpath="/predict")

    r.raise_for_status()

    if r.encoding is None:
        r.encoding = 'utf-8'

    return r.json()


def retrieve_domain(endpoint):
    r = endpoint.request(method="get",
                         subpath="domain")

    r.raise_for_status()

    if r.encoding is None:
        r.encoding = 'utf-8'

    return r.json()


def send_action(endpoint, sender_id, action_name):
    # type: (EndpointConfig, Text, Text) -> Dict[Text, Any]
    payload = {"action": action_name}
    subpath = "/conversations/{}/execute".format(sender_id)

    r = endpoint.request(payload,
                         method="post",
                         subpath=subpath)

    r.raise_for_status()

    if r.encoding is None:
        r.encoding = 'utf-8'

    return r.json()


def send_events(endpoint, sender_id, events):
    # type: (EndpointConfig, Text, List[Event]) -> Dict[Text, Any]
    payload = [e.as_dict() for e in events]
    subpath = "/conversations/{}/tracker/events".format(sender_id)

    r = endpoint.request(payload,
                         method="post",
                         subpath=subpath)

    r.raise_for_status()

    if r.encoding is None:
        r.encoding = 'utf-8'

    return r.json()


def send_fit_example(endpoint, tracker):
    # TODO: TB - implement
    pass


def extract_latest_message(tracker_dump):
    for e in reversed(tracker_dump.get("events", [])):
        if e.get("event") == "user":
            return e
    return None


def validate_prediction(endpoint,  # type: EndpointConfig
                        tracker_dump,  # type: Dict[Text, Any]
                        predictions,  # type: List[Dict[Text, Any]]
                        intents  # type:  List[Text]
                        ):
    # type: (...) -> None
    # given a state, predict next action via asking a human

    sender_id = tracker_dump.get("sender_id")
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

        if action_name == ACTION_LISTEN_NAME:
            print("Next user input:")

    elif user_input == "2":
        # max prob prediction was false, new action required
        # action wrong
        y = _request_action(predictions, tracker_dump)

        # update tracker with new action
        new_action_name = predictions[y].get("action")

        # need to copy tracker, because the tracker will be
        # updated with the new event somewhere else
        response = send_action(endpoint,
                               sender_id,
                               new_action_name)

        for response in response.get("messages", []):
            console.print_bot_output(response)

        send_fit_example(endpoint, training_tracker)

    elif user_input == "3":
        # intent wrong and maybe action wrong
        intent = _request_intent(tracker_dump, intents)
        latest_message = extract_latest_message(tracker_dump)
        latest_message.get("parse_data")["intent"] = intent

        send_events(endpoint,
                    sender_id,
                    [UserUtteranceReverted(),
                     UserUttered.from_parameters(latest_message)])

    elif user_input == "0":
        raise TrainingFinishedException()

    else:
        raise Exception(
                "Incorrect user input received '{}'".format(user_input))


def record_messages(endpoint,
                    sender_id=UserMessage.DEFAULT_SENDER_ID,
                    max_message_limit=None,
                    use_response_stream=True,
                    on_finish=None):
    """Read messages from the command line and print bot responses."""

    exit_text = INTENT_MESSAGE_PREFIX + 'stop'

    utils.print_color("Bot loaded. Type a message and press enter "
                      "(use '{}' to exit): ".format(exit_text),
                      utils.bcolors.OKGREEN)

    domain = retrieve_domain(endpoint)

    num_messages = 0
    while not console.is_msg_limit_reached(num_messages, max_message_limit):
        text = console.get_cmd_input()
        if text == exit_text:
            break

        response = send_message_receive_prediction(endpoint,
                                                   sender_id, text)
        validate_prediction(endpoint,
                            response.get("tracker"),
                            response.get("scores"),
                            domain.get("intents"))

        num_messages += 1

    if on_finish:
        on_finish()

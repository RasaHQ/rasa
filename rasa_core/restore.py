from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from builtins import str

import argparse
import json
import logging
import warnings
from difflib import SequenceMatcher
from typing import Text, Optional, List, Tuple

from rasa_core import utils, constants
from rasa_core.actions.action import ACTION_LISTEN_NAME
from rasa_core.agent import Agent
from rasa_core.channels import UserMessage, CollectingOutputChannel, console
from rasa_core.domain import Domain
from rasa_core.events import UserUttered, ActionExecuted
from rasa_core.interpreter import NaturalLanguageInterpreter
from rasa_core.run import load_agent
from rasa_core.trackers import DialogueStateTracker
from rasa_core.utils import AvailableEndpoints

logger = logging.getLogger()  # get the root logger


def create_argument_parser():
    """Parse all the command line arguments for the restore script."""

    parser = argparse.ArgumentParser(
            description='starts the bot')
    parser.add_argument(
            '-d', '--core',
            required=True,
            type=str,
            help="core model to run")
    parser.add_argument(
            '-u', '--nlu',
            type=str,
            help="nlu model to run")
    parser.add_argument(
            'tracker_dump',
            type=str,
            help="file that contains a dumped tracker state in json format")
    parser.add_argument(
            '--enable_api',
            action="store_true",
            help="Start the web server api in addition to the input channel")

    utils.add_logging_option_arguments(parser)

    return parser


def _check_prediction_aligns_with_story(last_prediction,
                                        actions_between_utterances):
    # type: (List[Text], List[Text]) -> None
    """Emit a warning if predictions do not align with expected actions."""

    p, a = align_lists(last_prediction, actions_between_utterances)
    if p != a:
        warnings.warn("Model predicted different actions than the "
                      "model used to create the story! Expected: "
                      "{} but got {}.".format(p, a))


def align_lists(predictions, golds):
    # type: (List[Text], List[Text]) -> Tuple[List[Text], List[Text]]
    """Align two lists trying to keep same elements at the same index.

    If lists contain different items at some indices, the algorithm will
    try to find the best alignment and pad with `None`
    values where necessary."""

    padded_predictions = []
    padded_golds = []
    s = SequenceMatcher(None, predictions, golds)

    for tag, i1, i2, j1, j2 in s.get_opcodes():
        padded_predictions.extend(predictions[i1:i2])
        padded_predictions.extend(["None"] * ((j2 - j1) - (i2 - i1)))

        padded_golds.extend(golds[j1:j2])
        padded_golds.extend(["None"] * ((i2 - i1) - (j2 - j1)))

    return padded_predictions, padded_golds


def actions_since_last_utterance(tracker):
    # type: (DialogueStateTracker) -> List[Text]
    """Extract all events after the most recent utterance from the user."""

    actions = []
    for e in reversed(tracker.events):
        if isinstance(e, UserUttered):
            break
        elif isinstance(e, ActionExecuted):
            actions.append(e.action_name)
    actions.reverse()
    return actions


def replay_events(tracker, agent):
    # type: (DialogueStateTracker, Agent) -> None
    """Take a tracker and replay the logged user utterances against an agent.

    During replaying of the user utterances, the executed actions and events
    created by the agent are compared to the logged ones of the tracker that
    is getting replayed. If they differ, a warning is logged.

    At the end, the tracker stored in the agent's tracker store for the
    same sender id will have quite the same state as the one
    that got replayed."""

    actions_between_utterances = []
    last_prediction = [ACTION_LISTEN_NAME]

    for i, event in enumerate(tracker.events_after_latest_restart()):
        if isinstance(event, UserUttered):
            _check_prediction_aligns_with_story(last_prediction,
                                                actions_between_utterances)

            actions_between_utterances = []
            print(utils.wrap_with_color(event.text, utils.bcolors.OKGREEN))
            out = CollectingOutputChannel()
            agent.handle_message(event.text, sender_id=tracker.sender_id,
                                 output_channel=out)
            for m in out.messages:
                console.print_bot_output(m)

            tracker = agent.tracker_store.retrieve(tracker.sender_id)
            last_prediction = actions_since_last_utterance(tracker)

        elif isinstance(event, ActionExecuted):
            actions_between_utterances.append(event.action_name)

    _check_prediction_aligns_with_story(last_prediction,
                                        actions_between_utterances)


def load_tracker_from_json(tracker_dump, domain):
    # type: (Text, Domain) -> DialogueStateTracker
    """Read the json dump from the file and instantiate a tracker it."""

    tracker_json = json.loads(utils.read_file(tracker_dump))
    sender_id = tracker_json.get("sender_id", UserMessage.DEFAULT_SENDER_ID)
    return DialogueStateTracker.from_dict(sender_id,
                                          tracker_json.get("events", []),
                                          domain.slots)


def serve_application(model_directory,  # type: Text
                      nlu_model=None,  # type: Optional[Text]
                      tracker_dump=None,  # type: Optional[Text]
                      port=constants.DEFAULT_SERVER_PORT,  # type: int
                      endpoints=None,  # type: Optional[Text]
                      enable_api=True  # type: bool
                      ):
    from rasa_core import run

    _endpoints = AvailableEndpoints.read_endpoints(endpoints)

    nlu = NaturalLanguageInterpreter.create(nlu_model, _endpoints.nlu)

    input_channels = run.create_http_input_channels("cmdline", None)

    agent = load_agent(model_directory, interpreter=nlu, endpoints=_endpoints)

    http_server = run.start_server(input_channels,
                                   None,
                                   None,
                                   port=port,
                                   initial_agent=agent,
                                   enable_api=enable_api)

    tracker = load_tracker_from_json(tracker_dump,
                                     agent.domain)

    run.start_cmdline_io(constants.DEFAULT_SERVER_FORMAT.format(port),
                         http_server.stop, sender_id=tracker.sender_id)

    replay_events(tracker, agent)

    try:
        http_server.serve_forever()
    except Exception as exc:
        logger.exception(exc)


if __name__ == '__main__':
    # Running as standalone python application
    arg_parser = create_argument_parser()
    cmdline_args = arg_parser.parse_args()

    utils.configure_colored_logging(cmdline_args.loglevel)

    print(utils.wrap_with_color(
            "We'll recreate the dialogue state. After that you can chat "
            "with the bot, continuing the input conversation.",
            utils.bcolors.OKGREEN + utils.bcolors.UNDERLINE))

    serve_application(cmdline_args.core,
                      cmdline_args.nlu,
                      cmdline_args.port,
                      cmdline_args.endpoints,
                      cmdline_args.enable_api)

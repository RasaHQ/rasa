from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import json
import logging
import warnings

from builtins import str
from typing import Text, Optional, List, Tuple

from rasa_core import utils, evaluate
from rasa_core.actions.action import ACTION_LISTEN_NAME
from rasa_core.agent import Agent
from rasa_core.channels import UserMessage
from rasa_core.channels.console import ConsoleInputChannel, ConsoleOutputChannel
from rasa_core.events import UserUttered, ActionExecuted
from rasa_core.trackers import DialogueStateTracker

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

    utils.add_logging_option_arguments(parser)

    return parser


def _check_prediction_aligns_with_story(last_prediction,
                                        actions_between_utterances):
    # type: (List[Text], List[Text]) -> None
    """Emit a warning if predictions do not align with expected actions."""

    p, a = evaluate.align_lists(last_prediction, actions_between_utterances)
    if p != a:
        warnings.warn("Model predicted different actions than the "
                      "model used to create the story! Expected: "
                      "{} but got {}.".format(p, a))


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
            agent.handle_message(event.text, sender_id=tracker.sender_id,
                                 output_channel=ConsoleOutputChannel())
            tracker = agent.tracker_store.retrieve(tracker.sender_id)
            last_prediction = evaluate.actions_since_last_utterance(tracker)

        elif isinstance(event, ActionExecuted):
            actions_between_utterances.append(event.action_name)

    _check_prediction_aligns_with_story(last_prediction,
                                        actions_between_utterances)


def load_tracker_from_json(tracker_dump, domain):
    # type: (Text, Agent) -> DialogueStateTracker
    """Read the json dump from the file and instantiate a tracker it."""

    tracker_json = json.loads(utils.read_file(tracker_dump))
    sender_id = tracker_json.get("sender_id", UserMessage.DEFAULT_SENDER_ID)
    return DialogueStateTracker.from_dict(sender_id,
                                          tracker_json.get("events", []),
                                          domain)


def recreate_agent(model_directory,  # type: Text
                   nlu_model=None,  # type: Optional[Text]
                   tracker_dump=None,  # type: Optional[Text]
                   endpoints=None
                   ):
    # type: (...) -> Tuple[Agent, DialogueStateTracker]
    """Recreate an agent instance."""

    nlg_endpoint = utils.read_endpoint_config(endpoints, "nlg")

    logger.debug("Loading Rasa Core Agent")
    agent = Agent.load(model_directory, nlu_model,
                       generator=nlg_endpoint)

    logger.debug("Finished loading agent. Loading stories now.")

    tracker = load_tracker_from_json(tracker_dump, agent.domain)
    replay_events(tracker, agent)

    return agent, tracker


if __name__ == '__main__':
    # Running as standalone python application
    arg_parser = create_argument_parser()
    cmdline_args = arg_parser.parse_args()

    utils.configure_colored_logging(cmdline_args.loglevel)

    agent, tracker = recreate_agent(cmdline_args.core,
                                    cmdline_args.nlu,
                                    cmdline_args.tracker_dump)

    print(utils.wrap_with_color(
            "You can now continue the dialogue. "
            "Use '/stop' to exit the conversation.",
            utils.bcolors.OKGREEN + utils.bcolors.UNDERLINE))

    agent.handle_channel(ConsoleInputChannel(tracker.sender_id))

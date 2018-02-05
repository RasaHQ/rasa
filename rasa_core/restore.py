from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import json
import logging

from builtins import str

from rasa_core import utils, evaluate
from rasa_core.actions.action import ActionListen
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


def check_prediction_aligns_with_story(last_prediction,
                                       actions_between_utterances):
    p, a = evaluate.min_list_distance(last_prediction,
                                      actions_between_utterances)
    if p != a:
        logger.warn("Model predicted different actions than the "
                    "model used to create the story! Expected: "
                    "{} but got {}.".format(p, a))


def main(model_directory, nlu_model=None, tracker_dump=None):
    """Run the agent."""

    logger.info("Rasa process starting")
    agent = Agent.load(model_directory, nlu_model)

    logger.info("Finished loading agent. Loading stories now.")

    tracker_json = json.loads(utils.read_file(tracker_dump))
    sender_id = tracker_json.get("sender_id", UserMessage.DEFAULT_SENDER_ID)
    tracker = DialogueStateTracker.from_dict(sender_id,
                                             tracker_json.get("events", []),
                                             agent.domain)

    actions_between_utterances = []
    last_prediction = [ActionListen().name()]

    for i, event in enumerate(tracker.events_after_latest_restart()):
        if isinstance(event, UserUttered):
            check_prediction_aligns_with_story(last_prediction,
                                               actions_between_utterances)

            actions_between_utterances = []
            print(utils.wrap_with_color(event.text, utils.bcolors.OKGREEN))
            agent.handle_message(event.text, sender_id=sender_id,
                                 output_channel=ConsoleOutputChannel())
            tracker = agent.tracker_store.retrieve(sender_id)
            last_prediction = evaluate.actions_since_last_utterance(tracker)

        elif isinstance(event, ActionExecuted):
            actions_between_utterances.append(event.action_name)

    check_prediction_aligns_with_story(last_prediction,
                                       actions_between_utterances)

    print(utils.wrap_with_color(
            "You can now continue the dialogue. "
            "Use '/stop' to exit the conversation.",
            utils.bcolors.OKGREEN + utils.bcolors.UNDERLINE))
    agent.handle_channel(ConsoleInputChannel(sender_id))

    return agent


if __name__ == '__main__':
    # Running as standalone python application
    arg_parser = create_argument_parser()
    cmdline_args = arg_parser.parse_args()

    utils.configure_colored_logging(cmdline_args.loglevel)

    main(cmdline_args.core,
         cmdline_args.nlu,
         cmdline_args.tracker_dump)

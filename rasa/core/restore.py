import argparse
import asyncio
import json
import logging
import warnings
from difflib import SequenceMatcher

import rasa.cli.utils
import rasa.core.cli.arguments
import rasa.utils.io
from typing import List, Optional, Text, Tuple

from rasa.cli import utils as cliutils
from rasa.core import constants, run
from rasa.core.actions.action import ACTION_LISTEN_NAME
from rasa.core.channels import UserMessage, CollectingOutputChannel, console
from rasa.core.domain import Domain
from rasa.core.events import ActionExecuted, UserUttered
from rasa.core.trackers import DialogueStateTracker
from rasa.core.utils import AvailableEndpoints

logger = logging.getLogger()  # get the root logger


def create_argument_parser():
    """Parse all the command line arguments for the restore script."""

    parser = argparse.ArgumentParser(description="starts the bot")
    parser.add_argument(
        "-d", "--core", required=True, type=str, help="core model to run"
    )
    parser.add_argument("-u", "--nlu", type=str, help="nlu model to run")
    parser.add_argument(
        "tracker_dump",
        type=str,
        help="file that contains a dumped tracker state in json format",
    )
    parser.add_argument(
        "--enable_api",
        action="store_true",
        help="Start the web server api in addition to the input channel",
    )

    rasa.core.cli.arguments.add_logging_option_arguments(parser)

    return parser


def _check_prediction_aligns_with_story(
    last_prediction: List[Text], actions_between_utterances: List[Text]
) -> None:
    """Emit a warning if predictions do not align with expected actions."""

    p, a = align_lists(last_prediction, actions_between_utterances)
    if p != a:
        warnings.warn(
            "Model predicted different actions than the "
            "model used to create the story! Expected: "
            "{} but got {}.".format(p, a)
        )


def align_lists(
    predictions: List[Text], golds: List[Text]
) -> Tuple[List[Text], List[Text]]:
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


def actions_since_last_utterance(tracker: DialogueStateTracker) -> List[Text]:
    """Extract all events after the most recent utterance from the user."""

    actions = []
    for e in reversed(tracker.events):
        if isinstance(e, UserUttered):
            break
        elif isinstance(e, ActionExecuted):
            actions.append(e.action_name)
    actions.reverse()
    return actions


async def replay_events(tracker: DialogueStateTracker, agent: "Agent") -> None:
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
            _check_prediction_aligns_with_story(
                last_prediction, actions_between_utterances
            )

            actions_between_utterances = []
            cliutils.print_success(event.text)
            out = CollectingOutputChannel()
            await agent.handle_text(
                event.text, sender_id=tracker.sender_id, output_channel=out
            )
            for m in out.messages:
                console.print_bot_output(m)

            tracker = agent.tracker_store.retrieve(tracker.sender_id)
            last_prediction = actions_since_last_utterance(tracker)

        elif isinstance(event, ActionExecuted):
            actions_between_utterances.append(event.action_name)

    _check_prediction_aligns_with_story(last_prediction, actions_between_utterances)


def load_tracker_from_json(tracker_dump: Text, domain: Domain) -> DialogueStateTracker:
    """Read the json dump from the file and instantiate a tracker it."""

    tracker_json = json.loads(rasa.utils.io.read_file(tracker_dump))
    sender_id = tracker_json.get("sender_id", UserMessage.DEFAULT_SENDER_ID)
    return DialogueStateTracker.from_dict(
        sender_id, tracker_json.get("events", []), domain.slots
    )


async def serve_application(
    core_model: Text = None,
    nlu_model: Optional[Text] = None,
    tracker_dump: Optional[Text] = None,
    port=constants.DEFAULT_SERVER_PORT,
    enable_api=True,
    endpoints=None,
):
    input_channels = run.create_http_input_channels("cmdline", None)

    app = run.configure_app(input_channels, enable_api=enable_api, port=port)

    logger.info(
        "Starting Rasa Core server on "
        "{}".format(constants.DEFAULT_SERVER_FORMAT.format(port))
    )

    # noinspection PyShadowingNames
    async def load_agent_and_tracker(app, loop):
        agent = await run.load_agent_on_start(
            core_model, endpoints, nlu_model, app, loop
        )

        tracker = load_tracker_from_json(tracker_dump, agent.domain)
        await replay_events(tracker, agent)

    app.register_listener(load_agent_and_tracker, "before_server_start")
    app.run(host="0.0.0.0", port=port, access_log=logger.isEnabledFor(logging.DEBUG))


if __name__ == "__main__":
    # Running as standalone python application
    arg_parser = create_argument_parser()
    cmdline_args = arg_parser.parse_args()

    rasa.utils.io.configure_colored_logging(cmdline_args.loglevel)
    _endpoints = AvailableEndpoints.read_endpoints(cmdline_args.endpoints)

    print (
        cliutils.wrap_with_color(
            "We'll recreate the dialogue state. After that you can chat "
            "with the bot, continuing the input conversation.",
            rasa.cli.utils.bcolors.OKGREEN + rasa.cli.utils.bcolors.UNDERLINE,
        )
    )

    _loop = asyncio.get_event_loop()
    _loop.run_until_complete(
        serve_application(
            cmdline_args.core,
            cmdline_args.nlu,
            cmdline_args.port,
            cmdline_args.enable_api,
            _endpoints,
        )
    )

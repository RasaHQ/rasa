import argparse
import logging
import os

from sanic import Sanic, response

from rasa.core.domain import Domain
from rasa.core.nlg import TemplatedNaturalLanguageGenerator
from rasa.core.trackers import DialogueStateTracker

logger = logging.getLogger(__name__)

DEFAULT_SERVER_PORT = 5056

DEFAULT_SANIC_WORKERS = 1


def create_argument_parser():
    """Parse all the command line arguments for the nlg server script."""

    parser = argparse.ArgumentParser(description="starts the nlg endpoint")
    parser.add_argument(
        "-p",
        "--port",
        default=DEFAULT_SERVER_PORT,
        type=int,
        help="port to run the server at",
    )
    parser.add_argument(
        "--workers",
        default=DEFAULT_SANIC_WORKERS,
        type=int,
        help="Number of processes to spin up",
    )
    parser.add_argument(
        "-d",
        "--domain",
        type=str,
        default=None,
        help="path of the domain file to load utterances from",
    )

    return parser


async def generate_response(nlg_call, domain):
    """Mock response generator.

    Generates the responses from the bot's domain file.
    """
    kwargs = nlg_call.get("arguments", {})
    template = nlg_call.get("template")
    sender_id = nlg_call.get("tracker", {}).get("sender_id")
    events = nlg_call.get("tracker", {}).get("events")
    tracker = DialogueStateTracker.from_dict(sender_id, events, domain.slots)
    channel_name = nlg_call.get("channel")

    return await TemplatedNaturalLanguageGenerator(domain.templates).generate(
        template, tracker, channel_name, **kwargs
    )


def run_server(domain, port, workers):
    app = Sanic(__name__)

    @app.route("/nlg", methods=["POST", "OPTIONS"])
    async def nlg(request):
        """Endpoint which processes the Core request for a bot response."""
        nlg_call = request.json
        bot_response = await generate_response(nlg_call, domain)

        return response.json(bot_response)

    app.run(
        host="0.0.0.0",
        port=port,
        workers=workers,
        backlog=int(os.environ.get("SANIC_BACKLOG", "100")),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # Running as standalone python application
    arg_parser = create_argument_parser()
    cmdline_args = arg_parser.parse_args()
    _domain = Domain.load(cmdline_args.domain)

    run_server(_domain, cmdline_args.port, cmdline_args.workers)

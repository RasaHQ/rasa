import argparse
import logging
import os
from functools import partial

from sanic import Sanic, response
from sanic.worker.loader import AppLoader

from rasa.shared.core.domain import Domain
from rasa.core.nlg import TemplatedNaturalLanguageGenerator
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.constants import ENV_SANIC_BACKLOG, DEFAULT_SANIC_WORKERS

logger = logging.getLogger(__name__)

DEFAULT_SERVER_PORT = 5056


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
    response = nlg_call.get("response")
    sender_id = nlg_call.get("tracker", {}).get("sender_id")
    events = nlg_call.get("tracker", {}).get("events")
    tracker = DialogueStateTracker.from_dict(sender_id, events, domain.slots)
    channel_name = nlg_call.get("channel", {}).get("name")

    return await TemplatedNaturalLanguageGenerator(domain.responses).generate(
        response, tracker, channel_name, **kwargs
    )


def create_app(domain):
    app = Sanic("nlg_server")

    @app.route("/nlg", methods=["POST", "OPTIONS"])
    async def nlg(request):
        """Endpoint which processes the Core request for a bot response."""
        nlg_call = request.json
        bot_response = await generate_response(nlg_call, domain)

        return response.json(bot_response)

    return app


def run_server(domain, port, workers):
    loader = AppLoader(factory=partial(create_app, domain=domain))
    app = loader.load()

    app.run(
        host="0.0.0.0",
        port=port,
        workers=workers,
        backlog=int(os.environ.get(ENV_SANIC_BACKLOG, "100")),
        legacy=True,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # Running as standalone python application
    arg_parser = create_argument_parser()
    cmdline_args = arg_parser.parse_args()
    _domain = Domain.load(cmdline_args.domain)

    run_server(_domain, cmdline_args.port, cmdline_args.workers)

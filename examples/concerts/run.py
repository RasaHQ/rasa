from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from rasa_core.agent import Agent
from rasa_core.channels.console import ConsoleInputChannel
from rasa_core.interpreter import RegexInterpreter


def run_concerts(serve_forever=True):
    agent = Agent.load("examples/concerts/models/policy/init",
                       interpreter=RegexInterpreter())

    if serve_forever:
        agent.handle_channel(ConsoleInputChannel())
    return agent


if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    run_concerts()

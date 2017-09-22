from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import six

from rasa_core.agent import Agent
from rasa_core.channels.console import ConsoleInputChannel
from rasa_core.interpreter import RasaNLUInterpreter

if six.PY2:
    nlu_model_path = 'examples/babi/models/nlu/current_py2'
else:
    nlu_model_path = 'examples/babi/models/nlu/current_py3'


def run_babi(serve_forever=True):
    agent = Agent.load("examples/babi/models/policy/current",
                       interpreter=RasaNLUInterpreter(nlu_model_path))

    if serve_forever:
        agent.handle_channel(ConsoleInputChannel())
    return agent


if __name__ == '__main__':
    logging.basicConfig(level="DEBUG")
    run_babi()

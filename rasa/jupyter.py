import pprint as pretty_print
from typing import Any, Dict, Text, TYPE_CHECKING
from rasa_core.utils import print_success, print_error
from rasa_core.interpreter import NaturalLanguageInterpreter

if TYPE_CHECKING:
    from rasa_core.agent import Agent


def pprint(object: Any):
    pretty_print.pprint(object, indent=2)


def chat(model: Text = None, agent: 'Agent' = None,
         interpreter: NaturalLanguageInterpreter = None):

    if model:
        from rasa.run import create_agent
        agent = create_agent(model)
    elif agent and interpreter:
        agent.interpreter = NaturalLanguageInterpreter.create(interpreter)
    else:
        print_error("You either have to define a model path or an agent and "
                    "an interpreter.")

    print("Your bot is ready to talk! Type your messages here or send '/stop'.")
    while True:
        message = input()
        if message == '/stop':
            break

        for response in agent.handle_text(message):
            _display_bot_response(response)


def _display_bot_response(response: Dict):
    from IPython.display import Image, display

    for response_type, value in response.items():
        if response_type == 'text':
            print_success(value)

        if response_type == 'image':
            image = Image(url=value)
            display(image,)

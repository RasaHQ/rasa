from contextlib import contextmanager
from datetime import datetime
from typing import Callable, Generator, List, Union

import structlog

from rasa.dialogue_understanding.commands.prompt_command import PromptCommand
from rasa.dialogue_understanding.generator import LLMBasedCommandGenerator
from rasa.shared.providers.llm.llm_response import LLMResponse

structlogger = structlog.get_logger()


def commands_as_string(commands: List[PromptCommand], delimiter: str = "\n") -> str:
    return delimiter.join([command.to_dsl() for command in commands])


def make_mock_invoke_llm(commands: str) -> Callable:
    """Capture the `commands` in a closure so the resulting async function
    can use it as its response.

    Args:
        commands: The commands to return from the mock LLM call.
    """

    async def _mock_invoke_llm(
        self: LLMBasedCommandGenerator, prompt: Union[List[dict], List[str], str]
    ) -> LLMResponse:
        structlogger.debug(
            f"LLM call intercepted, response mocked. "
            f"Responding with the following commands: '{commands}' "
            f"to the prompt: {prompt}"
        )
        fake_response_dict = {
            "id": "",
            "choices": [commands],
            "created": int(datetime.now().timestamp()),
            "model": "mocked-llm",
        }
        return LLMResponse.from_dict(fake_response_dict)

    return _mock_invoke_llm


@contextmanager
def patch_invoke_llm_in_generators(mock_impl: Callable) -> Generator:
    """Replace CommandGenerator.invoke_llm in the base class AND in all
    current subclasses (recursively).  Everything is restored on exit.
    """
    originals = {}

    def collect(cls: type[LLMBasedCommandGenerator]) -> None:
        # store current attribute, then recurse
        originals[cls] = cls.invoke_llm
        for sub in cls.__subclasses__():
            collect(sub)

    # collect every existing subclass of CommandGenerator
    collect(LLMBasedCommandGenerator)  # type: ignore[type-abstract]

    try:
        # apply the monkey-patch everywhere
        for cls in originals:
            cls.invoke_llm = mock_impl  # type: ignore[method-assign]
        yield
    finally:
        # restore originals (even if an exception happened)
        for cls, orig in originals.items():
            cls.invoke_llm = orig  # type: ignore[method-assign]

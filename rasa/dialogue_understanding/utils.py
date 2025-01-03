from contextlib import contextmanager
from typing import Generator

record_commands_and_prompts = False


@contextmanager
def set_record_commands_and_prompts() -> Generator:
    global record_commands_and_prompts
    record_commands_and_prompts = True
    try:
        yield
    finally:
        record_commands_and_prompts = False

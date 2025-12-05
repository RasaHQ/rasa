"""Pytest fixtures for shared/core/flows tests."""

from typing import Generator

import pytest

from rasa.core.config.configuration import Configuration


@pytest.fixture(autouse=True)
def reset_configuration_singleton() -> Generator[None, None, None]:
    """Ensure Configuration is initialized before each test and reset after.

    This is needed because some flow step operations (like
    CallFlowStep.is_calling_agent) access Configuration.get_instance(), which
    requires initialization. When tests run in parallel across multiple CI runners,
    each runner starts fresh without the Configuration singleton initialized.
    """
    # Initialize to empty before the test if not already initialized
    if Configuration._instance is None:
        Configuration.initialise_empty()
    yield
    # Reset after the test to ensure clean state for next test
    Configuration._instance = None

import pytest

from rasa.core.channels.channel import OutputChannel


@pytest.fixture
def output_channel() -> OutputChannel:
    """Create a mock OutputChannel with an empty name."""

    class MockOutputChannel(OutputChannel):
        @classmethod
        def name(cls) -> str:
            return ""

    return MockOutputChannel()


@pytest.fixture
def create_output_channel():
    """Factory fixture to create a mock OutputChannel with a specified name."""

    def _create(channel_name: str = "") -> OutputChannel:
        class MockOutputChannel(OutputChannel):
            @classmethod
            def name(cls) -> str:
                return channel_name

        return MockOutputChannel()

    return _create

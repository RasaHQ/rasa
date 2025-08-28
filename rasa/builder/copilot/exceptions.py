class CopilotFinalBufferReached(Exception):
    """Raised when the rolling buffer ends."""

    pass


class CopilotStreamEndedEarly(Exception):
    """Raised when the stream ended early.

    This happens when the stream ended before the max expected special response
    tokens were reached.
    """

    pass


class CopilotStreamError(Exception):
    """Raised when the stream fails."""

    pass


class InvalidCopilotChatHistorySignature(Exception):
    """Raised when the provided history signature does not match."""

    pass


class MissingCopilotChatHistorySignature(Exception):
    """Raised when a required history signature is missing."""

    pass

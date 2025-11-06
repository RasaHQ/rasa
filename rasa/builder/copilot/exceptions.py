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


class CopilotHistoryDatabaseError(Exception):
    """Raised when database operations fail."""

    pass

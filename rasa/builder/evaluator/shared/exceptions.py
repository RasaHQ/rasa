class EvaluationError(Exception):
    """Base exception for evaluation-related errors."""

    pass


class ClaimExtractionError(Exception):
    """Raised when a claim extraction error occurs."""

    pass


class FaithfulnessJudgingError(Exception):
    """Raised when a faithfulness judging error occurs."""

    pass

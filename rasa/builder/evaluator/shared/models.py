"""Shared models for the evaluator module."""

from pydantic import BaseModel, Field


class EvaluationFailure(BaseModel):
    """Model for an evaluation failure."""

    error_message: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Type of error that occurred")

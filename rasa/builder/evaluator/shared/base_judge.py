"""Base class for LLM-based judges in the evaluation framework.

This module provides a base class that handles common functionality for all
LLM-based judges, including client management, concurrent evaluation with
semaphores, batch processing, and error handling.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, TypeVar, Union

import openai
import structlog
from jinja2 import Template

from rasa.builder.copilot.constants import ROLE_USER
from rasa.builder.evaluator.shared.exceptions import EvaluationError
from rasa.builder.evaluator.shared.models import EvaluationFailure

structlogger = structlog.get_logger()

# Type variables for generic input/output types
TInput = TypeVar("TInput")
TResult = TypeVar("TResult")


class BaseLLMJudge(ABC, Generic[TInput, TResult]):
    """Base class for LLM-based judges.

    This class provides common functionality for all judges that use LLMs
    to evaluate claims, completeness, or other aspects of responses. It handles:
    - OpenAI client lifecycle management
    - Concurrent evaluation with semaphore-based rate limiting
    - Batch processing with error handling
    - LLM API calls with structured output

    Subclasses must implement:
    - _get_prompt_template(): Return the Jinja2 template for the judge
    - _get_response_schema(): Return the JSON schema for LLM responses
    - _render_prompt(): Render the prompt for a specific input
    - _get_model_config(): Return model configuration (name, temperature, etc.)
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 5000,
        timeout: int = 30,
        max_concurrent_evaluations: int = 10,
    ):
        """Initialize the judge.

        Args:
            max_concurrent_evaluations: Maximum number of concurrent LLM calls
                for batch evaluation. Defaults to 10.
        """
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout

        self._prompt_template = self._get_prompt_template()
        self._response_schema = self._get_response_schema()

        self._max_concurrent_evaluations = max_concurrent_evaluations
        self._semaphore = asyncio.Semaphore(max_concurrent_evaluations)

    @abstractmethod
    def _get_prompt_template(self) -> Template:
        """Get the Jinja2 template for rendering prompts.

        Returns:
            Jinja2 Template instance for this judge
        """
        pass

    @abstractmethod
    def _get_response_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for LLM responses.

        Returns:
            Dictionary containing the JSON schema
        """
        pass

    @abstractmethod
    def _render_prompt(self, input: TInput) -> str:
        """Render the prompt for a specific input.

        Args:
            input: The input to render the prompt for

        Returns:
            The rendered prompt string
        """
        pass

    @abstractmethod
    def _parse_result(self, response_data: Dict) -> TResult:
        """Parse the LLM response into a result object.

        Args:
            response_data: The parsed JSON response from the LLM

        Returns:
            A result object of type TResult
        """
        pass

    async def evaluate(
        self,
        inputs: List[TInput],
    ) -> List[Union[TResult, EvaluationFailure]]:
        """Evaluate multiple inputs concurrently.

        Uses a shared OpenAI client for all concurrent evaluations for efficiency.

        Args:
            inputs: List of inputs to evaluate

        Returns:
            A list containing one result per input, where each element is either
            a result (if evaluation succeeded) or a failure (if evaluation failed).
            The result at index i corresponds to the input at index i.
        """
        # Create a single shared client for all concurrent evaluations
        client = openai.AsyncOpenAI()
        try:
            # Create evaluation tasks and execute them concurrently
            tasks = [
                self._evaluate_single_input_with_error_handling(input, client)
                for input in inputs
            ]
            return await asyncio.gather(*tasks)
        finally:
            # Always close the client after all evaluations are done
            try:
                await client.close()
            except Exception as exc:
                # Closing should not break request processing, but we log it
                structlogger.warning(
                    f"evaluator.{self.__class__.__name__.lower()}.client_close_error",
                    error=str(exc),
                )

    async def _evaluate_single_input_with_error_handling(
        self,
        input: TInput,
        client: openai.AsyncOpenAI,
    ) -> Union[TResult, EvaluationFailure]:
        """Evaluate a single input and capture any errors.

        Args:
            input: Input to evaluate
            client: Shared OpenAI client for making LLM calls

        Returns:
            Result if evaluation succeeded, Failure if it failed
        """
        try:
            result = await self._evaluate_single_input(input, client)
            return result
        except Exception as e:
            structlogger.warning(
                f"{self.__class__.__name__.lower()}.evaluate.failed",
                error_type=type(e).__name__,
                error=str(e),
            )
            return EvaluationFailure(
                error_message=str(e),
                error_type=type(e).__name__,
            )

    async def _evaluate_single_input(
        self,
        input: TInput,
        client: openai.AsyncOpenAI,
    ) -> TResult:
        """Evaluate a single input against the judge's criteria.

        Args:
            input: Input to evaluate
            client: Shared OpenAI client for making LLM calls

        Returns:
            Result containing the evaluation

        Raises:
            JudgingError: If evaluation fails
        """
        try:
            # Render the prompt
            prompt = self._render_prompt(input)

            # Use semaphore to limit concurrent LLM calls
            async with self._semaphore:
                response = await self._call_llm(prompt, client)

            # Parse and validate response
            response_data = json.loads(response)
            result = self._parse_result(response_data)

            return result

        except Exception as e:
            structlogger.error(
                f"{self.__class__.__name__.lower()}.evaluate.error",
                error_type=type(e).__name__,
                error=str(e),
            )
            raise EvaluationError(f"Failed to evaluate: {e!s}") from e

    async def _call_llm(self, prompt: str, client: openai.AsyncOpenAI) -> str:
        """Call the LLM service with the given prompt.

        Args:
            prompt: The prompt to send to the LLM
            client: Shared OpenAI client for making the API call

        Returns:
            The LLM response text

        Raises:
            JudgingError: If LLM call fails
        """
        try:
            response = await client.chat.completions.create(
                messages=[{"role": ROLE_USER, "content": prompt}],
                model=self._model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                timeout=self._timeout,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": f"{self.__class__.__name__.lower()}_response",
                        "schema": self._response_schema,
                        "strict": True,
                    },
                },
            )

            content = response.choices[0].message.content or ""

            if not content:
                message = "LLM returned no content"
                structlogger.error(
                    f"{self.__class__.__name__.lower()}._call_llm.empty_response",
                    prompt=prompt,
                    event_info=message,
                    response=response,
                )
                raise EvaluationError(message)

            return content

        except Exception as e:
            structlogger.error(
                f"{self.__class__.__name__.lower()}._call_llm.error",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise EvaluationError(f"LLM call failed: {e!s}") from e

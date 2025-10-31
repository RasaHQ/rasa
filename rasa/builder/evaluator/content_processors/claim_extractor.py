"""Claim extractor for extracting atomic, verifiable claims from Copilot responses.

This module provides functionality to break down Copilot responses into atomic,
verifiable claims that can be evaluated against evidence sources.
"""

import asyncio
import importlib
from typing import List, Union

import openai
import structlog
from jinja2 import Template

from rasa.builder.copilot.constants import ROLE_USER
from rasa.builder.evaluator.constants import (
    CLAIM_EXTRACTOR_PROMPTS_FILE,
    CLAIM_EXTRACTOR_PROMPTS_PACKAGE_NAME,
    CLAIM_EXTRACTOR_RESPONSE_SCHEMA_PATH,
)
from rasa.builder.evaluator.content_processors.models import (
    ClaimExtractionFailure,
    Claims,
)
from rasa.builder.evaluator.exceptions import ClaimExtractionError
from rasa.shared.constants import PACKAGE_NAME
from rasa.shared.utils.io import read_json_file

structlogger = structlog.get_logger()


class ClaimExtractor:
    """Extracts atomic, verifiable claims from Copilot responses using LLM.

    This class uses an LLM to break down Copilot responses into individual,
    testable claims with importance labels and confidence scores.
    """

    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: int,
        max_concurrent_extractions: int = 10,
    ):
        """Initialize the claim extractor.

        Args:
            model: The LLM model to use for claim extraction.
            temperature: Temperature parameter for LLM generation.
            max_tokens: Maximum tokens to generate in LLM response.
            timeout: Timeout for the LLM call in seconds.
            max_concurrent_extractions: Maximum number of concurrent LLM calls
                for batch extraction. Defaults to 10.
        """
        self._prompt_template = Template(
            importlib.resources.read_text(
                f"{PACKAGE_NAME}.{CLAIM_EXTRACTOR_PROMPTS_PACKAGE_NAME}",
                CLAIM_EXTRACTOR_PROMPTS_FILE,
            )
        )
        self._response_schema = read_json_file(CLAIM_EXTRACTOR_RESPONSE_SCHEMA_PATH)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._max_concurrent_extractions = max_concurrent_extractions
        self._semaphore = asyncio.Semaphore(max_concurrent_extractions)

    async def extract(
        self, copilot_responses: List[str]
    ) -> List[Union[Claims, ClaimExtractionFailure]]:
        """Extract claims from multiple Copilot responses concurrently.

        Processes all responses and collects extraction errors.
        Uses a shared OpenAI client for all concurrent extractions for efficiency.

        Args:
            copilot_responses: List of Copilot response texts to analyze

        Returns:
            List of Claims objects or ClaimExtractionFailure objects
        """
        # Create a single shared client for all concurrent extractions
        client = openai.AsyncOpenAI()
        try:
            # Create extraction tasks and execute them concurrently
            tasks = [
                self._extract_claims_with_error_handling(response, client)
                for response in copilot_responses
            ]
            return await asyncio.gather(*tasks)
        finally:
            # Always close the client after all extractions are done
            try:
                await client.close()
            except Exception as exc:
                # Closing should not break request processing, but we log it
                structlogger.warning(
                    "evaluator.claim_extractor.client_close_error",
                    error=str(exc),
                )

    async def _extract_claims_with_error_handling(
        self, response: str, client: openai.AsyncOpenAI
    ) -> Union[Claims, ClaimExtractionFailure]:
        """Extract claims and capture any errors.

        Args:
            response: The Copilot response text to analyze.
            client: Shared OpenAI client for making LLM calls.

        Returns:
            Claims if the extraction was successful, or ClaimExtractionFailure if the
            extraction failed.
        """
        try:
            claims = await self._extract_claims(response, client)
            return claims
        except Exception as e:
            structlogger.warning(
                "claim_extractor.extract_with_error_capture.failed",
                error_type=type(e).__name__,
                error=str(e),
            )
            structlogger.debug(
                "claim_extractor.extract_with_error_capture.failed.debug",
                response_preview=response[:100] if response else None,
            )
            return ClaimExtractionFailure(
                error_message=str(e),
                error_type=type(e).__name__,
            )

    async def _extract_claims(
        self, copilot_response: str, client: openai.AsyncOpenAI
    ) -> Claims:
        """Extract atomic claims from a Copilot response.

        Args:
            copilot_response: The Copilot response text to analyze
            client: Shared OpenAI client for making LLM calls

        Returns:
            Claims object containing the extracted claims

        Raises:
            ClaimExtractionError: If claim extraction fails
        """
        try:
            structlogger.info(
                "claim_extractor.extract.start",
            )

            # Prepare the prompt
            prompt = self._prompt_template.render(response=copilot_response)

            # Use semaphore to limit concurrent LLM calls
            async with self._semaphore:
                llm_response = await self._call_llm(prompt, client)

            claims = Claims.model_validate_json(llm_response)

            structlogger.info(
                "claim_extractor.extract.success",
                total_claims=len(claims),
                high_importance_claims=len(claims.high_importance_claims),
                medium_importance_claims=len(claims.medium_importance_claims),
                low_importance_claims=len(claims.low_importance_claims),
            )

            return claims

        except Exception as e:
            structlogger.error(
                "claim_extractor.extract.error",
                error_type=type(e).__name__,
                error=str(e),
            )
            structlogger.debug(
                "claim_extractor.extract.error.debug",
                copilot_response=copilot_response,
            )
            raise ClaimExtractionError(f"Failed to extract claims: {e!s}") from e

    async def _call_llm(self, prompt: str, client: openai.AsyncOpenAI) -> str:
        """Call the LLM service with the given prompt.

        Args:
            prompt: The prompt to send to the LLM
            client: Shared OpenAI client for making the API call

        Returns:
            The LLM response text

        Raises:
            Exception: If LLM call fails
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
                        "name": "claim_extraction_response",
                        "schema": self._response_schema,
                        "strict": True,
                    },
                },
            )

            content = response.choices[0].message.content or ""

            if not content:
                message = "LLM returned no content"
                structlogger.error(
                    "claim_extractor._call_llm.empty_response",
                    event_info=message,
                )
                structlogger.debug(
                    "claim_extractor._call_llm.empty_response.debug",
                    prompt=prompt,
                )
                raise ClaimExtractionError(message)

            return content

        except Exception as e:
            structlogger.error(
                "claim_extractor._call_llm.error",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise ClaimExtractionError(f"LLM call failed: {e!s}") from e

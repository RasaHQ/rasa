"""Faithfulness Judge for evaluating claims against evidence.

This module provides functionality to evaluate whether claims extracted from
Copilot responses are grounded in the provided evidence (documentation and code).
"""

import importlib
import json
from typing import Any, Dict

import structlog
from jinja2 import Template

from rasa.builder.evaluator.faithfulness_judge.models import (
    FaithfulnessJudgeInput,
    FaithfulnessJudgeResult,
)
from rasa.builder.evaluator.shared.base_judge import BaseLLMJudge
from rasa.builder.evaluator.shared.constants import (
    FAITHFULNESS_JUDGE_PROMPTS_FILE,
    FAITHFULNESS_JUDGE_PROMPTS_PACKAGE_NAME,
    FAITHFULNESS_JUDGE_RESPONSE_SCHEMA_PATH,
)
from rasa.shared.constants import PACKAGE_NAME
from rasa.shared.utils.io import read_json_file

structlogger = structlog.get_logger()


class FaithfulnessJudge(BaseLLMJudge[FaithfulnessJudgeInput, FaithfulnessJudgeResult]):
    """Evaluates claims against evidence to determine faithfulness.

    This class uses an LLM to determine whether claims are supported,
    contradicted, or have insufficient information based on the provided
    documentation and code evidence.
    """

    def _get_prompt_template(self) -> Template:
        return Template(
            importlib.resources.read_text(
                f"{PACKAGE_NAME}.{FAITHFULNESS_JUDGE_PROMPTS_PACKAGE_NAME}",
                FAITHFULNESS_JUDGE_PROMPTS_FILE,
            )
        )

    def _get_response_schema(self) -> Dict[str, Any]:
        return read_json_file(FAITHFULNESS_JUDGE_RESPONSE_SCHEMA_PATH)

    def _render_prompt(self, input: FaithfulnessJudgeInput) -> str:
        """Render the prompt for the faithfulness judge.

        Args:
            input: The input to render the prompt for

        Returns:
            The rendered prompt
        """
        # Convert to JSON strings for template rendering
        documentation_evidence = json.dumps(
            {
                "documentation_evidence": [
                    evidence.model_dump(mode="json")
                    for evidence in input.documentation_evidence
                ]
            },
            indent=2,
        )
        code_evidence = json.dumps(
            {
                "code_evidence": [
                    evidence.model_dump(mode="json") for evidence in input.code_evidence
                ]
            },
            indent=2,
        )
        claims = input.claims.model_dump_json(indent=2)

        # Prepare the prompt
        prompt = self._prompt_template.render(
            claims=claims,
            documentation_evidence=documentation_evidence,
            code_evidence=code_evidence,
        )
        return prompt

    def _parse_result(self, response_data: Dict[str, Any]) -> FaithfulnessJudgeResult:
        """Parse the LLM response into a result object.

        Args:
            response_data: The parsed JSON response from the LLM in a form
            of the FaithfulnessJudgeResult schema

        Returns:
            FaithfulnessJudgeResult
        """
        result = FaithfulnessJudgeResult.model_validate(response_data)
        structlogger.info(
            "evaluator.faithfulness_judge.parse_result.success",
            total_verdicts=len(result.verdicts),
            supported_verdicts=len(result.supported_verdicts),
            contradicted_verdicts=len(result.contradicted_verdicts),
            not_enough_info_verdicts=len(result.not_enough_info_verdicts),
        )
        return result

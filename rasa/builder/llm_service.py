"""Service for handling LLM interactions."""

import asyncio
import importlib
import json
from contextlib import asynccontextmanager
from copy import deepcopy
from typing import Any, AsyncGenerator, Dict, List, Optional

import importlib_resources
import openai
import structlog
from jinja2 import Template

from rasa.builder import config
from rasa.builder.copilot.copilot import Copilot
from rasa.builder.copilot.copilot_response_handler import CopilotResponseHandler
from rasa.builder.copilot.copilot_templated_message_provider import (
    load_copilot_internal_message_templates,
)
from rasa.builder.copilot.history_store import (
    CopilotHistoryStore,
    SQLiteCopilotHistoryStore,
)
from rasa.builder.exceptions import LLMGenerationError
from rasa.builder.guardrails.clients import (
    GuardrailsClient,
    LakeraAIGuardrails,
)
from rasa.builder.guardrails.policy_checker import GuardrailsPolicyChecker
from rasa.constants import PACKAGE_NAME
from rasa.shared.constants import DOMAIN_SCHEMA_FILE, RESPONSES_SCHEMA_FILE
from rasa.shared.core.flows.yaml_flows_io import FLOWS_SCHEMA_FILE
from rasa.shared.utils.io import read_json_file
from rasa.shared.utils.yaml import read_schema_file

structlogger = structlog.get_logger()


class LLMService:
    """Handles OpenAI LLM interactions with caching for efficiency."""

    def __init__(self) -> None:
        self._client: Optional[openai.AsyncOpenAI] = None
        self._domain_schema: Optional[Dict[str, Any]] = None
        self._flows_schema: Optional[Dict[str, Any]] = None
        self._copilot: Optional[Copilot] = None
        self._guardrails: Optional[GuardrailsClient] = None
        self._guardrails_policy_checker: Optional[GuardrailsPolicyChecker] = None
        self._copilot_response_handler: Optional[CopilotResponseHandler] = None
        self._copilot_internal_message_templates: Optional[Dict[str, str]] = None
        self._history_store: Optional[CopilotHistoryStore] = None

    @property
    def copilot(self) -> Copilot:
        """Get or lazy create copilot instance."""
        if self._copilot is None:
            self._copilot = Copilot()

        try:
            return self._copilot
        except Exception as e:
            structlogger.error(
                "llm_service.copilot.error",
                event_info="LLM Service: Error getting copilot instance.",
                error=str(e),
            )
            raise

    @property
    def copilot_response_handler(self) -> CopilotResponseHandler:
        """Get or lazy create copilot response handler instance."""
        if self._copilot_response_handler is None:
            self._copilot_response_handler = CopilotResponseHandler(
                rolling_buffer_size=config.COPILOT_HANDLER_ROLLING_BUFFER_SIZE,
            )
        try:
            return self._copilot_response_handler
        except Exception as e:
            structlogger.error(
                "llm_service.copilot_response_handler.error",
                event_info=(
                    "LLM Service: Error getting copilot response handler instance."
                ),
                error=str(e),
            )
            raise

    @property
    def guardrails(self) -> Optional[GuardrailsClient]:
        """Get or lazy create guardrails instance."""
        if not config.ENABLE_GUARDRAILS:
            return None
        # TODO: Replace with Open Source guardrails implementation once it's ready
        try:
            if self._guardrails is None:
                self._guardrails = LakeraAIGuardrails()
            return self._guardrails
        except Exception as e:
            structlogger.error(
                "llm_service.guardrails.error",
                event_info="LLM Service: Error getting guardrails instance.",
                error=str(e),
            )
            raise

    @property
    def guardrails_policy_checker(self) -> Optional[GuardrailsPolicyChecker]:
        """Get or lazy create guardrails policy checker instance."""
        try:
            if self._guardrails_policy_checker is None and self.guardrails is not None:
                self._guardrails_policy_checker = GuardrailsPolicyChecker(
                    self.guardrails
                )
            return self._guardrails_policy_checker
        except Exception as e:
            structlogger.error(
                "llm_service.guardrails_policy_checker.error",
                event_info=(
                    "LLM Service: Error getting guardrails policy checker instance."
                ),
                error=str(e),
            )
            raise

    @property
    def copilot_internal_message_templates(self) -> Dict[str, str]:
        """Get or lazy load copilot internal message templates."""
        if self._copilot_internal_message_templates is None:
            self._copilot_internal_message_templates = (
                load_copilot_internal_message_templates()
            )
        return self._copilot_internal_message_templates

    @property
    def history_store(self) -> CopilotHistoryStore:
        """Get or lazy create history store instance."""
        if self._history_store is None:
            database_path = config.COPILOT_HISTORY_SQLITE_PATH
            structlogger.info("llm_service.history_store.backend", path=database_path)
            self._history_store = SQLiteCopilotHistoryStore(database_path)

        try:
            return self._history_store
        except Exception as e:
            structlogger.error(
                "llm_service.history_store.error",
                event_info="LLM Service: Error getting history store instance.",
                error=str(e),
            )
            raise

    @staticmethod
    def instantiate_copilot() -> Copilot:
        """Instantiate a new Copilot instance."""
        return Copilot()

    @staticmethod
    def instantiate_handler(rolling_buffer_size: int) -> CopilotResponseHandler:
        """Instantiate a new CopilotResponseHandler instance."""
        return CopilotResponseHandler(
            rolling_buffer_size=rolling_buffer_size,
        )

    @asynccontextmanager
    async def _get_client(self) -> AsyncGenerator[openai.AsyncOpenAI, None]:
        """Get or create OpenAI client with proper resource management."""
        if self._client is None:
            self._client = openai.AsyncOpenAI(timeout=config.OPENAI_TIMEOUT)

        try:
            yield self._client
        except Exception as e:
            structlogger.error("llm.client_error", error=str(e))
            raise

    def _prepare_schemas(self) -> None:
        """Prepare and cache schemas for LLM generation."""
        if self._domain_schema is None:
            self._domain_schema = _prepare_domain_schema()

        if self._flows_schema is None:
            self._flows_schema = _prepare_flows_schema()

    async def generate_rasa_project(
        self, messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate Rasa project data using OpenAI."""
        self._prepare_schemas()

        try:
            async with self._get_client() as client:
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "rasa_project",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "domain": self._domain_schema,
                                "flows": self._flows_schema,
                            },
                            "required": ["domain", "flows"],
                        },
                    },
                }
                response = await client.chat.completions.create(  # type: ignore
                    model=config.OPENAI_MODEL,
                    messages=messages,
                    temperature=config.OPENAI_TEMPERATURE,
                    response_format=response_format,
                )

                content = response.choices[0].message.content
                if not content:
                    raise LLMGenerationError("Empty response from LLM")

                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    raise LLMGenerationError(f"Invalid JSON from LLM: {e}")

        except openai.OpenAIError as e:
            raise LLMGenerationError(f"OpenAI API error: {e}")
        except asyncio.TimeoutError:
            raise LLMGenerationError("LLM request timed out")


# Schema preparation functions (stateless)
def _prepare_domain_schema() -> Dict[str, Any]:
    """Prepare domain schema by removing unnecessary parts."""
    domain_schema = deepcopy(read_schema_file(DOMAIN_SCHEMA_FILE, PACKAGE_NAME, False))

    if not isinstance(domain_schema, dict):
        raise ValueError("Domain schema is not a dictionary")

    # Remove parts not needed for CALM bots
    unnecessary_keys = ["intents", "entities", "forms", "config", "session_config"]

    for key in unnecessary_keys:
        domain_schema["mapping"].pop(key, None)

    # Remove problematic slot mappings
    slot_mapping = domain_schema["mapping"]["slots"]["mapping"]["regex;([A-Za-z]+)"][
        "mapping"
    ]
    slot_mapping.pop("mappings", None)
    slot_mapping.pop("validation", None)

    # Add responses schema
    responses_schema = read_schema_file(RESPONSES_SCHEMA_FILE, PACKAGE_NAME, False)
    if isinstance(responses_schema, dict):
        domain_schema["mapping"]["responses"] = responses_schema["schema;responses"]
    else:
        raise ValueError("Expected responses schema to be a dictionary.")

    return domain_schema


def _prepare_flows_schema() -> Dict[str, Any]:
    """Prepare flows schema by removing nlu_trigger."""
    schema_file = str(
        importlib_resources.files(PACKAGE_NAME).joinpath(FLOWS_SCHEMA_FILE)
    )
    flows_schema = deepcopy(read_json_file(schema_file))
    flows_schema["$defs"]["flow"]["properties"].pop("nlu_trigger", None)
    return flows_schema


# Template functions (stateless with caching)
_skill_template: Optional[Template] = None
_helper_template: Optional[Template] = None


def get_skill_generation_messages(
    skill_description: str, project_data: Dict[str, str]
) -> List[Dict[str, Any]]:
    """Get messages for skill generation."""
    global _skill_template

    if _skill_template is None:
        template_content = importlib.resources.read_text(
            "rasa.builder",
            "skill_to_bot_prompt.jinja2",
        )
        _skill_template = Template(template_content)

    system_prompt = _skill_template.render(
        skill_description=skill_description,
        project_data=project_data,
    )
    return [{"role": "system", "content": system_prompt}]


# Global service instance
llm_service = LLMService()

from typing import Any, Callable, Dict, Optional

from mcp import Tool
from pydantic import BaseModel

from rasa.agents.constants import (
    TOOL_ADDITIONAL_PROPERTIES_KEY,
    TOOL_DESCRIPTION_KEY,
    TOOL_EXECUTOR_KEY,
    TOOL_INPUT_SCHEMA_KEY,
    TOOL_NAME_KEY,
    TOOL_PARAMETERS_KEY,
    TOOL_PROPERTIES_KEY,
    TOOL_PROPERTY_TYPE_KEY,
    TOOL_REQUIRED_KEY,
    TOOL_STRICT_KEY,
    TOOL_TYPE_FUNCTION_KEY,
    TOOL_TYPE_KEY,
)


class AgentToolSchema(BaseModel):
    name: str
    parameters: dict[str, Any]
    strict: bool
    description: Optional[str] = None
    type: str = "function"

    @classmethod
    def from_mcp_tool(cls, tool: Tool) -> "AgentToolSchema":
        """Convert MCP Tool to AgentToolSchema."""
        parameters = tool.inputSchema.copy() if tool.inputSchema else {}

        if parameters:
            cls._validate_and_fix_parameters(parameters)

        return cls(
            name=tool.name,
            description=tool.description,
            parameters=parameters,
            strict=False,
            type=TOOL_TYPE_FUNCTION_KEY,
        )

    @classmethod
    def from_litellm_json_format(cls, tool: Dict[str, Any]) -> "AgentToolSchema":
        """Convert OpenAI dict format to AgentToolSchema."""
        references = (
            "Refer: \n"
            "- LiteLLM JSON Format - https://docs.litellm.ai/docs/completion/function_call#full-code---parallel-function-calling-with-gpt-35-turbo-1106\n"
            "- OpenAI Tool JSON Format - https://platform.openai.com/docs/guides/tools?tool-type=function-calling\n"
        )
        expected_structure = (
            "{\n"
            "    'type': 'function',\n"
            "    'function': {\n"
            "        'name': 'string',\n"
            "        'description': 'string',\n"
            "        'parameters': {\n"
            "            'type': 'object',\n"
            "            'properties': {'string': 'string'},\n"
            "            'required': ['string']\n"
            "        }\n"
            "    }\n"
            "}"
        )
        if (
            TOOL_NAME_KEY in tool
            and TOOL_DESCRIPTION_KEY in tool
            and TOOL_INPUT_SCHEMA_KEY in tool
        ):
            raise ValueError(
                "Anthropic Tool format is not supported yet. Please use the LiteLLM "
                "Tool format, which is based on OpenAI's format.\n"
                "The expected structure is:\n"
                f"{expected_structure}\n"
                f"{references}"
            )
        if not (TOOL_TYPE_FUNCTION_KEY in tool and TOOL_TYPE_KEY in tool):
            raise ValueError(
                "Invalid tool format. Expected a dictionary with 'type' and "
                "'function' keys. Expected LiteLLM Tool format that is based on OpenAI "
                "format which has the following structure: \n"
                f"{expected_structure}\n"
                f"{references}"
            )

        function_data = tool[TOOL_TYPE_FUNCTION_KEY]

        if not (
            TOOL_NAME_KEY in function_data and TOOL_DESCRIPTION_KEY in function_data
        ):
            raise ValueError(
                "Invalid tool format. 'function' must contain 'name' and "
                "'description' keys. Expected LiteLLM Tool format that is based on "
                "OpenAI format which has the following structure: \n"
                f"{expected_structure}\n"
                f"{references}"
            )
        parameters = function_data.get(TOOL_PARAMETERS_KEY, {})

        if parameters:
            cls._validate_and_fix_parameters(parameters)

        return cls(
            name=function_data[TOOL_NAME_KEY],
            description=function_data[TOOL_DESCRIPTION_KEY],
            parameters=parameters,
            strict=function_data.get(TOOL_STRICT_KEY, False),
            type=tool[TOOL_TYPE_KEY],
        )

    @staticmethod
    def _validate_and_fix_parameters(parameters: Dict[str, Any]) -> None:
        """Validate and fix parameters to ensure they meet OpenAI function calling
        requirements."""
        if not parameters:
            return

        # Ensure additionalProperties is set at the top level
        if TOOL_ADDITIONAL_PROPERTIES_KEY not in parameters:
            parameters[TOOL_ADDITIONAL_PROPERTIES_KEY] = False

        # Ensure required is set at the top level
        if TOOL_REQUIRED_KEY not in parameters:
            parameters[TOOL_REQUIRED_KEY] = []

        # Ensure all properties have types, required field and additionalProperties
        if TOOL_PROPERTIES_KEY in parameters:
            AgentToolSchema._ensure_property_types(parameters)

    @staticmethod
    def _ensure_property_types(parameters: Dict[str, Any]) -> None:
        """Ensure all properties in parameters have a type defined and
        additionalProperties is set."""
        properties = parameters[TOOL_PROPERTIES_KEY]

        if not properties:
            return

        for _, prop_schema in properties.items():
            if not isinstance(prop_schema, dict):
                continue

            # Ensure the property has a type
            if TOOL_PROPERTY_TYPE_KEY not in prop_schema:
                prop_schema[TOOL_PROPERTY_TYPE_KEY] = "string"

            # If it's an object type, ensure additionalProperties and required
            # fields are set
            if prop_schema[TOOL_PROPERTY_TYPE_KEY] == "object":
                if TOOL_ADDITIONAL_PROPERTIES_KEY not in prop_schema:
                    prop_schema[TOOL_ADDITIONAL_PROPERTIES_KEY] = False

                # Ensure required key exists for object properties
                if TOOL_REQUIRED_KEY not in prop_schema:
                    prop_schema[TOOL_REQUIRED_KEY] = []

    def to_litellm_json_format(self) -> Dict[str, Any]:
        """Convert AgentToolSchema to OpenAI format."""
        # Ensure the schema is valid before conversion
        return {
            TOOL_TYPE_KEY: TOOL_TYPE_FUNCTION_KEY,
            TOOL_TYPE_FUNCTION_KEY: self.model_dump(exclude={TOOL_TYPE_KEY}),
        }


class CustomToolSchema(BaseModel):
    """A class that represents the schema of a custom agent tool."""

    tool_name: str
    tool_definition: AgentToolSchema
    tool_executor: Callable

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "CustomToolSchema":
        """Convert a custom tool config to CustomToolSchema."""
        agent_tool_schema = AgentToolSchema.from_litellm_json_format(config)
        if TOOL_EXECUTOR_KEY not in config:
            raise ValueError("Custom tool executor is required.")

        return cls(
            tool_name=agent_tool_schema.name,
            tool_definition=agent_tool_schema,
            tool_executor=config[TOOL_EXECUTOR_KEY],
        )

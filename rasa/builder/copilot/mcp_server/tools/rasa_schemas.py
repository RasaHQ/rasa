"""Load Rasa flow, domain, and e2e schemas for MCP tools.

Flow schema is JSON Schema; domain and e2e schemas are YAML schema format.
"""

import json

from rasa.builder.copilot.mcp_server.models import SchemaResponse, SchemaType


def get_flow_schema(calm_only: bool = True) -> SchemaResponse:
    """Load the official Rasa flow schema (JSON Schema format).

    Args:
        calm_only: Whether to return only the CALM-related properties of the flow
            schema. If True, the schema excludes the NLU-related properties from flow
            definitions (CALM-style, intent-agnostic).

    Returns:
        The flow schema as a JSON string (from flows_yaml_schema.json).
    """
    from rasa.builder.shared.rasa_schema_utils import (
        get_flows_schema as _get_flows_schema,
    )

    schema = _get_flows_schema(calm_only=calm_only)
    return SchemaResponse(
        success=True,
        schema_type=SchemaType.FLOW,
        schema_content=json.dumps(schema, indent=2),
    )


def get_full_domain_schema(calm_only: bool = True) -> SchemaResponse:
    """Load the Rasa domain schema in YAML schema format (slots, actions, responses).

    Returns the same combined domain + responses schema used for validating
    and generating domain YAML (e.g. in project generation), with NLU
    parts (intents, entities, etc.) removed.

    Args:
        calm_only: Whether to return only the CALM-related properties of the domain
            schema.
    """
    from rasa.builder.shared.rasa_schema_utils import (
        get_domain_schema as _get_domain_schema,
    )

    schema = _get_domain_schema(calm_only=calm_only)
    return SchemaResponse(
        success=True,
        schema_type=SchemaType.DOMAIN,
        schema_content=json.dumps(schema, indent=2),
    )


def get_e2e_schema() -> SchemaResponse:
    """Load the Rasa e2e test schema in YAML schema format.

    Returns:
        The e2e test schema as a JSON-serialized string (schema is YAML schema format).
    """
    from rasa.builder.shared.rasa_schema_utils import (
        get_e2e_tests_schema as _get_e2e_tests_schema,
    )

    schema = _get_e2e_tests_schema()
    return SchemaResponse(
        success=True,
        schema_type=SchemaType.E2E,
        schema_content=json.dumps(schema, indent=2),
    )

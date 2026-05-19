# Rasa MCP Server

MCP server for building, validating, training, and testing Rasa assistants.

This server enables AI agents to:
- Search official Rasa documentation (RAG-powered)
- Inspect project structure (flows, slots, responses, actions)
- Validate configuration and training data
- Train models
- Test conversations and debug assistant behavior

## Documentation

RAG search over official Rasa docs. Use for any Rasa-related questions.

- `search_rasa_documentation` - Search docs and return relevant snippets with source links

Always ground your answers in the documentation.

## Introspection

List and get flows, slots, responses, and actions.

- `list_project_flow_definitions` - List all flows in the project
- `list_project_slot_definitions` - List all slots in the domain
- `list_project_response_definitions` - List all responses in the domain
- `get_flow` - Get a single flow by ID or name with full definition
- `get_slot` - Get a single slot by name with full definition
- `get_response` - Get a single response by name with full definition
- `list_project_custom_actions_in_domain` - List custom actions declared in domain
- `list_custom_action_implementations` - List custom action implementations from Python files
- `list_default_action_names` - List built-in Rasa actions

## Schemas

Get official Rasa schemas for validation and code generation.

- `get_flow_schema` - Flow JSON Schema (for flows.yml structure)
- `get_domain_schema` - Domain YAML schema (slots, actions, responses structure within the domain files)
- `get_e2e_schema` - Assistant E2E test YAML schema (test cases and steps)

## Build

Validate project and train models.

- `validate_project` - Check configuration and training data for errors
- `train_rasa_assistant` - Train a new model

Always validate before training. Fix any validation errors before proceeding.

## Test

Talk to assistant, get logs, and inspect tracker state.

- `talk_to_assistant` - Send messages and get responses
- `get_assistant_logs` - View recent logs for troubleshooting

Use these to verify your changes work as expected.

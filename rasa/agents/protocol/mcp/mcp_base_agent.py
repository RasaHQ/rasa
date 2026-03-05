import json
import math
from abc import abstractmethod
from datetime import datetime, timedelta
from inspect import isawaitable
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import anyio
import structlog
from jinja2 import Template
from mcp import ListToolsResult

from rasa.agents.constants import (
    AGENT_DEFAULT_MAX_RETRIES,
    AGENT_DEFAULT_TIMEOUT_SECONDS,
    AGENT_FILLER_MESSAGE_STREAM_DEFAULT_CHUNK_SIZE,
    AGENT_FILLER_MESSAGES_ENABLED_DEFAULT,
    AGENT_METADATA_AGENT_ID_KEY,
    AGENT_METADATA_AGENT_RESPONSE_KEY,
    AGENT_METADATA_MODEL_ID_KEY,
    AGENT_METADATA_RESUMED_AFTER_INTERRUPTION,
    AGENT_METADATA_SENDER_ID_KEY,
    AGENT_METADATA_STRUCTURED_RESULTS_KEY,
    BOT_UTTERANCE_AGENT_MESSAGE_TYPE_FILLER_MESSAGE,
    KEY_ARGUMENTS,
    KEY_CONTENT,
    KEY_FUNCTION,
    KEY_ID,
    KEY_NAME,
    KEY_ROLE,
    KEY_TOOL_CALL_ID,
    KEY_TOOL_CALLS,
    KEY_TYPE,
)
from rasa.agents.core.agent_protocol import AgentProtocol
from rasa.agents.core.types import AgentStatus, ProtocolType
from rasa.agents.schemas import (
    AgentInput,
    AgentInputSlot,
    AgentOutput,
    AgentToolResult,
    AgentToolSchema,
    CustomToolSchema,
)
from rasa.agents.utils import get_slot_value_from_agent_input
from rasa.core.available_agents import AgentConfig, AgentMCPServerConfig, ProtocolConfig
from rasa.core.channels import OutputChannel
from rasa.shared.agents.utils import make_agent_identifier
from rasa.shared.constants import (
    DEFAULT_INCLUDE_DATE_TIME,
    DEFAULT_TIMEZONE,
    MAX_COMPLETION_TOKENS_CONFIG_KEY,
    MODEL_CONFIG_KEY,
    OPENAI_PROVIDER,
    PROVIDER_CONFIG_KEY,
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
    ROLE_TOOL,
    ROLE_USER,
    TEMPERATURE_CONFIG_KEY,
    TIMEOUT_CONFIG_KEY,
)
from rasa.shared.core.constants import MOCKED_DATETIME_SLOT
from rasa.shared.core.events import BotUttered, Event, SlotSet, UserUttered
from rasa.shared.exceptions import AgentInitializationException, AuthenticationError
from rasa.shared.providers.llm.llm_response import LLMResponse, LLMToolCall
from rasa.shared.utils.constants import (
    LANGFUSE_METADATA_AGENT_ID,
    LANGFUSE_METADATA_COMPONENT_NAME,
    LANGFUSE_METADATA_CUSTOM_METADATA,
    LANGFUSE_METADATA_MODEL_ID,
    LANGFUSE_METADATA_REACT_SUB_AGENT_NAME,
    LANGFUSE_METADATA_SESSION_ID,
    LANGFUSE_METADATA_TAGS,
    LOG_COMPONENT_SOURCE_METHOD_INIT,
)
from rasa.shared.utils.datetime_utils import (
    resolve_datetime,
    validate_datetime_configuration,
)
from rasa.shared.utils.llm import (
    get_prompt_template,
    llm_factory,
    resolve_model_client_config,
    serialize_bot_response_for_prompt,
)
from rasa.shared.utils.mcp.server_connection import MCPServerConnection
from rasa.shared.utils.mcp.utils import build_mcp_meta, call_tool_with_meta

DEFAULT_OPENAI_MAX_GENERATED_TOKENS = 256
MODEL_NAME_GPT_4O_2024_11_20 = "gpt-4o-2024-11-20"
DEFAULT_LLM_CONFIG = {
    PROVIDER_CONFIG_KEY: OPENAI_PROVIDER,
    MODEL_CONFIG_KEY: MODEL_NAME_GPT_4O_2024_11_20,
    TEMPERATURE_CONFIG_KEY: 0.0,
    MAX_COMPLETION_TOKENS_CONFIG_KEY: DEFAULT_OPENAI_MAX_GENERATED_TOKENS,
    TIMEOUT_CONFIG_KEY: 7,
}

if TYPE_CHECKING:
    from rasa.core.config.available_endpoints import MCPMetaMapConfig

structlogger = structlog.get_logger()

_MESSAGE_CACHE_BASE_MESSAGES_KEY = "base_messages_key"
_MESSAGE_CACHE_BASE_MESSAGES = "base_messages"


class MCPBaseAgent(AgentProtocol):
    """MCP protocol implementation."""

    MAX_ITERATIONS = 10

    TOOL_CALL_DEFAULT_TIMEOUT = 10  # seconds

    # ============================================================================
    # Initialization & Setup
    # ============================================================================

    def __init__(
        self,
        name: str,
        description: str,
        protocol_type: ProtocolConfig,
        server_configs: List[AgentMCPServerConfig],
        llm_config: Optional[Dict[str, Any]] = None,
        prompt_template: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        include_date_time: Optional[bool] = None,
        timezone: Optional[str] = None,
        enable_filler_messages: Optional[bool] = None,
        tool_timeout: Optional[float] = None,
    ):
        self._name = name

        self._description = description

        self._protocol_type = protocol_type

        self._llm_config = resolve_model_client_config(
            llm_config, self.__class__.__name__
        )

        self.llm_client = llm_factory(self._llm_config, self.get_default_llm_config())

        self.prompt_template = get_prompt_template(
            prompt_template,
            self.get_default_prompt_template(),
            log_source_component=self.__class__.__name__,
            log_source_method=LOG_COMPONENT_SOURCE_METHOD_INIT,
        )

        self._timeout = timeout or AGENT_DEFAULT_TIMEOUT_SECONDS

        self._max_retries = max_retries or AGENT_DEFAULT_MAX_RETRIES

        self._include_date_time = (
            include_date_time
            if include_date_time is not None
            else DEFAULT_INCLUDE_DATE_TIME
        )
        self._timezone = timezone or DEFAULT_TIMEZONE

        self._enable_filler_messages = (
            enable_filler_messages
            if enable_filler_messages is not None
            else AGENT_FILLER_MESSAGES_ENABLED_DEFAULT
        )

        if tool_timeout is not None and (
            not math.isfinite(tool_timeout) or tool_timeout <= 0
        ):
            raise ValueError("`tool_timeout` must be a finite number greater than 0.")

        self._tool_timeout = (
            tool_timeout if tool_timeout is not None else self.TOOL_CALL_DEFAULT_TIMEOUT
        )

        self._server_configs = server_configs or []

        # Server name -> meta_map config from endpoints.yml.
        self._server_to_meta_map: Dict[str, "MCPMetaMapConfig"] = {
            server_config.name: server_config.meta_map
            for server_config in self._server_configs
            if server_config.meta_map is not None
        }

        # Stores the MCP tools for the agent.
        self._mcp_tools: List[AgentToolSchema] = []

        # Stores the custom tools for the agent.
        self._custom_tools: List[CustomToolSchema] = [
            CustomToolSchema.from_dict(tool)
            for tool in self.get_custom_tool_definitions()
        ]

        # Maps the tool names to the MCP servers that provide them.
        # key: tool name, value: server name.
        self._tool_to_server_mapper: Dict[str, str] = {}

        # Stores the connections to the MCP servers.
        # key: server name, value: connection object.
        self._server_connections: Dict[str, MCPServerConnection] = {}

    @classmethod
    def from_config(cls, config: AgentConfig) -> "MCPBaseAgent":
        """Initialize the MCP Open Agent with the given configuration."""
        # Warn if configuration.timeout is set for MCP agents
        if config.configuration and config.configuration.timeout is not None:
            structlogger.warning(
                "mcp_agent.configuration.timeout.not_implemented",
                event_info=(
                    "`configuration.timeout` is not supported for MCP agents. MCP "
                    "agents do not make external connections, so an agent-level timeout"
                    " does not apply. To control timeout behavior for LLM calls, set "
                    "the `timeout` value in the `model_group` section of endpoints.yml "
                    "and reference it through `configuration.llm.model_group`."
                ),
                agent_name=config.agent.name,
            )

        # Set datetime configuration
        include_date_time = DEFAULT_INCLUDE_DATE_TIME
        timezone = DEFAULT_TIMEZONE
        is_custom_timezone_provided = False
        if config.configuration:
            if config.configuration.include_date_time is not None:
                include_date_time = config.configuration.include_date_time
            if config.configuration.timezone is not None:
                timezone = config.configuration.timezone
                is_custom_timezone_provided = True

            # Validate datetime configuration
            validate_datetime_configuration(
                include_date_time,
                timezone,
                is_custom_timezone_provided,
                f"agent '{config.agent.name}'",
                error_code="agent.configuration.invalid_timezone",
                agent_name=config.agent.name,
            )

        return cls(
            name=config.agent.name,
            description=config.agent.description,
            protocol_type=config.agent.protocol,
            llm_config=config.configuration.llm if config.configuration else None,
            prompt_template=config.configuration.prompt_template
            if config.configuration
            else None,
            timeout=config.configuration.timeout if config.configuration else None,
            max_retries=config.configuration.max_retries
            if config.configuration
            else None,
            server_configs=config.connections.mcp_servers
            if config.connections
            else None,
            include_date_time=include_date_time,
            timezone=timezone,
            enable_filler_messages=config.configuration.enable_filler_messages
            if config.configuration
            else None,
            tool_timeout=config.configuration.tool_timeout
            if config.configuration
            else None,
        )

    # ============================================================================
    # Class Configuration & Properties
    # ============================================================================

    @classmethod
    @abstractmethod
    def get_default_prompt_template(cls) -> str: ...

    @property
    def agent_conforms_to(self) -> ProtocolConfig:
        return self._protocol_type

    @property
    @abstractmethod
    def protocol_type(self) -> ProtocolType: ...

    @staticmethod
    def get_default_llm_config() -> Dict[str, Any]:
        """Get the default LLM config for the command generator."""
        return DEFAULT_LLM_CONFIG

    @classmethod
    def get_agent_specific_built_in_tools(
        cls, agent_input: AgentInput
    ) -> List[AgentToolSchema]:
        """Get agentic specific built-in tools."""
        return []

    def get_custom_tool_definitions(self) -> List[Dict[str, Any]]:
        """Add custom tool definitions and their executors for MCP agents.

        This method can be overridden to provide custom tools that the agent
        can use during its operation. Each tool definition follows the LiteLLM JSON
        format and must include:
        - "type": should always be "function" for tools.
        - "function" → the tool metadata (name, description, and parameters).
        - "tool_executor" → a coroutine method that actually performs the tool's action.

        Refer:
        - LiteLLM JSON Format - https://docs.litellm.ai/docs/completion/function_call#full-code---parallel-function-calling-with-gpt-35-turbo-1106
        - OpenAI Tool JSON Format - https://platform.openai.com/docs/guides/tools?tool-type=function-calling

        Note:
            - In LiteLLM, the tool metadata is wrapped inside the "function" key.
            - This differs from OpenAI's format, where the metadata (name, description,
            parameters) sits at the top level. Be careful when copying examples from
            OpenAI docs.
            - The tool executor method should be a coroutine function that returns an
            AgentToolResult object.

        Returns:
            A list of tool definitions paired with their executors.

        Example:
            ```python
            def get_custom_tool_definitions(self) -> List[Dict[str, Any]]:
                return [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "description": "Get the current weather in given location",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "location": {
                                        "type": "string",
                                        "description": "The city, e.g. San Francisco",
                                    },
                                    "unit": {
                                        "type": "string",
                                        "enum": ["celsius", "fahrenheit"],
                                    },
                                },
                                "required": ["location"],
                            },
                        },
                        "tool_executor": self.get_current_weather,
                    }
                ]
            ```
        """
        return []

    # ============================================================================
    # Connection Management
    # ============================================================================

    async def connect(self) -> None:
        """Connect to the MCP servers and initialize the agent.

        This method establishes connections to the configured MCP servers,
        fetches the available tools, and prepares the agent for operation.
        It should be called before sending any messages to the agent.

        Retries:
            Retries connection N times if a ConnectionError is raised.

        Logs:
            Warning: If the connection to any server fails.
            Warning: If there is a duplicate tool name across servers.
            Warning: If there is an error fetching tools from any server.
        """
        for attempt in range(1, self._max_retries + 1):
            try:
                await self.connect_to_servers()
                await self.fetch_and_store_available_tools()
                break
            except ConnectionError as ce:
                structlogger.warning(
                    "mcp_agent.connect.connection_error",
                    event_info=f"Connection attempt {attempt} failed.",
                    error=str(ce),
                    attempt=attempt,
                    max_retries=self._max_retries,
                )
                if attempt == self._max_retries:
                    structlogger.error(
                        "mcp_agent.connect.failed_after_retries",
                        event_info="All connection attempts failed.",
                    )
                    raise AgentInitializationException(
                        f"Agent `{self._name}` failed to initialize. Failed to connect "
                        f"to MCP servers after {self._max_retries} attempts. {ce!s}"
                    ) from ce
            except (Exception, AuthenticationError) as e:
                if isinstance(e, AuthenticationError):
                    event_info = (
                        f"Authentication error during agent initialization. {e!s}"
                    )
                else:
                    event_info = f"Unexpected error during agent initialization. {e!s}"
                structlogger.error(
                    "mcp_agent.connect.unexpected_exception",
                    event_info=event_info,
                    error=str(e),
                    agent_name=self._name,
                    agent_id=str(make_agent_identifier(self._name, self.protocol_type)),
                )
                raise AgentInitializationException(event_info) from e

    async def connect_to_server(self, server_config: AgentMCPServerConfig) -> None:
        server_name = server_config.name
        connection = MCPServerConnection.from_config(server_config.model_dump())
        try:
            await connection.connect()
            self._server_connections[server_name] = connection
            structlogger.info(
                "mcp_agent.connect_to_server.connected",
                event_info=(
                    f"Agent `{self._name}` connected to MCP server - "
                    f"`{server_name}` @ `{connection.server_url}`"
                ),
                server_id=server_name,
                server_url=connection.server_url,
                agent_name=self._name,
                agent_id=str(make_agent_identifier(self._name, self.protocol_type)),
            )
        except Exception as e:
            event_info = (
                f"Agent `{self._name}` failed to connect to MCP server - "
                f"`{server_name}` @ `{server_config.url}`"
            )
            structlogger.error(
                "mcp_agent.connect.failed_to_connect",
                event_info=event_info,
                server_id=server_name,
                server_url=server_config.url,
                agent_name=self._name,
                agent_id=str(make_agent_identifier(self._name, self.protocol_type)),
            )

            # Wrap exceptions with extra info and raise the same type of exception.
            raise type(e)(f"{event_info} : {e!s}") from e

    async def connect_to_servers(self) -> None:
        """Connect to MCP servers."""
        for server_config in self._server_configs:
            await self.connect_to_server(server_config)

    async def disconnect_server(self, server_name: str) -> None:
        """Disconnect from an MCP server.

        Args:
            server_name: The name of the server to disconnect from.

        Logs:
            - An error if the server disconnect fails.
        """
        if server_name not in self._server_connections:
            return
        try:
            await self._server_connections[server_name].close()
        except Exception as e:
            structlogger.error(
                "mcp_agent.disconnect_server.error",
                event_info=f"Failed to disconnect from server `{server_name}`: {e!s}",
                server_name=server_name,
                agent_name=self._name,
                agent_id=str(make_agent_identifier(self._name, self.protocol_type)),
            )

    async def disconnect(self) -> None:
        """Close all MCP server connections."""
        server_names = list(self._server_connections.keys())
        for server_name in server_names:
            await self.disconnect_server(server_name)

    # ============================================================================
    # Tool Management
    # ============================================================================

    async def list_tools(self, connection: MCPServerConnection) -> ListToolsResult:
        """List the tools from the MCP server."""
        session = await connection.ensure_active_session()
        return await session.list_tools()

    def get_custom_tools(self) -> List[AgentToolSchema]:
        """Get the custom tools for the agent."""
        return [tool.tool_definition for tool in self._custom_tools]

    def get_available_tools(self, agent_input: AgentInput) -> List[AgentToolSchema]:
        """Get the available tools for the agent."""
        return (
            self._mcp_tools
            + self.get_agent_specific_built_in_tools(agent_input)
            + self.get_custom_tools()
        )

    async def _get_filtered_tools_from_server(
        self,
        server_name: str,
        connection: MCPServerConnection,
        include_tools: Optional[List[str]] = None,
        exclude_tools: Optional[List[str]] = None,
    ) -> List[AgentToolSchema]:
        """Get filtered tools from the MCP server.

        This method fetches the available tools from the MCP server and filters them
        based on the include_tools and exclude_tools parameters.

        Args:
            server_name: The name of the MCP server.
            connection: The MCP server connection.
            include_tools: List of tool names to include. If provided, only tools in
                this list will be fetched.
            exclude_tools: List of tool names to exclude. If provided, tools in this
                list will not be fetched.

        Returns:
            A list of AgentToolSchema objects representing the filtered tools.

        Logs:
            Warning: If there is a duplicate tool name across servers.
            Warning: If there is an error fetching tools from the server.
        """
        try:
            tools_response = await self.list_tools(connection)
            if not tools_response:
                return []

            filtered_tools = []
            for tool in tools_response.tools:
                if include_tools and tool.name not in include_tools:
                    continue
                if exclude_tools and tool.name in exclude_tools:
                    continue
                filtered_tools.append(AgentToolSchema.from_mcp_tool(tool))

            return filtered_tools

        except Exception as e:
            event_info = f"Failed to load tools from {server_name}"
            structlogger.warning(
                "mcp_agent.get_filtered_tools_from_server.failed_to_get_tools",
                event_info=event_info,
                server_name=server_name,
                server_url=connection.server_url,
                error=str(e),
            )
            return []

    def _get_include_exclude_tools_from_server_configs(
        self, server_name: str
    ) -> Tuple[Optional[List[str]], Optional[List[str]]]:
        """Get the include and exclude tools from the server configs."""
        for server_config in self._server_configs:
            if server_config.name == server_name:
                return server_config.include_tools, server_config.exclude_tools
        return None, None

    async def fetch_and_store_available_tools(self) -> None:
        """Fetch and store the available tools from the MCP servers.

        This method fetches the available tools from the MCP servers and stores them
        in the agent's internal state. It also maps the tool names to the MCP servers
        that provide them.

        Side effects:
            - Updates the `_mcp_tools` attribute.
            - Updates the `_tool_to_server_mapper` attribute.

        Logs:
            Warning: If there is a duplicate tool name across servers.
            Warning: If there is an error fetching tools from any server.
        """
        for server_name, connection in self._server_connections.items():
            # Get the include and exclude tools from the server configs.
            include_tools, exclude_tools = (
                self._get_include_exclude_tools_from_server_configs(server_name)
            )

            # Get the filtered tools from the server.
            tools = await self._get_filtered_tools_from_server(
                server_name, connection, include_tools, exclude_tools
            )

            # Add the tools to the tool_to_server_mapper and the available_tools.
            for tool in tools:
                if tool.name in self._tool_to_server_mapper:
                    structlogger.warning(
                        "mcp_agent.duplicate_tool_name",
                        event_info=(
                            f"Tool - {tool.name} from server {server_name} already "
                            f"exists in {self._tool_to_server_mapper[tool.name]}. "
                            f"Omitting the tool from server {server_name}."
                        ),
                        tool_name=tool.name,
                        server_name=server_name,
                        server_url=connection.server_url,
                    )
                    continue

                self._tool_to_server_mapper[tool.name] = server_name
                self._mcp_tools.append(tool)

        structlogger.debug(
            "mcp_agent.fetch_and_store_available_tools.success",
            event_info=(
                "Successfully fetched and stored available tools from MCP servers."
            ),
            agent_name=self._name,
            agent_id=str(make_agent_identifier(self._name, self.protocol_type)),
            mcp_tools_fetched=len(self._mcp_tools),
            mcp_tools=[tool.name for tool in self._mcp_tools],
        )

    # ============================================================================
    # LLM & Prompt Management
    # ============================================================================

    def _build_context_for_prompt(self, context: AgentInput) -> Dict[str, Any]:
        """Get the context dictionary for the prompt."""
        context_dict = context.model_dump(exclude={"id", "timestamp", "events"})
        if "slots" in context_dict and isinstance(context_dict["slots"], list):
            context_dict["slots"] = {
                slot.name: slot.value
                for slot in context.slots
                if slot.value is not None
            }

        if self._include_date_time:
            context_dict["current_datetime"] = self._get_current_datetime_for_prompt(
                context
            )

        # Expose resume-after-interruption for Jinja prompt template.
        metadata = context.metadata or {}
        context_dict["resumed_after_interruption"] = bool(
            metadata.get(AGENT_METADATA_RESUMED_AFTER_INTERRUPTION)
        )
        context_dict["resumed_last_request"] = (
            metadata.get(AGENT_METADATA_AGENT_RESPONSE_KEY, "") or ""
        )

        return {
            **context_dict,
            "description": self._description,
            "enable_filler_messages": self._enable_filler_messages,
        }

    def _get_current_datetime_for_prompt(self, context: AgentInput) -> datetime:
        """Get the current datetime for the prompt."""
        # Get the mocked datetime value from the context.
        mocked_datetime_value = get_slot_value_from_agent_input(
            context, MOCKED_DATETIME_SLOT
        )
        # Resolve the datetime.
        return resolve_datetime(mocked_datetime_value, timezone=self._timezone)

    def render_prompt_template(self, context: AgentInput) -> str:
        """Render the prompt template with the provided inputs."""
        # Build the context for the prompt.
        template_vars = self._build_context_for_prompt(context)
        # Render the prompt template.
        return Template(self.prompt_template).render(**template_vars)

    def _build_conversation_messages_for_llm_request(
        self, context: AgentInput, turns: int = 10
    ) -> List[Dict[str, str]]:
        """Build user/assistant messages from utterance events and user message."""
        messages: List[Dict[str, str]] = []

        # Collect up to `turns` most recent user and bot utterance events.
        utterance_events: List[Any] = []
        collected = 0
        for event in reversed(context.events):
            if not isinstance(event, (UserUttered, BotUttered)):
                continue
            utterance_events.append(event)
            collected += 1
            if collected >= turns:
                break

        # Reverse to restore chronological order (oldest first).
        utterance_events.reverse()

        # Track the last user text seen while building messages so we can
        # decide whether to append context.user_message without a second pass.
        last_user_content: Optional[str] = None
        for event in utterance_events:
            if isinstance(event, UserUttered):
                if not event.text:
                    continue
                last_user_content = event.text
                messages.append({KEY_ROLE: ROLE_USER, KEY_CONTENT: event.text})
            elif isinstance(event, BotUttered):
                bot_response = serialize_bot_response_for_prompt(event)
                if not bot_response:
                    continue
                messages.append({KEY_ROLE: ROLE_ASSISTANT, KEY_CONTENT: bot_response})

        # Append context.user_message if it is not already the most recent
        # user-role message built from events.
        #
        # Two cases require the append:
        #   1. context.events is empty (first turn — no tracker history yet).
        #   2. The turns limit cut off the window before reaching the current
        #      UserUttered, so it was never added by the loop above.
        if last_user_content != context.user_message:
            messages.append({KEY_ROLE: ROLE_USER, KEY_CONTENT: context.user_message})

        return messages

    def _get_conversation_cache_key(
        self, context: AgentInput, turns: int
    ) -> Tuple[int, Tuple[Tuple[str, str], ...], str]:
        """Return a cache key for conversation messages built from utterances."""
        # Build a compact signature of the effective conversation context that
        # actually reaches the LLM:
        # - same event types (`UserUttered`, `BotUttered`),
        # - same text serialization rules for bot messages,
        # - same `turns` truncation behavior.
        #
        # This keeps cache behavior aligned with `build_messages_for_llm_request`:
        # if the visible utterance window changes in content/order, or the
        # fallback `context.user_message` changes, the key changes and we rebuild.
        utterance_signature: List[Tuple[str, str]] = []
        collected = 0
        for event in reversed(context.events):
            if not isinstance(event, (UserUttered, BotUttered)):
                continue
            if isinstance(event, UserUttered):
                if event.text:
                    utterance_signature.append((ROLE_USER, event.text))
            else:
                bot_response = serialize_bot_response_for_prompt(event)
                if bot_response:
                    utterance_signature.append((ROLE_ASSISTANT, bot_response))
            collected += 1
            if collected >= turns:
                break

        utterance_signature.reverse()
        return turns, tuple(utterance_signature), context.user_message

    def _build_system_prompt_cache_key(self, context: AgentInput) -> Optional[str]:
        """Return cache key for system prompt content.

        The base implementation always returns a JSON-serialized key built from
        mutable `AgentInput` fields. Subclasses may override this and return
        `None` to explicitly disable base-message cache reuse for scenarios where
        prompt inputs are non-deterministic in a single run.
        """
        # Cache is per `send_message` run; use only mutable prompt inputs from
        # `context` so key changes track in-loop state updates (e.g. SlotSet).
        key_payload = {
            "context": context.model_dump(exclude={"id", "timestamp", "events"}),
        }
        return json.dumps(key_payload, sort_keys=True, default=str)

    def _get_base_messages_cache_key(
        self, context: AgentInput, turns: int
    ) -> Optional[str]:
        """Return cache key for full base messages.

        Returns `None` only when a subclass override of
        `_build_system_prompt_cache_key` opts out of cache reuse.
        """
        # Cache the whole "base messages" payload (system + conversation) behind
        # one combined key:
        # - `system_key`: prompt/template context (slots/metadata/config/date-time),
        # - `conversation_key`: utterance-derived dialogue context.
        #
        # Any change in either part invalidates the cached base message list.
        # A subclass can return `None` from `_build_system_prompt_cache_key`
        # to force per-iteration rebuilds.
        system_key = self._build_system_prompt_cache_key(context)
        if system_key is None:
            return None

        key_payload = {
            "system_key": system_key,
            "conversation_key": self._get_conversation_cache_key(context, turns),
        }
        return json.dumps(key_payload, sort_keys=True, default=str)

    def _build_messages_for_llm_request_with_cache(
        self,
        context: AgentInput,
        cache_state: Dict[str, Any],
        turns: int = 10,
    ) -> List[Dict[str, str]]:
        """Build LLM messages with loop-scoped cache reuse."""
        cache_key = self._get_base_messages_cache_key(context, turns)
        cached_base_messages = cache_state.get(_MESSAGE_CACHE_BASE_MESSAGES)
        if (
            cache_key is not None
            and cache_state.get(_MESSAGE_CACHE_BASE_MESSAGES_KEY) == cache_key
            and cached_base_messages is not None
        ):
            return [dict(message) for message in cached_base_messages]

        # Keep customer override behavior intact: cache wraps the public
        # `build_messages_for_llm_request` hook instead of bypassing it.
        # On cache miss, we call the override and store the returned list.
        base_messages = self.build_messages_for_llm_request(context, turns)
        if cache_key is not None:
            cache_state[_MESSAGE_CACHE_BASE_MESSAGES_KEY] = cache_key
            cache_state[_MESSAGE_CACHE_BASE_MESSAGES] = [
                dict(message) for message in base_messages
            ]
        else:
            cache_state.pop(_MESSAGE_CACHE_BASE_MESSAGES_KEY, None)
            cache_state.pop(_MESSAGE_CACHE_BASE_MESSAGES, None)

        return [dict(message) for message in base_messages]

    def build_messages_for_llm_request(
        self, context: AgentInput, turns: int = 10
    ) -> List[Dict[str, str]]:
        """Build messages for the LLM request from conversation history.

        Filters to user and bot utterance events only, then limits to the most
        recent `turns` events. Note: here "turns" counts individual user/bot
        messages (utterance events), not full conversation turns (user+assistant
        exchanges).

        Args:
            context: Agent input with events and current user message.
            turns: Maximum number of user and bot utterance events to include
                in the context (default 10). Applied after filtering to
                utterance events only.

        Returns:
            List of message dicts with "role" and "content" for the LLM.
        """
        return [
            {KEY_ROLE: ROLE_SYSTEM, KEY_CONTENT: self.render_prompt_template(context)}
        ] + self._build_conversation_messages_for_llm_request(context, turns)

    def _get_assistant_message_with_tool_calls(
        self, llm_response: LLMResponse
    ) -> Dict[str, Any]:
        """Get assistant message with tool calls."""
        if not llm_response.tool_calls:
            return {}
        return {
            KEY_ROLE: ROLE_ASSISTANT,
            KEY_CONTENT: llm_response.choices[0],
            KEY_TOOL_CALLS: [
                {
                    KEY_ID: tool_call.id,
                    KEY_TYPE: tool_call.type,
                    KEY_FUNCTION: {
                        KEY_NAME: tool_call.tool_name,
                        KEY_ARGUMENTS: json.dumps(tool_call.tool_args),
                    },
                }
                for tool_call in llm_response.tool_calls
            ],
        }

    def _apply_slot_set_events_to_agent_input(
        self, agent_input: AgentInput, events: List[Event]
    ) -> Dict[str, Any]:
        """Apply SlotSet events to input slots and return changed values.

        For each `SlotSet(key, value)` in `events`, this method updates the matching
        slot in `agent_input.slots` by name, or appends a new slot when it does not
        exist yet.
        """
        updated_slots: Dict[str, Any] = {}
        if not events:
            return updated_slots

        slot_indices = {
            slot.name: index for index, slot in enumerate(agent_input.slots)
        }
        for event in events:
            if not isinstance(event, SlotSet):
                continue

            if event.key not in slot_indices:
                agent_input.slots.append(
                    AgentInputSlot(
                        name=event.key,
                        value=event.value,
                        type="any",
                        allowed_values=None,
                    )
                )
                slot_indices[event.key] = len(agent_input.slots) - 1

            updated_slots[event.key] = event.value
            agent_input.slots[slot_indices[event.key]].value = event.value

        return updated_slots

    def _get_tool_call_message(self, tool_response: AgentOutput) -> Dict[str, Any]:
        """Get the tool call message."""
        return {
            KEY_ROLE: ROLE_TOOL,
            KEY_TOOL_CALL_ID: tool_response.id,
            KEY_CONTENT: tool_response.response_message,
        }

    def _get_system_message_for_malformed_tool_response(self) -> Dict[str, Any]:
        """Get the system message for a malformed tool response."""
        system_message = (
            "The previous tool response contained invalid or incomplete JSON and could"
            " not be parsed. Retry by generating a tool response in STRICT JSON string "
            "format only. Ensure the JSON is fully well-formed and corresponds exactly "
            "to the user's last request."
        )
        return {
            KEY_ROLE: ROLE_SYSTEM,
            KEY_CONTENT: system_message,
        }

    def get_llm_tracing_metadata(self, agent_input: AgentInput) -> Dict[str, Any]:
        return {
            LANGFUSE_METADATA_SESSION_ID: agent_input.metadata.get(
                AGENT_METADATA_SENDER_ID_KEY
            ),
            LANGFUSE_METADATA_TAGS: [self.__class__.__name__],
            LANGFUSE_METADATA_CUSTOM_METADATA: {
                LANGFUSE_METADATA_AGENT_ID: agent_input.metadata.get(
                    AGENT_METADATA_AGENT_ID_KEY
                ),
                LANGFUSE_METADATA_MODEL_ID: agent_input.metadata.get(
                    AGENT_METADATA_MODEL_ID_KEY
                ),
                LANGFUSE_METADATA_COMPONENT_NAME: self.__class__.__name__,
                LANGFUSE_METADATA_REACT_SUB_AGENT_NAME: self._name,
            },
        }

    # ============================================================================
    # Tool Execution
    # ============================================================================

    def _get_meta_for_mcp_server(
        self, server_id: str, agent_input: Optional[AgentInput]
    ) -> Dict[str, Any]:
        """Get _meta dict for an MCP tool call from meta_map config and agent input."""
        meta_map = self._server_to_meta_map.get(server_id)
        if not meta_map:
            return {}

        if not agent_input or not meta_map.from_slots:
            return build_mcp_meta(meta_map, {})

        # Build the slots dictionary from the agent input.
        slots = {
            meta_map_entry.slot: get_slot_value_from_agent_input(
                agent_input, meta_map_entry.slot
            )
            for meta_map_entry in meta_map.from_slots
        }

        # Build the _meta dict from the meta_map config and the slots dict.
        return build_mcp_meta(meta_map, slots)

    async def _execute_mcp_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        agent_input: Optional[AgentInput] = None,
    ) -> AgentToolResult:
        """Execute a tool call via MCP servers."""
        if tool_name not in self._tool_to_server_mapper:
            return AgentToolResult(
                tool_name=tool_name,
                result=None,
                is_error=True,
                error_message=f"Tool `{tool_name}` not found in the server.",
            )

        server_id = self._tool_to_server_mapper[tool_name]
        connection = self._server_connections[server_id]
        try:
            session = await connection.ensure_active_session()
            meta = self._get_meta_for_mcp_server(server_id, agent_input)
            result = await call_tool_with_meta(
                session,
                tool_name,
                arguments,
                timedelta(seconds=self._tool_timeout),
                meta,
            )
            return AgentToolResult.from_mcp_tool_result(tool_name, result)
        except Exception as e:
            return AgentToolResult(
                tool_name=tool_name,
                result=None,
                is_error=True,
                error_message=(
                    f"Failed to execute tool `{tool_name}` via MCP server `{server_id}`"
                    f" @ `{connection.server_url}`: {e!s}"
                ),
            )

    async def _run_custom_tool(
        self, custom_tool: CustomToolSchema, arguments: Dict[str, Any]
    ) -> AgentToolResult:
        """Run a custom tool and return the result.

        Args:
            custom_tool: The custom tool schema containing the tool executor.
            arguments: The arguments to pass to the tool executor.

        Returns:
            The result of the tool execution as an AgentToolResult.
        """
        result = custom_tool.tool_executor(arguments)
        return await result if isawaitable(result) else result

    async def _execute_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        agent_input: Optional[AgentInput] = None,
    ) -> AgentToolResult:
        """Execute a tool call.

        This method first checks if the tool is a built-in tool. If it is, it executes
        the built-in tool. If it is not, it executes the tool via MCP servers.

        Args:
            tool_name: The name of the tool to execute.
            arguments: The arguments to pass to the tool.
            agent_input: Optional agent input for building MCP _meta from slots.

        Returns:
            The result of the tool execution as an AgentToolResult object.
        """
        try:
            for custom_tool in self._custom_tools:
                if custom_tool.tool_name == tool_name:
                    try:
                        with anyio.fail_after(self._tool_timeout):
                            return await self._run_custom_tool(custom_tool, arguments)

                    except TimeoutError:
                        return AgentToolResult(
                            tool_name=tool_name,
                            result=None,
                            is_error=True,
                            error_message=(
                                f"Built-in tool `{tool_name}` timed out after "
                                f"{self._tool_timeout} seconds."
                            ),
                        )
        except Exception as e:
            return AgentToolResult(
                tool_name=tool_name,
                result=None,
                is_error=True,
                error_message=f"Failed to execute built-in tool `{tool_name}`: {e!s}",
            )
        return await self._execute_mcp_tool(tool_name, arguments, agent_input)

    def _generate_agent_error_output(
        self,
        tool_output: AgentToolResult,
        agent_input: AgentInput,
        tool_call: LLMToolCall,
    ) -> AgentOutput:
        """Generate an agent error output."""
        structlogger.error(
            "mcp_agent.send_message.tool_execution_error",
            event_info=(
                f"Tool `{tool_output.tool_name}` returned an error: "
                f"{tool_output.error_message}"
            ),
            tool_name=tool_output.tool_name,
            tool_args=json.dumps(tool_call.tool_args),
        )
        if tool_output.is_error:
            return AgentOutput(
                id=agent_input.id,
                status=AgentStatus.FATAL_ERROR,
                error_message=tool_output.error_message,
            )
        else:
            return AgentOutput(
                id=agent_input.id,
                status=AgentStatus.RECOVERABLE_ERROR,
                error_message=tool_output.error_message,
            )

    def _get_structured_results_for_agent_output(
        self,
        agent_input: AgentInput,
        current_tool_results: Dict[str, AgentToolResult],
    ) -> List[List[Dict[str, Any]]]:
        """Get the tool results for the agent output."""
        structured_results_of_current_iteration: List[Dict[str, Any]] = []
        for tool_result in current_tool_results.values():
            structured_results_of_current_iteration.append(
                {"name": tool_result.tool_name, "result": tool_result.result}
            )

        previous_structured_results: List[List[Dict[str, Any]]] = (
            agent_input.metadata.get(AGENT_METADATA_STRUCTURED_RESULTS_KEY, []) or []
        )
        previous_structured_results.append(structured_results_of_current_iteration)

        return previous_structured_results

    # ============================================================================
    # Streaming Filler Message Methods
    # ============================================================================

    async def _send_filler_message(
        self,
        agent_input: AgentInput,
        filler_message_text: str,
        output_channel: Optional[OutputChannel],
        generated_events: List[Event],
    ) -> None:
        """Send a filler message to the user before tool execution.

        This method sends a brief filler message to the user via the output channel
        to reduce perceived latency while tools are being executed.

        Args:
            agent_input: The agent input containing user information.
            filler_message_text: The filler message text to send.
            output_channel: The output channel for sending messages.
            generated_events: List of events to append BotUttered event to.
        """
        if not output_channel:
            structlogger.debug(
                "mcp_agent.send_filler_message.no_output_channel",
                event_info="No output channel provided, skipping filler message",
                agent_name=self._name,
            )
            return

        if not filler_message_text or not filler_message_text.strip():
            structlogger.debug(
                "mcp_agent.send_filler_message.empty_message",
                event_info="Empty filler message text, skipping",
                agent_name=self._name,
            )
            return

        recipient_id = self._get_recipient_id(agent_input)
        if not recipient_id:
            structlogger.debug(
                "mcp_agent.send_filler_message.no_recipient_id",
                event_info="No recipient ID found, skipping filler message",
                agent_name=self._name,
            )
            return

        try:
            # Use streaming if the channel supports it, otherwise fall back
            # to send_text_message for reliability.
            if output_channel.supports_streaming:
                await self._stream_filler_message_chunks(
                    output_channel, recipient_id, filler_message_text
                )
            else:
                await output_channel.send_text_message(
                    recipient_id=recipient_id,
                    text=filler_message_text,
                )

            # Create BotUttered event for the filler message
            generated_events.append(
                BotUttered(
                    text=filler_message_text,
                    metadata=self._create_filler_message_metadata(agent_input),
                )
            )

            structlogger.debug(
                "mcp_agent.send_filler_message.sent",
                event_info="Filler message sent successfully",
                agent_name=self._name,
                recipient_id=recipient_id,
                filler_message_text=filler_message_text,
            )
        except Exception as e:
            structlogger.error(
                "mcp_agent.send_filler_message.error",
                event_info="Error sending filler message",
                agent_name=self._name,
                error=str(e),
            )

    async def _stream_filler_message_chunks(
        self,
        output_channel: OutputChannel,
        recipient_id: str,
        text: str,
    ) -> None:
        """Stream filler message text chunk by chunk.

        Args:
            output_channel: The output channel to stream to.
            recipient_id: The recipient ID.
            text: The text to stream.
        """
        await output_channel.send_response_chunk_start(recipient_id)

        for i in range(0, len(text), AGENT_FILLER_MESSAGE_STREAM_DEFAULT_CHUNK_SIZE):
            chunk = text[i : i + AGENT_FILLER_MESSAGE_STREAM_DEFAULT_CHUNK_SIZE]
            await output_channel.send_response_chunk(
                recipient_id=recipient_id,
                chunk=chunk,
            )

        await output_channel.send_response_chunk_end(recipient_id, is_intermediate=True)

    def _get_recipient_id(self, agent_input: AgentInput) -> Optional[str]:
        """Extract recipient ID from agent input.

        Args:
            agent_input: The agent input.

        Returns:
            The recipient ID or None if not found.
        """
        # First try the recipient_id from the agent input
        # If not found, try the sender_id from the metadata
        # If not found, return None
        return agent_input.recipient_id or agent_input.metadata.get(
            AGENT_METADATA_SENDER_ID_KEY
        )

    def _create_filler_message_metadata(
        self, agent_input: AgentInput
    ) -> Dict[str, Any]:
        """Create metadata for filler message BotUttered events.

        Args:
            agent_input: The agent input.

        Returns:
            Metadata dictionary for the BotUttered event.
        """
        return {
            "utter_source": self.__class__.__name__,
            "agent_name": self._name,
            "message_type": BOT_UTTERANCE_AGENT_MESSAGE_TYPE_FILLER_MESSAGE,
            AGENT_METADATA_AGENT_ID_KEY: agent_input.metadata.get(
                AGENT_METADATA_AGENT_ID_KEY
            ),
            AGENT_METADATA_MODEL_ID_KEY: agent_input.metadata.get(
                AGENT_METADATA_MODEL_ID_KEY
            ),
        }

    def _extract_filler_message_from_response(
        self, llm_response: LLMResponse
    ) -> Optional[str]:
        """Extract filler message text from LLM response.

        When the LLM returns both content and tool_calls, the content
        is typically the filler message.

        Args:
            llm_response: The LLM response.

        Returns:
            The filler message text or None if not present.
        """
        if not llm_response.choices:
            return None

        content = llm_response.choices[0]
        if content and content.strip():
            return content.strip()

        return None

    # ============================================================================
    # Core Protocol Methods
    # ============================================================================

    @abstractmethod
    async def send_message(
        self, agent_input: AgentInput, output_channel: Optional[OutputChannel] = None
    ) -> AgentOutput:
        """Send a message to the agent."""
        ...

    async def run(
        self, input: AgentInput, output_channel: Optional[OutputChannel] = None
    ) -> AgentOutput:
        """Send a message to Agent/server and return response."""
        return await self.send_message(input, output_channel)

    # ============================================================================
    # Message Processing
    # ============================================================================

    async def process_input(self, input: AgentInput) -> AgentInput:
        """Pre-process the input before sending it to the agent."""
        return input

    async def process_tool_output(
        self,
        current_iteration_tool_results: Dict[str, AgentToolResult],
        cumulative_tool_results: Dict[str, AgentToolResult],
        output_channel: Optional[OutputChannel] = None,
    ) -> List[Event]:
        """Post-process MCP tool results for the current LLM iteration.

        This method is called after an LLM iteration where at least one external tool
        call completed successfully.

        Args:
            current_iteration_tool_results: Mapping of tool call ID to tool result
                (`tool_call.id -> AgentToolResult`) for tool calls completed in the
                current iteration.
            cumulative_tool_results: Mapping of tool call ID to tool result
                (`tool_call.id -> AgentToolResult`) across all completed iterations in
                this run.
            output_channel: Channel that can be used by overrides to emit user-facing
                messages while deriving events. If an override emits an intermediate
                message through `output_channel`, it should also return a matching
                `BotUttered` event so the same message can be included in subsequent
                LLM-iteration context.

        Returns:
            A list of **new events for this iteration only**. The runtime appends these
            to the accumulated `AgentOutput.events` and also exposes them to subsequent
            LLM iterations.

        Override this in custom MCP agents to emit additional events based on
        tool execution.
        """
        return []

    async def _process_tool_output_or_raise(
        self,
        current_iteration_tool_results: Dict[str, AgentToolResult],
        cumulative_tool_results: Dict[str, AgentToolResult],
        output_channel: Optional[OutputChannel],
    ) -> List[Event]:
        """Process tool results for this iteration into events.

        Returns an empty list when no tool results are available.

        Raises:
            RuntimeError: If `process_tool_output` fails. Callers are expected
                to handle this in the agent loop and map it to `FATAL_ERROR`
                `AgentOutput`.
        """
        if not current_iteration_tool_results:
            return []

        structlogger.debug(
            "mcp_base_agent.process_tool_output.start",
            agent_name=self._name,
            num_current_iteration_tool_results=len(current_iteration_tool_results),
            num_cumulative_tool_results=len(cumulative_tool_results),
        )
        try:
            new_events = await self.process_tool_output(
                current_iteration_tool_results,
                cumulative_tool_results,
                output_channel,
            )
            structlogger.debug(
                "mcp_base_agent.process_tool_output.completed",
                agent_name=self._name,
                num_events=len(new_events),
            )
            return new_events
        except Exception as e:
            structlogger.error(
                "mcp_base_agent.process_tool_output.failed",
                agent_name=self._name,
                error=str(e),
            )
            raise RuntimeError(
                f"Failed to process MCP tool output for agent `{self._name}`: {e!s}"
            ) from e

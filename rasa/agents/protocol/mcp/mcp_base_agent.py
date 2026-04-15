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
    AGENT_FILLER_MESSAGES_ENABLED_DEFAULT,
    AGENT_METADATA_AGENT_ID_KEY,
    AGENT_METADATA_AGENT_RESPONSE_KEY,
    AGENT_METADATA_MODEL_ID_KEY,
    AGENT_METADATA_RESTARTED_KEY,
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
    AgentToolContext,
    AgentToolResult,
    AgentToolSchema,
    CustomToolSchema,
)
from rasa.agents.utils import get_slot_value_from_agent_input
from rasa.core.available_agents import AgentConfig, AgentMCPServerConfig, ProtocolConfig
from rasa.core.channels import OutputChannel
from rasa.core.constants import (
    BOT_UTTERANCE_AGENT_MESSAGE_TYPE_FINAL_RESPONSE,
    BOT_UTTERANCE_AGENT_MESSAGE_TYPE_INPUT_REQUIRED,
    BOT_UTTERANCE_AGENT_MESSAGE_TYPE_KEY,
    BOT_UTTERANCE_AGENT_NAME_KEY,
    UTTER_SOURCE_METADATA_KEY,
)
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
from rasa.shared.core.events import (
    AgentCompleted,
    AgentStarted,
    BotUttered,
    Event,
    McpToolExecuted,
    SlotSet,
    UserUttered,
)
from rasa.shared.exceptions import (
    AgentInitializationException,
    AuthenticationError,
    LLMToolResponseDecodeError,
    ProviderClientAPIException,
)
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
    DEFAULT_OPENAI_CHAT_MODEL_NAME,
    DEFAULT_OPENAI_TEMPERATURE,
    REASONING_EFFORT_CONFIG_KEY,
    REASONING_EFFORT_NONE,
    acompletion_with_streaming,
    get_prompt_template,
    invoke_llm_and_send_non_streaming_response,
    llm_factory,
    resolve_model_client_config,
    serialize_bot_response_for_prompt,
)
from rasa.shared.utils.mcp.server_connection import MCPServerConnection
from rasa.shared.utils.mcp.utils import build_mcp_meta, call_tool_with_meta

# Marker text for "previous run (completed)" in message content when agent restarted.
# Must match the instruction text in MCP prompt templates.
PREVIOUS_RUN_MARKER = "--- Previous run (completed). ---"
END_PREVIOUS_RUN_MARKER = "--- End of previous run ---"

DEFAULT_OPENAI_MAX_GENERATED_TOKENS = 256
DEFAULT_LLM_CONFIG = {
    PROVIDER_CONFIG_KEY: OPENAI_PROVIDER,
    MODEL_CONFIG_KEY: DEFAULT_OPENAI_CHAT_MODEL_NAME,
    REASONING_EFFORT_CONFIG_KEY: REASONING_EFFORT_NONE,
    TEMPERATURE_CONFIG_KEY: DEFAULT_OPENAI_TEMPERATURE,
    MAX_COMPLETION_TOKENS_CONFIG_KEY: DEFAULT_OPENAI_MAX_GENERATED_TOKENS,
    TIMEOUT_CONFIG_KEY: 7,
}

if TYPE_CHECKING:
    from rasa.agents.core.cancellation import CancellationToken
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
        - "tool_executor" → a coroutine method (args, context) that performs the
          tool's action. Receives tool arguments and AgentToolContext (with
          context.metadata) as separate parameters.

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

            Tool executor signature: (args: Dict[str, Any], context: AgentToolContext)
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
        context_dict = context.model_dump(
            exclude={"id", "timestamp", "events", "metadata"}
        )
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
        context_dict["restarted"] = bool(metadata.get(AGENT_METADATA_RESTARTED_KEY))

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

    @staticmethod
    def _completed_run_region(
        events: List[Event], agent_id: str
    ) -> Tuple[Optional[int], Optional[int]]:
        """Return (start_idx, end_idx) for the last completed run of this agent.

        The region is from the AgentStarted(agent_id) that started the run, up to
        and including the AgentCompleted(agent_id) that ended it. We find the last
        AgentCompleted, then the earliest AgentStarted(agent_id) with no
        AgentCompleted(agent_id) between it and that end.
        """
        # Find end_idx: last AgentCompleted(agent_id) in the list.
        end_idx: Optional[int] = None
        for i in range(len(events) - 1, -1, -1):
            ev = events[i]
            if isinstance(ev, AgentCompleted) and ev.agent_id == agent_id:
                end_idx = i
                break
        if end_idx is None:
            return (None, None)
        # Find start_idx: the earliest AgentStarted(agent_id) with no
        # AgentCompleted(agent_id) between it and end_idx (the run that contains
        # the conversation).
        start_idx: Optional[int] = None
        for j in range(0, end_idx):
            ev = events[j]
            if not (isinstance(ev, AgentStarted) and ev.agent_id == agent_id):
                continue
            if any(
                isinstance(evk := events[k], AgentCompleted)
                and evk.agent_id == agent_id
                for k in range(j + 1, end_idx)
            ):
                continue
            start_idx = j
            break
        return (start_idx, end_idx)

    def build_messages_for_llm_request(
        self, context: AgentInput, turns: int = 10
    ) -> List[Dict[str, str]]:
        """Build messages for the LLM request from conversation history.

        Filters to user and bot utterance events only, then limits to the most
        recent `turns` events. Note: here "turns" counts individual user/bot
        messages (utterance events), not full conversation turns (user+assistant
        exchanges).

        When the agent was restarted, messages that fall in the last completed
        run are prefixed/suffixed with markers so the model does not reuse them.

        Args:
            context: Agent input with events and current user message.
            turns: Maximum number of user and bot utterance events to include
                in the context (default 10). Applied after filtering to
                utterance events only.

        Returns:
            List of message dicts with "role" and "content" for the LLM.
        """
        system_content = self.render_prompt_template(context)
        messages = [{KEY_ROLE: ROLE_SYSTEM, KEY_CONTENT: system_content}]

        is_restarted = bool(
            context.metadata.get(AGENT_METADATA_RESTARTED_KEY) and context.id
        )
        if is_restarted:
            conversation, last_user_content = (
                self._build_conversation_messages_after_restart(context, turns)
            )
        else:
            conversation, last_user_content = self._build_conversation_messages(
                context, turns
            )

        messages.extend(conversation)
        # Append current user message if it was not already in the last N events
        # (e.g. first turn or the turns window did not include the latest utterance).
        if last_user_content != context.user_message:
            messages.append({KEY_ROLE: ROLE_USER, KEY_CONTENT: context.user_message})
        return messages

    def create_bot_uttered_for_streamed_content(
        self,
        text: str,
        agent_input: AgentInput,
        message_type: Optional[str] = BOT_UTTERANCE_AGENT_MESSAGE_TYPE_FINAL_RESPONSE,
    ) -> BotUttered:
        """Create a BotUttered event for content streamed directly to the channel.

        Attaches standard agent metadata so the event is correctly attributed
        in the tracker and downstream analytics.

        Args:
            text: The streamed text content.
            agent_input: The current agent input (used to read agent/model IDs).
            message_type: The agent message type to set in the event metadata.

        Returns:
            A BotUttered event with agent attribution metadata.
        """
        return BotUttered(
            text=text,
            metadata={
                UTTER_SOURCE_METADATA_KEY: self.__class__.__name__,
                BOT_UTTERANCE_AGENT_NAME_KEY: self._name,
                BOT_UTTERANCE_AGENT_MESSAGE_TYPE_KEY: message_type,
                AGENT_METADATA_AGENT_ID_KEY: agent_input.metadata.get(
                    AGENT_METADATA_AGENT_ID_KEY
                ),
                AGENT_METADATA_MODEL_ID_KEY: agent_input.metadata.get(
                    AGENT_METADATA_MODEL_ID_KEY
                ),
            },
        )

    def _build_conversation_messages(
        self, context: AgentInput, turns: int
    ) -> Tuple[List[Dict[str, str]], Optional[str]]:
        """Build user/assistant messages from the last `turns` utterance events.

        Returns:
            (list of message dicts, last user message text or None).
        """
        utterance_events = [
            e for e in context.events if isinstance(e, (UserUttered, BotUttered))
        ]
        utterance_events = utterance_events[-turns:] if utterance_events else []

        messages: List[Dict[str, str]] = []
        # Caller uses this to decide whether to append context.user_message.
        last_user_content: Optional[str] = None
        for event in utterance_events:
            content = self._content_from_utterance_event(event)
            if content is None:
                continue
            if isinstance(event, UserUttered):
                last_user_content = event.text
            messages.append(
                {
                    KEY_ROLE: ROLE_USER
                    if isinstance(event, UserUttered)
                    else ROLE_ASSISTANT,
                    KEY_CONTENT: content,
                }
            )
        return (messages, last_user_content)

    def _build_conversation_messages_after_restart(
        self, context: AgentInput, turns: int
    ) -> Tuple[List[Dict[str, str]], Optional[str]]:
        """Build user/assistant messages with restart markers around the last run.

        Messages that fall in the last completed run (AgentStarted..AgentCompleted)
        get PREVIOUS_RUN_MARKER / END_PREVIOUS_RUN_MARKER so the model does not
        reuse them. Returns (list of message dicts, last user message text or None).
        """
        start_idx, end_idx = self._completed_run_region(context.events, context.id)
        utterance_entries = self._utterance_entries_with_restart_flags(
            context.events, start_idx, end_idx
        )
        utterance_entries = utterance_entries[-turns:] if utterance_entries else []

        # Last message in the completed run gets END_PREVIOUS_RUN_MARKER.
        last_completed_index = self._last_index_in_completed_run(utterance_entries)

        messages: List[Dict[str, str]] = []
        last_user_content: Optional[str] = None
        for i, (event, in_completed_run) in enumerate(utterance_entries):
            content = self._content_from_utterance_event(event)
            if content is None:
                continue
            if isinstance(event, UserUttered):
                last_user_content = event.text

            if in_completed_run:
                # First message in the window that belongs to the completed run.
                is_first_in_run = not any(utterance_entries[j][1] for j in range(i))
                if is_first_in_run:
                    content = f"{PREVIOUS_RUN_MARKER}\n{content}"
                if last_completed_index is not None and i == last_completed_index:
                    content = f"{content}\n{END_PREVIOUS_RUN_MARKER}"

            messages.append(
                {
                    KEY_ROLE: ROLE_USER
                    if isinstance(event, UserUttered)
                    else ROLE_ASSISTANT,
                    KEY_CONTENT: content,
                }
            )
        return (messages, last_user_content)

    def _content_from_utterance_event(self, event: Event) -> Optional[str]:
        """Return display content for an utterance event, or None to skip.

        Returns None for empty user text or bot responses that serialize to empty,
        so the caller omits them from the message list.
        """
        if isinstance(event, UserUttered):
            return event.text or None
        if isinstance(event, BotUttered):
            return serialize_bot_response_for_prompt(event) or None
        return None

    def _utterance_entries_with_restart_flags(
        self,
        events: List[Event],
        start_idx: Optional[int],
        end_idx: Optional[int],
    ) -> List[Tuple[Event, bool]]:
        """Pair each utterance event with whether it lies in the completed run.

        start_idx/end_idx are the last AgentStarted and next AgentCompleted
        indices for this agent. The bool is True when the event index is in
        [start_idx, end_idx], so we can add markers only around that run.
        """
        result: List[Tuple[Event, bool]] = []
        for idx, event in enumerate(events):
            if not isinstance(event, (UserUttered, BotUttered)):
                continue
            in_run = (
                start_idx is not None
                and end_idx is not None
                and start_idx <= idx <= end_idx
            )
            result.append((event, in_run))
        return result

    def _last_index_in_completed_run(
        self,
        utterance_entries: List[Tuple[Any, bool]],
    ) -> Optional[int]:
        """Index of the last entry with in_completed_run=True, or None.

        Used to append END_PREVIOUS_RUN_MARKER only to the final message of
        the completed run, so the model sees a clear end to the marked section.
        """
        for i in range(len(utterance_entries) - 1, -1, -1):
            if utterance_entries[i][1]:
                return i
        return None

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
            "context": context.model_dump(
                exclude={"id", "timestamp", "events", "metadata"}
            ),
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
        strip_original_system_prompt: bool = False,
        turns: int = 10,
    ) -> List[Dict[str, str]]:
        """Build LLM messages with loop-scoped cache reuse.

        When strip_original_system_prompt is True (e.g. after task_completed was
        called or when retrying after empty content at task completion), the
        first message (original system prompt) is omitted so the model sees
        only the conversation and any later system message (e.g. retry instruction).
        """
        cache_key = self._get_base_messages_cache_key(context, turns)
        cached_base_messages = cache_state.get(_MESSAGE_CACHE_BASE_MESSAGES)
        if (
            cache_key is not None
            and cache_state.get(_MESSAGE_CACHE_BASE_MESSAGES_KEY) == cache_key
            and cached_base_messages is not None
        ):
            base = cached_base_messages
            if strip_original_system_prompt and base:
                return [dict(m) for m in base[1:]]
            return [dict(message) for message in base]

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

        if strip_original_system_prompt and base_messages:
            return [dict(message) for message in base_messages[1:]]
        return [dict(message) for message in base_messages]

    def _get_assistant_message_with_tool_calls(
        self, llm_response: LLMResponse
    ) -> Dict[str, Any]:
        """Get assistant message with tool calls."""
        if not llm_response.tool_calls:
            return {}
        # When the LLM returns only tool calls (no content), choices can be empty
        # (e.g. from streaming with no text deltas).
        content = (
            llm_response.choices[0] if llm_response and llm_response.choices else None
        )
        return {
            KEY_ROLE: ROLE_ASSISTANT,
            KEY_CONTENT: content,
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

    async def generate_and_send_response(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        metadata: Dict[str, Any],
        agent_input: AgentInput,
        output_channel: Optional[OutputChannel],
    ) -> Tuple[Optional[LLMResponse], Optional[BotUttered]]:
        """Obtain the LLM response and optionally deliver it to the output channel.

        Routes to one of two paths:
        - Streaming channel     → ``acompletion_with_streaming`` (real-time chunks)
        - Non-streaming channel → ``invoke_llm_and_send_non_streaming_response``
          (single message)

        Args:
            messages: The conversation messages to send to the LLM.
            tools: Available tools in OpenAI JSON format.
            metadata: Tracing / metadata dict passed through to the LLM client.
            agent_input: The current agent input (used to read agent/model IDs).
            output_channel: Channel to deliver content to.

        Returns:
            A single ``LLMResponse`` with:
            - ``choices``: the fully accumulated content string (or ``[]`` if none).
            - ``tool_calls``: the assembled tool calls (or ``None`` if none).
            and a ``BotUttered`` event if the LLM response has content.

        Raises:
            ValueError: If no output channel or recipient ID is provided.
        """
        bot_uttered: Optional[BotUttered] = None
        llm_response: Optional[LLMResponse] = None
        recipient_id = self._get_recipient_id(agent_input)
        if not output_channel or not recipient_id:
            structlogger.debug(
                "mcp_agent.generate_and_send_response.no_channel",
                event_info=(
                    "No output channel or recipient ID provided; "
                    "calling LLM without channel delivery."
                ),
                agent_name=self._name,
            )
            raise ValueError("No output channel or recipient ID provided.")

        if output_channel.supports_streaming:
            structlogger.debug(
                "mcp_base_agent.generate_and_send_response.streaming",
                event_info="Sending message to LLM with streaming support.",
                agent_name=self._name,
                agent_id=str(make_agent_identifier(self._name, self.protocol_type)),
            )
            llm_response = await acompletion_with_streaming(
                self.llm_client,
                messages,
                output_channel,
                recipient_id,
                tools=tools,
                metadata=metadata,
            )
        else:
            llm_response = await invoke_llm_and_send_non_streaming_response(
                self.llm_client, output_channel, recipient_id, messages, tools, metadata
            )

        llm_content = (
            llm_response.choices[0] if llm_response and llm_response.choices else None
        )
        if llm_content is not None and isinstance(llm_content, str):
            llm_content = llm_content.strip()
        if llm_content:
            bot_uttered = self.create_bot_uttered_for_streamed_content(
                llm_content, agent_input
            )
        structlogger.debug(
            "mcp_base_agent.generate_and_send_response.response",
            agent_name=self._name,
            agent_id=str(make_agent_identifier(self._name, self.protocol_type)),
            bot_uttered=bot_uttered,
            llm_response=llm_response.to_dict() if llm_response else None,
        )
        return llm_response, bot_uttered

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
        self,
        custom_tool: CustomToolSchema,
        arguments: Dict[str, Any],
        agent_input: Optional[AgentInput] = None,
    ) -> AgentToolResult:
        """Run a custom tool and return the result.

        Args:
            custom_tool: The custom tool schema containing the tool executor.
            arguments: The arguments from the LLM to pass to the tool executor.
            agent_input: Optional agent input.

        Returns:
            The result of the tool execution as an AgentToolResult.
        """
        executor_args = dict(arguments)
        context = AgentToolContext(
            metadata=(
                agent_input.metadata
                if agent_input and agent_input.metadata is not None
                else {}
            )
        )
        result = custom_tool.tool_executor(executor_args, context)
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
                            return await self._run_custom_tool(
                                custom_tool, arguments, agent_input
                            )

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

    # ============================================================================
    # Output Creation Helpers
    # ============================================================================

    def _is_filler_bot_utterance(
        self,
        llm_response: LLMResponse,
        bot_uttered: Optional[BotUttered],
    ) -> bool:
        """Whether streamed assistant text should be recorded as a filler message."""
        return (
            bot_uttered is not None
            and self._enable_filler_messages
            and bool(llm_response.tool_calls)
        )

    def _record_input_required_bot_uttered(
        self,
        bot_uttered: Optional[BotUttered],
        generated_events: List[Event],
    ) -> None:
        """Mark `bot_uttered` as input-required and append it."""
        if bot_uttered:
            bot_uttered.metadata[BOT_UTTERANCE_AGENT_MESSAGE_TYPE_KEY] = (
                BOT_UTTERANCE_AGENT_MESSAGE_TYPE_INPUT_REQUIRED
            )
            generated_events.append(bot_uttered)

    def _record_filler_bot_uttered(
        self,
        bot_uttered: BotUttered,
        generated_events: List[Event],
    ) -> None:
        """Mark `bot_uttered` as filler and append it when enabled."""
        bot_uttered.metadata[BOT_UTTERANCE_AGENT_MESSAGE_TYPE_KEY] = (
            BOT_UTTERANCE_AGENT_MESSAGE_TYPE_FILLER_MESSAGE
        )
        generated_events.append(bot_uttered)

    def _create_recoverable_error_output(
        self,
        agent_input: AgentInput,
        error_message: str,
        *,
        generated_events: Optional[List[Event]] = None,
        tool_results: Optional[Dict[str, AgentToolResult]] = None,
        log_event_name: Optional[str] = None,
        **log_kwargs: Any,
    ) -> AgentOutput:
        """Create an AgentOutput for a recoverable error (e.g. no LLM response).

        Logs a warning with log_event_name, event_info=error_message, agent
        context, and any extra log_kwargs before returning the output.
        """
        structlogger.warning(
            log_event_name,
            event_info=error_message,
            agent_name=self._name,
            agent_id=str(make_agent_identifier(self._name, self.protocol_type)),
            **log_kwargs,
        )
        return AgentOutput(
            id=agent_input.id,
            status=AgentStatus.RECOVERABLE_ERROR,
            error_message=error_message,
            events=generated_events if generated_events else None,
            structured_results=self._get_structured_results_for_agent_output(
                agent_input, tool_results or {}
            ),
        )

    def _create_input_required_output(
        self,
        agent_input: AgentInput,
        response_message: Optional[str],
        output_channel: Optional[OutputChannel],
        events: Optional[List[Event]],
        tool_results: Optional[Dict[str, AgentToolResult]] = None,
    ) -> AgentOutput:
        """Create an AgentOutput for content-only response (INPUT_REQUIRED).

        When output_channel is set, response_message is omitted to prevent the
        downstream pipeline from re-sending the same message as a duplicate
        (content was already sent directly to the output channel).
        """
        return AgentOutput(
            id=agent_input.id,
            status=AgentStatus.INPUT_REQUIRED,
            response_message=None if output_channel else response_message,
            events=events,
            structured_results=self._get_structured_results_for_agent_output(
                agent_input, tool_results or {}
            ),
        )

    def _create_fatal_error_output(
        self,
        agent_input: AgentInput,
        error_message: str,
        log_event_name: Optional[str] = None,
        *,
        events: Optional[List[Event]] = None,
        tool_results: Optional[Dict[str, AgentToolResult]] = None,
        **log_kwargs: Any,
    ) -> AgentOutput:
        """Create an AgentOutput for a fatal error in the agent loop.

        Logs a structlogger.error with log_event_name, event_info=error_message,
        agent_name, agent_id, and any extra log_kwargs before returning the output.
        """
        structlogger.error(
            log_event_name,
            event_info=error_message,
            agent_name=self._name,
            agent_id=str(make_agent_identifier(self._name, self.protocol_type)),
            **log_kwargs,
        )
        return AgentOutput(
            id=agent_input.id,
            status=AgentStatus.FATAL_ERROR,
            response_message=f"I encountered an error: {error_message!s}",
            events=events,
            structured_results=self._get_structured_results_for_agent_output(
                agent_input, tool_results or {}
            ),
            error_message=error_message,
        )

    def _create_max_iterations_reached_output(
        self,
        agent_input: AgentInput,
        *,
        events: Optional[List[Event]] = None,
        tool_results: Optional[Dict[str, AgentToolResult]] = None,
    ) -> AgentOutput:
        """Create an AgentOutput when max iterations reached without completion."""
        return AgentOutput(
            id=agent_input.id,
            status=AgentStatus.COMPLETED,
            response_message=(
                "I've completed my research but couldn't provide a final answer within "
                "the allowed steps."
            ),
            events=events,
            structured_results=self._get_structured_results_for_agent_output(
                agent_input, tool_results or {}
            ),
        )

    # ============================================================================
    # Tool Output Handling
    # ============================================================================

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

    def _append_tool_result_message(
        self,
        tool_call_messages: List[Dict[str, Any]],
        tool_call_id: str,
        content: str,
    ) -> None:
        """Append a tool result message to the conversation messages."""
        tool_call_messages.append(
            {
                KEY_ROLE: ROLE_TOOL,
                KEY_TOOL_CALL_ID: tool_call_id,
                KEY_CONTENT: content,
            }
        )

    async def _process_tool_call(
        self,
        tool_call: LLMToolCall,
        agent_input: AgentInput,
        tool_call_messages: List[Dict[str, Any]],
        tool_results: Dict[str, AgentToolResult],
        current_iteration_tool_results: Dict[str, AgentToolResult],
        log_event_name: str,
        *,
        events: Optional[List[Event]] = None,
    ) -> Optional[AgentOutput]:
        """Execute a tool call and append its result to the conversation.

        Returns an AgentOutput on tool failure (caller should return it);
        returns None to continue.
        """
        tool_output = await self._execute_tool_call(
            tool_call.tool_name,
            tool_call.tool_args,
            agent_input=agent_input,
        )

        structlogger.debug(
            log_event_name,
            event_info=f"Tool output for tool call {tool_call.tool_name}",
            tool_output=tool_output.model_dump(),
            json_formatting=["tool_output"],
            tool_name=tool_call.tool_name,
            agent_name=self._name,
            agent_id=str(make_agent_identifier(self._name, self.protocol_type)),
        )

        # If the tool call failed, generate an agent error output.
        if tool_output.is_error or tool_output.result is None:
            agent_id = str(make_agent_identifier(self._name, self.protocol_type))
            error_event = McpToolExecuted(
                tool_name=tool_call.tool_name,
                arguments=tool_call.tool_args,
                result=None,
                is_error=True,
                error_message=tool_output.error_message,
                metadata={"agent_id": agent_id},
            )
            if events is not None:
                events.append(error_event)
            else:
                events = [error_event]

            log_event_name = "mcp_agent.send_message.tool_execution_error"
            log_kwargs = {
                "tool_name": tool_output.tool_name,
                "tool_args": json.dumps(tool_call.tool_args),
            }
            if tool_output.is_error:
                return self._create_fatal_error_output(
                    agent_input,
                    tool_output.error_message,
                    log_event_name=log_event_name,
                    events=events,
                    tool_results=tool_results,
                    **log_kwargs,
                )
            return self._create_recoverable_error_output(
                agent_input,
                tool_output.error_message,
                log_event_name=log_event_name,
                generated_events=events,
                tool_results=tool_results,
                **log_kwargs,
            )

        # Store the tool output in the tool_results.
        tool_results[tool_call.id] = tool_output
        current_iteration_tool_results[tool_call.id] = tool_output

        # Add the tool call message to the messages.
        self._append_tool_result_message(
            tool_call_messages, tool_call.id, tool_output.result
        )
        return None

    def _is_malformed_tool_response_exception(self, e: Exception) -> bool:
        """Return True if the exception is a malformed tool response (retryable)."""
        return (
            isinstance(e, ProviderClientAPIException)
            and isinstance(e.original_exception, LLMToolResponseDecodeError)
        ) or isinstance(e, LLMToolResponseDecodeError)

    def _append_malformed_tool_response_system_message(
        self,
        messages: List[Dict[str, Any]],
        agent_input: AgentInput,
        exception: Exception,
        component_name: str,
    ) -> None:
        """Log the malformed tool response and append the system message for retry."""
        log_event_name = f"{component_name}.send_message.malformed_tool_response_error"
        original_exception_str = (
            str(exception.original_exception)
            if isinstance(exception, ProviderClientAPIException)
            else str(exception)
        )
        structlogger.debug(
            log_event_name,
            event_info=(
                "Malformed tool response received from LLM "
                "(JSON decode error). Retrying the LLM call."
            ),
            user_message=agent_input.user_message,
            agent_name=self._name,
            agent_id=str(make_agent_identifier(self._name, self.protocol_type)),
            original_exception=original_exception_str,
        )
        messages.append(self._get_system_message_for_malformed_tool_response())

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
    # Core Protocol Methods
    # ============================================================================

    @abstractmethod
    async def send_message(
        self, agent_input: AgentInput, output_channel: Optional[OutputChannel] = None
    ) -> AgentOutput:
        """Send a message to the agent."""
        ...

    async def run(
        self,
        input: AgentInput,
        output_channel: Optional[OutputChannel] = None,
        cancellation_token: Optional["CancellationToken"] = None,
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

    def _get_mcp_tool_executed_events(
        self,
        tool_calls: List[LLMToolCall],
        current_iteration_tool_results: Dict[str, AgentToolResult],
    ) -> List[Event]:
        """Return inspector events for this iteration's executed MCP tools."""
        agent_id = str(make_agent_identifier(self._name, self.protocol_type))
        events: List[Event] = []
        for tool_call in tool_calls:
            if tool_call.id not in current_iteration_tool_results:
                continue

            tool_result = current_iteration_tool_results[tool_call.id]
            events.append(
                McpToolExecuted(
                    tool_name=tool_call.tool_name,
                    arguments=tool_call.tool_args,
                    result=tool_result.result,
                    is_error=tool_result.is_error,
                    error_message=tool_result.error_message,
                    metadata={"agent_id": agent_id},
                )
            )
        return events

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

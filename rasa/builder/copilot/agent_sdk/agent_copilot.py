"""Main Agents SDK copilot implementation.

This copilot uses OpenAI's Agents SDK with MCP (Model Context Protocol) integration
while maintaining full compatibility with the existing copilot interface.
"""

import asyncio
import importlib.resources
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Tuple

import structlog
from agents import Agent, ModelSettings, Runner, StreamEvent
from jinja2 import Template

from rasa.builder import config
from rasa.builder.copilot.agent_sdk.hooks import RasaCopilotHooks
from rasa.builder.copilot.agent_sdk.planning_tools import PLANNING_TOOLS
from rasa.builder.copilot.base_copilot import BaseCopilot
from rasa.builder.copilot.constants import (
    COPILOT_LAST_USER_MESSAGE_CONTEXT_PROMPT_FILE_AGENT_SDK,
    COPILOT_PROMPTS_DIR,
    COPILOT_PROMPTS_FILE_AGENT_SDK,
    COPILOT_TRAINING_ERROR_HANDLER_PROMPT_FILE_AGENT_SDK,
)
from rasa.builder.copilot.models import (
    CopilotContext,
    CopilotGenerationContext,
    EventContent,
    FileContent,
    InternalCopilotRequestChatMessage,
    MCPToolCall,
    TodoPlanUpdate,
    UsageStatistics,
    UserChatMessage,
)
from rasa.builder.copilot.response_handling.agent_copilot_response_handler import (
    AgentCopilotResponseHandler,
)
from rasa.builder.copilot.response_handling.utils import is_response_completed_event
from rasa.builder.telemetry.langfuse.agent_copilot_langfuse_telemetry import (
    AgentCopilotLangfuseTelemetry,
)
from rasa.builder.telemetry.langfuse.traced_mcp_server import (
    TracedMCPServerWrapper,
)
from rasa.shared.constants import PACKAGE_NAME

structlogger = structlog.get_logger()


class AgentCopilot(BaseCopilot):
    """Agents SDK-based copilot implementation with MCP integration."""

    def __init__(self) -> None:
        """Initialize the Agent SDK copilot."""
        # Load system prompt template (Agent SDK optimized version)
        self._system_message_prompt_template = Template(
            importlib.resources.read_text(
                f"{PACKAGE_NAME}.{COPILOT_PROMPTS_DIR}",
                COPILOT_PROMPTS_FILE_AGENT_SDK,
            )
        )

        self._last_user_message_context_prompt_template = Template(
            importlib.resources.read_text(
                f"{PACKAGE_NAME}.{COPILOT_PROMPTS_DIR}",
                COPILOT_LAST_USER_MESSAGE_CONTEXT_PROMPT_FILE_AGENT_SDK,
            )
        )

        self._training_error_handler_prompt_template = Template(
            importlib.resources.read_text(
                f"{PACKAGE_NAME}.{COPILOT_PROMPTS_DIR}",
                COPILOT_TRAINING_ERROR_HANDLER_PROMPT_FILE_AGENT_SDK,
            )
        )

        # Usage statistics tracking
        self._usage_statistics = UsageStatistics(
            input_token_price=config.COPILOT_INPUT_TOKEN_PRICE,
            output_token_price=config.COPILOT_OUTPUT_TOKEN_PRICE,
            cached_token_price=config.COPILOT_CACHED_TOKEN_PRICE,
        )

        # Queue for MCP tool call events from hooks
        self._mcp_tool_queue: asyncio.Queue[MCPToolCall] = asyncio.Queue()

        # Queue for task planning updates from function tools
        self._plan_queue: asyncio.Queue[TodoPlanUpdate] = asyncio.Queue()

    @property
    def llm_config(self) -> Dict[str, Any]:
        """The LLM config used to generate the response."""
        return {
            "model": config.OPENAI_MODEL,
            "temperature": config.OPENAI_TEMPERATURE,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

    @property
    def usage_statistics(self) -> UsageStatistics:
        """Get usage statistics for the copilot."""
        return self._usage_statistics

    @asynccontextmanager
    async def _create_mcp_server(self) -> AsyncGenerator[TracedMCPServerWrapper, None]:
        """Context manager to create and manage MCP server connection.

        Yields:
            Connected TracedMCPServerWrapper instance with Langfuse tracing
        """
        mcp_url = f"http://{config.MCP_SERVER_HOST}:{config.MCP_SERVER_PORT}/mcp"
        structlogger.info(
            "agent_sdk.mcp_client.connecting",
            event_info="Creating traced MCP client connection",
            mcp_url=mcp_url,
            timeout=config.MCP_TOOL_CALL_TIMEOUT,
        )

        try:
            async with TracedMCPServerWrapper(
                name="Rasa MCP Server",
                params={
                    "url": mcp_url,
                    "timeout": config.MCP_TOOL_CALL_TIMEOUT,
                },
                client_session_timeout_seconds=120,
                cache_tools_list=True,
                max_retry_attempts=config.MCP_MAX_RETRY_ATTEMPTS,
            ) as server:
                structlogger.info(
                    "agent_sdk.mcp_client.connected",
                    event_info="Traced MCP client connected successfully",
                )
                yield server
                structlogger.info(
                    "agent_sdk.mcp_client.disconnecting",
                    event_info="Closing traced MCP client connection",
                )
        except Exception as e:
            structlogger.error(
                "agent_sdk.mcp_client.connection_error",
                event_info="Failed to connect to MCP server",
                error=str(e),
                mcp_url=mcp_url,
            )
            raise
        finally:
            structlogger.info(
                "agent_sdk.mcp_client.disconnected",
                event_info="Traced MCP client connection closed or errored",
            )

    @asynccontextmanager
    async def _create_agent(
        self, system_instructions: str
    ) -> AsyncGenerator[Agent, None]:
        """Context manager to yield an Agents SDK agent instance.

        Usage:
            async with self._create_agent(system_instructions) as agent:
                ...

        Args:
            system_instructions: The system prompt for the agent

        Yields:
            Configured Agent instance
        """
        model_settings = ModelSettings(
            temperature=config.OPENAI_TEMPERATURE,
        )

        # Create agent with hooks to track MCP tool calls
        # Include both MCP server tools and local planning tools
        async with self._create_mcp_server() as server:
            yield Agent(
                name="Rasa Copilot",
                instructions=system_instructions,
                model=config.OPENAI_MODEL,
                model_settings=model_settings,
                mcp_servers=[server],
                tools=PLANNING_TOOLS,  # Local function tools for task planning
                hooks=RasaCopilotHooks(self._mcp_tool_queue),
            )

    def _get_last_user_message(self, context: CopilotContext) -> str:
        """Get the last user message from the context."""
        if not context.copilot_chat_history:
            return ""
        last_message = context.copilot_chat_history[-1]
        if hasattr(last_message, "get_flattened_text_content"):
            return last_message.get_flattened_text_content()
        return ""

    def _get_tracker_event_attachments(
        self, context: CopilotContext
    ) -> List[EventContent]:
        """Get the tracker event attachments from the context."""
        if not context.copilot_chat_history:
            return []
        return self._extract_tracker_event_attachments(context.copilot_chat_history[-1])

    async def generate_response(
        self,
        context: CopilotContext,
    ) -> Tuple[AgentCopilotResponseHandler, CopilotGenerationContext]:
        """Generate a response from the copilot.

        This method matches the signature of the existing copilot's generate_response
        to maintain API compatibility.

        Args:
            context: The context of the copilot

        Returns:
            A tuple containing the async response stream and a
            CopilotGenerationContext object with relevant documents and messages
        """
        # Reset event translator for new response
        self.usage_statistics.reset()

        # Render system prompt
        system_prompt = self._system_message_prompt_template.render()

        # Get the user's message
        user_message = self._get_last_user_message(context)
        tracker_event_attachments = self._get_tracker_event_attachments(context)

        messages = await self._build_messages(context)

        # Create generation context for telemetry/tracking
        generation_context = CopilotGenerationContext(
            system_message={"role": "system", "content": system_prompt},
            chat_history=messages[:-1],
            last_user_message={"role": "user", "content": user_message},
            tracker_event_attachments=tracker_event_attachments,
        )

        copilot_response_handler = AgentCopilotResponseHandler(
            response_stream=self._stream_response(system_prompt, messages),
            rolling_buffer_size=config.COPILOT_HANDLER_ROLLING_BUFFER_SIZE,
            mcp_tool_queue=self._mcp_tool_queue,
            plan_queue=self._plan_queue,
        )

        # Return the stream and generation context
        return (
            copilot_response_handler,
            generation_context,
        )

    async def _build_messages(
        self,
        context: CopilotContext,
    ) -> List[Dict[str, Any]]:
        """Build the complete message list for the OpenAI API.

        Args:
            context: The context of the copilot.

        Returns:
            A list of messages in OpenAI format.
        """
        if not context.copilot_chat_history:
            return []

        past_messages = self._create_chat_history_messages(
            context.copilot_chat_history[:-1]
        )

        latest_message = self._process_latest_message(
            context.copilot_chat_history[-1], context
        )

        messages = [*past_messages, latest_message]
        return self._convert_to_responses_api_format(messages)

    def _convert_to_responses_api_format(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert Chat Completions format to Responses API format."""
        converted = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if isinstance(content, str):
                # Simple string content is fine as-is
                converted.append(msg)
            elif isinstance(content, list):
                new_content = []
                for item in content:
                    if item.get("type") == "text":
                        # user → input_text, assistant → output_text
                        new_type = "input_text" if role == "user" else "output_text"
                        new_content.append({"type": new_type, "text": item["text"]})
                    else:
                        new_content.append(item)
                converted.append({"role": role, "content": new_content})
            else:
                converted.append(msg)
        return converted

    @AgentCopilotLangfuseTelemetry.trace_streaming_generation
    async def _stream_response(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream response events from the agent.

        This is a simple passthrough from the Agent SDK to the response handler.
        All event processing, queue draining, and context management happens in
        the CopilotResponseHandler.

        Args:
            system_prompt: The system prompt for the agent
            messages: The messages to send to the agent

        Yields:
            StreamEvent objects from the Agent SDK
        """
        structlogger.debug(
            "agent_sdk.agent_copilot.stream_response.start",
            messages_count=len(messages),
            max_turns=config.COPILOT_MAX_AGENT_STEPS,
        )

        try:
            # Run the agent with streaming enabled
            async with self._create_agent(system_prompt) as agent:
                result = Runner.run_streamed(
                    agent,
                    input=messages,
                    max_turns=config.COPILOT_MAX_AGENT_STEPS,
                )

                # Simple passthrough of StreamEvents to the response handler
                async for event in result.stream_events():
                    # Extract usage statistics from ResponseCompletedEvent
                    if is_response_completed_event(event):
                        # event.data is ResponseCompletedEvent
                        self.usage_statistics.update_from_response_completed_event(
                            event.data  # type: ignore[union-attr]
                        )
                    yield event

        except Exception as e:
            structlogger.error(
                "agent_sdk.agent_copilot.stream_response.error",
                event_info="Error streaming agent response",
                error=str(e),
            )
            raise

    # HELPERS

    def _process_latest_message(
        self,
        latest_message: Any,
        context: CopilotContext,
    ) -> Dict[str, Any]:
        """Process the latest message and convert it to OpenAI format.

        Args:
            latest_message: The most recent message from the chat history.
            context: The copilot context containing conversation state.
            relevant_documents: List of relevant documents for context.

        Returns:
            Message in OpenAI format.

        Raises:
            ValueError: If the message type is not supported.
        """
        if isinstance(latest_message, UserChatMessage):
            tracker_event_attachments = latest_message.get_content_blocks_by_type(
                EventContent
            )
            # TODO: Update the render method not to take the context once the
            #       tracker context is available through a tool call. In other words,
            #       the context should only contain the attachments.
            rendered_prompt = self._render_last_user_message_context_prompt(
                context, tracker_event_attachments
            )
            return latest_message.build_openai_message(prompt=rendered_prompt)

        elif isinstance(latest_message, InternalCopilotRequestChatMessage):
            rendered_prompt = self._render_training_error_handler_prompt(latest_message)
            return latest_message.build_openai_message(prompt=rendered_prompt)

        else:
            raise ValueError(f"Unexpected message type: {type(latest_message)}")

    def _render_last_user_message_context_prompt(
        self,
        context: CopilotContext,
        tracker_event_attachments: List[EventContent],
    ) -> str:
        # TODO: Make this available through a tool call. Once available, remove the
        #       context from the prompt.
        conversation = self._format_conversation_history(context.tracker_context)
        # TODO: Make this available through a tool call. Once available, remove the
        #       context from the prompt.
        current_state = self._format_current_state(context.tracker_context)
        # Format tracker events
        attachments = self._format_tracker_event_attachments(tracker_event_attachments)

        rendered_prompt = self._last_user_message_context_prompt_template.render(
            current_conversation=conversation,
            current_state=current_state,
            attachments=attachments,
        )
        return rendered_prompt

    def _render_training_error_handler_prompt(
        self,
        internal_request_message: InternalCopilotRequestChatMessage,
    ) -> str:
        """Render the training error handler prompt with documentation and context.

        Args:
            internal_request_message: Internal request message.
            context: The copilot context.

        Returns:
            Rendered prompt string for training error analysis.
        """
        modified_files_dicts: Dict[str, str] = {
            file.file_path: file.file_content
            for file in internal_request_message.get_content_blocks_by_type(FileContent)
        }
        rendered_prompt = self._training_error_handler_prompt_template.render(
            logs=internal_request_message.get_flattened_log_content(),
            modified_files=modified_files_dicts,
        )

        return rendered_prompt

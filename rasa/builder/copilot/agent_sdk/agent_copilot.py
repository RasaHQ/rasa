"""Main Agents SDK copilot implementation.

This copilot uses OpenAI's Agents SDK with MCP (Model Context Protocol) integration
while maintaining full compatibility with the existing copilot interface.
"""

import importlib.resources
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import structlog
from agents import Agent, ModelSettings, RawResponsesStreamEvent, Runner, StreamEvent
from agents.mcp import MCPServerStreamableHttp
from jinja2 import Template
from openai.types.responses import ResponseTextDeltaEvent

from rasa.builder import config
from rasa.builder.copilot.base_copilot import BaseCopilot
from rasa.builder.copilot.constants import (
    COPILOT_PROMPTS_DIR,
    COPILOT_PROMPTS_FILE_AGENT_SDK,
)
from rasa.builder.copilot.copilot_response_handler import (
    CopilotResponseHandler,
)
from rasa.builder.copilot.models import (
    CopilotContext,
    CopilotGenerationContext,
    EventContent,
    UsageStatistics,
)
from rasa.builder.document_retrieval.models import Document
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

        # Usage statistics tracking
        self._usage_statistics = UsageStatistics(
            input_token_price=config.COPILOT_INPUT_TOKEN_PRICE,
            output_token_price=config.COPILOT_OUTPUT_TOKEN_PRICE,
            cached_token_price=config.COPILOT_CACHED_TOKEN_PRICE,
        )

    @property
    def usage_statistics(self) -> UsageStatistics:
        """Get usage statistics for the copilot."""
        return self._usage_statistics

    # TODO: (agent-sdk) Implement custom hooks by subclassing AgentHooks
    # def _create_hooks(self) -> AgentHooks:
    #     """Create custom AgentHooks subclass for observability."""
    #     class RasaCopilotHooks(AgentHooks):
    #         async def on_tool_start(self, context, agent, tool) -> None:
    #             structlogger.info("agent_sdk.tool.start", tool_name=tool.name)
    #
    #         async def on_tool_end(self, context, agent, tool) -> None:
    #             structlogger.info("agent_sdk.tool.end", tool_name=tool.name)
    #
    #     return RasaCopilotHooks()

    @asynccontextmanager
    async def _create_mcp_server(self) -> AsyncGenerator[MCPServerStreamableHttp, None]:
        """Context manager to create and manage MCP server connection.

        Yields:
            Connected MCPServerStreamableHttp instance
        """
        mcp_url = f"http://{config.MCP_SERVER_HOST}:{config.MCP_SERVER_PORT}/mcp"
        structlogger.info(
            "agent_sdk.mcp_client.connecting",
            event_info="Creating MCP client connection",
            mcp_url=mcp_url,
            timeout=config.MCP_TOOL_CALL_TIMEOUT,
        )

        try:
            async with MCPServerStreamableHttp(
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
                    event_info="MCP client connected successfully",
                )
                yield server
                structlogger.info(
                    "agent_sdk.mcp_client.disconnecting",
                    event_info="Closing MCP client connection",
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
                event_info="MCP client connection closed or errored",
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

        # Note: AgentHooks in openai-agents 0.4.2 needs to be subclassed, not
        # instantiated with params. For now, we create agent without hooks.
        async with self._create_mcp_server() as server:
            yield Agent(
                name="Rasa Copilot",
                instructions=system_instructions,
                model=config.OPENAI_MODEL,
                model_settings=model_settings,
                mcp_servers=[server],
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
    ) -> Tuple[CopilotResponseHandler, CopilotGenerationContext]:
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

        messages = await self._build_messages(context, relevant_documents=[])

        # Create generation context for telemetry/tracking
        generation_context = CopilotGenerationContext(
            system_message={"role": "system", "content": system_prompt},
            chat_history=messages[:-1],
            last_user_message={"role": "user", "content": user_message},
            tracker_event_attachments=tracker_event_attachments,
        )

        copilot_response_handler = CopilotResponseHandler(
            self._stream_response(system_prompt, messages),
            rolling_buffer_size=config.COPILOT_HANDLER_ROLLING_BUFFER_SIZE,
        )

        # Return the stream and generation context
        return (
            copilot_response_handler,
            generation_context,
        )

    async def _build_messages(
        self,
        context: CopilotContext,
        relevant_documents: List[Document],
    ) -> List[Dict[str, Any]]:
        """Build the complete message list for the OpenAI API.

        Args:
            context: The context of the copilot.
            relevant_documents: The relevant documents to use in the context.

        Returns:
            A list of messages in OpenAI format.
        """
        if not context.copilot_chat_history:
            return []

        past_messages = self._create_chat_history_messages(
            context.copilot_chat_history[:-1]
        )

        latest_message = self._process_latest_message(
            context.copilot_chat_history[-1], context, relevant_documents
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

    def extract_text_response_delta(self, event: StreamEvent) -> Optional[str]:
        """Extract the text response delta from the event."""
        if not isinstance(event, RawResponsesStreamEvent) or not isinstance(
            event.data, ResponseTextDeltaEvent
        ):
            return None
        return event.data.delta

    async def _stream_response(
        self,
        system_prompt: str,
        messages: List[Dict[str, Any]],
    ) -> AsyncGenerator[str, None]:
        """Stream response tokens from the agent.

        Args:
            system_prompt: The system prompt for the agent
            messages: The messages to send to the agent

        Yields:
            Response tokens as strings
        """
        structlogger.debug(
            "agent_sdk.agent_copilot.stream_response.start",
            messages_count=len(messages),
        )

        try:
            # Run the agent with streaming enabled
            async with self._create_agent(system_prompt) as agent:
                result = Runner.run_streamed(
                    agent,
                    input=messages,
                )

                # Stream the response tokens
                # TODO: (agent-sdk) Ticket https://rasahq.atlassian.net/browse/SWI-859
                async for event in result.stream_events():
                    text_delta = self.extract_text_response_delta(event)
                    if text_delta:
                        yield text_delta

        except Exception as e:
            structlogger.error(
                "agent_sdk.agent_copilot.stream_response.error",
                event_info="Error streaming agent response",
                error=str(e),
            )
            raise

    @property
    def llm_config(self) -> Dict[str, Any]:
        """The LLM config used to generate the response."""
        return {
            "model": config.OPENAI_MODEL,
            "temperature": config.OPENAI_TEMPERATURE,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

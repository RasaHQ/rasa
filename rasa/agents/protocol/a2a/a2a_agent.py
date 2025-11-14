import asyncio
import json
import os
import time
import uuid
from contextlib import aclosing
from typing import Any, ClassVar, Dict, List, Optional, Set
from urllib.parse import urlparse

import httpx
import structlog
from a2a.client import (
    A2ACardResolver,
    A2AClientError,
    A2AClientHTTPError,
    A2AClientJSONError,
    Client,
    ClientConfig,
    ClientEvent,
    ClientFactory,
)
from a2a.client.errors import A2AClientJSONRPCError
from a2a.types import (
    AgentCard,
    Artifact,
    DataPart,
    FilePart,
    FileWithUri,
    InternalError,
    InvalidAgentResponseError,
    Message,
    Part,
    Role,
    Task,
    TaskQueryParams,
    TaskState,
    TextPart,
    TransportProtocol,
)
from pydantic import ValidationError

from rasa.agents.constants import (
    A2A_AGENT_CONTEXT_ID_KEY,
    A2A_AGENT_TASK_ID_KEY,
    AGENT_DEFAULT_MAX_RETRIES,
    AGENT_DEFAULT_TIMEOUT_SECONDS,
    AGENT_METADATA_STRUCTURED_RESULTS_KEY,
    MAX_AGENT_RETRY_DELAY_SECONDS,
)
from rasa.agents.core.agent_protocol import AgentProtocol
from rasa.agents.core.types import AgentStatus, ProtocolType
from rasa.agents.schemas import AgentInput, AgentOutput
from rasa.core.available_agents import AgentConfig
from rasa.core.channels import OutputChannel
from rasa.core.constants import (
    AGENT_MESSAGE_TYPE_INTERMEDIATE_MESSAGE,
    AGENT_MESSAGE_TYPE_KEY,
    INTERMEDIATE_MESSAGE_AGENT_NAME_KEY,
    INTERMEDIATE_MESSAGE_AGENT_TASK_ID_KEY,
    INTERMEDIATE_MESSAGE_ID_KEY,
    INTERMEDIATE_MESSAGE_TIMESTAMP_KEY,
    UTTER_SOURCE_METADATA_KEY,
)
from rasa.shared.agents.auth.agent_auth_manager import AgentAuthManager
from rasa.shared.core.events import BotUttered, Event
from rasa.shared.exceptions import (
    AgentInitializationException,
    InvalidParameterException,
    RasaException,
)

A2A_TASK_POOLING_INITIAL_DELAY = 0.5
A2A_TASK_POOLING_MAX_WAIT = 60

structlogger = structlog.get_logger()


class A2AAgent(AgentProtocol):
    """A2A client implementation."""

    __SUPPORTED_OUTPUT_MODES: ClassVar[list[str]] = [
        "text",
        "text/plain",
        "application/json",
    ]

    # ============================================================================
    # Initialization & Setup
    # ============================================================================

    def __init__(
        self,
        name: str,
        description: str,
        agent_card_path: str,
        timeout: int,
        max_retries: int,
        auth_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._name = name
        self._description = description
        self._agent_card_path = agent_card_path
        self._timeout = timeout
        self._max_retries = max_retries
        self._auth_config = auth_config

        self.agent_card: Optional[AgentCard] = None
        self._client: Optional[Client] = None

    @classmethod
    def from_config(cls, config: AgentConfig) -> AgentProtocol:
        """Initialize the A2A Agent with the given configuration."""
        agent_card_path = (
            config.configuration.agent_card if config.configuration else None
        )
        if not agent_card_path:
            raise InvalidParameterException(
                "Agent card path or URL must be provided in the configuration "
                "for A2A agents."
            )

        timeout = (
            config.configuration.timeout
            if config.configuration and config.configuration.timeout
            else AGENT_DEFAULT_TIMEOUT_SECONDS
        )
        max_retries = (
            config.configuration.max_retries
            if config.configuration and config.configuration.max_retries
            else AGENT_DEFAULT_MAX_RETRIES
        )

        _auth_config = config.configuration.auth if config.configuration else None
        return cls(
            name=config.agent.name,
            description=config.agent.description,
            agent_card_path=agent_card_path,
            timeout=timeout,
            max_retries=max_retries,
            auth_config=_auth_config,
        )

    @property
    def protocol_type(self) -> ProtocolType:
        return ProtocolType.A2A

    # ============================================================================
    # Connection Management
    # ============================================================================

    async def connect(self) -> None:
        """Fetch the AgentCard and initialize the A2A client."""
        from rasa.nlu.utils import is_url

        if is_url(self._agent_card_path):
            self.agent_card = await A2AAgent._resolve_agent_card_with_retry(
                self._agent_card_path, self._timeout, self._max_retries
            )
        else:
            self.agent_card = A2AAgent._load_agent_card_from_file(self._agent_card_path)
        structlogger.debug(
            "a2a_agent.from_config",
            event_info=f"Loaded agent card from {self._agent_card_path}",
            agent_card=self.agent_card.model_dump(),
            json_formatting=["agent_card"],
        )

        try:
            self._client = self._init_client()
            structlogger.debug(
                "a2a_agent.connect.agent_client_initialized",
                event_info=f"Initialized A2A client for agent '{self._name}'. "
                f"Performing health check using the URL {self.agent_card.url}",
            )
        except Exception as exception:
            structlogger.error(
                "a2a_agent.connect.error",
                event_info="Failed to initialize A2A client",
                agent_name=self._name,
                error=str(exception),
            )
            raise AgentInitializationException(
                f"Failed to initialize A2A client "
                f"for agent '{self._name}': {exception}",
            ) from exception

        await self._perform_health_check()
        structlogger.debug(
            "a2a_agent.connect.success",
            event_info=f"Connected to A2A server '{self._name}' "
            f"at {self.agent_card.url}",
        )

    async def disconnect(self) -> None:
        """We don't need to explicitly disconnect the A2A client."""
        return

    # ============================================================================
    # Core Protocol Methods
    # ============================================================================

    async def process_input(self, agent_input: AgentInput) -> AgentInput:
        """Pre-process the input before sending it to the agent."""
        # A2A-specific input processing logic
        return agent_input

    async def run(
        self, agent_input: AgentInput, output_channel: Optional[OutputChannel] = None
    ) -> AgentOutput:
        """Send a message to Agent/server and return response."""
        generated_events: List[Event] = []
        if not self._client or not self.agent_card:
            structlogger.error(
                "a2a_agent.run.error",
                event_info="A2A client is not initialized. Call connect() first.",
            )
            return AgentOutput(
                id=agent_input.id,
                status=AgentStatus.FATAL_ERROR,
                error_message="Client not initialized",
                events=generated_events or None,
            )

        structlogger.info(
            "a2a_agent.run.start",
            event_info="Running A2A agent",
            agent_name=self._name,
        )
        message = self._prepare_message(agent_input)

        task_id: Optional[str] = None
        events_received = 0
        try:
            # Use aclosing to ensure proper cleanup of the async generator
            stream = self._client.send_message(message)
            async with aclosing(stream) as stream:  # type: ignore[type-var]
                async for event in stream:
                    events_received += 1
                    agent_output = self._handle_send_message_response(
                        agent_input,
                        event,
                        generated_events,
                        output_channel=output_channel,
                    )
                    if agent_output is not None:
                        return agent_output
                    else:
                        # Not a terminal response, save taskID (in case that's the only
                        # event, and we need to pool) and continue waiting for events
                        if (
                            isinstance(event, tuple)
                            and len(event) == 2
                            and isinstance(event[0], Task)
                        ):
                            task_id = event[0].id
                        continue
        except A2AClientJSONRPCError as e:
            return self._handle_json_rpc_error_response(
                agent_input, e.error, generated_events
            )
        except A2AClientError as exception:
            structlogger.error(
                "a2a_agent.run.send_message.error",
                event_info="Error during sending message to A2A agent",
                agent_name=self._name,
                error=str(exception),
            )
            return AgentOutput(
                id=agent_input.id,
                status=AgentStatus.FATAL_ERROR,
                error_message=f"Send message error: {exception!s}",
                events=generated_events or None,
            )

        # The stream has ended, but we didn't get a terminal response.
        # Check if we received any events at all.
        if events_received == 0:
            structlogger.error(
                "a2a_agent.run.no_events_received",
                event_info="No events received from A2A agent after sending message",
                agent_name=self._name,
            )
            return AgentOutput(
                id=agent_input.id,
                status=AgentStatus.RECOVERABLE_ERROR,
                error_message="No events received from A2A agent",
                events=generated_events or None,
            )

        # Now we need to poll the task until it reaches a terminal state.
        if not task_id:
            structlogger.error(
                "a2a_agent.run.pooling.missing_id",
                event_info="Missing task_id for polling",
                agent_name=self._name,
                task_id=task_id,
            )
            return AgentOutput(
                id=agent_input.id,
                status=AgentStatus.FATAL_ERROR,
                error_message="Missing task_id for polling",
                events=generated_events or None,
            )
        return await self._pool_task_until_terminal(
            agent_input=agent_input,
            task_id=task_id,
            generated_events=generated_events,
            max_wait=A2A_TASK_POOLING_MAX_WAIT,
            initial_delay=A2A_TASK_POOLING_INITIAL_DELAY,
            max_delay=MAX_AGENT_RETRY_DELAY_SECONDS,
            output_channel=output_channel,
        )

    async def process_output(self, output: AgentOutput) -> AgentOutput:
        """Post-process the output before returning it to Rasa."""
        # A2A-specific output processing logic
        return output

    # ============================================================================
    # Message Processing & Response Handling
    # ============================================================================

    def _handle_send_message_response(
        self,
        agent_input: AgentInput,
        response: ClientEvent | Message,
        generated_events: List[Event],
        output_channel: Optional[OutputChannel] = None,
    ) -> Optional[AgentOutput]:
        """Handle possible response types from the A2A client.

        In case of streaming, the response can be either exactly *one* Message,
        or a *series* of tuples of (Task, Optional[TaskUpdateEvent]).

        In case of pooling, the response can be either exactly *one* Message,
        or exactly *one* tuple of (Task, None).

        If the agent response is terminal (i.e., completed, failed, etc.),
        this method will return an AgentOutput.
        Otherwise, the task is still in progress (i.e., submitted, working), so this
        method will return None, so that the streaming or pooling agent can continue
        to wait for updates.
        """
        if isinstance(response, Message):
            return self._handle_message_response(
                agent_input, response, generated_events
            )
        elif (
            isinstance(response, tuple)
            and len(response) == 2
            and isinstance(response[0], Task)
        ):
            return self._handle_client_event(
                agent_input, response, generated_events, output_channel=output_channel
            )
        else:
            # Currently, no other response types exist, so this branch is
            # unreachable. It is kept as a safeguard against future changes
            # to the A2A protocol: if new response types are introduced,
            # the agent will log an error instead of crashing.
            return self._handle_unexpected_response_type(
                agent_input, response, generated_events
            )

    def _handle_json_rpc_error_response(
        self, agent_input: AgentInput, error: Any, generated_events: List[Event]
    ) -> AgentOutput:
        structlogger.error(
            "a2a_agent.run.error",
            event_info="Received JSON-RPC error response from A2A agent",
            agent_name=self._name,
            error=str(error),
        )
        if isinstance(
            error,
            (
                InternalError,
                InvalidAgentResponseError,
            ),
        ):
            return AgentOutput(
                id=agent_input.id,
                status=AgentStatus.RECOVERABLE_ERROR,
                error_message=str(error),
                events=generated_events or None,
            )
        else:
            return AgentOutput(
                id=agent_input.id,
                status=AgentStatus.FATAL_ERROR,
                error_message=str(error),
                events=generated_events or None,
            )

    def _handle_client_event(
        self,
        agent_input: AgentInput,
        client_event: ClientEvent,
        generated_events: List[Event],
        output_channel: Optional[OutputChannel] = None,
    ) -> Optional[AgentOutput]:
        task = client_event[0]
        update_event = client_event[1]
        structlogger.debug(
            "a2a_agent.run.client_event_received",
            event_info="Received client event from A2A",
            task=task.model_dump() if task else None,
            update_event=update_event.model_dump() if update_event else None,
            json_formatting=["task", "update_event"],
        )

        return self._handle_task(
            agent_input=agent_input,
            task=task,
            generated_events=generated_events,
            sent_intermediate_messages=None,
            output_channel=output_channel,
        )

    def _handle_message_response(
        self, agent_input: AgentInput, message: Message, generated_events: List[Event]
    ) -> Optional[AgentOutput]:
        structlogger.debug(
            "a2a_agent.run.message_received",
            event_info="Received message from A2A",
            agent_name=self._name,
            message=message.model_dump(),
            json_formatting=["message"],
        )
        metadata = agent_input.metadata or {}
        metadata[A2A_AGENT_CONTEXT_ID_KEY] = message.context_id

        return AgentOutput(
            id=agent_input.id,
            status=AgentStatus.INPUT_REQUIRED,
            response_message=self._generate_response_message_from_parts(message.parts),
            metadata=metadata,
            events=generated_events or None,
        )

    def _handle_unexpected_response_type(
        self,
        agent_input: AgentInput,
        response_result: Any,
        generated_events: List[Event],
    ) -> AgentOutput:
        structlogger.error(
            "a2a_agent.run.unexpected_response_type",
            event_info="Received unexpected response type from A2A server "
            "during streaming",
            agent_name=self._name,
            response_type=type(response_result),
        )
        return AgentOutput(
            id=agent_input.id,
            status=AgentStatus.FATAL_ERROR,
            error_message=f"Unexpected response type: {type(response_result)}",
            events=generated_events or None,
        )

    def _handle_task(
        self,
        agent_input: AgentInput,
        task: Task,
        generated_events: List[Event],
        sent_intermediate_messages: Optional[Set[str]] = None,
        output_channel: Optional[OutputChannel] = None,
    ) -> Optional[AgentOutput]:
        """If task status is terminal (e.g. completed, failed) return AgentOutput.

        If the task is still in progress (i.e., submitted, working), return None,
        so that the streaming or pooling agent can continue to wait for updates.
        """
        state = task.status.state

        metadata = agent_input.metadata or {}
        metadata[A2A_AGENT_CONTEXT_ID_KEY] = task.context_id
        metadata[A2A_AGENT_TASK_ID_KEY] = task.id

        if state == TaskState.input_required:
            response_message = (
                self._generate_response_message_from_parts(task.status.message.parts)
                if task.status.message
                else ""
            )  # This should not happen, but as type of message property
            # is optional, so we need to handle it
            return AgentOutput(
                id=agent_input.id,
                status=AgentStatus.INPUT_REQUIRED,
                response_message=response_message,
                metadata=metadata,
                events=generated_events or None,
            )
        elif state == TaskState.completed:
            response_message = self._generate_completed_response_message(task)
            structured_results = (
                self._generate_structured_results_from_artifacts(
                    agent_input, task.artifacts
                )
                if task.artifacts
                else None
            )
            return AgentOutput(
                id=agent_input.id,
                status=AgentStatus.COMPLETED,
                response_message=response_message,
                structured_results=structured_results,
                metadata=metadata,
                events=generated_events or None,
            )
        elif (
            state == TaskState.failed
            or state == TaskState.canceled
            or state == TaskState.rejected
            or state == TaskState.auth_required
        ):
            structlogger.error(
                "a2a_agent.run_streaming_agent.unsuccessful_task_state",
                event_info="Task execution finished with an unsuccessful state",
                agent_name=self._name,
                state=state,
            )
            return AgentOutput(
                id=agent_input.id,
                status=AgentStatus.RECOVERABLE_ERROR,
                error_message=f"Task state: {state}",
                metadata=metadata,
                events=generated_events or None,
            )
        elif state == TaskState.submitted or state == TaskState.working:
            # The task is still in progress, send intermediate status update
            # to the user and return None to continue waiting for updates
            self._send_intermediate_message(
                agent_input,
                task,
                generated_events,
                sent_intermediate_messages,
                output_channel,
            )
            return None
        elif state == TaskState.unknown:
            # The task has an unknown state. Perhaps this is a transient condition.
            # Return None to continue waiting for updates
            structlogger.warning(
                "a2a_agent.run_streaming_agent.unknown_task_state",
                event_info="Task is in unknown state, continuing to wait for updates",
                agent_name=self._name,
                state=state,
            )
            return None
        else:
            structlogger.error(
                "a2a_agent.run_streaming_agent.unexpected_task_state",
                event_info="Unexpected task state received from A2A",
                agent_name=self._name,
                state=state,
            )
            return AgentOutput(
                id=agent_input.id,
                status=AgentStatus.FATAL_ERROR,
                error_message=f"Unexpected task state: {state}",
                metadata=metadata,
                events=generated_events or None,
            )

    def _send_intermediate_message(
        self,
        agent_input: AgentInput,
        task: Task,
        generated_events: List[Event],
        sent_intermediate_messages: Optional[Set[str]] = None,
        output_channel: Optional[OutputChannel] = None,
    ) -> None:
        """Send an intermediate message to the user if the task is in progress.
        This allows the user to see that the agent is working on their request,
        providing better UX for long-running operations.

        Args:
            agent_input: The agent input containing user information
            task: The task from the A2A stream
            generated_events: List of events generated so far for this run
            sent_intermediate_messages: Set of intermediate messages already sent,
            to avoid duplicates.
            output_channel: The output channel for sending messages
        """
        if output_channel is None:
            structlogger.debug(
                "a2a_agent.send_intermediate_message.no_output_channel",
                event_info=(
                    "No output channel provided, cannot send intermediate message.",
                ),
                agent_name=self._name,
            )
            return

        recipient_id = agent_input.recipient_id
        if not recipient_id:
            structlogger.debug(
                "a2a_agent.send_intermediate_message.no_recipient_id",
                event_info=(
                    "No recipient ID provided in agent input, cannot send "
                    "intermediate message."
                ),
                agent_name=self._name,
            )
            return

        try:
            message = (
                self._generate_response_message_from_parts(task.status.message.parts)
                if task.status.message
                else ""
            )
            if len(message) == 0:
                structlogger.debug(
                    "a2a_agent.send_intermediate_message.empty_message",
                    event_info="Skipping sending empty intermediate message",
                    agent_name=self._name,
                    recipient_id=recipient_id,
                )
                return

            # In polling mode, avoid sending duplicate intermediate messages
            if sent_intermediate_messages is not None:
                if message in sent_intermediate_messages:
                    structlogger.debug(
                        "a2a_agent.send_intermediate_message.duplicate_skipped",
                        event_info="Skipping duplicate intermediate message"
                        " during polling",
                        agent_name=self._name,
                        recipient_id=recipient_id,
                        message=message,
                    )
                    return

            structlogger.debug(
                "a2a_agent.send_intermediate_message.sending",
                event_info="Sending intermediate message to output channel",
                agent_name=self._name,
                recipient_id=recipient_id,
                message=message,
            )

            # Send the message in background without awaiting it
            async_task = asyncio.create_task(
                output_channel.send_text_message(
                    recipient_id=recipient_id, text=message
                )
            )

            # Append BotUttered only if the async send succeeded
            def _on_send_done(t: asyncio.Task) -> None:
                exc = t.exception()
                if exc:
                    structlogger.error(
                        "a2a_agent.send_intermediate_message.async_task_error",
                        event_info="Sending intermediate message in async task failed",
                        agent_name=self._name,
                        recipient_id=recipient_id,
                        error=str(exc),
                    )
                else:
                    generated_events.append(
                        BotUttered(
                            text=message,
                            metadata=self.create_bot_uttered_event_metadata(task),
                        )
                    )
                    if sent_intermediate_messages is not None:
                        sent_intermediate_messages.add(message)

            async_task.add_done_callback(_on_send_done)
        except Exception as e:
            structlogger.error(
                "a2a_agent.send_intermediate_message.error",
                event_info="Error sending intermediate message to output channel",
                agent_name=self._name,
                recipient_id=recipient_id,
                error=str(e),
            )

    def create_bot_uttered_event_metadata(self, task: Task) -> Dict[str, str]:
        bot_uttered_metadata = {
            UTTER_SOURCE_METADATA_KEY: self.__class__.__name__,
            INTERMEDIATE_MESSAGE_AGENT_NAME_KEY: self._name,
            INTERMEDIATE_MESSAGE_AGENT_TASK_ID_KEY: task.id,
            AGENT_MESSAGE_TYPE_KEY: AGENT_MESSAGE_TYPE_INTERMEDIATE_MESSAGE,
        }
        if task.status.timestamp:
            bot_uttered_metadata[INTERMEDIATE_MESSAGE_TIMESTAMP_KEY] = (
                task.status.timestamp
            )
        if task.status.message:
            bot_uttered_metadata[INTERMEDIATE_MESSAGE_ID_KEY] = (
                task.status.message.message_id
            )
        return bot_uttered_metadata

    # ============================================================================
    # Message Preparation & Formatting
    # ============================================================================

    @staticmethod
    def _prepare_message(agent_input: AgentInput) -> Message:
        parts: List[Part] = []
        if agent_input.metadata and A2A_AGENT_CONTEXT_ID_KEY in agent_input.metadata:
            # Agent knows the conversation history already, send the last
            # user message only
            parts.append(Part(root=TextPart(text=agent_input.user_message)))
        else:
            # Send the full conversation history
            parts.append(Part(root=TextPart(text=agent_input.conversation_history)))

        if len(agent_input.slots) > 0:
            slots_dict: Dict[str, Any] = {
                "slots": [
                    slot.model_dump(exclude={"type", "allowed_values"})
                    for slot in agent_input.slots
                    if slot.value is not None
                ]
            }
            parts.append(Part(root=DataPart(data=slots_dict)))

        agent_message = Message(
            role=Role.user,
            parts=parts,
            message_id=str(uuid.uuid4()),
            context_id=agent_input.metadata.get(A2A_AGENT_CONTEXT_ID_KEY, None),
            task_id=agent_input.metadata.get(A2A_AGENT_TASK_ID_KEY, None),
        )
        structlogger.debug(
            "a2a_agent.prepare_message",
            event_info="Prepared message to send to A2A server",
            agent_name=agent_input.id,
            message=agent_message.model_dump(),
            json_formatting=["message"],
        )
        return agent_message

    # ============================================================================
    # Task Management & Polling
    # ============================================================================

    async def _pool_task_until_terminal(
        self,
        agent_input: AgentInput,
        task_id: str,
        generated_events: List[Event],
        max_wait: int,
        initial_delay: float,
        max_delay: int,
        output_channel: Optional[OutputChannel] = None,
    ) -> AgentOutput:
        """Poll the task status until it reaches a terminal state or times out."""
        if not self._client:
            structlogger.error(
                "a2a_agent.pool_task_until_terminal.error",
                event_info="A2A client is not initialized. Call connect() first.",
            )
            return AgentOutput(
                id=agent_input.id,
                status=AgentStatus.FATAL_ERROR,
                error_message="Client not initialized",
            )

        structlogger.debug(
            "a2a_agent.pool_task_until_terminal.start",
            event_info="Start polling task from A2A server",
            agent_name=self._name,
            task_id=task_id,
            max_wait=max_wait,
            initial_delay=initial_delay,
            max_delay=max_delay,
        )
        start_time = time.monotonic()
        delay = initial_delay
        sent_intermediate_messages: Set[str] = set()

        while True:
            try:
                task = await self._client.get_task(TaskQueryParams(id=task_id))
                agent_output = self._handle_task(
                    agent_input=agent_input,
                    task=task,
                    generated_events=generated_events,
                    sent_intermediate_messages=sent_intermediate_messages,
                    output_channel=output_channel,
                )
                if agent_output is not None:
                    # Reached a terminal state, return the output
                    return agent_output

                elapsed = time.monotonic() - start_time
                if elapsed >= max_wait:
                    structlogger.debug(
                        "a2a_agent.pool_task_until_terminal.timeout",
                        event_info="Polling task from A2A server timed out",
                        agent_name=self._name,
                        task_id=task_id,
                        elapsed=elapsed,
                        max_wait=max_wait,
                    )
                    return AgentOutput(
                        id=agent_input.id,
                        status=AgentStatus.FATAL_ERROR,
                        error_message="Polling timed out",
                        events=generated_events or None,
                    )

                structlogger.error(
                    "a2a_agent.pool_task_until_terminal.waiting",
                    event_info="Task not in terminal state yet, waiting to poll again",
                    delay=delay,
                    agent_name=self._name,
                    task_id=task_id,
                    elapsed=elapsed,
                    max_wait=max_wait,
                )
                await asyncio.sleep(delay)
                # Exponential backoff with cap
                delay = min(delay * 2, max_delay)

            except A2AClientError as exception:
                structlogger.error(
                    "a2a_agent.pool_task_until_terminal.error",
                    event_info="Error during polling task from A2A server",
                    agent_name=self._name,
                    error=str(exception),
                )
                return AgentOutput(
                    id=agent_input.id,
                    status=AgentStatus.FATAL_ERROR,
                    error_message=f"Polling error: {exception!s}",
                    events=generated_events or None,
                )

    # ============================================================================
    # Response Generation & Formatting
    # ============================================================================

    @staticmethod
    def _generate_response_message_from_parts(parts: Optional[List[Part]]) -> str:
        """Convert a list of Part objects to a single string message."""
        result = ""
        if not parts:
            return result
        for part in parts:
            if isinstance(part.root, TextPart):
                result += part.root.text + "\n"
            elif isinstance(part.root, DataPart):
                # DataPart results will be returned as a part of the structured results,
                # we don't need to include it in the response message
                continue
            elif isinstance(part.root, FilePart) and isinstance(
                part.root.file, FileWithUri
            ):
                # If the file is a FileWithUri, we can include the URI
                result += f"File: {part.root.file.uri}\n"
            else:
                structlogger.warning(
                    "a2a_agent._parts_to_single_message.warning",
                    event_info="Unsupported part type encountered",
                    part_type=type(part.root),
                )
        return result.strip()

    @staticmethod
    def _generate_completed_response_message(task: Task) -> str:
        """Generate a response message for a completed task.

        In case of completed tasks, the final message might be in
        the task status message or in the artifacts (or both).
        """
        # We need to preserve the order of the message,
        # but also make sure to remove any duplicates.
        result: List[str] = []
        if task.status.message:
            message = A2AAgent._generate_response_message_from_parts(
                task.status.message.parts
            )
            if message and message not in result:
                result.append(message)
        if task.artifacts:
            for artifact in task.artifacts:
                message = A2AAgent._generate_response_message_from_parts(artifact.parts)
                if message and message not in result:
                    result.append(message)
        return "\n".join(result)

    @staticmethod
    def _generate_structured_results_from_artifacts(
        agent_input: AgentInput, artifacts: List[Artifact]
    ) -> Optional[List[List[Dict[str, Any]]]]:
        structured_results_of_current_iteration: List[Dict[str, Any]] = []
        # There might be multiple artifacts in the response, each of them might
        # contain multiple parts. We will treat each DataPart in each artifact
        # as a separate tool result. The tool name will be the agent ID + index
        # of the artifact + index of the part.
        # E.g., foo_0_1, foo_0_2, foo_1_0, etc.
        for artifact_index, artifact in enumerate(artifacts):
            for part_index, part in enumerate(artifact.parts):
                if isinstance(part.root, DataPart) and len(part.root.data) > 0:
                    structured_result = {
                        "name": f"{agent_input.id}_{artifact_index}_{part_index}",
                        "type": "data",
                        "result": part.root.data,
                    }
                    structured_results_of_current_iteration.append(structured_result)
                elif isinstance(part.root, FilePart) and isinstance(
                    part.root.file, FileWithUri
                ):
                    structured_result = {
                        "name": f"{agent_input.id}_{artifact_index}_{part_index}",
                        "type": "file",
                        "result ": {
                            "uri": part.root.file.uri,
                            "name": part.root.file.name,
                            "mime_type": part.root.file.mime_type,
                        },
                    }
                    structured_results_of_current_iteration.append(structured_result)

        previous_structured_results: List[List[Dict[str, Any]]] = (
            agent_input.metadata.get(AGENT_METADATA_STRUCTURED_RESULTS_KEY, []) or []
        )
        previous_structured_results.append(structured_results_of_current_iteration)

        return previous_structured_results

    # ============================================================================
    # Agent Card Management
    # ============================================================================

    @staticmethod
    def _load_agent_card_from_file(agent_card_path: str) -> AgentCard:
        """Load agent card from JSON file."""
        try:
            with open(os.path.abspath(agent_card_path), "r") as f:
                card_data = json.load(f)
            return AgentCard.model_validate(card_data)

        except FileNotFoundError as e:
            raise AgentInitializationException(
                f"Agent card file not found: {agent_card_path}",
            ) from e
        except (IOError, PermissionError) as e:
            raise AgentInitializationException(
                f"Error reading agent card file {agent_card_path}: {e}",
            ) from e
        except json.JSONDecodeError as e:
            raise AgentInitializationException(
                f"Invalid JSON in agent card file {agent_card_path}: {e}",
            ) from e
        except ValidationError as e:
            raise AgentInitializationException(
                f"Failed to load agent card from {agent_card_path}: {e}",
            ) from e

    @staticmethod
    async def _resolve_agent_card_with_retry(
        agent_card_path: str, timeout: int, max_retries: int
    ) -> AgentCard:
        """Resolve the agent card from a given path or URL."""
        # split agent_card_path into base URL and path
        try:
            url_parts = urlparse(agent_card_path)
            base_url = f"{url_parts.scheme}://{url_parts.netloc}"
            path = url_parts.path
        except ValueError:
            raise RasaException(f"Could not parse the URL: '{agent_card_path}'.")
        structlogger.debug(
            "a2a_agent.resolve_agent_card",
            event_info="Resolving agent card from remote URL",
            agent_card_path=agent_card_path,
            base_url=base_url,
            path=path,
            timeout=timeout,
        )

        for attempt in range(max_retries):
            if attempt > 0:
                structlogger.debug(
                    "a2a_agent.resolve_agent_card.retrying",
                    agent_card_path=f"{base_url}/{path}",
                    attempt=attempt + 1,
                    num_retries=max_retries,
                )

            try:
                agent_card = await A2ACardResolver(
                    httpx.AsyncClient(timeout=timeout),
                    base_url=base_url,
                    agent_card_path=path,
                ).get_agent_card()

                if agent_card:
                    return agent_card
            except (A2AClientHTTPError, A2AClientJSONError) as exception:
                structlogger.warning(
                    "a2a_agent.resolve_agent_card.error",
                    event_info="Error while resolving agent card",
                    agent_card_path=agent_card_path,
                    attempt=attempt + 1,
                    num_retries=max_retries,
                    error=str(exception),
                )
                if attempt < max_retries - 1:
                    # exponential backoff - wait longer with each retry
                    # 1 second, 2 seconds, 4 seconds, etc.
                    await asyncio.sleep(min(2**attempt, MAX_AGENT_RETRY_DELAY_SECONDS))

        raise AgentInitializationException(
            f"Failed to resolve agent card from {agent_card_path} after "
            f"{max_retries} attempts.",
        )

    # ============================================================================
    # Client Initialization & Health Checks
    # ============================================================================

    def _init_client(self) -> Client:
        _agent_manager = AgentAuthManager.load_auth(self._auth_config)
        auth_strategy = _agent_manager.get_auth() if _agent_manager else None
        factory = ClientFactory(
            config=ClientConfig(
                httpx_client=httpx.AsyncClient(
                    timeout=self._timeout, auth=auth_strategy
                ),
                streaming=True,
                supported_transports=[
                    TransportProtocol.jsonrpc,
                    TransportProtocol.http_json,
                    TransportProtocol.grpc,
                ],
                accepted_output_modes=self.__SUPPORTED_OUTPUT_MODES,
            )
        )
        return factory.create(self.agent_card)

    async def _perform_health_check(self) -> None:
        if not self._client or not self.agent_card:
            structlogger.error(
                "a2a_agent.health_check.error",
                event_info="A2A client is not initialized. Call connect() first.",
            )
            raise AgentInitializationException("Client not initialized")

        try:
            test_message = Message(
                role=Role.user,
                parts=[Part(root=TextPart(text="hello"))],
                message_id=str(uuid.uuid4()),
            )
            # Use aclosing to ensure proper cleanup of the async generator
            stream = self._client.send_message(test_message)
            async with aclosing(stream) as stream:  # type: ignore[type-var]
                async for event in stream:
                    if (
                        isinstance(event, Message)
                        or isinstance(event, tuple)
                        and len(event) == 2
                        and isinstance(event[0], Task)
                    ):
                        # We got a valid response, health check succeeded
                        return

                    event_info = "Unexpected response type during health check"
                    structlogger.error(
                        "a2a_agent.health_check.unexpected_response",
                        event_info=event_info,
                        agent_name=self._name,
                        response=event,
                        url=str(self.agent_card.url),
                    )
                    raise AgentInitializationException(f"{event_info}: {event}")
                # If the loop completes with no return, no events were received
                event_info = (
                    f"Health check failed for A2A agent '{self._name}' "
                    f"at {self.agent_card.url}: no events received"
                )
                structlogger.error(
                    "a2a_agent.health_check.no_events",
                    event_info=event_info,
                    agent_name=self._name,
                    url=str(self.agent_card.url),
                )
                raise AgentInitializationException(event_info)
        except Exception as exception:
            event_info = (
                f"Health check failed for A2A agent '{self._name}' at "
                f"{self.agent_card.url}: {exception!s}"
            )
            structlogger.error(
                "a2a_agent.health_check.failed",
                event_info=event_info,
                agent_name=self._name,
                error=str(exception),
                url=str(self.agent_card.url),
            )
            raise AgentInitializationException(event_info) from exception

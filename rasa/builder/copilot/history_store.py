import asyncio
import json
import os
import sqlite3
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Generator, List, Optional, Type

import structlog

from rasa.builder.copilot.constants import (
    DEFAULT_COPILOT_CHAT_ID,
    ROLE_COPILOT,
    ROLE_COPILOT_INTERNAL,
    ROLE_SYSTEM,
    ROLE_USER,
)
from rasa.builder.copilot.exceptions import CopilotHistoryDatabaseError
from rasa.builder.copilot.models import (
    ButtonContent,
    ChatMessage,
    CodeContent,
    ContentBlock,
    ConversationKey,
    CopilotChatMessage,
    CopilotSystemMessage,
    EventContent,
    FileContent,
    InternalCopilotRequestChatMessage,
    LinkContent,
    LogContent,
    LogItem,
    LogsContent,
    ReferenceEntry,
    ReferenceItem,
    ReferencesContent,
    ResponseCategory,
    TextContent,
    UserChatMessage,
)

structlogger = structlog.get_logger()


class CopilotHistoryStore(ABC):
    """Abstract interface for storing and retrieving copilot chat history."""

    @abstractmethod
    async def get(self, key: ConversationKey) -> List[ChatMessage]:
        """Get the conversation history for the given key.

        Args:
            key: The conversation key.

        Returns:
            A list of chat messages in the conversation history.
        """
        raise NotImplementedError

    @abstractmethod
    async def append(self, key: ConversationKey, message: ChatMessage) -> None:
        """Append a message to the conversation history for the given key.

        Args:
            key: The conversation key.
            message: The chat message to append.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete(self, key: ConversationKey) -> None:
        """Delete the conversation history for the given key.

        Args:
            key: The conversation key.
        """
        raise NotImplementedError


class SQLiteCopilotHistoryStore(CopilotHistoryStore):
    """SQLite-based implementation of copilot history storage."""

    def __init__(self, database_path: str) -> None:
        """Initialize the SQLite store.

        Args:
            database_path: Path to the SQLite database file.
        """
        super().__init__()
        self._database_path = database_path

        # Ensure the database directory exists
        if database_dir := os.path.dirname(database_path):
            os.makedirs(database_dir, exist_ok=True)

        self._initialize_database()

    @contextmanager
    def _database_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for SQLite database connections.

        Yields:
            SQLite connection that will be automatically closed.
        """
        connection = sqlite3.connect(self._database_path)
        try:
            yield connection
        finally:
            connection.close()

    def _initialize_database(self) -> None:
        """Create the database schema if it doesn't exist."""
        with self._database_connection() as connection:
            # Create the messages table
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS copilot_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content_json TEXT NOT NULL,
                    response_category TEXT,
                    created_at REAL NOT NULL
                )
                """
            )

            # Create index for efficient conversation queries
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversation
                ON copilot_messages(chat_id, created_at)
                """
            )
            connection.commit()

    @staticmethod
    def _get_message_class_for_role(role: Optional[str]) -> Type[ChatMessage]:
        """Get the appropriate message class for a given role.

        Args:
            role: The role string (e.g., 'user', 'copilot', 'copilot_internal') or None.

        Returns:
            The corresponding message class.
        """
        if role is None:
            structlogger.warning("copilot_history_store.get_message_class.missing_role")
            return CopilotChatMessage

        if role == ROLE_SYSTEM:
            return CopilotSystemMessage
        elif role == ROLE_USER:
            return UserChatMessage
        elif role == ROLE_COPILOT:
            return CopilotChatMessage
        elif role == ROLE_COPILOT_INTERNAL:
            return InternalCopilotRequestChatMessage
        else:
            structlogger.warning(
                "copilot_history_store.get_message_class.unknown_role",
                role=role,
            )
            return CopilotChatMessage

    @staticmethod
    def _deserialize_content_block(content_dict: Dict[str, Any]) -> ContentBlock:
        """Deserialize a content block from dictionary.

        Args:
            content_dict: Dictionary containing content block data.

        Returns:
            The appropriate ContentBlock instance.
        """
        raw_type = content_dict.get("type")

        if raw_type is None:
            structlogger.warning(
                "copilot_history_store.deserialize_content.missing_type",
                content_dict=content_dict,
            )
            return TextContent(type="text", text=str(content_dict))

        if not isinstance(raw_type, str):
            structlogger.warning(
                "copilot_history_store.deserialize_content.non_string_type",
                provided_type=type(raw_type).__name__,
                value=raw_type,
            )
            return TextContent(type="text", text=str(content_dict))

        content_type = raw_type.strip().lower()

        constructors: Dict[str, Callable[..., ContentBlock]] = {
            "text": TextContent,
            "code": CodeContent,
            "file": FileContent,
            "log": LogContent,
            "link": LinkContent,
            "button": ButtonContent,
            "references": ReferencesContent,
            "logs": LogsContent,
        }

        if content_type in constructors:
            return constructors[content_type](**content_dict)

        if content_type == "event":
            # Special handling for EventContent to avoid double nesting
            if "event_data" in content_dict:
                content_copy = content_dict.copy()
                event_data = content_copy.pop("event_data", {}) or {}
                result_dict = {
                    "type": content_copy.get("type"),
                    "event": content_copy.get("event"),
                }
                result_dict.update(event_data)
                return EventContent(**result_dict)
            return EventContent(**content_dict)

        structlogger.warning(
            "copilot_history_store.deserialize_content.unknown_type",
            content_type=content_type,
        )
        return TextContent(type="text", text=str(content_dict))

    @staticmethod
    def _create_message_from_db_row(
        role: str,
        content: List,
        response_category: Optional[str],
        timestamp: Optional[float] = None,
    ) -> ChatMessage:
        """Create a chat message from database row data.

        Args:
            role: The role string (e.g., 'user', 'copilot', 'copilot_internal').
            content: The content list (will be ignored for system messages).
            response_category: Optional response category string.
            timestamp: Optional unix timestamp (UTC) when the message was created.

        Returns:
            The appropriate ChatMessage instance.
        """
        message_class = SQLiteCopilotHistoryStore._get_message_class_for_role(role)

        kwargs: Dict[str, Any] = {
            "role": role,
            "response_category": ResponseCategory(response_category)
            if response_category
            else None,
            "timestamp": timestamp,
        }

        # System messages don't have content field
        if role != ROLE_SYSTEM:
            # Deserialize content blocks properly
            deserialized_content = [
                SQLiteCopilotHistoryStore._deserialize_content_block(block)
                for block in content
            ]
            kwargs["content"] = deserialized_content

        return message_class(**kwargs)

    @staticmethod
    def _serialize_messages(messages: List[CopilotChatMessage]) -> str:
        """Serialize messages to JSON string.

        Args:
            messages: List of copilot chat messages.

        Returns:
            JSON string representation of the messages.
        """
        return json.dumps(
            [message.model_dump() for message in messages], ensure_ascii=False
        )

    @staticmethod
    def _deserialize_messages(json_payload: str) -> List[ChatMessage]:
        """Deserialize messages from JSON string.

        Args:
            json_payload: JSON string containing serialized messages.

        Returns:
            List of chat messages.
        """
        raw_messages = json.loads(json_payload or "[]")
        messages: List[ChatMessage] = []
        for message_data in raw_messages:
            role = message_data.get("role")
            message_class = SQLiteCopilotHistoryStore._get_message_class_for_role(role)
            messages.append(message_class(**message_data))
        return messages

    def _read_conversation_from_database(
        self, key: ConversationKey
    ) -> List[ChatMessage]:
        """Read conversation messages from SQLite database.

        Args:
            key: Conversation key.

        Returns:
            List of chat messages for the conversation.
        """
        with self._database_connection() as connection:
            cursor = connection.execute(
                """
                SELECT role, content_json, response_category, created_at
                FROM copilot_messages
                WHERE chat_id=?
                ORDER BY created_at
                """,
                key.to_tuple(),
            )
            messages: List[ChatMessage] = []
            for row in cursor.fetchall():
                role, content_json, response_category, created_at = row
                content = json.loads(content_json)

                message = self._create_message_from_db_row(
                    role=role,
                    content=content,
                    response_category=response_category,
                    timestamp=created_at,
                )
                messages.append(message)
            return messages

    def _write_single_message_to_database(
        self, key: ConversationKey, message: ChatMessage
    ) -> None:
        """Write a single message to SQLite database.

        Args:
            key: Conversation key.
            message: Chat message to store.
        """
        with self._database_connection() as connection:
            timestamp = (
                message.timestamp
                if message.timestamp is not None
                else datetime.now(timezone.utc).timestamp()
            )

            if message.timestamp is None:
                message.timestamp = timestamp

            # CopilotSystemMessage doesn't have a content field
            content = []
            if message.role != ROLE_SYSTEM:
                content = [c.model_dump() for c in message.content]

            content_json = json.dumps(content)

            connection.execute(
                """
                INSERT INTO copilot_messages
                (
                    chat_id,
                    role,
                    content_json,
                    response_category,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    *key.to_tuple(),
                    message.role,
                    content_json,
                    message.response_category.value
                    if message.response_category
                    else None,
                    timestamp,
                ),
            )
            connection.commit()

    def _delete_conversation_from_database(self, key: ConversationKey) -> None:
        """Delete conversation from SQLite database.

        Args:
            key: Conversation key.
        """
        with self._database_connection() as connection:
            connection.execute(
                """
                DELETE FROM copilot_messages
                WHERE chat_id=?
                """,
                key.to_tuple(),
            )
            connection.commit()

    def _handle_missing_table_error(self, exc: sqlite3.OperationalError) -> bool:
        """Check if error is due to missing table and reinitialize if needed.

        Args:
            exc: The SQLite operational error.

        Returns:
            True if table was missing and reinitialized, False otherwise.
        """
        if "no such table" in str(exc).lower():
            structlogger.warning("copilot_history_store.table_missing", error=str(exc))
            self._initialize_database()
            return True
        return False

    async def get(self, key: ConversationKey) -> List[ChatMessage]:
        """Get conversation history from SQLite database.

        Args:
            key: The conversation key.

        Returns:
            A list of chat messages in the conversation history.

        Raises:
            CopilotHistoryDatabaseError: If database operation fails.
        """
        try:
            return await asyncio.to_thread(self._read_conversation_from_database, key)
        except sqlite3.OperationalError as exc:
            if self._handle_missing_table_error(exc):
                return []
            raise CopilotHistoryDatabaseError(
                f"Failed to get conversation {key.chat_id}: {exc}"
            )
        except Exception as exc:
            structlogger.error(
                "copilot_history_store.get_failed",
                conversation_key=key.chat_id,
                error=str(exc),
            )
            raise CopilotHistoryDatabaseError(
                f"Failed to get conversation {key.chat_id}: {exc}"
            )

    async def append(self, key: ConversationKey, message: ChatMessage) -> None:
        """Append a message to SQLite conversation history.

        Args:
            key: The conversation key.
            message: The chat message to append.

        Raises:
            CopilotHistoryDatabaseError: If database operation fails.
        """
        try:
            await asyncio.to_thread(
                self._write_single_message_to_database,
                key,
                message,
            )
        except sqlite3.OperationalError as exc:
            if self._handle_missing_table_error(exc):
                await asyncio.to_thread(
                    self._write_single_message_to_database, key, message
                )
                return
            raise CopilotHistoryDatabaseError(
                f"Failed to append to conversation {key.chat_id}: {exc}"
            )
        except Exception as exc:
            structlogger.error(
                "copilot_history_store.append_failed",
                conversation_key=key.chat_id,
                message_role=getattr(message, "role", "unknown"),
                error=str(exc),
            )
            raise CopilotHistoryDatabaseError(
                f"Failed to append to conversation {key.chat_id}: {exc}"
            )

    async def delete(self, key: ConversationKey) -> None:
        """Delete conversation history from SQLite database.

        Args:
            key: The conversation key.

        Raises:
            CopilotHistoryDatabaseError: If database operation fails.
        """
        try:
            await asyncio.to_thread(self._delete_conversation_from_database, key)
        except Exception as exc:
            structlogger.error(
                "copilot_history_store.delete_failed",
                conversation_key=key.chat_id,
                error=str(exc),
            )
            raise CopilotHistoryDatabaseError(
                f"Failed to delete conversation {key.chat_id}: {exc}"
            )


async def persist_copilot_message_to_history(
    content: Optional[List[ContentBlock]] = None,
    text: Optional[str] = None,
    references: Optional[List[ReferenceEntry]] = None,
    chat_id: str = DEFAULT_COPILOT_CHAT_ID,
    response_category: ResponseCategory = ResponseCategory.COPILOT,
) -> None:
    """Persist a copilot message to conversation history.

    This is the general function for persisting any copilot message.
    You can either provide a pre-built content array, or text with optional references.

    Args:
        content: Optional list of ContentBlock objects (takes precedence over text)
        text: Optional text content for simple text-only messages
        references: Optional list of ReferenceEntry objects to include
        chat_id: The chat ID to persist the message to
        response_category: The response category for the message
    """
    # If neither content nor text provided, nothing to persist
    if not content and not text:
        return

    try:
        # Import here to avoid circular dependency
        from rasa.builder.llm_service import llm_service

        conversation_key = ConversationKey(chat_id=chat_id)

        # Build content blocks
        if content:
            message_content = content
        else:
            message_content = [TextContent(type="text", text=text)]

            # Add references as a content block if provided
            if references:
                reference_items = [
                    ReferenceItem(index=ref.index, title=ref.title, url=ref.url)
                    for ref in references
                ]
                message_content.append(
                    ReferencesContent(type="references", references=reference_items)
                )

        copilot_message = CopilotChatMessage(
            role="copilot",
            content=message_content,
            response_category=response_category,
        )

        await llm_service.history_store.append(conversation_key, copilot_message)

        structlogger.debug(
            "copilot_message_persisted",
            chat_id=chat_id,
            response_category=response_category.value,
            content_blocks=len(message_content),
        )
    except Exception as persist_exc:
        structlogger.error(
            "copilot_message_persist_failed",
            error=str(persist_exc),
            chat_id=chat_id,
        )


async def persist_training_error_analysis_to_history(
    text: Optional[str] = None,
    logs: Optional[List[LogContent]] = None,
    references: Optional[List[ReferenceEntry]] = None,
    chat_id: str = DEFAULT_COPILOT_CHAT_ID,
    response_category: ResponseCategory = ResponseCategory.TRAINING_ERROR_LOG_ANALYSIS,
) -> None:
    """Persist training error analysis as a single message with all content types.

    Args:
        text: Optional main analysis text
        logs: Optional list of log content blocks
        references: Optional list of reference entries
        chat_id: The chat ID to persist the message to
        response_category: The response category for the message
    """
    if not text and not logs and not references:
        return

    # Build content array with all content types
    content: List[ContentBlock] = []

    if text:
        content.append(TextContent(type="text", text=text))

    if logs:
        log_items = [
            LogItem(
                type="log",
                content=log.content,
                context=log.context,
                metadata=log.metadata or {},
            )
            for log in logs
        ]
        content.append(LogsContent(type="logs", logs=log_items))

    if references:
        reference_items = [
            ReferenceItem(index=ref.index, title=ref.title, url=ref.url)
            for ref in references
        ]
        content.append(ReferencesContent(type="references", references=reference_items))

    await persist_copilot_message_to_history(
        content=content,
        chat_id=chat_id,
        response_category=response_category,
    )


async def persist_user_message_to_history(
    text: str,
    chat_id: str = DEFAULT_COPILOT_CHAT_ID,
) -> None:
    """Persist a user message to conversation history.

    Args:
        text: The text content of the user message
        chat_id: The chat ID to persist the message to
    """
    if not text:
        return

    try:
        # Import here to avoid circular dependency
        from rasa.builder.llm_service import llm_service

        conversation_key = ConversationKey(chat_id=chat_id)

        user_message = UserChatMessage(
            role=ROLE_USER,
            content=[TextContent(type="text", text=text)],
        )

        await llm_service.history_store.append(conversation_key, user_message)

        structlogger.debug(
            "user_message_persisted",
            chat_id=chat_id,
        )
    except Exception as persist_exc:
        structlogger.error(
            "user_message_persist_failed",
            error=str(persist_exc),
            chat_id=chat_id,
        )

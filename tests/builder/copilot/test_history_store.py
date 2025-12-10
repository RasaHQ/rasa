import asyncio
import os
import sqlite3
import tempfile

import pytest

from rasa.builder.copilot.exceptions import CopilotHistoryDatabaseError
from rasa.builder.copilot.history_store import (
    CopilotHistoryStore,
    SQLiteCopilotHistoryStore,
    persist_copilot_message_to_history,
    persist_training_error_analysis_to_history,
)
from rasa.builder.copilot.models import (
    ButtonContent,
    CodeContent,
    ConversationKey,
    CopilotChatMessage,
    EventContent,
    FileContent,
    LinkContent,
    LogContent,
    LogsContent,
    ReferenceEntry,
    ReferenceItem,
    ReferencesContent,
    ResponseCategory,
    TextContent,
    UserChatMessage,
)
from rasa.builder.llm_service import LLMService, llm_service


@pytest.fixture
def conversation_key() -> ConversationKey:
    return ConversationKey(
        chat_id="test-chat",
    )


@pytest.fixture
def user_message() -> UserChatMessage:
    return UserChatMessage(
        role="user",
        content=[TextContent(type="text", text="Hello, how are you?")],
    )


@pytest.fixture
def copilot_message() -> CopilotChatMessage:
    return CopilotChatMessage(
        role="copilot",
        content=[TextContent(type="text", text="I'm doing well, thank you!")],
    )


class TestSQLiteCopilotHistoryStore:
    @pytest.fixture
    def temp_db_path(self) -> str:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_path = f.name

        yield temp_path

        # Cleanup after test
        if os.path.exists(temp_path):
            os.remove(temp_path)

    @pytest.fixture
    def store(self, temp_db_path: str) -> SQLiteCopilotHistoryStore:
        return SQLiteCopilotHistoryStore(temp_db_path)

    @pytest.mark.asyncio
    async def test_database_initialization(self, temp_db_path: str):
        # Create store to trigger database initialization
        SQLiteCopilotHistoryStore(temp_db_path)

        # Verify database file was created
        assert os.path.exists(temp_db_path)

        # Verify table exists by attempting a query
        conn = sqlite3.connect(temp_db_path)
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='copilot_messages'"
            )
            table_exists = cursor.fetchone() is not None
            assert table_exists
        finally:
            conn.close()

    @pytest.mark.asyncio
    async def test_get_empty_conversation(
        self, store: SQLiteCopilotHistoryStore, conversation_key: ConversationKey
    ):
        messages = await store.get(conversation_key)
        assert messages == []

    @pytest.mark.asyncio
    async def test_append_single_message(
        self,
        store: SQLiteCopilotHistoryStore,
        conversation_key: ConversationKey,
        user_message: CopilotChatMessage,
    ):
        await store.append(conversation_key, user_message)

        messages = await store.get(conversation_key)
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert messages[0].content[0].text == "Hello, how are you?"

    @pytest.mark.asyncio
    async def test_append_multiple_messages(
        self,
        store: SQLiteCopilotHistoryStore,
        conversation_key: ConversationKey,
        user_message: CopilotChatMessage,
        copilot_message: CopilotChatMessage,
    ):
        await store.append(conversation_key, user_message)
        await store.append(conversation_key, copilot_message)

        messages = await store.get(conversation_key)
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content[0].text == "Hello, how are you?"
        assert messages[1].role == "copilot"
        assert messages[1].content[0].text == "I'm doing well, thank you!"

    @pytest.mark.asyncio
    async def test_persistence_across_store_instances(
        self,
        temp_db_path: str,
        conversation_key: ConversationKey,
        user_message: CopilotChatMessage,
    ):
        # Create first store instance and add data
        store1 = SQLiteCopilotHistoryStore(temp_db_path)
        await store1.append(conversation_key, user_message)

        # Create second store instance and verify data is still there
        store2 = SQLiteCopilotHistoryStore(temp_db_path)
        messages = await store2.get(conversation_key)

        assert len(messages) == 1
        assert messages[0].role == "user"
        assert messages[0].content[0].text == "Hello, how are you?"

    @pytest.mark.asyncio
    async def test_delete_existing_conversation(
        self,
        store: SQLiteCopilotHistoryStore,
        conversation_key: ConversationKey,
        user_message: CopilotChatMessage,
    ):
        # Add a message first
        await store.append(conversation_key, user_message)
        assert len(await store.get(conversation_key)) == 1

        # Delete the conversation
        await store.delete(conversation_key)
        messages = await store.get(conversation_key)
        assert messages == []

    @pytest.mark.asyncio
    async def test_delete_non_existent_conversation(
        self, store: SQLiteCopilotHistoryStore, conversation_key: ConversationKey
    ):
        # Should not raise any exception
        await store.delete(conversation_key)

        # Should still return empty list
        messages = await store.get(conversation_key)
        assert messages == []

    @pytest.mark.asyncio
    async def test_multiple_conversations_isolated(
        self,
        store: SQLiteCopilotHistoryStore,
        user_message: CopilotChatMessage,
        copilot_message: CopilotChatMessage,
    ):
        key1 = ConversationKey(chat_id="chat1")
        key2 = ConversationKey(chat_id="chat2")
        key3 = ConversationKey(chat_id="chat3")

        # Add different messages to different conversations
        await store.append(key1, user_message)
        await store.append(key2, copilot_message)

        # Verify isolation
        messages1 = await store.get(key1)
        messages2 = await store.get(key2)
        messages3 = await store.get(key3)

        assert len(messages1) == 1
        assert len(messages2) == 1
        assert len(messages3) == 0

        assert messages1[0].role == "user"
        assert messages2[0].role == "copilot"

    @pytest.mark.asyncio
    async def test_concurrent_append_operations(
        self, store: SQLiteCopilotHistoryStore, conversation_key: ConversationKey
    ):
        async def append_message(index: int) -> None:
            message = UserChatMessage(
                role="user",
                content=[TextContent(type="text", text=f"Concurrent message {index}")],
            )
            await store.append(conversation_key, message)

        # Run multiple concurrent appends
        await asyncio.gather(*[append_message(i) for i in range(5)])

        messages = await store.get(conversation_key)
        assert len(messages) == 5

    @pytest.mark.asyncio
    async def test_all_content_types_preserved(
        self,
        store: SQLiteCopilotHistoryStore,
        conversation_key: ConversationKey,
    ):
        # Create a message with all content types
        complex_message = CopilotChatMessage(
            role="copilot",
            content=[
                TextContent(type="text", text="Here's the analysis:"),
                LogContent(
                    type="log",
                    content="Error: Training failed\nTraceback: ...",
                    context="training session",
                    metadata={"severity": "error", "component": "trainer"},
                ),
                EventContent(
                    type="event",
                    event="training_error",
                    error_type="config_invalid",
                    line=42,
                ),
                CodeContent(
                    type="code",
                    text=(
                        "config:\n  policies:\n    "
                        "- name: TEDPolicy\n      max_history: 5"
                    ),
                ),
                FileContent(
                    type="file",
                    file_path="config.yml",
                    file_content=(
                        "# Configuration file\npipeline:\n  "
                        "- name: WhitespaceTokenizer"
                    ),
                ),
                LinkContent(
                    type="link",
                    url="https://rasa.com/docs/rasa/policies/ted-policy/",
                    label="TEDPolicy Documentation",
                ),
                ButtonContent(
                    type="button",
                    payload="/fix_config",
                    label="Fix Configuration",
                ),
            ],
        )

        # Store the message
        await store.append(conversation_key, complex_message)

        # Retrieve the message
        messages = await store.get(conversation_key)
        assert len(messages) == 1

        retrieved_message = messages[0]
        assert retrieved_message.role == "copilot"
        assert len(retrieved_message.content) == 7

        # Verify each content type is properly preserved
        content_blocks = retrieved_message.content

        # Check TextContent
        assert isinstance(content_blocks[0], TextContent)
        assert content_blocks[0].text == "Here's the analysis:"

        # Check LogContent with all fields
        assert isinstance(content_blocks[1], LogContent)
        assert content_blocks[1].content == "Error: Training failed\nTraceback: ..."
        assert content_blocks[1].context == "training session"
        assert content_blocks[1].metadata == {
            "severity": "error",
            "component": "trainer",
        }

        # Check EventContent with event_data
        assert isinstance(content_blocks[2], EventContent)
        assert content_blocks[2].event == "training_error"
        assert content_blocks[2].event_data == {
            "error_type": "config_invalid",
            "line": 42,
        }

        # Check CodeContent
        assert isinstance(content_blocks[3], CodeContent)
        assert "TEDPolicy" in content_blocks[3].text

        # Check FileContent
        assert isinstance(content_blocks[4], FileContent)
        assert content_blocks[4].file_path == "config.yml"
        assert "WhitespaceTokenizer" in content_blocks[4].file_content

        # Check LinkContent
        assert isinstance(content_blocks[5], LinkContent)
        assert (
            content_blocks[5].url == "https://rasa.com/docs/rasa/policies/ted-policy/"
        )
        assert content_blocks[5].label == "TEDPolicy Documentation"

        # Check ButtonContent
        assert isinstance(content_blocks[6], ButtonContent)
        assert content_blocks[6].payload == "/fix_config"
        assert content_blocks[6].label == "Fix Configuration"

    def test_message_serialization_deserialization(self):
        """Test that message serialization and deserialization work correctly."""
        # Create messages with complex content
        messages = [
            UserChatMessage(
                role="user",
                content=[TextContent(type="text", text="Simple message")],
            ),
            CopilotChatMessage(
                role="copilot",
                content=[
                    TextContent(type="text", text="Here's some code:"),
                    TextContent(type="text", text="```python\nprint('hello')\n```"),
                    TextContent(type="text", text="And some unicode: 🤖 café naïve"),
                ],
            ),
        ]

        # Test serialization
        serialized = SQLiteCopilotHistoryStore._serialize_messages(messages)
        assert isinstance(serialized, str)
        assert "🤖" in serialized  # Unicode should be preserved
        assert "café naïve" in serialized
        assert "```python" in serialized

        # Test deserialization
        deserialized = SQLiteCopilotHistoryStore._deserialize_messages(serialized)
        assert len(deserialized) == 2
        assert deserialized[0].role == "user"
        assert deserialized[1].role == "copilot"
        assert len(deserialized[1].content) == 3
        assert deserialized[1].content[0].text == "Here's some code:"
        assert deserialized[1].content[1].text == "```python\nprint('hello')\n```"
        assert deserialized[1].content[2].text == "And some unicode: 🤖 café naïve"

    @pytest.mark.asyncio
    async def test_complex_message_content_persistence(
        self, store: SQLiteCopilotHistoryStore, conversation_key: ConversationKey
    ):
        # Create a message with multiple content blocks
        complex_message = CopilotChatMessage(
            role="copilot",
            content=[
                TextContent(type="text", text="Here's some code:"),
                TextContent(type="text", text="```python\nprint('hello')\n```"),
                TextContent(type="text", text="And some unicode: 🤖 café naïve"),
            ],
        )

        await store.append(conversation_key, complex_message)

        messages = await store.get(conversation_key)
        assert len(messages) == 1

        retrieved_message = messages[0]
        assert len(retrieved_message.content) == 3
        assert retrieved_message.content[0].text == "Here's some code:"
        assert retrieved_message.content[1].text == "```python\nprint('hello')\n```"
        assert retrieved_message.content[2].text == "And some unicode: 🤖 café naïve"

    @pytest.mark.asyncio
    async def test_whitespace_handling_in_keys(
        self, store: SQLiteCopilotHistoryStore, user_message: CopilotChatMessage
    ):
        # Keys with whitespace should be stripped and treated as equivalent
        key_with_spaces = ConversationKey(
            chat_id="  test-chat  ",
        )
        key_without_spaces = ConversationKey(
            chat_id="test-chat",
        )

        # Add message with key that has spaces
        await store.append(key_with_spaces, user_message)

        # Should be retrievable with key without spaces
        messages = await store.get(key_without_spaces)
        assert len(messages) == 1
        assert messages[0].role == "user"

    @pytest.mark.asyncio
    async def test_large_conversation_history(
        self, store: SQLiteCopilotHistoryStore, conversation_key: ConversationKey
    ):
        # Create a conversation with many messages
        for i in range(100):
            if i % 2 == 0:
                message = UserChatMessage(
                    role="user",
                    content=[TextContent(type="text", text=f"Message number {i}")],
                )
            else:
                message = CopilotChatMessage(
                    role="copilot",
                    content=[TextContent(type="text", text=f"Message number {i}")],
                )
            await store.append(conversation_key, message)

        messages = await store.get(conversation_key)
        assert len(messages) == 100

        # Verify order is maintained
        for i, message in enumerate(messages):
            expected_role = "user" if i % 2 == 0 else "copilot"
            assert message.role == expected_role
            assert message.content[0].text == f"Message number {i}"

    @pytest.mark.asyncio
    async def test_database_directory_creation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = os.path.join(temp_dir, "nested", "subdir", "test.db")

            # Directory shouldn't exist initially
            assert not os.path.exists(os.path.dirname(nested_path))

            # Creating store should create the directory
            SQLiteCopilotHistoryStore(nested_path)
            assert os.path.exists(os.path.dirname(nested_path))
            assert os.path.exists(nested_path)

    @pytest.mark.asyncio
    async def test_empty_and_none_content_handling(
        self, store: SQLiteCopilotHistoryStore, conversation_key: ConversationKey
    ):
        # Message with empty text
        empty_message = UserChatMessage(
            role="user",
            content=[TextContent(type="text", text="")],
        )

        # Message with only whitespace
        whitespace_message = CopilotChatMessage(
            role="copilot",
            content=[TextContent(type="text", text="   \n\t   ")],
        )

        await store.append(conversation_key, empty_message)
        await store.append(conversation_key, whitespace_message)

        messages = await store.get(conversation_key)
        assert len(messages) == 2
        assert messages[0].content[0].text == ""
        assert messages[1].content[0].text == "   \n\t   "

    @pytest.mark.asyncio
    async def test_corrupted_data_handling(
        self, store: SQLiteCopilotHistoryStore, conversation_key: ConversationKey
    ):
        # First, insert valid data
        user_message = UserChatMessage(
            role="user",
            content=[TextContent(type="text", text="Hello")],
        )
        await store.append(conversation_key, user_message)

        # Now corrupt the data directly in the database
        tuple_key = conversation_key.to_tuple()
        with store._database_connection() as connection:
            # Insert invalid JSON that's not a list
            connection.execute(
                """
                UPDATE copilot_messages
                SET content_json = '{"invalid": "not a list"}'
                WHERE chat_id=?
                """,
                tuple_key,
            )
            connection.commit()

        with pytest.raises(CopilotHistoryDatabaseError) as exc_info:
            await store.get(conversation_key)

        assert "Failed to get conversation" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_invalid_json_handling(
        self, store: SQLiteCopilotHistoryStore, conversation_key: ConversationKey
    ):
        # Insert invalid JSON directly
        tuple_key = conversation_key.to_tuple()
        with store._database_connection() as connection:
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
                (*tuple_key, "user", "invalid json {", None, 1234567890.0),
            )
            connection.commit()

        with pytest.raises(CopilotHistoryDatabaseError) as exc_info:
            await store.get(conversation_key)

        assert "Failed to get conversation" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_database_connection_error_propagation(self, temp_db_path: str):
        store = SQLiteCopilotHistoryStore(temp_db_path)

        # Close the database file and make it inaccessible
        os.chmod(temp_db_path, 0o000)  # Remove all permissions

        try:
            conversation_key = ConversationKey(chat_id="test")

            # This should raise a database error due to permission issues
            with pytest.raises(CopilotHistoryDatabaseError):
                await store.get(conversation_key)

        finally:
            # Restore permissions for cleanup
            os.chmod(temp_db_path, 0o644)

    @pytest.mark.asyncio
    async def test_auto_recovery_on_missing_table_get(
        self, temp_db_path: str, conversation_key: ConversationKey
    ):
        # Create store and verify it initializes properly
        store = SQLiteCopilotHistoryStore(temp_db_path)

        # Add a message to verify the table exists
        user_message = UserChatMessage(
            role="user",
            content=[TextContent(type="text", text="Test message")],
        )
        await store.append(conversation_key, user_message)

        # Manually drop the table to simulate the issue
        with store._database_connection() as connection:
            connection.execute("DROP TABLE copilot_messages")
            connection.commit()

        # Verify table is gone
        with store._database_connection() as connection:
            cursor = connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='copilot_messages'"
            )
            assert cursor.fetchone() is None

        # The get() should auto-recover and return empty list
        messages = await store.get(conversation_key)
        assert messages == []

        # Verify the table was recreated
        with store._database_connection() as connection:
            cursor = connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='copilot_messages'"
            )
            assert cursor.fetchone() is not None

    @pytest.mark.asyncio
    async def test_auto_recovery_on_missing_table_append(
        self, temp_db_path: str, conversation_key: ConversationKey
    ):
        # Create store and verify it initializes properly
        store = SQLiteCopilotHistoryStore(temp_db_path)

        # Manually drop the table to simulate the issue
        with store._database_connection() as connection:
            connection.execute("DROP TABLE copilot_messages")
            connection.commit()

        # Create a new message
        user_message = UserChatMessage(
            role="user",
            content=[TextContent(type="text", text="Test after recovery")],
        )

        # The append() should auto-recover and successfully write
        await store.append(conversation_key, user_message)

        # Verify the table was recreated and message was saved
        messages = await store.get(conversation_key)
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert messages[0].content[0].text == "Test after recovery"

    @pytest.mark.asyncio
    async def test_handle_missing_table_error_helper(self, temp_db_path: str):
        # Create store
        store = SQLiteCopilotHistoryStore(temp_db_path)

        # Drop the table
        with store._database_connection() as connection:
            connection.execute("DROP TABLE copilot_messages")
            connection.commit()

        # Create an OperationalError similar to what SQLite raises
        try:
            with store._database_connection() as connection:
                connection.execute("SELECT * FROM copilot_messages")
        except sqlite3.OperationalError as exc:
            # Test that the helper correctly identifies and handles the error
            result = store._handle_missing_table_error(exc)
            assert result is True

            # Verify table was recreated
            with store._database_connection() as connection:
                cursor = connection.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name='copilot_messages'"
                )
                assert cursor.fetchone() is not None

        # Test with a different operational error (should return False)
        other_error = sqlite3.OperationalError("database is locked")
        result = store._handle_missing_table_error(other_error)
        assert result is False

    def test_llm_service_history_store_integration(self):
        # Create a fresh LLMService instance
        service = LLMService()

        # Initially, _history_store should be None (lazy loading)
        assert service._history_store is None

        # First access should initialize the store
        store1 = service.history_store
        assert isinstance(store1, CopilotHistoryStore)
        assert isinstance(store1, SQLiteCopilotHistoryStore)

        # Second access should return the same instance (not create new one)
        store2 = service.history_store
        assert store2 is store1


def test_llm_service_creates_history_store():
    store = llm_service.history_store
    assert isinstance(store, CopilotHistoryStore)
    assert isinstance(store, SQLiteCopilotHistoryStore)


@pytest.fixture
async def initialized_history_store():
    # Create a temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_path = f.name

    # Create and initialize the store
    temp_store = SQLiteCopilotHistoryStore(temp_path)

    # Replace llm_service's history store with our temp store
    original_store = llm_service._history_store
    llm_service._history_store = temp_store

    yield temp_store

    # Restore original store
    llm_service._history_store = original_store

    # Clean up temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.mark.asyncio
async def test_persist_copilot_message_to_history(initialized_history_store):
    test_chat_id = "test_persist_general"
    conversation_key = ConversationKey(chat_id=test_chat_id)

    # Clean up any existing messages
    await llm_service.history_store.delete(conversation_key)

    # Test 1: Persist a simple text message
    await persist_copilot_message_to_history(
        text="This is a simple text message",
        chat_id=test_chat_id,
        response_category=ResponseCategory.COPILOT,
    )

    # Verify the message was persisted
    messages = await llm_service.history_store.get(conversation_key)
    assert len(messages) == 1
    assert messages[0].role == "copilot"
    assert messages[0].response_category == ResponseCategory.COPILOT
    assert len(messages[0].content) == 1
    assert isinstance(messages[0].content[0], TextContent)
    assert messages[0].content[0].text == "This is a simple text message"

    # Test 2: Persist a message with pre-built content blocks
    content_blocks = [
        TextContent(type="text", text="Message with multiple blocks"),
        CodeContent(type="code", text="print('hello world')"),
        LinkContent(type="link", url="https://example.com", label="Example Link"),
    ]
    await persist_copilot_message_to_history(
        content=content_blocks,
        chat_id=test_chat_id,
        response_category=ResponseCategory.COPILOT,
    )

    # Verify both messages exist
    messages = await llm_service.history_store.get(conversation_key)
    assert len(messages) == 2

    # Check the second message
    second_message = messages[1]
    assert second_message.role == "copilot"
    assert len(second_message.content) == 3
    assert isinstance(second_message.content[0], TextContent)
    assert isinstance(second_message.content[1], CodeContent)
    assert isinstance(second_message.content[2], LinkContent)
    assert second_message.content[1].text == "print('hello world')"

    # Test 3: Content parameter takes precedence over text
    await persist_copilot_message_to_history(
        content=[TextContent(type="text", text="Content wins")],
        text="This should be ignored",
        chat_id=test_chat_id,
    )

    messages = await llm_service.history_store.get(conversation_key)
    assert len(messages) == 3
    assert messages[2].content[0].text == "Content wins"

    # Clean up
    await llm_service.history_store.delete(conversation_key)


@pytest.mark.asyncio
async def test_persist_training_error_analysis_to_history(initialized_history_store):
    test_chat_id = "test_training_error_analysis"
    conversation_key = ConversationKey(chat_id=test_chat_id)

    # Clean up any existing messages
    await llm_service.history_store.delete(conversation_key)

    # Test: Persist a complete training error analysis
    text = "Training failed due to a configuration error in the flow definition"
    logs = [
        LogContent(
            type="log",
            content="Error: Undefined action 'check_balanc' in flow",
            context="training_error",
            metadata={"line": 42, "file": "check_balance.yml"},
        ),
        LogContent(
            type="log",
            content="Validation failed with exception",
            context="validation",
        ),
    ]
    references = [
        ReferenceEntry(
            index=1,
            title="Flow Configuration Guide",
            url="https://rasa.com/docs/flows",
        ),
        ReferenceEntry(
            index=2,
            title="Training Documentation",
            url="https://rasa.com/docs/training",
        ),
    ]

    await persist_training_error_analysis_to_history(
        text=text,
        logs=logs,
        references=references,
        chat_id=test_chat_id,
        response_category=ResponseCategory.TRAINING_ERROR_LOG_ANALYSIS,
    )

    # Verify the message was persisted correctly
    messages = await llm_service.history_store.get(conversation_key)
    assert len(messages) == 1

    message = messages[0]
    assert message.role == "copilot"
    assert message.response_category == ResponseCategory.TRAINING_ERROR_LOG_ANALYSIS
    assert (
        len(message.content) == 3
    )  # text, logs, references (order based on implementation)

    # Verify text content
    text_content = message.content[0]
    assert isinstance(text_content, TextContent)
    assert text_content.text == text

    # Verify logs content (second in the implementation)
    logs_content = message.content[1]
    assert isinstance(logs_content, LogsContent)
    assert len(logs_content.logs) == 2
    assert (
        logs_content.logs[0].content == "Error: Undefined action 'check_balanc' in flow"
    )
    assert logs_content.logs[0].context == "training_error"
    assert logs_content.logs[0].metadata == {"line": 42, "file": "check_balance.yml"}
    assert logs_content.logs[1].content == "Validation failed with exception"

    # Verify references content (third in the implementation)
    references_content = message.content[2]
    assert isinstance(references_content, ReferencesContent)
    assert len(references_content.references) == 2
    assert references_content.references[0].index == 1
    assert references_content.references[0].title == "Flow Configuration Guide"
    assert references_content.references[1].index == 2
    assert references_content.references[1].url == "https://rasa.com/docs/training"

    # Test with only some content types
    await persist_training_error_analysis_to_history(
        text="Only text, no logs or references",
        chat_id=test_chat_id,
    )

    messages = await llm_service.history_store.get(conversation_key)
    assert len(messages) == 2
    assert len(messages[1].content) == 1  # Only text content
    assert isinstance(messages[1].content[0], TextContent)

    # Clean up
    await llm_service.history_store.delete(conversation_key)


@pytest.mark.asyncio
async def test_copilot_message_with_references(initialized_history_store):
    test_chat_id = "test_copilot_references"
    conversation_key = ConversationKey(chat_id=test_chat_id)

    # Clean up any existing messages
    await llm_service.history_store.delete(conversation_key)

    # Build content blocks as the copilot endpoint would
    content_blocks = [
        TextContent(
            type="text",
            text="A flow is a conversational pattern that your assistant can follow.",
        ),
        ReferencesContent(
            type="references",
            references=[
                ReferenceItem(
                    index=1,
                    title="Business Logic with Flows",
                    url="https://rasa.com/docs/rasa/flows",
                ),
                ReferenceItem(
                    index=2,
                    title="Flow Builder",
                    url="https://rasa.com/docs/rasa/flow-builder",
                ),
            ],
        ),
    ]

    # Persist as the copilot endpoint would
    await persist_copilot_message_to_history(
        content=content_blocks,
        chat_id=test_chat_id,
        response_category=ResponseCategory.COPILOT,
    )

    # Verify the message was stored correctly
    messages = await llm_service.history_store.get(conversation_key)
    assert len(messages) == 1

    message = messages[0]
    assert message.role == "copilot"
    assert message.response_category == ResponseCategory.COPILOT
    assert len(message.content) == 2

    # Verify text content
    assert isinstance(message.content[0], TextContent)
    assert "flow is a conversational pattern" in message.content[0].text

    # Verify references content
    assert isinstance(message.content[1], ReferencesContent)
    assert len(message.content[1].references) == 2
    assert message.content[1].references[0].title == "Business Logic with Flows"
    assert message.content[1].references[1].index == 2

    # Verify serialization for frontend
    serialized = message.model_dump()
    assert serialized["content"][1]["type"] == "references"
    assert len(serialized["content"][1]["references"]) == 2

    # Clean up
    await llm_service.history_store.delete(conversation_key)

import base64
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Generator, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, PropertyMock, patch

import pytest
from pytest import MonkeyPatch
from sanic import Sanic

from rasa.builder import config
from rasa.builder.copilot.history_store import SQLiteCopilotHistoryStore
from rasa.builder.copilot.models import (
    ConversationKey,
    CopilotChatMessage,
    CopilotGenerationContext,
    GeneratedContent,
    ResponseCategory,
    ResponseCompleteness,
    TextContent,
    UserChatMessage,
)
from rasa.builder.document_retrieval.models import Document
from rasa.builder.guardrails.clients import LakeraAIGuardrails
from rasa.builder.guardrails.models import (
    GuardrailResponse,
)
from rasa.builder.job_manager import job_manager
from rasa.builder.models import JobStatus, JobStatusEvent
from rasa.builder.service import bp, setup_project_generator
from rasa.cli.scaffold import ProjectTemplateName
from rasa.core.actions.direct_custom_actions_executor import DirectCustomActionExecutor
from rasa.shared.core.domain import Domain
from rasa.utils.endpoints import EndpointConfig


@pytest.fixture()
def sanic_app() -> Sanic:
    app = Sanic("bot_builder_test")
    app.blueprint(bp)
    # add a dummy input channel expected by service.get_input_channel
    app.ctx.input_channel = MagicMock(latest_tracker_session_id=None)
    return app


@pytest.fixture(autouse=True)
def patch_copilot_dependencies(monkeypatch):
    """Patch Copilot/LLM for offline testing.

    Patches all Copilot/LLM bits so that the /api/copilot route can run
    entirely offline and without hitting third-party services.
    """
    # 1. Patch project generator to return a mock project
    project_folder = SimpleNamespace(name="proj")
    mock_git_service = MagicMock()
    mock_git_service.git_operation = MagicMock(
        return_value=AsyncMock().__aenter__.return_value
    )
    mock_git_service.get_current_commit_sha = AsyncMock(return_value="mock_sha_before")
    pg = SimpleNamespace(
        project_folder=project_folder,
        get_bot_files=lambda *args, **kwargs: {},
        git_service=mock_git_service,
        unsafe_commit_changes=AsyncMock(return_value="mock_sha_after"),
    )
    monkeypatch.setattr("rasa.builder.service.get_project_generator", lambda _: pg)

    # 2. Patch Copilot's generate_response method to return a mock handler
    # make sure all keys expected by log_copilot_from_handler are present
    usage_stats = SimpleNamespace(
        model_dump=lambda: {
            "model": "gpt-mock",
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        }
    )

    # 3. Create a mock handler (returned by Copilot.generate_response)
    token = SimpleNamespace(
        content="hi",
        response_category=ResponseCategory.COPILOT,
        response_completeness=ResponseCompleteness.COMPLETE,
        to_sse_event=lambda: SimpleNamespace(format=lambda: ""),
    )

    async def _stream():
        yield token

    handler = SimpleNamespace(
        generated_responses=[token],
        stream=_stream,
        extract_references=lambda *_: SimpleNamespace(
            to_sse_event=lambda: SimpleNamespace(format=lambda: "")
        ),
        extract_response_category=lambda: ResponseCategory.COPILOT,
        extract_text_from_generated_responses=lambda: "hi",
    )

    fake_copilot = SimpleNamespace(
        generate_response=AsyncMock(
            return_value=(
                handler,
                CopilotGenerationContext(
                    relevant_documents=[
                        Document(
                            title="test doc",
                            content="test doc",
                            url="https://doc",
                        )
                    ],
                    system_message={"role": "system", "content": "test system message"},
                    chat_history=[{"role": "user", "content": "test history"}],
                    last_user_message={"role": "user", "content": "test user message"},
                    tracker_event_attachments=[],
                ),
            )
        ),
        usage_statistics=usage_stats,
    )

    # Patch get_copilot_class to return a lambda that returns fake_copilot
    monkeypatch.setattr(
        "rasa.builder.service.get_copilot_class",
        lambda: lambda: fake_copilot,
    )

    # 4. Additional patches to avoid errors in the service
    monkeypatch.setattr("rasa.builder.service.get_recent_logs", lambda: "")
    monkeypatch.setattr(
        "rasa.builder.service.current_tracker_from_input_channel",
        AsyncMock(return_value=None),
    )

    # Guard-rails must not raise or flag
    async def _no_flag(self, request):
        return GuardrailResponse(
            hello_rasa_user_id="test",
            hello_rasa_project_id="test",
            flagged=False,
            detections=[],
        )

    monkeypatch.setattr(LakeraAIGuardrails, "send_request", _no_flag)


@pytest.fixture
def self_removable_path(tmp_path: Path) -> Generator[Path, Any, None]:
    """Fixture that ensures temporary directory is cleaned up after test completion."""
    yield tmp_path

    # Clean up filesystem
    shutil.rmtree(tmp_path, ignore_errors=True)

    # Clean up sys.path
    tmp_path_str = str(tmp_path)

    # Remove all occurrences safely
    for _ in range(sys.path.count(tmp_path_str)):
        sys.path.remove(tmp_path_str)


def _setup_copilot_mocks(
    monkeypatch: MonkeyPatch,
    expected_response: str,
    history_store: Optional[SQLiteCopilotHistoryStore] = None,
) -> None:
    # Mock response handler
    mock_generated_response = GeneratedContent(
        content=expected_response, response_category=ResponseCategory.COPILOT
    )

    async def mock_stream():
        for text in expected_response.split():
            token = MagicMock()
            token.to_sse_event.return_value.format.return_value = f"data: {text}\n\n"
            yield token

    def mock_instantiate_handler(*args, **kwargs):
        handler = MagicMock()
        handler.generated_responses = [mock_generated_response]
        handler.stream = mock_stream
        mock_reference = MagicMock()
        mock_reference.to_sse_event.return_value.format.return_value = ""
        handler.extract_references.return_value = mock_reference
        handler.extract_response_category.return_value = ResponseCategory.COPILOT
        handler.extract_text_from_generated_responses.return_value = expected_response
        return handler

    # Mock copilot client
    async def mock_generate_response(context):
        # Create a handler instance
        handler = mock_instantiate_handler()

        # Return the handler and a proper generation context
        mock_generation_context = MagicMock()
        mock_generation_context.relevant_documents = []
        return handler, mock_generation_context

    mock_copilot = MagicMock()
    mock_copilot.generate_response = mock_generate_response

    # Patch get_copilot_class to return a lambda that returns mock_copilot
    monkeypatch.setattr(
        "rasa.builder.service.get_copilot_class", lambda: lambda: mock_copilot
    )

    # Mock llm_service for history_store and guardrails
    mock_llm_service = MagicMock()
    mock_llm_service.history_store = history_store

    # Mock guardrails_policy_checker to return None (no violations)
    mock_guardrails_checker = MagicMock()
    mock_guardrails_checker.check_copilot_chat_for_policy_violations = AsyncMock(
        return_value=None
    )
    mock_guardrails_checker.check_assistant_chat_for_policy_violations = AsyncMock(
        return_value=None
    )
    mock_llm_service.guardrails_policy_checker = mock_guardrails_checker

    # Configure history_store if provided
    if history_store is not None:
        type(mock_llm_service).history_store = PropertyMock(return_value=history_store)

    monkeypatch.setattr("rasa.builder.service.llm_service", mock_llm_service)


def test_setup_project_generator_adds_to_sys_path(self_removable_path: Path):
    tmp_path_str = str(self_removable_path)
    assert tmp_path_str not in sys.path

    setup_project_generator(tmp_path_str)
    assert tmp_path_str in sys.path


def test_setup_project_generator_avoids_duplicate_sys_path_entries(
    self_removable_path: Path,
):
    tmp_path_str = str(self_removable_path)
    assert tmp_path_str not in sys.path

    setup_project_generator(tmp_path_str)
    setup_project_generator(tmp_path_str)

    assert sys.path.count(tmp_path_str) == 1


async def test_template_loads_actions_module(
    self_removable_path: Path,
    monkeypatch: MonkeyPatch,
    default_event_loop_policy,
):
    """Test that template loads actions module correctly."""
    # Set up isolated environment
    tmp_path_str = str(self_removable_path)
    monkeypatch.chdir(tmp_path_str)
    # this is required for the human handoff action in the telco example
    monkeypatch.setenv("OPENAI_API_KEY", "test-foo-bar")

    # Initialize project from template, and fetch action names
    project_generator = setup_project_generator(tmp_path_str)
    await project_generator.init_from_template(ProjectTemplateName.TELCO)
    project_action_names = Domain.from_path(tmp_path_str).action_names_or_texts

    # Set up action executor with mock endpoint, and fetch registered actions
    mock_endpoint = EndpointConfig.from_dict({"actions_module": "actions"})
    actions_executor = DirectCustomActionExecutor("test", mock_endpoint)
    registered_actions = actions_executor.action_executor.actions
    registered_action_names = list(registered_actions.keys())

    # Ensure all registered actions are in the project actions
    for action in registered_action_names:
        assert action in project_action_names


@pytest.mark.asyncio
async def test_job_events_unknown_id_returns_404(sanic_app: Sanic):
    async with sanic_app.asgi_client as client:
        _, response = await client.get("/api/job-events/does-not-exist")

    assert response.status == 404
    payload = json.loads(response.body)
    assert payload["error"] == "Job not found"
    assert payload["details"]["job_id"] == "does-not-exist"


@pytest.mark.asyncio
async def test_job_events_streams_events(sanic_app: Sanic):
    job = job_manager.create_job()
    await job.put(JobStatusEvent.from_status(JobStatus.generating))
    await job.put(JobStatusEvent.eof())

    async with sanic_app.asgi_client as client:
        _, response = await client.get(f"/api/job-events/{job.id}")

    assert response.status == 200
    body = response.body.decode()

    assert "event: progress" in body
    assert '"status": "generating"' in body

    # Ensure the stream terminates with an EOF event
    assert body.endswith("\n\n")


@pytest.mark.asyncio
async def test_job_events_stops_after_eof(sanic_app: Sanic):
    # Create a job with two progress events followed by EOF
    job = job_manager.create_job()
    await job.put(JobStatusEvent.from_status(JobStatus.generating))
    await job.put(JobStatusEvent.from_status(JobStatus.generation_success))
    await job.put(JobStatusEvent.eof())
    await job.put(JobStatusEvent.from_status(JobStatus.training))
    await job.put(JobStatusEvent.from_status(JobStatus.train_success))

    async with sanic_app.asgi_client as client:
        _, response = await client.get(f"/api/job-events/{job.id}")

    # Assert that only the first two events are streamed
    body = response.body.decode()
    assert body.count("event: progress") == 2


@pytest.mark.asyncio
async def test_get_assistant_info_returns_json(sanic_app: Sanic):
    assistant_id = "test-assistant"
    agent_mock = MagicMock()
    processor_mock = MagicMock()
    model_metadata_mock = MagicMock()
    model_metadata_mock.assistant_id = assistant_id
    processor_mock.model_metadata = model_metadata_mock
    agent_mock.processor = processor_mock

    sanic_app.ctx.agent = agent_mock

    async with sanic_app.asgi_client as client:
        _, response = await client.get("/api/assistant")

    assert response.status == 200
    payload = json.loads(response.body)
    assert payload == {"assistant_id": assistant_id}


class TestFilesEndpoint:
    """Test the POST /api/files endpoint."""

    @pytest.fixture
    def sample_bot_files(self):
        """Sample bot files for testing."""
        return {
            "config.yml": "version: '3.1'\npipeline: []",
            "domain.yml": "version: '3.1'\nintents: []",
            "data/nlu.yml": "version: '3.1'\nnlu: []",
        }

    @pytest.fixture
    def mock_project_generator(self, monkeypatch):
        """Mock project generator for testing."""
        mock_pg = Mock()
        mock_pg.replace_all_bot_files = Mock()
        mock_pg.get_training_input = Mock()
        monkeypatch.setattr(
            "rasa.builder.service.get_project_generator", lambda _: mock_pg
        )
        return mock_pg

    @pytest.mark.asyncio
    async def test_post_files_creates_job(
        self, sanic_app: Sanic, sample_bot_files, mock_project_generator
    ):
        """Test POST /api/files creates a job and returns job_id."""
        with patch("rasa.builder.service.run_replace_all_files_job") as mock_job:
            async with sanic_app.asgi_client as client:
                _, response = await client.post("/api/files", json=sample_bot_files)

            assert response.status == 200
            payload = json.loads(response.body)
            assert "job_id" in payload
            assert payload["status"] == "received"

            # Verify job was created and scheduled
            mock_job.assert_called_once()
            call_args = mock_job.call_args[0]
            assert call_args[2] == sample_bot_files  # bot_files argument

    @pytest.mark.asyncio
    async def test_post_files_invalid_json(
        self, sanic_app: Sanic, mock_project_generator
    ):
        """Test POST /api/files with invalid JSON returns 400."""
        async with sanic_app.asgi_client as client:
            _, response = await client.post("/api/files", data="invalid json")

        assert response.status == 400
        payload = json.loads(response.body)
        assert payload["error"] == "Invalid request"

    @pytest.mark.asyncio
    async def test_post_files_empty_payload(
        self, sanic_app: Sanic, mock_project_generator
    ):
        """Test POST /api/files with empty payload."""
        async with sanic_app.asgi_client as client:
            _, response = await client.post("/api/files", json={})

        assert response.status == 200
        payload = json.loads(response.body)
        assert "job_id" in payload

    @pytest.mark.asyncio
    async def test_post_files_with_none_values(
        self, sanic_app: Sanic, mock_project_generator
    ):
        """Test POST /api/files handles None values correctly."""
        files_with_none = {
            "config.yml": "version: '3.1'",
            "domain.yml": None,
            "data/nlu.yml": "nlu data",
        }

        async with sanic_app.asgi_client as client:
            _, response = await client.post("/api/files", json=files_with_none)

        assert response.status == 200
        payload = json.loads(response.body)
        assert "job_id" in payload

    @pytest.mark.asyncio
    async def test_post_files_job_creation_error(
        self, sanic_app: Sanic, mock_project_generator
    ):
        """Test POST /api/files handles job creation errors."""
        with patch("rasa.builder.service.job_manager.create_job") as mock_create_job:
            mock_create_job.side_effect = Exception("Job creation failed")

            async with sanic_app.asgi_client as client:
                _, response = await client.post(
                    "/api/files", json={"config.yml": "test"}
                )

            assert response.status == 500
            payload = json.loads(response.body)
            assert payload["error"] == "Failed to replace bot files"

    @pytest.mark.asyncio
    async def test_get_files_endpoint_still_works(self, sanic_app: Sanic):
        """Test that GET /api/files endpoint still works after POST changes."""
        mock_files = {"config.yml": "test config", "domain.yml": "test domain"}

        with patch("rasa.builder.service.get_project_generator") as mock_get_pg:
            mock_pg = Mock()
            mock_pg.get_bot_files.return_value = mock_files
            mock_get_pg.return_value = mock_pg

            async with sanic_app.asgi_client as client:
                _, response = await client.get("/api/files")

            assert response.status == 200
            payload = json.loads(response.body)
            assert payload == mock_files


class TestFilesEndpointIntegration:
    """Integration tests for the files endpoint with job execution."""

    @pytest.fixture
    def temp_project_dir(self, self_removable_path: Path):
        """Create a temporary project directory."""
        return self_removable_path / "project"

    @pytest.mark.asyncio
    async def test_full_replace_workflow(self, sanic_app: Sanic, temp_project_dir):
        """Test the full file replacement workflow."""
        temp_project_dir.mkdir()

        # Create initial files
        (temp_project_dir / "config.yml").write_text("old config")
        (temp_project_dir / "old_file.txt").write_text("old file")

        # Mock project generator with real directory
        from rasa.builder.project_generator.project_generator import ProjectGenerator

        real_pg = ProjectGenerator(str(temp_project_dir))
        sanic_app.ctx.project_generator = real_pg

        with (
            patch(
                "rasa.builder.jobs.validate_project", new_callable=AsyncMock
            ) as mock_validate,
            patch(
                "rasa.builder.job_helpers.train_and_load_agent",
                new_callable=AsyncMock,
            ) as mock_train,
            patch(
                "rasa.builder.job_helpers.link_model_to_commit",
                new_callable=AsyncMock,
            ),
            patch("rasa.builder.jobs.update_agent"),
        ):
            mock_validate.return_value = None
            mock_train.return_value = Mock()

            new_files = {"config.yml": "new config", "domain.yml": "new domain"}

            async with sanic_app.asgi_client as client:
                _, response = await client.post("/api/files", json=new_files)

            assert response.status == 200
            payload = json.loads(response.body)
            job_id = payload["job_id"]

            # Wait deterministically for the background job to complete
            job = job_manager.get_job(job_id)

            async def drain():
                async for _ in job.event_stream():
                    pass

            await drain()

            # Verify files were replaced correctly
            assert (temp_project_dir / "config.yml").read_text() == "new config"
            assert (temp_project_dir / "domain.yml").read_text() == "new domain"
            # Old file should be deleted
            assert not (temp_project_dir / "old_file.txt").exists()

    @pytest.mark.asyncio
    async def test_restricted_files_preserved(self, sanic_app: Sanic, temp_project_dir):
        """Test that restricted files are preserved during replacement."""
        temp_project_dir.mkdir()

        # Create restricted directories and files
        rasa_dir = temp_project_dir / ".rasa"
        rasa_dir.mkdir()
        (rasa_dir / "cache").write_text("cache data")

        models_dir = temp_project_dir / "models"
        models_dir.mkdir()
        (models_dir / "model.tar.gz").write_text("model data")

        # Create regular file
        (temp_project_dir / "config.yml").write_text("old config")

        from rasa.builder.project_generator.project_generator import ProjectGenerator

        real_pg = ProjectGenerator(str(temp_project_dir))
        sanic_app.ctx.project_generator = real_pg

        with (
            patch(
                "rasa.builder.jobs.validate_project", new_callable=AsyncMock
            ) as mock_validate,
            patch(
                "rasa.builder.job_helpers.train_and_load_agent",
                new_callable=AsyncMock,
            ) as mock_train,
            patch(
                "rasa.builder.job_helpers.link_model_to_commit",
                new_callable=AsyncMock,
            ),
            patch("rasa.builder.jobs.update_agent"),
        ):
            mock_validate.return_value = None
            mock_train.return_value = Mock()

            new_files = {"config.yml": "new config", "domain.yml": "new domain"}

            async with sanic_app.asgi_client as client:
                _, response = await client.post("/api/files", json=new_files)

            assert response.status == 200
            payload = json.loads(response.body)
            job_id = payload["job_id"]

            # Wait deterministically for the background job to complete
            job = job_manager.get_job(job_id)

            async def drain():
                async for _ in job.event_stream():
                    pass  # consume until EOF

            await drain()

            # Verify restricted files are preserved
            assert (rasa_dir / "cache").exists()
            assert (rasa_dir / "cache").read_text() == "cache data"
            assert (models_dir / "model.tar.gz").exists()
            assert (models_dir / "model.tar.gz").read_text() == "model data"

            # Verify new files were written
            assert (temp_project_dir / "config.yml").read_text() == "new config"
            assert (temp_project_dir / "domain.yml").read_text() == "new domain"


@pytest.mark.asyncio
async def test_copilot_endpoint_stores_messages_to_sqlite(
    sanic_app: Sanic, monkeypatch: MonkeyPatch
) -> None:
    """Test that /api/copilot stores both user and copilot messages to SQLite."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_db:
        temp_db_path = temp_db.name

    try:
        # Configure SQLite backend
        monkeypatch.setattr(config, "COPILOT_HISTORY_SQLITE_PATH", temp_db_path)
        monkeypatch.setattr(config, "HELLO_RASA_PROJECT_ID", "hello-rasa")
        monkeypatch.setattr("rasa.builder.service.HELLO_RASA_PROJECT_ID", "hello-rasa")

        # Setup mocks and store
        expected_response = "Hello! I can help you build a bot."
        test_store = SQLiteCopilotHistoryStore(temp_db_path)

        # Configure the llm_service singleton to use our test store
        # (both service.py and history_store.py import from llm_service module)
        monkeypatch.setattr(
            "rasa.builder.llm_service.llm_service._history_store", test_store
        )
        _setup_copilot_mocks(monkeypatch, expected_response, history_store=test_store)

        # Make request
        user_id, session_id = "test-user", "test-session"
        user_message_text = "Hello, can you help me build a bot?"
        payload = {
            "session_id": session_id,
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": user_message_text}],
            },
        }

        _, response = await sanic_app.asgi_client.post(
            "/api/copilot",
            json=payload,
            headers={"X-User-Id": user_id, "Accept": "text/event-stream"},
        )
        assert response.status == 200

        # Verify both messages stored
        conv_key = ConversationKey(chat_id="default")
        stored_messages = await test_store.get(conv_key)
        assert len(stored_messages) == 2

        # Verify user message
        user_msg = stored_messages[0]
        assert user_msg.role == "user"
        assert user_msg.content[0].text == user_message_text
        assert user_msg.response_category is None

        # Verify copilot message
        copilot_msg = stored_messages[1]
        assert copilot_msg.role == "copilot"
        assert copilot_msg.content[0].text == expected_response
        assert copilot_msg.response_category == ResponseCategory.COPILOT
    finally:
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)


@pytest.mark.asyncio
async def test_get_copilot_history_success(
    sanic_app: Sanic, monkeypatch: MonkeyPatch
) -> None:
    """Test GET /api/copilot/history returns stored messages successfully."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_db:
        temp_db_path = temp_db.name

    try:
        # Configure SQLite backend
        monkeypatch.setattr(config, "COPILOT_HISTORY_SQLITE_PATH", temp_db_path)
        monkeypatch.setattr(config, "HELLO_RASA_PROJECT_ID", "hello-rasa")
        monkeypatch.setattr("rasa.builder.service.HELLO_RASA_PROJECT_ID", "hello-rasa")

        # Create store and add test messages
        test_store = SQLiteCopilotHistoryStore(temp_db_path)
        monkeypatch.setattr(
            "rasa.builder.service.llm_service._history_store", test_store
        )

        # Make GET request
        _, response = await sanic_app.asgi_client.get(
            "/api/copilot/history",
            headers={"X-User-Id": "test-user"},
        )

        # Verify response
        assert response.status == 200
        response_data = response.json
        assert response_data["messages"] == []

        conversation_key = ConversationKey(chat_id="default")

        # Add test messages
        user_message = UserChatMessage(
            role="user",
            content=[TextContent(type="text", text="Hello, how are you?")],
        )
        await test_store.append(conversation_key, user_message)

        copilot_message = CopilotChatMessage(
            role="copilot",
            content=[TextContent(type="text", text="I'm doing well, thank you!")],
            response_category=ResponseCategory.COPILOT,
        )
        await test_store.append(conversation_key, copilot_message)

        # Make GET request
        _, response = await sanic_app.asgi_client.get(
            "/api/copilot/history",
            headers={"X-User-Id": "test-user"},
        )

        # Verify response
        assert response.status == 200
        response_data = response.json
        assert len(response_data["messages"]) == 2

        # Verify user message
        assert response_data["messages"][0]["role"] == "user"
        assert (
            response_data["messages"][0]["content"][0]["text"] == "Hello, how are you?"
        )
        assert response_data["messages"][0]["response_category"] is None

        # Verify copilot message
        assert response_data["messages"][1]["role"] == "copilot"
        assert (
            response_data["messages"][1]["content"][0]["text"]
            == "I'm doing well, thank you!"
        )
        assert response_data["messages"][1]["response_category"] == "copilot"
    finally:
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)


@pytest.mark.asyncio
async def test_delete_copilot_history_success(
    sanic_app: Sanic, monkeypatch: MonkeyPatch
) -> None:
    """Test DELETE /api/copilot/history successfully deletes stored messages."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_db:
        temp_db_path = temp_db.name

    try:
        # Configure SQLite backend
        monkeypatch.setattr(config, "COPILOT_HISTORY_SQLITE_PATH", temp_db_path)
        monkeypatch.setattr(config, "HELLO_RASA_PROJECT_ID", "hello-rasa")
        monkeypatch.setattr("rasa.builder.service.HELLO_RASA_PROJECT_ID", "hello-rasa")

        # Create store and add test messages
        test_store = SQLiteCopilotHistoryStore(temp_db_path)
        monkeypatch.setattr(
            "rasa.builder.service.llm_service._history_store", test_store
        )

        conv_key = ConversationKey(chat_id="default")

        # Make DELETE request
        _, response = await sanic_app.asgi_client.delete(
            "/api/copilot/history",
            headers={"X-User-Id": "test-user"},
        )

        # Verify response (should succeed)
        assert response.status == 200
        response_data = response.json
        assert response_data["status"] == "deleted"

        # Add test messages
        user_message = UserChatMessage(
            role="user",
            content=[TextContent(type="text", text="Hello, how are you?")],
        )
        await test_store.append(conv_key, user_message)

        # Verify messages exist before deletion
        messages_before = await test_store.get(conv_key)
        assert len(messages_before) == 1

        # Make DELETE request
        _, response = await sanic_app.asgi_client.delete(
            "/api/copilot/history",
            headers={"X-User-Id": "test-user"},
        )

        # Verify response
        assert response.status == 200
        response_data = response.json
        assert response_data["status"] == "deleted"

        # Verify messages are actually deleted
        messages_after = await test_store.get(conv_key)
        assert len(messages_after) == 0
    finally:
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)


class TestBackupToBotEndpoint:
    """Test /backup-to-bot endpoint."""

    @staticmethod
    def create_test_backup_data(files: Dict[str, str]) -> str:
        # Create a valid tar.gz backup data for testing
        with tempfile.NamedTemporaryFile(suffix=".tar.gz") as temp_file:
            with tarfile.open(temp_file.name, "w:gz") as tar:
                for filename, content in files.items():
                    info = tarfile.TarInfo(filename)
                    info.size = len(content.encode("utf-8"))
                    tar.addfile(info, fileobj=io.BytesIO(content.encode("utf-8")))

            temp_file.seek(0)
            backup_bytes = temp_file.read()
            return base64.b64encode(backup_bytes).decode("utf-8")

    async def test_backup_to_bot_endpoint_success(self, sanic_app, monkeypatch):
        mock_run_job = AsyncMock()
        monkeypatch.setattr("rasa.builder.service.run_backup_to_bot_job", mock_run_job)

        request_data = {
            "presigned_url": "https://s3.amazonaws.com/bucket/path?signature=test"
        }

        request, response = await sanic_app.asgi_client.post(
            "/api/backup-to-bot", json=request_data
        )

        assert response.status == 200
        response_data = response.json
        assert "job_id" in response_data
        mock_run_job.assert_called_once()

    async def test_backup_to_bot_endpoint_invalid_json(self, sanic_app):
        request, response = await sanic_app.asgi_client.post(
            "/api/backup-to-bot", data="invalid json"
        )

        assert response.status == 400
        response_data = response.json
        assert "error" in response_data
        assert "Invalid request" in response_data["error"]

    async def test_backup_to_bot_endpoint_missing_presigned_url(self, sanic_app):
        request_data = {}

        request, response = await sanic_app.asgi_client.post(
            "/api/backup-to-bot", json=request_data
        )

        assert response.status == 400
        response_data = response.json
        assert "error" in response_data

    async def test_backup_to_bot_endpoint_empty_presigned_url(self, sanic_app):
        request_data = {"presigned_url": ""}

        request, response = await sanic_app.asgi_client.post(
            "/api/backup-to-bot", json=request_data
        )

        assert response.status == 400
        response_data = response.json
        assert "error" in response_data
        assert "String should have at least 1 character" in str(response_data)

    async def test_backup_to_bot_endpoint_whitespace_presigned_url(self, sanic_app):
        request_data = {"presigned_url": "   "}

        request, response = await sanic_app.asgi_client.post(
            "/api/backup-to-bot", json=request_data
        )

        assert response.status == 400
        response_data = response.json
        assert "error" in response_data
        assert "Presigned URL cannot be empty" in str(response_data)

    async def test_backup_to_bot_endpoint_invalid_url_format(self, sanic_app):
        request_data = {"presigned_url": "not-a-valid-url"}

        request, response = await sanic_app.asgi_client.post(
            "/api/backup-to-bot", json=request_data
        )

        assert response.status == 400
        response_data = response.json
        assert "error" in response_data
        assert "must be a valid HTTP/HTTPS URL" in str(response_data)


class TestDownloadEndpoint:
    """Test download endpoint query parameter handling."""

    @pytest.fixture(autouse=True)
    def setup_download_mocks(self, monkeypatch):
        # Mock authentication to bypass auth requirements
        def mock_is_auth_required(**kwargs):
            return False

        monkeypatch.setattr(
            "rasa.builder.auth.is_auth_required_now", mock_is_auth_required
        )

        # Setup service mocks
        self.mock_pg = MagicMock()
        self.mock_pg.get_bot_files.return_value = {"config.yml": "version: '3.1'"}
        self.mock_get_pg = MagicMock(return_value=self.mock_pg)
        self.mock_create_archive = MagicMock(return_value=b"fake archive data")

        monkeypatch.setattr(
            "rasa.builder.service.get_project_generator", self.mock_get_pg
        )
        monkeypatch.setattr(
            "rasa.builder.service.create_bot_project_archive", self.mock_create_archive
        )

    async def test_download_exclude_models_default(self, sanic_app):
        # Make request without query parameters
        request, response = await sanic_app.asgi_client.get("/api/download")
        assert response.status == 200

        # Verify get_bot_files was called with default exclude_models_directory=True
        self.mock_pg.get_bot_files.assert_called_once_with(
            exclude_models_directory=True
        )

    async def test_download_exclude_models_true(self, sanic_app):
        request, response = await sanic_app.asgi_client.get(
            "/api/download?exclude_models_directory=true"
        )

        assert response.status == 200
        self.mock_pg.get_bot_files.assert_called_once_with(
            exclude_models_directory=True
        )

    async def test_download_combined_parameters(self, sanic_app):
        # Make request with both parameters
        request, response = await sanic_app.asgi_client.get(
            "/api/download?exclude_models_directory=false&project_name=full-bot"
        )
        assert response.status == 200

        # Verify both parameters are respected
        self.mock_pg.get_bot_files.assert_called_once_with(
            exclude_models_directory=False
        )
        call_args = self.mock_create_archive.call_args
        assert call_args[0][1] == "full-bot"


class TestInternalEndpoints:
    """Test internal endpoints for MCP server communication."""

    @pytest.fixture
    def sanic_app_with_agent(self, sanic_app: Sanic, tmp_path: Path) -> Sanic:
        """Create a Sanic app with a mock agent."""
        # Use a mock project generator with a real path
        mock_pg = MagicMock()
        mock_pg.project_folder = tmp_path
        sanic_app.ctx.project_generator = mock_pg

        mock_agent = MagicMock()
        mock_agent.is_ready.return_value = True
        mock_agent.tracker_store = MagicMock()
        sanic_app.ctx.agent = mock_agent
        return sanic_app

    @pytest.mark.asyncio
    async def test_reload_agent_internal_from_localhost(
        self, sanic_app_with_agent: Sanic, monkeypatch: MonkeyPatch
    ) -> None:
        """Test reload_agent_internal endpoint from localhost."""
        # Mock _is_localhost_request to return True (ASGI test client uses "mockserver")
        monkeypatch.setattr(
            "rasa.builder.service._is_localhost_request", lambda _: True
        )

        # Mock try_load_existing_agent to return a mock agent
        mock_agent = MagicMock()
        monkeypatch.setattr(
            "rasa.builder.service.try_load_existing_agent",
            AsyncMock(return_value=mock_agent),
        )
        monkeypatch.setattr("rasa.builder.service.update_agent", MagicMock())

        async with sanic_app_with_agent.asgi_client as client:
            _, response = await client.post("/api/internal/reload-agent")

        # Should succeed since we're from localhost
        assert response.status == 200
        payload = json.loads(response.body)
        assert payload["success"] is True
        assert payload["message"] == "Agent reloaded"

    @pytest.mark.asyncio
    async def test_reload_agent_internal_no_agent(
        self, sanic_app_with_agent: Sanic, monkeypatch: MonkeyPatch
    ) -> None:
        """Test reload_agent_internal when no agent exists."""
        # Mock _is_localhost_request to return True
        monkeypatch.setattr(
            "rasa.builder.service._is_localhost_request", lambda _: True
        )

        # Mock try_load_existing_agent to return None
        monkeypatch.setattr(
            "rasa.builder.service.try_load_existing_agent",
            AsyncMock(return_value=None),
        )

        async with sanic_app_with_agent.asgi_client as client:
            _, response = await client.post("/api/internal/reload-agent")

        assert response.status == 404
        payload = json.loads(response.body)
        assert payload["success"] is False
        assert "No agent found" in payload["message"]

    @pytest.mark.asyncio
    async def test_reload_agent_internal_forbidden_non_localhost(
        self, sanic_app_with_agent: Sanic, monkeypatch: MonkeyPatch
    ) -> None:
        """Test reload_agent_internal rejects non-localhost requests."""
        # Mock _is_localhost_request to return False
        monkeypatch.setattr(
            "rasa.builder.service._is_localhost_request", lambda _: False
        )

        async with sanic_app_with_agent.asgi_client as client:
            _, response = await client.post("/api/internal/reload-agent")

        assert response.status == 403
        payload = json.loads(response.body)
        assert payload["error"] == "Forbidden"
        assert "localhost" in payload["details"]["message"]

    @pytest.mark.asyncio
    async def test_get_tracker_internal_success(
        self, sanic_app_with_agent: Sanic, monkeypatch: MonkeyPatch
    ) -> None:
        """Test get_tracker_internal endpoint with valid session."""
        # Mock _is_localhost_request to return True
        monkeypatch.setattr(
            "rasa.builder.service._is_localhost_request", lambda _: True
        )

        # Mock the tracker store to return a mock tracker
        mock_tracker = MagicMock()
        mock_tracker.events = []
        sanic_app_with_agent.ctx.agent.tracker_store.retrieve = AsyncMock(
            return_value=mock_tracker
        )

        async with sanic_app_with_agent.asgi_client as client:
            _, response = await client.get("/api/internal/tracker/test-session-123")

        assert response.status == 200
        payload = json.loads(response.body)
        assert "conversation_turns" in payload or "current_state" in payload

    @pytest.mark.asyncio
    async def test_get_tracker_internal_session_not_found(
        self, sanic_app_with_agent: Sanic, monkeypatch: MonkeyPatch
    ) -> None:
        """Test get_tracker_internal when session doesn't exist."""
        # Mock _is_localhost_request to return True
        monkeypatch.setattr(
            "rasa.builder.service._is_localhost_request", lambda _: True
        )

        # Mock tracker store to return None for unknown session
        sanic_app_with_agent.ctx.agent.tracker_store.retrieve = AsyncMock(
            return_value=None
        )

        async with sanic_app_with_agent.asgi_client as client:
            _, response = await client.get("/api/internal/tracker/nonexistent-session")

        assert response.status == 404
        payload = json.loads(response.body)
        assert payload["error"] == "Tracker not found"

    @pytest.mark.asyncio
    async def test_get_tracker_internal_forbidden_non_localhost(
        self, sanic_app_with_agent: Sanic, monkeypatch: MonkeyPatch
    ) -> None:
        """Test get_tracker_internal rejects non-localhost requests."""
        # Mock _is_localhost_request to return False
        monkeypatch.setattr(
            "rasa.builder.service._is_localhost_request", lambda _: False
        )

        async with sanic_app_with_agent.asgi_client as client:
            _, response = await client.get("/api/internal/tracker/test-session")

        assert response.status == 403
        payload = json.loads(response.body)
        assert payload["error"] == "Forbidden"
        assert "localhost" in payload["details"]["message"]


class TestIsLocalhostRequest:
    """Test _is_localhost_request helper function."""

    def test_localhost_127_0_0_1(self) -> None:
        """Test that 127.0.0.1 is recognized as localhost."""
        from rasa.builder.service import _is_localhost_request

        mock_request = MagicMock()
        mock_request.ip = "127.0.0.1"

        assert _is_localhost_request(mock_request) is True

    def test_localhost_ipv6(self) -> None:
        """Test that ::1 (IPv6 localhost) is recognized as localhost."""
        from rasa.builder.service import _is_localhost_request

        mock_request = MagicMock()
        mock_request.ip = "::1"

        assert _is_localhost_request(mock_request) is True

    def test_non_localhost_ip(self) -> None:
        """Test that external IPs are not recognized as localhost."""
        from rasa.builder.service import _is_localhost_request

        mock_request = MagicMock()
        mock_request.ip = "192.168.1.100"

        assert _is_localhost_request(mock_request) is False

    def test_public_ip_not_localhost(self) -> None:
        """Test that public IPs are not recognized as localhost."""
        from rasa.builder.service import _is_localhost_request

        mock_request = MagicMock()
        mock_request.ip = "8.8.8.8"

        assert _is_localhost_request(mock_request) is False


class TestCopilotModeEndpoints:
    """Test copilot mode GET and POST endpoints."""

    @pytest.mark.asyncio
    async def test_get_copilot_mode_returns_current_mode(
        self, sanic_app: Sanic, monkeypatch: MonkeyPatch
    ) -> None:
        """Test GET /api/copilot/mode returns the current copilot mode."""
        mock_get_mode = MagicMock(return_value="legacy")
        monkeypatch.setattr("rasa.builder.copilot.get_copilot_mode", mock_get_mode)

        async with sanic_app.asgi_client as client:
            _, response = await client.get("/api/copilot/mode")

        assert response.status == 200
        payload = json.loads(response.body)
        assert payload == {"mode": "legacy"}

    @pytest.mark.asyncio
    async def test_get_copilot_mode_returns_agent_sdk(
        self, sanic_app: Sanic, monkeypatch: MonkeyPatch
    ) -> None:
        """Test GET /api/copilot/mode returns agent_sdk mode."""
        mock_get_mode = MagicMock(return_value="agent_sdk")
        monkeypatch.setattr("rasa.builder.copilot.get_copilot_mode", mock_get_mode)

        async with sanic_app.asgi_client as client:
            _, response = await client.get("/api/copilot/mode")

        assert response.status == 200
        payload = json.loads(response.body)
        assert payload == {"mode": "agent_sdk"}

    @pytest.mark.asyncio
    async def test_switch_copilot_mode_success_to_agent_sdk(
        self, sanic_app: Sanic, monkeypatch: MonkeyPatch
    ) -> None:
        """Test POST /api/copilot/mode successfully switches to agent_sdk."""
        mock_set_mode = MagicMock()
        monkeypatch.setattr("rasa.builder.copilot.set_copilot_mode", mock_set_mode)

        request_data = {"mode": "agent_sdk"}

        async with sanic_app.asgi_client as client:
            _, response = await client.post("/api/copilot/mode", json=request_data)

        assert response.status == 200
        payload = json.loads(response.body)
        assert payload["mode"] == "agent_sdk"
        assert payload["message"] == "Copilot mode switched to agent_sdk"
        mock_set_mode.assert_called_once_with("agent_sdk")

    @pytest.mark.asyncio
    async def test_switch_copilot_mode_success_to_legacy(
        self, sanic_app: Sanic, monkeypatch: MonkeyPatch
    ) -> None:
        """Test POST /api/copilot/mode successfully switches to legacy."""
        mock_set_mode = MagicMock()
        monkeypatch.setattr("rasa.builder.copilot.set_copilot_mode", mock_set_mode)

        request_data = {"mode": "legacy"}

        async with sanic_app.asgi_client as client:
            _, response = await client.post("/api/copilot/mode", json=request_data)

        assert response.status == 200
        payload = json.loads(response.body)
        assert payload["mode"] == "legacy"
        assert payload["message"] == "Copilot mode switched to legacy"
        mock_set_mode.assert_called_once_with("legacy")

    @pytest.mark.asyncio
    async def test_switch_copilot_mode_empty_body_returns_400(
        self, sanic_app: Sanic
    ) -> None:
        """Test POST /api/copilot/mode with empty body returns 400."""
        async with sanic_app.asgi_client as client:
            _, response = await client.post("/api/copilot/mode", data="")

        assert response.status == 400
        payload = json.loads(response.body)
        assert payload["error"] == "Invalid request"
        assert payload["details"]["message"] == "Request body is required"

    @pytest.mark.asyncio
    async def test_switch_copilot_mode_missing_mode_parameter_returns_400(
        self, sanic_app: Sanic
    ) -> None:
        """Test POST /api/copilot/mode with missing mode parameter returns 400."""
        request_data = {}

        async with sanic_app.asgi_client as client:
            _, response = await client.post("/api/copilot/mode", json=request_data)

        assert response.status == 400
        payload = json.loads(response.body)
        assert payload["error"] == "Invalid request"
        assert payload["details"]["message"] == "Mode parameter is required"

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Generator
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from pytest import MonkeyPatch
from sanic import Sanic

from rasa.builder.copilot.models import (
    CopilotGenerationContext,
    ResponseCategory,
    ResponseCompleteness,
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
    """
    Patch all Copilot/LLM bits so that the /api/copilot route can run
    entirely offline and without hitting third-party services.
    """
    # 1. Patch project generator to return a mock project
    project_folder = SimpleNamespace(name="proj")
    pg = SimpleNamespace(project_folder=project_folder, get_bot_files=lambda: {})
    monkeypatch.setattr("rasa.builder.service.get_project_generator", lambda _: pg)

    # 2. Patch Copilot's generate_response method to return a mock stream
    async def fake_stream():
        yield "token"

    # make sure all keys expected by log_copilot_from_handler are present
    usage_stats = SimpleNamespace(
        model_dump=lambda: {
            "model": "gpt-mock",
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        }
    )

    fake_copilot = SimpleNamespace(
        generate_response=AsyncMock(
            return_value=(
                fake_stream(),
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

    monkeypatch.setattr(
        "rasa.builder.llm_service.LLMService.instantiate_copilot",
        lambda _self: fake_copilot,
    )

    # 3. Patch `instantiate_handler` to return a mock CopilotResponseHandler
    token = SimpleNamespace(
        content="hi",
        response_category=ResponseCategory.COPILOT,
        response_completeness=ResponseCompleteness.COMPLETE,
        to_sse_event=lambda: SimpleNamespace(format=lambda: ""),
    )

    async def _handle(_):
        yield token

    handler = SimpleNamespace(
        generated_responses=[token],
        handle_response=_handle,
        extract_references=lambda *_: SimpleNamespace(
            to_sse_event=lambda: SimpleNamespace(format=lambda: "")
        ),
    )

    monkeypatch.setattr(
        "rasa.builder.llm_service.LLMService.instantiate_handler",
        lambda _self, *_args, **_kwargs: handler,
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
    self_removable_path: Path, monkeypatch: MonkeyPatch
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
        from rasa.builder.project_generator import ProjectGenerator

        real_pg = ProjectGenerator(str(temp_project_dir))
        sanic_app.ctx.project_generator = real_pg

        with (
            patch(
                "rasa.builder.jobs.validate_project", new_callable=AsyncMock
            ) as mock_validate,
            patch(
                "rasa.builder.jobs.train_and_load_agent", new_callable=AsyncMock
            ) as mock_train,
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

        from rasa.builder.project_generator import ProjectGenerator

        real_pg = ProjectGenerator(str(temp_project_dir))
        sanic_app.ctx.project_generator = real_pg

        with (
            patch(
                "rasa.builder.jobs.validate_project", new_callable=AsyncMock
            ) as mock_validate,
            patch(
                "rasa.builder.jobs.train_and_load_agent", new_callable=AsyncMock
            ) as mock_train,
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

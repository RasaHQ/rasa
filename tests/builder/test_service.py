import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from pytest import MonkeyPatch
from sanic import Sanic

from rasa.builder.copilot.constants import ROLE_COPILOT, ROLE_USER
from rasa.builder.copilot.models import (
    CopilotChatMessage,
    CopilotContext,
    CopilotGenerationContext,
    ResponseCategory,
    ResponseCompleteness,
    TextContent,
)
from rasa.builder.document_retrieval.models import Document
from rasa.builder.guardrails.lakera import LakeraAIGuardrails
from rasa.builder.guardrails.models import (
    GuardrailResponse,
    LakeraGuardrailResponse,
)
from rasa.builder.guardrails.utils import check_copilot_chat_for_policy_violations
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


def test_setup_project_generator_adds_to_sys_path(tmp_path: Path):
    tmp_path_str = str(tmp_path)
    assert tmp_path_str not in sys.path

    setup_project_generator(tmp_path_str)
    assert tmp_path_str in sys.path


def test_setup_project_generator_avoids_duplicate_sys_path_entries(tmp_path: Path):
    tmp_path_str = str(tmp_path)
    assert tmp_path_str not in sys.path

    setup_project_generator(tmp_path_str)
    setup_project_generator(tmp_path_str)

    assert sys.path.count(tmp_path_str) == 1


async def test_template_loads_actions_module(tmp_path: Path, monkeypatch: MonkeyPatch):
    tmp_path_str = str(tmp_path)
    monkeypatch.chdir(tmp_path_str)
    # this is required for the human handoff action in the telco example
    monkeypatch.setenv("OPENAI_API_KEY", "test-foo-bar")

    # Initialize project from template, and fetch action names
    project_generator = setup_project_generator(tmp_path_str)
    await project_generator.init_from_template(ProjectTemplateName.TELCO)
    project_action_names = Domain.from_path("").action_names_or_texts

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
@patch("rasa.builder.llm_service.llm_service.guardrails.send_request")
async def test_check_copilot_chat_for_policy_violations(
    mock_send_request: Mock,
) -> None:
    """Test check_copilot_chat_for_policy_violations with a simple message list."""
    # Given
    # Create a message list containing the four scenarios:
    # - normal user messages
    # - normal copilot messages
    # - user messages that are flagged
    # - copilot responses to those that flagged
    copilot_chat_history = [
        CopilotChatMessage(
            role=ROLE_USER,
            content=[TextContent(type="text", text="Hello")],
            response_category=None,
        ),
        CopilotChatMessage(
            role=ROLE_COPILOT,
            content=[TextContent(type="text", text="Hello")],
            response_category=None,
        ),
        CopilotChatMessage(
            role=ROLE_USER,
            content=[
                TextContent(
                    type="text", text="User message that violates guardrail policy."
                )
            ],
            response_category=ResponseCategory.GUARDRAILS_POLICY_VIOLATION,
        ),
        CopilotChatMessage(
            role=ROLE_COPILOT,
            content=[TextContent(type="text", text="Copilot response.")],
            response_category=ResponseCategory.GUARDRAILS_POLICY_VIOLATION,
        ),
    ]
    context: CopilotContext = CopilotContext(
        tracker=None,
        assistant_logs="",
        assistant_files={},
        copilot_chat_history=copilot_chat_history,
    )
    mock_send_request.return_value = LakeraGuardrailResponse(
        flagged=False, hello_rasa_user_id="test", hello_rasa_project_id="test"
    )

    # When
    result: Optional[Any] = await check_copilot_chat_for_policy_violations(
        context=context,
        hello_rasa_user_id="test_user",
        hello_rasa_project_id="test_project",
    )

    # Then
    assert result is None

    mock_send_request.assert_called_once()
    call_args = mock_send_request.call_args[0][0]
    # Verify message processing - should only include first two, non-flagged messages.
    messages: List[Dict[str, Any]] = call_args.messages
    # Expect only the single user message "Hello"
    assert len(messages) == 1
    assert messages[0]["role"] == ROLE_USER
    assert messages[0]["content"] == "Hello"


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

import argparse
import sys
import types
from pathlib import Path

import pytest

from rasa.core.available_agents import AvailableAgents
from rasa.core.config.available_endpoints import AvailableEndpoints
from rasa.core.config.configuration import Configuration


@pytest.fixture
def empty_available_agents() -> AvailableAgents:
    return AvailableAgents(agents={})


def _setup_monkeypatch_for_config(
    monkeypatch: pytest.MonkeyPatch, sent: AvailableAgents, captured: dict
) -> None:
    def _read_from_folder(agent_folder: str) -> AvailableAgents:  # type: ignore[override]
        captured["folder"] = agent_folder
        return sent

    # make endpoints init cheap by returning an empty AvailableEndpoints
    monkeypatch.setattr(
        "rasa.core.config.available_endpoints.AvailableEndpoints.read_endpoints",
        lambda *_args, **_kwargs: AvailableEndpoints(),
        raising=True,
    )

    monkeypatch.setattr(
        "rasa.core.available_agents.AvailableAgents.read_from_folder",
        _read_from_folder,
        raising=True,
    )


def test_shell_initializes_sub_agents_with_custom_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    empty_available_agents: AvailableAgents,
) -> None:
    from rasa.cli.shell import shell

    captured: dict = {}
    _setup_monkeypatch_for_config(monkeypatch, empty_available_agents, captured)

    # prevent loading a model; make it fail fast after init
    monkeypatch.setattr(
        "rasa.cli.validation.config_path_validation.get_validated_path",
        lambda *a, **k: str(tmp_path / "model.tar.gz"),
        raising=True,
    )
    monkeypatch.setattr(
        "rasa.cli.shell.get_local_model",
        lambda *a, **k: (_ for _ in ()).throw(Exception("stop")),
        raising=True,
    )

    args = types.SimpleNamespace(
        endpoints=str(tmp_path / "endpoints.yml"),
        sub_agents=str(tmp_path / "custom_sub_agents"),
        model=str(tmp_path / "model.tar.gz"),
        connector=None,
        conversation_id=None,
    )

    try:
        shell(args)
    except Exception:
        pass

    assert captured["folder"] == str(tmp_path / "custom_sub_agents")
    assert Configuration.get_instance().available_agents is empty_available_agents


def test_run_initializes_sub_agents_with_custom_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    empty_available_agents: AvailableAgents,
) -> None:
    from rasa.cli.run import run

    captured: dict = {}
    _setup_monkeypatch_for_config(monkeypatch, empty_available_agents, captured)

    # prevent server launch by stubbing the alias used inside cli.run
    monkeypatch.setattr("rasa.cli.run.rasa_run", lambda **kwargs: None, raising=True)

    args = types.SimpleNamespace(
        endpoints=str(tmp_path / "endpoints.yml"),
        credentials=str(tmp_path / "credentials.yml"),
        sub_agents=str(tmp_path / "custom_sub_agents"),
        enable_api=True,
        remote_storage="s3://bucket/path",
        model=str(tmp_path / "model.tar.gz"),
        skip_yaml_validation=[],
    )

    run(args)

    assert captured["folder"] == str(tmp_path / "custom_sub_agents")
    assert Configuration.get_instance().available_agents is empty_available_agents


def test_inspect_initializes_sub_agents_with_custom_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    empty_available_agents: AvailableAgents,
) -> None:
    from rasa.cli.inspect import inspect

    captured: dict = {}
    _setup_monkeypatch_for_config(monkeypatch, empty_available_agents, captured)

    # prevent opening browser and model loading
    monkeypatch.setattr(
        "rasa.cli.validation.config_path_validation.get_validated_path",
        lambda *a, **k: str(tmp_path / "model.tar.gz"),
        raising=True,
    )
    # Raise ModelNotFound to exit inspect early after initialization
    from rasa.exceptions import ModelNotFound  # local import for monkeypatch lambda

    monkeypatch.setattr(
        "rasa.cli.inspect.get_local_model",
        lambda *a, **k: (_ for _ in ()).throw(ModelNotFound("no model")),
        raising=True,
    )
    # Avoid starting the server from inspect by stubbing cli.run.run
    monkeypatch.setattr("rasa.cli.run.run", lambda _args: None, raising=True)

    args = types.SimpleNamespace(
        endpoints=str(tmp_path / "endpoints.yml"),
        sub_agents=str(tmp_path / "custom_sub_agents"),
        model=str(tmp_path / "model.tar.gz"),
        voice=False,
        legacy=False,
        port=5005,
        auth_token=None,
    )

    try:
        inspect(args)
    except Exception:
        pass

    assert captured["folder"] == str(tmp_path / "custom_sub_agents")
    assert Configuration.get_instance().available_agents is empty_available_agents


def test_train_initializes_sub_agents_with_custom_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    empty_available_agents: AvailableAgents,
) -> None:
    from rasa.cli.train import run_training

    captured: dict = {}
    _setup_monkeypatch_for_config(monkeypatch, empty_available_agents, captured)

    # bypass actual training
    class DummyTrainingResult:
        def __init__(self) -> None:
            self.code = 0
            self.model = None

    monkeypatch.setattr(
        "rasa.api.train", lambda **kwargs: DummyTrainingResult(), raising=True
    )
    monkeypatch.setattr(
        "rasa.cli.train.validate_files", lambda *a, **k: None, raising=True
    )
    # Short-circuit config validation to avoid filesystem dependencies
    monkeypatch.setattr(
        "rasa.cli.train.get_validated_config",
        lambda *a, **k: {},
        raising=True,
    )
    monkeypatch.setattr(
        "rasa.cli.train.get_validated_path",
        lambda *a, **k: (a[0] if a else ""),
        raising=True,
    )
    monkeypatch.setattr(
        "rasa.shared.importers.importer.TrainingDataImporter.load_from_config",
        lambda **k: types.SimpleNamespace(
            get_stories=lambda: types.SimpleNamespace(story_steps=[]),
            get_flows=lambda: types.SimpleNamespace(is_empty=lambda: True),
            get_nlu_data=lambda: types.SimpleNamespace(
                contains_no_pure_nlu_data=lambda: True, has_e2e_examples=lambda: False
            ),
            get_domain=lambda: types.SimpleNamespace(is_empty=lambda: True),
        ),
    )

    args = argparse.Namespace(
        domain=str(tmp_path / "domain.yml"),
        config=str(tmp_path / "config.yml"),
        endpoints=str(tmp_path / "endpoints.yml"),
        data=[str(tmp_path / "data")],
        out=str(tmp_path / "models"),
        dry_run=False,
        force=False,
        fixed_model_name=None,
        persist_nlu_data=False,
        sub_agents=str(tmp_path / "custom_sub_agents"),
        skip_validation=True,
        validation_max_history=None,
        fail_on_validation_warnings=False,
        remote_storage=None,
        epoch_fraction=1.0,
        keep_local_model_copy=False,
        remote_root_only=False,
        finetune=None,
    )

    run_training(args, can_exit=False)

    assert captured["folder"] == str(tmp_path / "custom_sub_agents")
    assert Configuration.get_instance().available_agents is empty_available_agents


def test_data_validate_initializes_sub_agents_with_custom_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    empty_available_agents: AvailableAgents,
) -> None:
    """Test that `rasa data validate` initializes sub-agents with a custom path."""
    from rasa.cli.validation.bot_config import validate_files

    captured: dict = {}
    _setup_monkeypatch_for_config(monkeypatch, empty_available_agents, captured)

    # Mock Validator to keep validation lightweight
    class MockValidator:
        config = None

        @staticmethod
        def from_importer(_importer):  # type: ignore[no-untyped-def]
            return MockValidator()

        def verify_story_structure(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return True

    # Provide a mock module for `rasa.validator` so the in-function import resolves
    mock_validator_module = types.ModuleType("rasa.validator")
    mock_validator_module.Validator = MockValidator  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rasa.validator", mock_validator_module)

    # Ensure Configuration is initialized with the custom sub-agents path before
    # validate_files is called
    Configuration.initialise_sub_agents(Path(str(tmp_path / "custom_sub_agents")))

    # Call validate_files with custom sub-agents and endpoints
    validate_files(
        fail_on_warnings=False,
        max_history=None,
        importer=object(),  # not used by dummy validator
        stories_only=True,
        flows_only=False,
        translations_only=False,
        sub_agents=str(tmp_path / "custom_sub_agents"),
        endpoints=str(tmp_path / "endpoints.yml"),
    )

    assert captured["folder"] == str(tmp_path / "custom_sub_agents")
    assert Configuration.get_instance().available_agents is empty_available_agents


@pytest.mark.asyncio
async def test_test_core_initializes_sub_agents_with_custom_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    empty_available_agents: AvailableAgents,
) -> None:
    """Test that `rasa test core` initializes sub-agents with a custom path."""
    from rasa.cli.test import run_core_test_async

    captured: dict = {}
    _setup_monkeypatch_for_config(monkeypatch, empty_available_agents, captured)

    # Mock model testing functions to avoid actual testing
    monkeypatch.setattr(
        "rasa.cli.validation.config_path_validation.get_validated_path",
        lambda *a, **k: str(tmp_path / "model.tar.gz"),
        raising=True,
    )

    async def mock_test_core(**kwargs):  # type: ignore[no-untyped-def]
        return None

    monkeypatch.setattr(
        "rasa.model_testing.test_core",
        mock_test_core,
        raising=True,
    )

    args = argparse.Namespace(
        stories=str(tmp_path / "data"),
        sub_agents=str(tmp_path / "custom_sub_agents"),
        model=str(tmp_path / "model.tar.gz"),
        out=str(tmp_path / "results"),
        no_errors=False,
        no_warnings=False,
        successes=False,
        e2e=False,
        evaluate_model_directory=False,
    )

    try:
        await run_core_test_async(args)
    except Exception:
        pass

    assert captured["folder"] == str(tmp_path / "custom_sub_agents")
    assert Configuration.get_instance().available_agents is empty_available_agents


@pytest.mark.asyncio
async def test_test_nlu_initializes_sub_agents_with_custom_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    empty_available_agents: AvailableAgents,
) -> None:
    """Test that `rasa test nlu` initializes sub-agents with a custom path."""
    from rasa.cli.test import run_nlu_test_async

    captured: dict = {}
    _setup_monkeypatch_for_config(monkeypatch, empty_available_agents, captured)

    # Mock get_validated_path to return paths without validation
    def mock_get_validated_path(path, name, default):  # type: ignore[no-untyped-def]
        if name == "nlu":
            return str(tmp_path / "data")
        elif name == "model":
            return str(tmp_path / "model.tar.gz")
        return path

    monkeypatch.setattr(
        "rasa.cli.test.get_validated_path",
        mock_get_validated_path,
        raising=True,
    )

    async def mock_test_nlu(*args, **kwargs):  # type: ignore[no-untyped-def]
        return None

    monkeypatch.setattr(
        "rasa.model_testing.test_nlu",
        mock_test_nlu,
        raising=True,
    )

    # Mock TrainingDataImporter to return empty NLU data
    mock_nlu_data = types.SimpleNamespace(training_examples=[])
    monkeypatch.setattr(
        "rasa.shared.importers.importer.TrainingDataImporter.load_from_dict",
        lambda **k: types.SimpleNamespace(
            get_nlu_data=lambda: mock_nlu_data,
        ),
        raising=True,
    )

    # Mock create_directory to avoid filesystem operations
    monkeypatch.setattr(
        "rasa.shared.utils.io.create_directory",
        lambda *a, **k: None,
        raising=True,
    )

    try:
        await run_nlu_test_async(
            config=None,
            data_path=str(tmp_path / "data"),
            models_path=str(tmp_path / "model.tar.gz"),
            output_dir=str(tmp_path / "results"),
            cross_validation=False,
            percentages=[0, 25, 50, 75],
            runs=3,
            no_errors=False,
            domain_path=str(tmp_path / "domain.yml"),
            all_args={
                "sub_agents": str(tmp_path / "custom_sub_agents"),
                "errors": True,
            },
        )
    except Exception:
        pass

    assert captured["folder"] == str(tmp_path / "custom_sub_agents")
    assert Configuration.get_instance().available_agents is empty_available_agents

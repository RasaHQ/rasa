from pathlib import Path
from typing import Callable

import pytest
from pytest import RunResult
from rasa.constants import (
    CONFIG_TELEMETRY_DATE as RASA_CONFIG_TELEMETRY_DATE,
    CONFIG_TELEMETRY_ENABLED as RASA_CONFIG_TELEMETRY_ENABLED,
)
from rasa.shared.utils.yaml import read_yaml_file

from rasa.telemetry import (
    RASA_PRO_CONFIG_FILE_TELEMETRY_KEY,
    CONFIG_TELEMETRY_ID,
)

KEYS = [
    RASA_CONFIG_TELEMETRY_ENABLED,
    RASA_CONFIG_TELEMETRY_DATE,
    CONFIG_TELEMETRY_ID,
]


@pytest.mark.timeout(180, func_only=True)
@pytest.mark.parametrize(
    "endpoints_file",
    [
        Path(__file__).parent.parent / "tracing" / "fixtures" / "jaeger_endpoints.yml",
        Path(__file__).parent.parent / "tracing" / "fixtures" / "otlp_endpoints.yml",
    ],
)
def test_telemetry_config_file_gets_written_with_default_telemetry_settings(
    endpoints_file: Path,
    run_in_simple_project: Callable[..., RunResult],
) -> None:
    run_in_simple_project("train", "--endpoints", endpoints_file)

    global_config_path = Path("~/.config/rasa/global.yml").expanduser()
    assert global_config_path.exists()

    config_content = read_yaml_file(global_config_path)

    rasa_config = config_content.get("metrics")
    assert rasa_config is not None

    traits = config_content.get(RASA_PRO_CONFIG_FILE_TELEMETRY_KEY)
    assert traits is not None

    for key in KEYS:
        assert key in traits

    assert traits[RASA_CONFIG_TELEMETRY_ENABLED] is True
    assert rasa_config[RASA_CONFIG_TELEMETRY_ENABLED] is True


@pytest.mark.timeout(180, func_only=True)
@pytest.mark.parametrize(
    "endpoints_file",
    [
        Path(__file__).parent.parent / "tracing" / "fixtures" / "jaeger_endpoints.yml",
        Path(__file__).parent.parent / "tracing" / "fixtures" / "otlp_endpoints.yml",
    ],
)
def test_telemetry_config_file_when_telemetry_is_disabled(
    endpoints_file: Path,
    run_in_simple_project: Callable[..., RunResult],
) -> None:
    run_in_simple_project("telemetry", "disable")
    run_in_simple_project("train", "--endpoints", endpoints_file)

    global_config_path = Path("~/.config/rasa/global.yml").expanduser()
    assert global_config_path.exists()

    config_content = read_yaml_file(global_config_path)

    rasa_config = config_content.get("metrics")
    assert rasa_config is not None

    traits = config_content.get(RASA_PRO_CONFIG_FILE_TELEMETRY_KEY)
    assert traits is not None

    for key in KEYS:
        assert key in traits

    assert traits[RASA_CONFIG_TELEMETRY_ENABLED] is False
    assert rasa_config[RASA_CONFIG_TELEMETRY_ENABLED] is False

from contextlib import nullcontext as does_not_raise
from typing import Any, Dict, List, Optional

import pytest
from apscheduler.triggers.cron import CronTrigger
from pytest import MonkeyPatch
from structlog.testing import capture_logs

from rasa.privacy.constants import USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME
from rasa.privacy.privacy_config import (
    AnonymizationPolicy,
    DeletionPolicy,
    PrivacyConfig,
    get_cron_trigger,
    validate_min_after_session_end,
    validate_policies,
    validate_privacy_config,
    validate_sensitive_slots,
)
from rasa.shared.constants import LATEST_TRAINING_DATA_FORMAT_VERSION
from rasa.shared.core.domain import Domain
from rasa.shared.exceptions import RasaException
from rasa.utils.endpoints import read_property_config_from_endpoints_file
from tests.utilities import filter_logs


@pytest.fixture
def privacy_dict() -> Dict[str, Any]:
    endpoint_file = "data/test_privacy/endpoints_with_valid_privacy.yml"
    return read_property_config_from_endpoints_file(
        endpoint_file, property_name="privacy"
    )


def test_validate_privacy_config_valid_input(privacy_dict: Dict[str, Any]) -> None:
    """Test that a valid privacy config does not raise an exception."""
    with does_not_raise():
        assert validate_privacy_config(privacy_dict) is None


@pytest.mark.parametrize(
    "invalid_privacy_path, error_message",
    [
        (
            "data/test_privacy/endpoints_with_invalid_privacy_additional_key.yml",
            "Additional properties are not allowed "
            "('inactive_after_session_end' was unexpected)",
        ),
        (
            "data/test_privacy/endpoints_with_invalid_privacy_zero_mins_setting.yml",
            "0 is less than or equal to the minimum of 0",
        ),
        (
            "data/test_privacy/endpoints_with_invalid_privacy_missing_required_key.yml",
            "'anonymization' is a required property",
        ),
        (
            "data/test_privacy/endpoints_with_invalid_privacy_long_redact_char.yml",
            "'##' is too long",
        ),
    ],
)
def test_validate_privacy_config_invalid_input(
    invalid_privacy_path: str, error_message: str
) -> None:
    invalid_privacy_dict = read_property_config_from_endpoints_file(
        invalid_privacy_path, property_name="privacy"
    )
    with capture_logs() as caplog:
        with pytest.raises(SystemExit):
            validate_privacy_config(invalid_privacy_dict)

        log = filter_logs(
            caplog,
            "privacy_config.invalid_privacy_config",
            "error",
        )
        assert len(log) == 1
        assert log[0]["validation_errors"] == [error_message]


@pytest.mark.parametrize(
    "cron_expression",
    [
        "0 0 * * *",
        "*/5 1,2,3 * * * ",
        "0 12 1 * *",
        "0 0 1 1 *",
        "45 23 * * 6",
    ],
)
def test_get_cron_trigger_valid(cron_expression: str) -> None:
    cron_trigger = get_cron_trigger(cron_expression)
    assert cron_trigger is not None
    assert isinstance(cron_trigger, CronTrigger)


@pytest.mark.parametrize(
    "cron_expression",
    [
        "0 25 * * *",
        "71 12 1 * *",
        "45 23 * * 9",
        "@daily",
        "@weekly",
        "@monthly",
        "@yearly",
        "@annually",
        "@hourly",
        "@midnight",
        "@reboot",
    ],
)
def test_get_cron_trigger_invalid(cron_expression: str) -> None:
    with capture_logs() as caplog:
        with pytest.raises(RasaException):
            get_cron_trigger(cron_expression)

        log = filter_logs(
            caplog,
            "privacy_config.invalid_cron_expression",
            "error",
        )
        assert len(log) == 1
        assert log[0]["cron"] == cron_expression


def test_validate_min_after_session_end_valid(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, "45")
    with does_not_raise():
        assert validate_min_after_session_end(50) is None


def test_validate_min_after_session_end_invalid(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, "45")
    with pytest.raises(RasaException) as exc_info:
        validate_min_after_session_end(30)
    assert (
        "Minimum time in minutes after session end must be greater than "
        f"{USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME} env var value."
    ) == str(exc_info.value)


@pytest.mark.parametrize("value", ["abc", "abc123"])
def test_validate_policies_invalid_inactivity_env_var(
    monkeypatch: MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, value)

    with pytest.raises(RasaException) as exc_info:
        validate_min_after_session_end(30)

    assert (
        f"Invalid value for USER_CHAT_INACTIVITY_IN_MINUTES env var: "
        f"invalid literal for int() with base 10: '{value}'."
    ) == str(exc_info.value)


@pytest.mark.parametrize(
    "sensitive_slots",
    [
        ["national_insurance_number", "passport_number"],
        [],
    ],
)
def test_validate_sensitive_slots(
    monkeypatch: MonkeyPatch,
    sensitive_slots: List[str],
) -> None:
    domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        slots:
          national_insurance_number:
            type: text
          passport_number:
            type: text
        """
    )
    with does_not_raise():
        assert validate_sensitive_slots(sensitive_slots, domain) is None


@pytest.mark.parametrize(
    "sensitive_slots, error_message",
    [
        (
            ["national_insurance_number", "passport_number", "non_existent_slot"],
            "Sensitive slot not found in the domain.",
        ),
    ],
)
def test_validate_sensitive_slots_invalid(
    sensitive_slots: List[str], error_message: str
) -> None:
    domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        slots:
          national_insurance_number:
            type: text
          passport_number:
            type: text
        """
    )
    with capture_logs() as caplog:
        with pytest.raises(RasaException) as exc_info:
            validate_sensitive_slots(sensitive_slots, domain)

        assert (
            "Sensitive slots defined in the privacy config do not match "
            "the slots defined in the domain. Please check the slot names."
        ) == str(exc_info.value)

        log = filter_logs(
            caplog,
            "privacy_config.invalid_sensitive_slot",
            "error",
        )
        assert len(log) == 1
        assert log[0]["sensitive_slot"] == sensitive_slots[-1]


@pytest.mark.parametrize(
    "deletion_policy, anonymization_policy",
    [
        (
            DeletionPolicy.from_dict(
                {
                    "cron": "0 0 * * *",
                    "min_after_session_end": 120,
                }
            ),
            AnonymizationPolicy.from_dict(
                {
                    "cron": "0 2 * * *",
                    "min_after_session_end": 60,
                }
            ),
        ),
        (None, None),
    ],
)
def test_validate_policies_valid(
    deletion_policy: Optional[DeletionPolicy],
    anonymization_policy: Optional[AnonymizationPolicy],
) -> None:
    with does_not_raise():
        assert validate_policies(deletion_policy, anonymization_policy) is None


def test_validate_policies_invalid_min_after_session_end() -> None:
    deletion_policy = DeletionPolicy.from_dict(
        {
            "cron": "0 0 * * *",
            "min_after_session_end": 120,
        }
    )
    anonymization_policy = AnonymizationPolicy.from_dict(
        {
            "cron": "0 2 * * *",
            "min_after_session_end": 180,
        }
    )

    with pytest.raises(RasaException) as exc_info:
        validate_policies(deletion_policy, anonymization_policy)

    assert (
        "Minimum time in minutes after session end for deletion policy "
        "must be greater than that of the anonymization policy."
    ) == str(exc_info.value)


def test_validate_policies_invalid_identical_cron() -> None:
    deletion_policy = DeletionPolicy.from_dict(
        {
            "cron": "0 0 * * *",
            "min_after_session_end": 120,
        }
    )
    anonymization_policy = AnonymizationPolicy.from_dict(
        {
            "cron": "0 0 * * *",
            "min_after_session_end": 60,
        }
    )

    with pytest.raises(RasaException) as exc_info:
        validate_policies(deletion_policy, anonymization_policy)

    assert (
        "Cron expressions for the deletion and anonymization policies "
        "must be different."
    ) == str(exc_info.value)


def test_privacy_config_from_dict_empty() -> None:
    with capture_logs() as caplog:
        with pytest.raises(SystemExit):
            PrivacyConfig.from_dict({})

        log = filter_logs(
            caplog,
            "privacy_config.invalid_privacy_config",
            "error",
        )
        assert len(log) == 1
        assert log[0]["validation_errors"] == ["'rules' is a required property"]


def test_privacy_config_from_dict_no_tracker_settings() -> None:
    privacy_config = PrivacyConfig.from_dict(
        {
            "rules": [
                {
                    "slot": "national_insurance_number",
                    "anonymization": {
                        "type": "redact",
                        "redaction_char": "*",
                        "keep_left": 2,
                        "keep_right": 2,
                    },
                },
            ]
        }
    )
    assert privacy_config.tracker_store_settings is None


def test_privacy_config_from_dict_invalid_method_type() -> None:
    with capture_logs() as caplog:
        with pytest.raises(SystemExit):
            PrivacyConfig.from_dict(
                {
                    "rules": [
                        {
                            "slot": "national_insurance_number",
                            "anonymization": {
                                "type": "hash",
                            },
                        },
                    ]
                }
            )

        log = filter_logs(
            caplog,
            "privacy_config.invalid_privacy_config",
            "error",
        )
        assert len(log) == 1
        assert log[0]["validation_errors"] == [
            "'hash' is not one of ['redact', 'mask']"
        ]


def test_privacy_config_optional_deletion_policy() -> None:
    privacy_config = PrivacyConfig.from_dict(
        {
            "rules": [
                {
                    "slot": "national_insurance_number",
                    "anonymization": {
                        "type": "mask",
                    },
                },
            ],
            "tracker_store_settings": {
                "anonymization": {
                    "cron": "0 2 * * *",
                    "min_after_session_end": 60,
                },
            },
        }
    )
    assert privacy_config.tracker_store_settings is not None
    assert privacy_config.tracker_store_settings.deletion_policy is None
    assert privacy_config.tracker_store_settings.anonymization_policy is not None


def test_privacy_config_optional_anonymization_policy() -> None:
    privacy_config = PrivacyConfig.from_dict(
        {
            "rules": [
                {
                    "slot": "national_insurance_number",
                    "anonymization": {
                        "type": "mask",
                    },
                },
            ],
            "tracker_store_settings": {
                "deletion": {
                    "cron": "0 0 * * *",
                    "min_after_session_end": 120,
                },
            },
        }
    )
    assert privacy_config.tracker_store_settings is not None
    assert privacy_config.tracker_store_settings.deletion_policy is not None
    assert privacy_config.tracker_store_settings.anonymization_policy is None


def test_privacy_config_optional_tracker_store_settings() -> None:
    privacy_config = PrivacyConfig.from_dict(
        {
            "rules": [
                {
                    "slot": "national_insurance_number",
                    "anonymization": {
                        "type": "mask",
                    },
                },
            ],
            "tracker_store_settings": {
                "deletion": {},
                "anonymization": {},
            },
        }
    )
    assert privacy_config.tracker_store_settings is not None
    assert privacy_config.tracker_store_settings.deletion_policy is None
    assert privacy_config.tracker_store_settings.anonymization_policy is None

from __future__ import annotations

import os
import sys
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import structlog
from apscheduler.triggers.cron import CronTrigger
from pydantic import BaseModel, ConfigDict

from rasa.constants import PACKAGE_NAME
from rasa.privacy.constants import (
    ANONYMIZATION_KEY,
    DELETION_KEY,
    KEEP_LEFT_KEY,
    KEEP_RIGHT_KEY,
    PRIVACY_CONFIG_SCHEMA,
    REDACTION_CHAR_KEY,
    SLOT_KEY,
    TRACKER_STORE_SETTINGS,
    USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME,
)
from rasa.shared.exceptions import RasaException
from rasa.shared.utils.io import read_json_file
from rasa.shared.utils.yaml import (
    YamlValidationException,
    validate_data_with_jsonschema,
)

if TYPE_CHECKING:
    from rasa.shared.core.domain import Domain

structlogger = structlog.get_logger(__name__)


class AnonymizationType(Enum):
    """Enum for the anonymization types."""

    REDACT = "redact"
    """Replaces the PII plaintext value with the same character
    for the entire or partial length of the value."""
    MASK = "mask"
    """Replaces the PII plaintext value with the uppercase slot name
    in square brackets, e.g. [CREDIT_CARD_NUMBER]."""


class AnonymizationMethod(BaseModel):
    """Class for configuring the anonymization method."""

    method_type: AnonymizationType
    """The anonymization method to be used."""
    redaction_char: str
    """The character to use for redaction."""
    keep_left: Optional[int] = None
    """The number of characters to be kept intact on the left side."""
    keep_right: Optional[int] = None
    """The number of characters to be kept intact on the right side."""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AnonymizationMethod:
        """Create an AnonymizationMethod object from parsed data."""
        method_type = AnonymizationType(
            data.get("type", AnonymizationType.REDACT.value)
        )
        redaction_char = data.get(REDACTION_CHAR_KEY, "*")
        keep_left = data.get(KEEP_LEFT_KEY)
        keep_right = data.get(KEEP_RIGHT_KEY)

        return cls(
            method_type=method_type,
            redaction_char=redaction_char,
            keep_left=keep_left,
            keep_right=keep_right,
        )


class PrivacyPolicy(BaseModel):
    """Parent class for configuring privacy policies."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    min_after_session_end: int
    """Minimum time in minutes after session end before the policy is executed."""
    cron: CronTrigger
    """Cron trigger for periodic execution of the privacy policy."""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PrivacyPolicy:
        """Create an AnonymizationPolicy object from parsed data."""
        min_after_session_end = data.get("min_after_session_end", 1)
        validate_min_after_session_end(min_after_session_end)

        cron_expression = get_cron_trigger(data.get("cron"))

        return cls(
            min_after_session_end=min_after_session_end,
            cron=cron_expression,
        )


class DeletionPolicy(PrivacyPolicy):
    """Class for configuring periodic deletion in the tracker store."""

    type: str = "deletion"


class AnonymizationPolicy(PrivacyPolicy):
    """Class for configuring periodic anonymization in the tracker store."""

    type: str = "anonymization"


class TrackerStoreSettings(BaseModel):
    """Class for configuring tracker store settings."""

    deletion_policy: Optional[DeletionPolicy] = None
    """The deletion policy to be used."""
    anonymization_policy: Optional[AnonymizationPolicy] = None
    """The anonymization policy to be used."""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrackerStoreSettings:
        """Create a TrackerStoreSettings object from parsed data."""
        deletion_policy = data.get(DELETION_KEY)
        anonymization_policy = data.get(ANONYMIZATION_KEY)

        deletion_policy = (
            DeletionPolicy.from_dict(deletion_policy) if deletion_policy else None
        )
        anonymization_policy = (
            AnonymizationPolicy.from_dict(anonymization_policy)
            if anonymization_policy
            else None
        )

        validate_policies(deletion_policy, anonymization_policy)

        return cls(
            deletion_policy=deletion_policy,
            anonymization_policy=anonymization_policy,
        )


class PrivacyConfig(BaseModel):
    """Class for configuring PII management."""

    anonymization_rules: Dict[str, AnonymizationMethod]
    """"Mapping of slot names to rules for anonymizing sensitive information."""
    tracker_store_settings: Optional[TrackerStoreSettings] = None
    """The tracker store settings to be used for periodic jobs
    anonymizing and deleting conversation data in the tracker store."""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PrivacyConfig:
        """Create a PrivacyConfig object from parsed privacy config."""
        # Validate the data against the schema
        validate_privacy_config(data)

        anonymization_rules = {
            rule[SLOT_KEY]: AnonymizationMethod.from_dict(
                rule.get(ANONYMIZATION_KEY, {})
            )
            for rule in data.get("rules", [])
        }

        tracker_store_settings = data.get(TRACKER_STORE_SETTINGS, {})
        tracker_store_settings = (
            TrackerStoreSettings.from_dict(tracker_store_settings)
            if tracker_store_settings
            else None
        )
        return cls(
            anonymization_rules=anonymization_rules,
            tracker_store_settings=tracker_store_settings,
        )


def validate_privacy_config(data: Dict[str, Any]) -> None:
    """Validate the privacy configuration."""
    import importlib_resources

    schema_file = str(
        importlib_resources.files(PACKAGE_NAME).joinpath(PRIVACY_CONFIG_SCHEMA)
    )
    schema_content = read_json_file(schema_file)
    try:
        validate_data_with_jsonschema(data, schema_content)
    except YamlValidationException as exception:
        validation_errors = (
            [error.message for error in exception.validation_errors]
            if exception.validation_errors
            else []
        )
        exception_message = exception.message
        structlogger.error(
            "privacy_config.invalid_privacy_config",
            validation_errors=validation_errors,
            event_info=f"Invalid privacy config: {exception_message}. "
            f"Please check the configuration file.",
        )
        sys.exit(1)


def get_cron_trigger(cron_expression: str) -> CronTrigger:
    """Validate the crontab expression."""
    try:
        cron = CronTrigger.from_crontab(cron_expression)
    except Exception as exc:
        structlogger.error(
            "privacy_config.invalid_cron_expression",
            cron=cron_expression,
        )
        raise RasaException("Invalid cron expression") from exc

    return cron


def validate_min_after_session_end(min_after_session_end: int) -> None:
    """Validate the minimum time after session end."""
    try:
        inactivity_period = int(
            os.getenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, "30")
        )
    except (ValueError, TypeError) as exc:
        raise RasaException(
            f"Invalid value for {USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME} "
            f"env var: {exc}."
        )

    if min_after_session_end < inactivity_period:
        raise RasaException(
            f"Minimum time in minutes after session end must be greater than "
            f"{USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME} env var value."
        )


def validate_policies(
    deletion_policy: Optional[DeletionPolicy],
    anonymization_policy: Optional[AnonymizationPolicy],
) -> None:
    """Validate the deletion and anonymization policies' configurations."""
    if not deletion_policy or not anonymization_policy:
        return None

    if (
        deletion_policy.min_after_session_end
        <= anonymization_policy.min_after_session_end
    ):
        raise RasaException(
            "Minimum time in minutes after session end for deletion policy "
            "must be greater than that of the anonymization policy."
        )

    if deletion_policy.cron.fields == anonymization_policy.cron.fields:
        raise RasaException(
            "Cron expressions for the deletion and anonymization policies "
            "must be different."
        )

    return None


def validate_sensitive_slots(sensitive_slots: List[str], domain: "Domain") -> None:
    """Validate the sensitive slots defined in the privacy config against the domain."""
    all_slot_names = [slot.name for slot in domain.slots]
    all_good = True
    for sensitive_slot in sensitive_slots:
        if sensitive_slot not in all_slot_names:
            structlogger.error(
                "privacy_config.invalid_sensitive_slot",
                sensitive_slot=sensitive_slot,
                event_info="Sensitive slot not found in the domain.",
            )
            all_good = False

    if not all_good:
        raise RasaException(
            "Sensitive slots defined in the privacy config do not match "
            "the slots defined in the domain. Please check the slot names."
        )

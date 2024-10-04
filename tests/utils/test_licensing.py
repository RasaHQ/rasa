from typing import Text
from unittest.mock import AsyncMock, patch

import pytest
from freezegun import freeze_time
from pytest import MonkeyPatch

from rasa.core.tracker_store import InMemoryTrackerStore
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import SessionStarted
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.utils import licensing
from rasa.utils.licensing import (
    LICENSE_ENV_VAR,
    PRODUCT_AREA,
    License,
    LicenseEncodingException,
    LicenseExpiredException,
    LicenseNotYetValidException,
    LicenseSchemaException,
    LicenseScopeException,
    LicenseSignatureInvalidException,
    LicenseValidationException,
    get_license_expiration_date,
    is_valid_license_scope,
    property_of_active_license,
    validate_license_from_env,
)

BLOCKED_JTI = "8e0d440f-704a-44c3-b7b4-a6f2357e9768"


def test_decode_valid_license(valid_license: Text) -> None:
    License.decode(valid_license)


def test_validate_valid_license(monkeypatch: MonkeyPatch, valid_license: Text) -> None:
    monkeypatch.setenv(LICENSE_ENV_VAR, valid_license)
    validate_license_from_env()


def test_validate_license_env_var_not_set(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv(LICENSE_ENV_VAR, raising=False)
    with pytest.raises(SystemExit):
        validate_license_from_env()


@pytest.mark.parametrize(
    "license_fixture, exception",
    [
        ("expired_license", LicenseExpiredException),
        ("invalid_signature_license", LicenseSignatureInvalidException),
        ("immature_license", LicenseNotYetValidException),
        ("invalid_schema_license", LicenseSchemaException),
        ("non_jwt_license", LicenseEncodingException),
        ("unscoped_license", LicenseScopeException),
    ],
)
def test_decode_invalid_license_raises_exception(
    license_fixture: str,
    exception: LicenseValidationException,
    request: pytest.FixtureRequest,
) -> None:
    license_value = request.getfixturevalue(license_fixture)
    with pytest.raises(exception):  # type: ignore
        License.decode(license_value)


@pytest.mark.parametrize(
    "license_fixture, exception",
    [
        ("blocked_license", LicenseExpiredException),
    ],
)
def test_decode_blocked_license_raises_exception(
    license_fixture: str,
    exception: LicenseValidationException,
    monkeypatch: MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    license_value = request.getfixturevalue(license_fixture)

    monkeypatch.setattr("rasa.utils.licensing.JTI_BLOCKLIST", {BLOCKED_JTI})
    with pytest.raises(exception):  # type: ignore
        License.decode(license_value)


@pytest.mark.parametrize(
    "license_scope, expected",
    [
        ("rasa:enterprise", False),
        ("rasa:studio", False),
        ("rasa:pro:analytics", False),
        ("rasa:pro:analytics rasa:pro:some-future-feature", False),
        ("rasa", True),
        ("rasa:pro", True),
        ("rasa:pro:plus", True),
        ("rasa:pro rasa:enterprise", True),
        ("rasa:pro rasa:studio", True),
        ("rasa:pro rasa:studio rasa:enterprise", True),
    ],
)
def test_is_valid_license_scope(license_scope: Text, expected: bool) -> None:
    actual = is_valid_license_scope(PRODUCT_AREA, license_scope)

    assert actual == expected


def test_property_of_active_license(
    monkeypatch: MonkeyPatch, valid_license: str
) -> None:
    monkeypatch.setenv(LICENSE_ENV_VAR, valid_license)

    assert (
        property_of_active_license(lambda active_license: active_license.jti)
        is not None
    )


@pytest.mark.parametrize(
    "product_area, license_scope, expected",
    [
        ("rasa:pro:plus rasa:voice", "rasa:pro rasa:voice", True),
        ("rasa:pro:plus", "rasa:pro rasa:voice", True),
        ("rasa:pro:plus rasa:voice", "rasa:pro", False),
    ],
)
def test_is_valid_license_scope_with_voice(
    product_area: Text, license_scope: Text, expected: bool
) -> None:
    actual = is_valid_license_scope(
        product_area=product_area,
        license_scope=license_scope,
    )
    assert actual is expected


# valid license expires on 3000-01-01
@freeze_time("2999-12-31")
def test_license_about_to_expire_warning_during_validation(
    monkeypatch: MonkeyPatch, valid_license: str
) -> None:
    monkeypatch.setenv(LICENSE_ENV_VAR, valid_license)

    # validate_license_from_env should issue a structlogger warning
    with patch.object(licensing.structlogger, "warning") as mock_warning:
        validate_license_from_env()

    # check the warning contains "license.expiration.warning"
    mock_warning.assert_called_once()
    assert "license.expiration.warning" in mock_warning.call_args[0][0]


def test_is_developer_license(monkeypatch: MonkeyPatch, valid_license: str) -> None:
    # default valid license is not a developer license
    monkeypatch.setenv(LICENSE_ENV_VAR, valid_license)

    assert not licensing.is_champion_server_license()


@pytest.mark.parametrize(
    "developer_license_fixture",
    [
        "champion_server_limited_license",
        "champion_server_internal_license",
    ],
)
def test_is_developer_license_with_developer_license(
    monkeypatch: MonkeyPatch,
    developer_license_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    developer_license = request.getfixturevalue(developer_license_fixture)
    monkeypatch.setenv(LICENSE_ENV_VAR, developer_license)

    assert licensing.is_champion_server_license()


@pytest.mark.parametrize(
    "developer_license_fixture, limit",
    [
        ("champion_license", None),
        ("champion_server_limited_license", 1000),
        ("champion_server_internal_license", 100),
        ("valid_license", None),
    ],
)
def test_conversation_limits_for_license(
    monkeypatch: MonkeyPatch,
    developer_license_fixture: str,
    limit: int,
    request: pytest.FixtureRequest,
) -> None:
    developer_license = request.getfixturevalue(developer_license_fixture)
    monkeypatch.setenv(LICENSE_ENV_VAR, developer_license)

    assert licensing.conversation_limit_for_license() == limit


async def test_conversation_counting_job_triggers_limits(
    monkeypatch: MonkeyPatch,
    champion_server_limited_license: str,
) -> None:
    tracker_store = InMemoryTrackerStore(Domain.empty())
    monkeypatch.setenv(LICENSE_ENV_VAR, champion_server_limited_license)

    # mock the hard & soft limit handlers to validate they get called

    mocked_handle_hard_limit_reached = AsyncMock()
    monkeypatch.setattr(
        "rasa.utils.licensing.handle_hard_limit_reached",
        mocked_handle_hard_limit_reached,
    )

    mocked_handle_soft_limit_reached = AsyncMock()
    monkeypatch.setattr(
        "rasa.utils.licensing.handle_soft_limit_reached",
        mocked_handle_soft_limit_reached,
    )

    # test setup done, let's go

    await licensing.run_conversation_counting(tracker_store, 10)

    mocked_handle_soft_limit_reached.assert_not_called()
    mocked_handle_hard_limit_reached.assert_not_called()

    # create a tracker store with 11 conversations
    for i in range(11):
        tracker = DialogueStateTracker.from_events(f"{i}", [SessionStarted()])
        await tracker_store.save(tracker)

    await licensing.run_conversation_counting(tracker_store, 10)

    mocked_handle_soft_limit_reached.assert_called_once()
    mocked_handle_hard_limit_reached.assert_not_called()

    # add more conversations to reach the hard limit
    for i in range(101):
        tracker = DialogueStateTracker.from_events(f"{i + 11}", [SessionStarted()])
        await tracker_store.save(tracker)

    await licensing.run_conversation_counting(tracker_store, 10)

    mocked_handle_hard_limit_reached.assert_called_once()
    # also assert that the soft limit was not called again (count should
    # still be 1 from before)
    mocked_handle_soft_limit_reached.assert_called_once()


def test_get_license_expiration_date(
    monkeypatch: MonkeyPatch, valid_license: str
) -> None:
    monkeypatch.setenv(LICENSE_ENV_VAR, valid_license)

    assert get_license_expiration_date() == "3000-01-01T00:00:00+00:00"

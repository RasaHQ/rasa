from typing import Text

import pytest
from pytest import MonkeyPatch

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
    monkeypatch.delenv(LICENSE_ENV_VAR)
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

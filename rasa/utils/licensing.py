from asyncio import AbstractEventLoop
import hashlib
import os
import random
import re
import time
import typing
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Set, Text, TypeVar

import jwt
from dotenv import dotenv_values
from sanic import Sanic
import structlog
from rasa import telemetry

from rasa.core import jobs
from rasa.shared.utils.cli import print_error_and_exit


if typing.TYPE_CHECKING:
    from rasa.core.tracker_store import TrackerStore


LICENSE_ENV_VAR = "RASA_PRO_LICENSE"
ALGORITHM = "RS256"
# deepcode ignore HardcodedKey: This is a public key - not a security issue.
PUBLIC_KEY = """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA6L90HBMIeiEUkLw85aRx
L8qisVwiJwy3E4E/MPKHEuzguUJG3KwQE3Yb37HWi6I8EXOI5UfP2RvbNPKbmYFo
90P27rKpvhZRCG9sy3cNu3Xr1XcQ6Eue2e12LWBAgzBZSqjzwkCOtt+L6LIm3lPf
2QYSFORTZR9PtOvI1b677W1lVjioRrbg1IG6UXGVOTCmeSFT/JzbsYzR7QFzFdWe
ytjHVgeb/R9inY1/LeiP2KCHYcYUF2sGc+6CsGvr9Kkio5KS10jBF27EeBaeVpXO
JH5viXVuRPCu5ymvnih5Rk4VYK3X3rG1rf5oju9OBDPLq7lKklY1pPZjBHADPU3o
/QIDAQAB
-----END PUBLIC KEY-----"""

# If a license must be revoked before expiration, add its JTI to this list.
# Adding to this list requires cutting a new release for the change to affect users.
# Versions prior to the one including the blocked JTI will still
# be available to blocked users until the regular expiration date
JTI_BLOCKLIST: Set[Text] = set([])

SCOPE_DELIMITER = ":"
PRODUCT_AREA = "rasa:pro:plus"
VOICE_SCOPE = "rasa:voice"
CHAMPION_SERVER_LIMITED_SCOPE = "rasa:pro:champion-server-limited"
CHAMPION_SERVER_INTERNAL_SCOPE = "rasa:pro:champion-server-internal"
# defines multiple of the soft limit that triggers the hard limit
HARD_LIMIT_FACTOR = 10
structlogger = structlog.get_logger()


class LicenseValidationException(Exception):
    """Parent class for exceptions raised when handling licenses."""


class LicenseSchemaException(LicenseValidationException):
    """Exception raised when a license does not contain the correct fields."""


class LicenseScopeException(LicenseValidationException):
    """Exception raised when a license does not contain the correct scope."""


class LicenseEncodingException(LicenseValidationException):
    """Exception raised when the JWT representing a license is not well-formed."""


class LicenseSignatureInvalidException(LicenseValidationException):
    """Exception raised when a license signature could not be verified."""


class LicenseExpiredException(LicenseValidationException):
    """Exception raised when a license has expired (exp)."""


class LicenseNotYetValidException(LicenseValidationException):
    """Exception raised when a license is not valid yet (nbf)."""


class LicenseNotFoundException(LicenseValidationException):
    """Exception raised when a license is not available."""


class License:
    """Represents a Rasa Pro license.

    There a are two ways of instancing an `License` object:
    - Via `decode`: This is the option that should be used 99% of the times.
      This option allows callers to decode a JWT into an `License`
      object, which means that only correctly signed JWT will be decodable.
      Other checks will be performed on the JWT as well. This implies that any
      `License` obtained via `decode` is guaranteed to represent a
      valid license, that is, a license that was obtained via a
      contract/deal/etc. with Rasa Inc (i.e. the owner of the private key).
    - Via `__init__`: This creates an `License` object directly, without
      performing any validations. Useful only for creating new licenses (which
      requires access to the private key to then encode) or for testing
      purposes.

    The scopes of the license define what it unlocks. The scopes are
    hierarchical, meaning that a license with a scope `rasa:pro` also
    unlocks the scopes `rasa:pro:plus`.

    Scopes can be used to enable or disable features in Rasa, the following
    scopes are used:

    - `rasa:pro` - Rasa Pro features
    - `rasa:pro:plus` - Rasa Pro Plus features
    - `rasa:voice` - Voice features

    In addition to scopes focused on feature flagging, there are also scopes
    that are used to limited by the license agreement they are issued under:

    - `rasa:pro:champion` - Champion license, can only be used for local
      development

    - `rasa:pro:champion-server-internal` - Champion license, can be used
      to deploy an assistant on a server for an internal use case. The license
      limits the number of conversations per month to 100.

    - `rasa:pro:champion-server-limited` - Champion license, can be used
        to deploy an assistant on a server for a limited external use case.
        The license allows for a maximum of 1000 conversations per month.

    The champion scopes on their own do not unlock any features, they are
    used to limit the use of the license to the agreed upon number of
    conversations. The champion scopes can be combined with the feature
    scopes to enable the features for the champion license, e.g.
    `rasa:pro:champion rasa:pro` would enable the champion license for
    local development and the Rasa Pro features.
    """

    __slots__ = ["jti", "iat", "nbf", "scope", "exp", "email", "company"]

    def __init__(
        self,
        *,
        company: Text,
        email: Text,
        exp: int,
        scope: Text,
        jti: Optional[Text] = None,
        iat: Optional[int] = None,
        nbf: Optional[int] = None,
    ) -> None:
        """Initializes an instance of `License`.

        Args:
            company: Company this license is issued to.
            email: Contact email for this license.
            exp: Expiration date (UNIX epoch time).
            scope: The license scope
            jti: JWT unique identifier - a unique identifier for this
                license. Defaults to a UUID4 if not set.
            iat: Created at (UNIX epoch time).
                Defaults to current time if not set.
            nbf: Time at which the license starts being valid (UNIX epoch time).
                Defaults to current time if not set.
        """
        self.company = company
        self.email = email
        self.exp = exp
        self.jti = jti or str(uuid.uuid4())
        self.iat = iat or int(time.time())
        self.nbf = nbf or self.iat
        self.scope = scope

    def as_dict(self) -> Dict[Text, Any]:
        """Returns this license as a dictionary object.

        Returns:
            License represented using a `dict`.
        """
        return {attr: getattr(self, attr) for attr in License.__slots__}

    def __str__(self) -> Text:
        """Returns a text representation of this license.

        Returns:
            String representing this license.
        """
        return f"License <{self.as_dict()}>"

    @staticmethod
    def decode(
        encoded_license: Text,
        check_not_before: Optional[bool] = True,
        check_expiration: Optional[bool] = True,
        product_area: Text = PRODUCT_AREA,
    ) -> "License":
        """Returns an instance of `License` from an encoded JWT.

        Args:
            encoded_license: JWT in encoded form.
            check_not_before: Check if not_before is in the future.
            check_expiration: Check if token has expired.
            product_area: The product scope of the license.

        Raises:
            LicenseSignatureInvalidException: If the license signature could
                not be validated.
            LicenseNotYetValidException: If the license is not valid yet.
            LicenseExpiredException: If the license has expired, or has been
                blocklisted.
            LicenseEncodingException: If the JWT was not correctly encoded.
            LicenseSchemaException: If the license contains unknown or extra
                fields, or if it is missing fields.

        Returns:
            A validated enterprise license.
        """
        try:
            decoded = jwt.decode(
                encoded_license,
                key=PUBLIC_KEY,
                algorithms=[ALGORITHM],
                options={
                    "verify_nbf": check_not_before,
                    "verify_exp": check_expiration,
                },
            )
        except jwt.exceptions.InvalidSignatureError:
            raise LicenseSignatureInvalidException(
                "Could not verify the license's signature."
            )
        except jwt.exceptions.ImmatureSignatureError:
            raise LicenseNotYetValidException("The license is not valid yet (nbf).")
        except jwt.exceptions.ExpiredSignatureError:
            raise LicenseExpiredException("The license has already expired (exp).")
        except jwt.exceptions.DecodeError:
            # Handle `DecodeError` last since other more specific exceptions
            # such as `InvalidSignatureError` inherit from it.
            raise LicenseEncodingException("Could not decode license as JWT.")

        if set(decoded.keys()) != set(License.__slots__):
            raise LicenseSchemaException("Invalid license schema.")

        license_scope = decoded.get("scope", "")
        if not is_valid_license_scope(product_area, license_scope):
            raise LicenseScopeException(
                f"The product scope of your issued license does not "
                f"include {product_area}."
            )

        if decoded["jti"] in JTI_BLOCKLIST:
            raise LicenseExpiredException("The license has already expired.")

        return License(**decoded)

    def encode(self, private_key: Text) -> Text:
        """Encodes this license into a JWT.

        NOTE: This method is only useful in the context of testing, or when
        using administrative scripts such as `license.py`. It
        should not be used otherwise.

        Args:
            private_key: Private key to use. Should correspond to `PUBLIC_KEY`.

        Returns:
            Encoded license.
        """
        return jwt.encode(self.as_dict(), key=private_key, algorithm=ALGORITHM)


def retrieve_license_from_env() -> Text:
    """Return the license found in the env var."""
    stored_env_values = dotenv_values(".env")
    license_from_env = os.environ.get(LICENSE_ENV_VAR)
    license = license_from_env or stored_env_values.get(LICENSE_ENV_VAR)
    if not license:
        raise LicenseNotFoundException()
    return license


def is_license_expiring_soon(license: License) -> bool:
    """Check if the license is about to expire in less than 30 days."""
    return license.exp - time.time() < 30 * 24 * 60 * 60


def validate_license_from_env(product_area: Text = PRODUCT_AREA) -> None:
    try:
        license_text = retrieve_license_from_env()
        license = License.decode(license_text, product_area=product_area)

        if is_license_expiring_soon(license):
            structlogger.warning(
                "license.expiration.warning",
                event_info=(
                    "Your license is about to expire. "
                    "Please contact Rasa for a renewal."
                ),
                expiration_date=datetime.fromtimestamp(
                    license.exp, timezone.utc
                ).isoformat(),
            )
    except LicenseNotFoundException:
        structlogger.error("license.not_found.error")
        raise SystemExit(
            f"A Rasa Pro license is required. "
            f"Please set the environmental variable "
            f"`{LICENSE_ENV_VAR}` to a valid license string. "
        )
    except LicenseValidationException as e:
        structlogger.error("license.validation.error", error=e)
        raise SystemExit(
            f"Failed to validate Rasa Pro license "
            f"which was read from environmental variable `{LICENSE_ENV_VAR}`. "
            f"Please ensure `{LICENSE_ENV_VAR}` is set to a valid license string. "
        )


def get_license_expiration_date() -> Text:
    """Return the expiration date of the license."""
    license_exp = property_of_active_license(lambda active_license: active_license.exp)
    return datetime.fromtimestamp(license_exp, timezone.utc).isoformat()


def is_valid_license_scope(product_area: Text, license_scope: Text) -> bool:
    """Verifies that the license scope matches the rasa-plus product area."""
    required_scopes = derive_scope_hierarchy(product_area)
    licensed_product_areas = derive_scope_hierarchy(license_scope)

    # update scopes that are required but not present in the license scope
    required_scopes.difference_update(licensed_product_areas)

    # this is dependent on a format where each product area is separated
    # by whitespace in the license scope field value
    licensed_scopes = license_scope.split()

    # initialise a variable to count matches of
    # licensed sub product scope pattern found in the required scopes set
    sub_product_scope_match_count = 0

    for required in required_scopes:
        for licensed in licensed_scopes:
            if re.search(licensed, required) is not None:
                sub_product_scope_match_count += 1

    return sub_product_scope_match_count == len(required_scopes)


def derive_scope_hierarchy(scope: Text) -> Set[Text]:
    """Derives all upper levels of the specified scopes and adds to a resulting set.

    For example, the `rasa:pro:plus` scope would result in the following set:
    {rasa, rasa:pro, rasa:pro:plus}.
    """
    product_hierarchy = [area.split(SCOPE_DELIMITER) for area in scope.split()]
    required_scopes = [
        SCOPE_DELIMITER.join(hierarchy[0:end])
        for hierarchy in product_hierarchy
        for end in range(1, len(hierarchy) + 1)
    ]

    return set(required_scopes)


T = TypeVar("T")


def property_of_active_license(prop: Callable[[License], T]) -> Optional[T]:
    """Return a property for this installation based on license.

    Returns:
    The property of the license if it exists, otherwise None.
    """
    try:
        retrieved_license = retrieve_license_from_env()
        if not retrieved_license:
            return None
        decoded = License.decode(retrieved_license)
        return prop(decoded)
    except LicenseNotFoundException:
        return None
    except LicenseValidationException as e:
        structlogger.warn("licensing.active_license.invalid", error=e)
        return None


def get_license_hash() -> Optional[Text]:
    """Return the hash of the current active license."""
    license_value = retrieve_license_from_env()
    return hashlib.sha256(license_value.encode("utf-8")).hexdigest()


def is_champion_server_license() -> bool:
    """Return whether the current license is a developer license."""

    def has_developer_license_scope(license: License) -> bool:
        if CHAMPION_SERVER_LIMITED_SCOPE in license.scope:
            return True
        if CHAMPION_SERVER_INTERNAL_SCOPE in license.scope:
            return True
        return False

    # or False is needed to handle the case where the license is not set
    # this is to please the type checker
    return property_of_active_license(has_developer_license_scope) or False


def conversation_limit_for_license() -> Optional[int]:
    def conversations_limit(license: License) -> Optional[int]:
        """Return maximum number of conversations per month for this license."""
        scope = license.scope

        if CHAMPION_SERVER_LIMITED_SCOPE in scope:
            return 1000
        if CHAMPION_SERVER_INTERNAL_SCOPE in scope:
            return 100
        return None

    return property_of_active_license(conversations_limit)


async def validate_limited_server_license(app: Sanic, loop: AbstractEventLoop) -> None:
    """Validate a limited server license and schedule conversation counting job."""
    max_number_conversations = conversation_limit_for_license()

    if app.ctx.agent:
        if max_number_conversations is None:
            structlogger.debug("licensing.server_limit.unlimited")
        else:
            store = app.ctx.agent.tracker_store
            await run_conversation_counting(store, max_number_conversations)
            await _schedule_conversation_counting(app, store, max_number_conversations)
    else:
        structlogger.warn("licensing.validate_limited_server_license.no_agent")


async def handle_soft_limit_reached(
    conversation_count: int, max_number_conversations: int, month: datetime
) -> None:
    """Log a warning when the number of conversations exceeds the soft limit."""
    structlogger.error(
        "licensing.conversation_count.exceeded",
        event_info=(
            "The number of conversations has exceeded the limit granted "
            "by your license. Please contact us to upgrade your license."
        ),
        conversation_count=conversation_count,
        max_number_conversations=max_number_conversations,
    )
    telemetry.track_conversation_count_soft_limit(conversation_count, month)


async def handle_hard_limit_reached(
    conversation_count: int, max_number_conversations: int, month: datetime
) -> None:
    """Log an error when the number of conversations exceeds the hard limit."""
    structlogger.error(
        "licensing.conversation_count.exceeded",
        event_info=(
            "The number of conversations has exceeded the limit granted "
            "by your license. Please contact us to upgrade your license."
        ),
        conversation_count=conversation_count,
        max_number_conversations=max_number_conversations,
    )
    telemetry.track_conversation_count_hard_limit(conversation_count, month)
    print_error_and_exit(
        "The number of conversations has exceeded the limit granted by your "
        "license. Please contact us to upgrade your license."
    )


def current_utc_month() -> datetime:
    """Return the current month in UTC timezone."""
    return datetime.now(timezone.utc).replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    )


async def run_conversation_counting(
    tracker_store: Optional["TrackerStore"], max_number_conversations: int
) -> None:
    """Count the number of conversations started in the current month and log it."""
    start_of_month_utc = current_utc_month()
    start_of_month_timestamp = start_of_month_utc.timestamp()

    conversation_count = await _count_conversations_after(
        tracker_store, start_of_month_timestamp
    )

    if conversation_count >= max_number_conversations * HARD_LIMIT_FACTOR:
        # user is above the hard limit
        await handle_hard_limit_reached(
            conversation_count, max_number_conversations, start_of_month_utc
        )
    elif conversation_count > max_number_conversations:
        await handle_soft_limit_reached(
            conversation_count, max_number_conversations, start_of_month_utc
        )

    structlogger.debug(
        "licensing.conversation_count",
        conversation_count=conversation_count,
        max_number_conversations=max_number_conversations,
    )
    telemetry.track_conversation_count(conversation_count, start_of_month_utc)


async def _schedule_conversation_counting(
    app: Sanic, tracker_store: Optional["TrackerStore"], max_number_conversations: int
) -> None:
    """Schedule a job counting the number of conversations in the current month."""

    async def conversation_counting_job(
        app: Sanic,
        tracker_store: Optional["TrackerStore"],
        max_number_conversations: int,
    ) -> None:
        try:
            await run_conversation_counting(tracker_store, max_number_conversations)
        except SystemExit:
            # we've reached the conversation limit
            app.stop()

    (await jobs.scheduler()).add_job(
        conversation_counting_job,
        "interval",
        # every 24 hours with a random offset of max 1 hour
        # clusters tend to get started at the same time, this avoids them
        # doing the "conversation counting work" at exactly the same time
        seconds=24 * 60 * 60 + random.randint(0, 60 * 60),
        args=[app, tracker_store, max_number_conversations],
        id="count-conversations",
        replace_existing=True,
    )


async def _count_conversations_after(
    tracker_store: Optional["TrackerStore"], after_timestamp: float
) -> int:
    """Count the number of conversations started in the current month."""
    if tracker_store is None:
        return 0

    return await tracker_store.count_conversations(after_timestamp=after_timestamp)

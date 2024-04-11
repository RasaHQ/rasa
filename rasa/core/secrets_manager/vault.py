import base64
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Text, Union, cast

import hvac.exceptions
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from rasa.shared.exceptions import RasaException
from rasa.utils.endpoints import EndpointConfig

from rasa.core.secrets_manager.constants import (
    TRACKER_STORE_ENDPOINT_TYPE,
    TRANSIT_KEY_FOR_ENCRYPTION_LABEL,
    VAULT_SECRET_MANAGER_NAME,
)
from rasa.core.secrets_manager.endpoints import (
    CredentialsLocation,
    EndpointReader,
    EndpointTrait,
    TrackerStoreEndpointValidator,
)
from rasa.core.secrets_manager.secret_manager import SecretManagerConfig, SecretsManager

logger = logging.getLogger(__name__)


class VaultCredentialsLocation:
    """Represents where the secret is stored in Vault.

    Secret can be encrypted with vault's transit engine.

    source: The source of the secret.
    secret_key: The key of the secret.
    transit_key: The key used to encrypt the secret.
    """

    def __init__(
        self,
        source: Text,
        secret_key: Text,
        transit_key: Optional[Text] = None,
    ) -> None:
        """Initialise the secret.

        Args:
            source: The source of the secret.
            secret_key: The key of the secret.
            transit_key: The key used to encrypt the secret.
        """
        self.source = source
        self.secret_key = secret_key
        self.transit_key = transit_key

    @classmethod
    def from_credentials_location(
        cls, credentials_location: CredentialsLocation
    ) -> "VaultCredentialsLocation":
        """Initialise the secret from CredentialsLocation.

        Args:
            credentials_location: The CredentialsLocation.
        """
        transit_key = (
            credentials_location.kwargs.get(TRANSIT_KEY_FOR_ENCRYPTION_LABEL)
            if credentials_location.kwargs
            else None
        )

        return cls(
            source=credentials_location.source,
            secret_key=credentials_location.secret_key,
            transit_key=transit_key,
        )

    @staticmethod
    def is_vault_credential_location_instance(
        value: Union[str, "VaultCredentialsLocation"],
    ) -> bool:
        """Check if the config key is a vault encrypted secret.

        Args:
            value: The config key.

        Returns:
            True if value is an instance of VaultCredentialsLocation.
        """
        return isinstance(value, VaultCredentialsLocation)


class VaultEndpointConfigReader:
    """Used to read Vault location specification in the endpoint config."""

    def __init__(self, endpoint_config: "EndpointConfig") -> None:
        """Initialize the Vault endpoint processor.

        Args:
            endpoint_config: The endpoint config.
        """
        self._endpoint_config_reader = EndpointReader(endpoint_config=endpoint_config)

    def get_credentials_location(
        self,
        property_name: Text,
    ) -> Optional[VaultCredentialsLocation]:
        """Get the VaultCredentialsLocation for an endpoint property.

        Args:
            property_name: The name of the endpoint's property

        Returns:
            The VaultCredentialsLocation for the property.
        """
        property_value = self._endpoint_config_reader.get_property_value(property_name)

        if property_value and CredentialsLocation.is_credentials_location_instance(
            property_value
        ):
            credentials_location = cast(CredentialsLocation, property_value)
            if (
                credentials_location.get_secret_manager_name()
                == VAULT_SECRET_MANAGER_NAME
            ):
                return VaultCredentialsLocation.from_credentials_location(
                    credentials_location=credentials_location
                )
            else:
                raise RasaException(
                    f"Secret manager {credentials_location.source} is not supported."
                )

        return None

    def get_transit_keys_per_endpoint_property(self) -> Optional[Dict[Text, Text]]:
        """Get the transit keys for endpoint's properties.

        Returns:
            A dictionary of the transit keys per endpoint's properties.

        Example:
        {
            "username": "transit_key_for_username",
            "password": "transit_key_for_password"
        }

        """
        transit_keys = {}

        for config_key_name in self._endpoint_config_reader.get_keys():
            credentials_location = self.get_credentials_location(
                property_name=config_key_name
            )

            if (
                credentials_location
                and VaultCredentialsLocation.is_vault_credential_location_instance(
                    credentials_location
                )
            ):
                if credentials_location.transit_key:
                    transit_keys[credentials_location.secret_key] = (
                        credentials_location.transit_key
                    )

        return transit_keys if transit_keys else None


class VaultSecretsManager(SecretsManager):
    """Secrets Manager for Vault.

    It supports both transit secret and kv engines.
    """

    def __init__(
        self,
        host: Text,
        token: Text,
        secrets_path: Text,
        transit_mount_point: Optional[Text] = None,
        namespace: Optional[Text] = None,
    ):
        """Initialise the VaultSecretsManager.

        Args:
            host: The host of the vault server.
            token: The token to authenticate with the vault server.
            secrets_path: The path to the secrets in the vault server.
            transit_mount_point: The mount point of the transit engine.
            namespace: The namespace in which secrets reside in.
        """
        self.host = host
        self.transit_mount_point = transit_mount_point
        self.token = token
        self.secrets_path = secrets_path
        self.namespace = namespace

        # Create client
        self.client = hvac.Client(
            url=self.host,
            token=self.token,
            namespace=self.namespace,
        )

        self.vault_token_manager = VaultTokenManager(host=self.host, token=self.token)

        self.vault_token_manager.start()

    def name(self) -> Text:
        """Return unique identifier of the secret manager.

        Returns:
            The unique identifier of the secret manager.
        """
        return VAULT_SECRET_MANAGER_NAME

    def load_secrets(self, endpoint_trait: "EndpointTrait") -> Dict[Text, Any]:
        """Load secrets from vault server.

        If secrets are encrypted with transit engine,
        they will be decrypted if transit_keys are provided.

        Args:
            endpoint_trait: The endpoint trait.

        Returns:
            The secrets mapped to their endpoint property.

        Example:
        {
            "username": "username stored in vault
            "password": "password stored in vault"
        }
        """
        logger.info(f"Loading secrets from vault server at {self.host}.")
        read_response = self.client.secrets.kv.read_secret_version(
            mount_point="secret", path=self.secrets_path
        )

        secrets = read_response["data"]["data"]

        reader = get_vault_endpoint_reader(endpoint_trait=endpoint_trait)

        if reader is None:
            raise RasaException(
                f"Failed to create endpoint processor "
                f"for endpoint {endpoint_trait.endpoint_type}"
            )

        transit_keys = reader.get_transit_keys_per_endpoint_property()

        if transit_keys is not None:
            self._decode_transit_secrets(secrets=secrets, transit_keys=transit_keys)

        logger.info(f"Successfully loaded secrets from vault server at {self.host}.")

        return secrets

    @staticmethod
    def _is_vault_transit_cipher(cipher: Text) -> bool:
        """Check if the cipher is a transit cipher.

        Args:
            cipher: The cipher to check.

        Returns:
            True if the cipher is a transit cipher, False otherwise.
        """
        # if cipher starts with vault:v/d: it is a transit cipher
        regexp = re.compile(r"^vault:v\d+:")
        return re.search(regexp, cipher) is not None

    def _decrypt_transit_cipher(self, cipher: Text, transit_key_name: Text) -> Text:
        """Decrypt the cipher using the transit key.

        Args:
            cipher: The cipher to decrypt.
            transit_key_name: The name of the transit key to use.

        Returns:
            The cipher decrypted with the transit key.
        """
        logger.info("Decrypting cipher using transit key.")
        decrypted_data = self.client.secrets.transit.decrypt_data(
            name=transit_key_name,
            ciphertext=cipher,
            mount_point=self.transit_mount_point,
        )
        logger.info("Finished decrypting cipher using transit key.")

        return (
            base64.standard_b64decode(decrypted_data.get("data").get("plaintext"))
            .decode("utf-8")
            .strip()
        )

    def _decode_transit_secrets(
        self, secrets: Dict[Text, Text], transit_keys: Optional[Dict[Text, Text]] = None
    ) -> None:
        """Decode values from tracker_store if they are encrypted with transit engine.

        Args:
            secrets:  Map of secrets to decode. The values will be updated in place.
            transit_keys: Map of secrets to transit keys.
        """
        if transit_keys is None:
            return

        logger.info("Start to decrypt secrets using transit secrets engine.")
        for secret_name, secret_value in secrets.items():
            if self._is_vault_transit_cipher(secret_value):
                transit_key_name = transit_keys.get(secret_name)
                if transit_key_name is not None:
                    secrets[secret_name] = self._decrypt_transit_cipher(
                        cipher=secret_value, transit_key_name=transit_key_name
                    )
        logger.info("Finished decrypting secrets using transit secrets engine.")


class VaultTokenManager:
    """VaultTokenManager renews the vault token if it renewable."""

    TOKEN_MINIMAL_REMAINING_LIFE_IN_SECONDS = 1
    TOKEN_MINIMAL_REFRESH_TIME_IN_SECONDS = 1
    # The bottom threshold for token's TTL (time to live) in seconds.
    # Set to 15 seconds to compensate for possible network latency
    TOKEN_TTL_BOTTOM_THRESHOLD_IN_SECONDS = 15
    DEFAULT_NUMBER_OF_RETRIES_FOR_TOKEN_REFRESH = 5
    TIME_TO_WAIT_BETWEEN_RETRIES_IN_SECONDS = 1

    def __init__(
        self,
        host: Text,
        token: Text,
        number_of_retries: int = DEFAULT_NUMBER_OF_RETRIES_FOR_TOKEN_REFRESH,
    ):
        """Initialise the VaultTokenManager.

        Args:
            host: The host of the vault server.
            token: The token to authenticate with the vault server.
                   If the token is expiring, it will be automatically refreshed.
            number_of_retries: The number of retries to refresh the token.
        """
        self.host = host
        self.token = token
        self.number_of_retries = number_of_retries

        self.client = hvac.Client(
            url=self.host,
            token=self.token,
        )

    def start(self) -> None:
        """Start refreshing the token if it is expiring."""
        renew_response: Dict[Text, Dict[Text, Any]] = (
            self.client.auth.token.lookup_self()
        )
        is_token_expiring = renew_response["data"]["renewable"]
        if is_token_expiring:
            refresh_interval_in_seconds = renew_response["data"]["creation_ttl"]
            first_expiring_period_in_seconds = renew_response["data"]["ttl"]
            logger.info("Token is expiring. Starting periodic token refresh.")
            self._start_token_refresh(
                first_expiring_period_in_seconds, refresh_interval_in_seconds
            )

    def _renew_token(self) -> None:
        """Renew the token."""
        index = 0
        for index in range(self.number_of_retries):
            try:
                logger.info("Renewing vault token.")
                self.client.auth.token.renew_self()
                logger.info("Finished renewing vault token.")
                break
            except Exception as e:
                if index == self.number_of_retries - 1:
                    raise RasaException(
                        f"Failed to renew vault token after "
                        f"{self.number_of_retries} retries."
                    )

                logger.warning(
                    f"Failed to renew vault token. "
                    f"Error: {e}. "
                    f"Trying again in "
                    f"{self.TIME_TO_WAIT_BETWEEN_RETRIES_IN_SECONDS} second(s)"
                )
                time.sleep(self.TIME_TO_WAIT_BETWEEN_RETRIES_IN_SECONDS)

    def _start_token_refresh(
        self, remaining_life_in_seconds: int, refresh_interval_in_seconds: int
    ) -> None:
        """Start a background job to refresh the token.

        Args:
            remaining_life_in_seconds: The remaining life of the token in seconds.
                                       This period can be shorter than the token's TTL.
            refresh_interval_in_seconds: The refresh interval in seconds.
                                         It matches the token's TTL (creation_ttl).
                                         This interval is used after the remaining life.
        """
        self.scheduler = BackgroundScheduler()
        self.scheduler.start()

        # Reduce the refresh interval and the remaining life by
        # TOKEN_TTL_BOTTOM_THRESHOLD_IN_SECONDS
        # to compensate for possible network latency.
        # If token's TTL and remaining TTL is less than
        # TOKEN_TTL_BOTTOM_THRESHOLD_IN_SECONDS
        # we must not reduce the refresh interval and the remaining life
        if refresh_interval_in_seconds > self.TOKEN_TTL_BOTTOM_THRESHOLD_IN_SECONDS:
            refresh_interval_in_seconds -= self.TOKEN_TTL_BOTTOM_THRESHOLD_IN_SECONDS
        else:
            logger.info(
                f"Token's refresh interval is less than "
                f"{self.TOKEN_TTL_BOTTOM_THRESHOLD_IN_SECONDS} seconds. "
                f"Readjusting the refresh interval to "
                f"{self.TOKEN_MINIMAL_REFRESH_TIME_IN_SECONDS} seconds."
            )
            refresh_interval_in_seconds = self.TOKEN_MINIMAL_REFRESH_TIME_IN_SECONDS

        if remaining_life_in_seconds > self.TOKEN_TTL_BOTTOM_THRESHOLD_IN_SECONDS:
            remaining_life_in_seconds -= self.TOKEN_TTL_BOTTOM_THRESHOLD_IN_SECONDS
        else:
            logger.info(
                f"Token's TTL is less than "
                f"{self.TOKEN_TTL_BOTTOM_THRESHOLD_IN_SECONDS} seconds. "
                f"Readjusting the refresh interval to "
                f"{self.TOKEN_MINIMAL_REMAINING_LIFE_IN_SECONDS} seconds."
            )
            remaining_life_in_seconds = self.TOKEN_MINIMAL_REMAINING_LIFE_IN_SECONDS

        self.scheduler.add_job(
            func=self._renew_token,
            trigger=IntervalTrigger(
                # add seconds to current time to make sure that the first refresh
                start_date=datetime.now()
                + timedelta(seconds=remaining_life_in_seconds),
                seconds=refresh_interval_in_seconds,
            ),
        )


class VaultSecretManagerConfig(SecretManagerConfig):
    """Configuration for the Vault secret manager."""

    def __init__(
        self,
        url: Text,
        token: Text,
        secrets_path: Text,
        transit_mount_point: Text = "transit",
        namespace: Optional[Text] = None,
    ) -> None:
        """Initialise the VaultSecretManagerConfig.

        Args:
            url: The URL of the vault server.
            token: The token to authenticate with the vault server.
            secrets_path: The path to the secrets in the vault server.
            transit_mount_point: The mount point of the transit engine.
            namespace: The namespace in which secrets reside in.
        """
        super().__init__(VAULT_SECRET_MANAGER_NAME)
        self.url = url
        self.token = token
        self.secrets_path = secrets_path
        self.transit_mount_point = transit_mount_point
        self.namespace = namespace


@dataclass
class VaultSecretManagerNonStrictConfig:
    """Non-Strict configuration for the Vault secret manager.

    It is used to validate and merge configurations
    from environment variables and endpoints.yml file.
    """

    url: Optional[Text]
    token: Optional[Text]
    secrets_path: Optional[Text]
    transit_mount_point: Optional[Text]
    namespace: Optional[Text] = None

    def is_empty(self) -> bool:
        """Check if all the values are empty."""
        return (
            (self.url is None or self.url == "")
            and (self.token is None or self.token == "")
            and (self.secrets_path is None or self.secrets_path == "")
            and (self.transit_mount_point is None or self.transit_mount_point == "")
            and (self.namespace is None or self.namespace == "")
        )

    def is_valid(self) -> bool:
        """Check if all the values are valid.

        url and token are required,
        secrets_path is required,
        transit_mount_point is optional, but if provided, it must not be empty.

        Returns:
            True if all the values are valid, False otherwise.
        """
        return (
            self.url is not None
            and self.url != ""
            and self.token is not None
            and self.token != ""
            and self.secrets_path is not None
            and self.secrets_path != ""
            and self._is_optional_value_valid(self.transit_mount_point)
            and self._is_optional_value_valid(self.namespace)
        )

    @staticmethod
    def _is_optional_value_valid(value: Optional[Text]) -> bool:
        """Check if the optional value is valid.

        Args:
            value: The optional value to check.

        Returns:
            True if the optional value is valid, False otherwise.
        """
        return value is None or value != ""

    def merge(
        self, other: "VaultSecretManagerNonStrictConfig"
    ) -> "VaultSecretManagerNonStrictConfig":
        """Merge two VaultSecretManagerEnvConfig objects.

        Args:
            other: The other VaultSecretManagerEnvConfig object.

        Returns:
            The merged VaultSecretManagerEnvConfig object.
        """
        return VaultSecretManagerNonStrictConfig(
            url=self.url or other.url,
            token=self.token or other.token,
            secrets_path=self.secrets_path or other.secrets_path,
            transit_mount_point=self.transit_mount_point or other.transit_mount_point,
            namespace=self.namespace or other.namespace,
        )


def get_vault_endpoint_reader(
    endpoint_trait: EndpointTrait,
) -> Optional[VaultEndpointConfigReader]:
    """Return VaultEndpointConfigReader for a given endpoint type.

    Args:
        endpoint_trait: The endpoint trait to create the reader for.

    Returns:
        The reader associated with the endpoint's type.
    """
    if (
        endpoint_trait.endpoint_type == TRACKER_STORE_ENDPOINT_TYPE
        and TrackerStoreEndpointValidator(
            endpoint_reader=EndpointReader(
                endpoint_config=endpoint_trait.endpoint_config
            )
        ).is_endpoint_config_valid()
    ):
        return VaultEndpointConfigReader(endpoint_config=endpoint_trait.endpoint_config)

    return None

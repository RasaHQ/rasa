import os
from typing import Optional, Text, cast

from rasa.core.secrets_manager.constants import (
    SECRET_MANAGER_ENV_NAME,
    VAULT_DEFAULT_RASA_SECRETS_PATH,
    VAULT_ENDPOINT_MOUNT_POINT_LABEL,
    VAULT_ENDPOINT_NAMESPACE_LABEL,
    VAULT_ENDPOINT_SECRETS_PATH_LABEL,
    VAULT_ENDPOINT_TRANSIT_MOUNT_POINT_LABEL,
    VAULT_MOUNT_POINT_ENV_NAME,
    VAULT_NAMESPACE_ENV_NAME,
    VAULT_RASA_SECRETS_PATH_ENV_NAME,
    VAULT_SECRET_MANAGER_NAME,
    VAULT_TOKEN_ENV_NAME,
    VAULT_TRANSIT_MOUNT_POINT_ENV_NAME,
    VAULT_URL_ENV_NAME,
)
from rasa.core.secrets_manager.secret_manager import (
    SecretManagerConfig,
    SecretsManager,
    SecretsManagerProvider,
)
from rasa.core.secrets_manager.vault import (
    VaultSecretManagerConfig,
    VaultSecretManagerNonStrictConfig,
    VaultSecretsManager,
)
from rasa.shared.exceptions import RasaException
from rasa.utils.endpoints import EndpointConfig, read_endpoint_config


def create(config: SecretManagerConfig) -> Optional[SecretsManager]:
    """Create a SecretsManager based on the configuration.

    Args:
        config: SecretManagerConfig

    Returns:
        SecretsManager created based on the configuration
    """
    secret_manager = None

    if config.secret_manager_type == VAULT_SECRET_MANAGER_NAME:
        vault_config = cast(VaultSecretManagerConfig, config)
        secret_manager = VaultSecretsManager(
            host=vault_config.url,
            token=vault_config.token,
            transit_mount_point=vault_config.transit_mount_point,
            secrets_path=vault_config.secrets_path,
            namespace=vault_config.namespace,
            mount_point=vault_config.mount_point,
        )

    return secret_manager


def read_vault_endpoint_config(
    endpoints_file: Optional[Text],
) -> Optional[VaultSecretManagerNonStrictConfig]:
    """Read endpoints file to discover vault config.

    Args:
        endpoints_file: Path to the endpoints file

    Returns:
        A configuration for the vault secret manager
    """
    if endpoints_file is None:
        return None

    endpoint_config = read_endpoint_config(
        filename=endpoints_file, endpoint_type="secrets_manager"
    )

    if endpoint_config:
        url = endpoint_config.url
        token = endpoint_config.token
        transit_mount_point = endpoint_config.kwargs.get(
            VAULT_ENDPOINT_TRANSIT_MOUNT_POINT_LABEL
        )
        secrets_path = endpoint_config.kwargs.get(VAULT_ENDPOINT_SECRETS_PATH_LABEL)
        namespace = endpoint_config.kwargs.get(VAULT_ENDPOINT_NAMESPACE_LABEL)
        mount_point = endpoint_config.kwargs.get(VAULT_ENDPOINT_MOUNT_POINT_LABEL)

        return VaultSecretManagerNonStrictConfig(
            url=url,
            token=token,
            transit_mount_point=transit_mount_point,
            secrets_path=secrets_path or VAULT_DEFAULT_RASA_SECRETS_PATH,
            namespace=namespace,
            mount_point=mount_point,
        )

    return None


def read_vault_env_vars() -> VaultSecretManagerNonStrictConfig:
    """Read environment variables to discover vault config.

    Returns:
        A configuration for the vault secret manager
    """
    url = os.getenv(VAULT_URL_ENV_NAME)
    token = os.getenv(VAULT_TOKEN_ENV_NAME)
    transit_mount_point = os.getenv(VAULT_TRANSIT_MOUNT_POINT_ENV_NAME)
    secrets_path = os.getenv(VAULT_RASA_SECRETS_PATH_ENV_NAME)
    namespace = os.getenv(VAULT_NAMESPACE_ENV_NAME)
    mount_point = os.getenv(VAULT_MOUNT_POINT_ENV_NAME)

    return VaultSecretManagerNonStrictConfig(
        url=url,
        token=token,
        transit_mount_point=transit_mount_point,
        secrets_path=secrets_path,
        namespace=namespace,
        mount_point=mount_point,
    )


def read_vault_config(
    endpoints_file: Optional[Text],
) -> Optional[VaultSecretManagerConfig]:
    """Read endpoints file to discover vault config.

    Args:
        endpoints_file: Path to the endpoints file

    Returns:
        A configuration for the vault secret manager
    """
    env_config = read_vault_env_vars()
    endpoint_config = read_vault_endpoint_config(endpoints_file)

    if env_config.is_valid() and endpoint_config is None:
        return VaultSecretManagerConfig(
            **env_config.__dict__,
        )

    if endpoint_config is not None:
        vault_config = env_config.merge(endpoint_config)

        if vault_config.is_empty():
            return None

        if vault_config.is_valid():
            return VaultSecretManagerConfig(
                **vault_config.__dict__,
            )

    raise RasaException(
        f"Cannot start Vault secret manager based on configuration from env vars "
        f"{VAULT_URL_ENV_NAME} = {env_config.url}, "
        f"{VAULT_TOKEN_ENV_NAME} = {env_config.token}, "
        f"{VAULT_RASA_SECRETS_PATH_ENV_NAME} = {env_config.secrets_path}, "
        f"{VAULT_TRANSIT_MOUNT_POINT_ENV_NAME} = {env_config.transit_mount_point}. "
        f"{VAULT_NAMESPACE_ENV_NAME} = {env_config.namespace}. "
        f"{VAULT_MOUNT_POINT_ENV_NAME} = {env_config.mount_point}. "
    )


def read_secret_manager_from_endpoint_config(
    endpoints_file: Text,
) -> Optional[EndpointConfig]:
    """Read endpoints file to discover secret manager.

    Args:
        endpoints_file: Path to the endpoints file

    Returns:
        A secret manager
    """
    secret_manager_config = read_endpoint_config(
        filename=endpoints_file, endpoint_type="secrets_manager"
    )
    if secret_manager_config is None:
        return None

    return secret_manager_config


def read_secret_manager_config(
    endpoints_file: Optional[Text],
) -> Optional[SecretManagerConfig]:
    """Read endpoints file to discover secret manager config.

    If the secret manager is configured, set the environment variables required
    to connect to the secret manager.
    The above steps are skipped if the environment variables are already set.

    Args:
        endpoints_file: Path to the endpoints file

    Returns:
        A configuration for the secret manager
    """
    secret_manager_name = os.getenv(SECRET_MANAGER_ENV_NAME)

    if secret_manager_name is None:
        if endpoints_file is None:
            return None

        secret_manager_config = read_secret_manager_from_endpoint_config(
            endpoints_file=endpoints_file
        )
        if secret_manager_config is None:
            return None

        secret_manager_name = secret_manager_config.type

    if secret_manager_name == VAULT_SECRET_MANAGER_NAME:
        return read_vault_config(endpoints_file)

    return None


def load_secret_manager(endpoints_file: Optional[Text]) -> Optional[SecretsManager]:
    """Create secret manager based on the configuration in the endpoints file.

    Args:
        endpoints_file: Path to the endpoints file.

    Returns:
        The secret manager or `None` if no secret manager is configured.
    """
    secret_manager_config = read_secret_manager_config(endpoints_file=endpoints_file)

    if (
        secret_manager_config is None
        or secret_manager_config.secret_manager_type is None
    ):
        return None

    provider = SecretsManagerProvider()
    secret_manager = create(secret_manager_config)

    if secret_manager is not None:
        provider.register_secret_manager(secret_manager)

    return secret_manager

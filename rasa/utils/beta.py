import logging
from textwrap import dedent

from rasa.shared.exceptions import RasaException
from rasa.utils.common import get_bool_env_variable

logger = logging.getLogger(__name__)


class BetaNotEnabledException(SystemExit, RasaException):
    """Raised when a beta feature is not enabled."""

    def __init__(self, feature_name: str, env_flag: str) -> None:
        """Initializes the exception.

        Args:
            feature_name: The name of the feature.
            env_flag: Name of the environment variable that enables the feature.
        """
        self.feature_name = feature_name
        self.env_flag = env_flag
        super().__init__(
            dedent(
                f"""
            This release, including the {feature_name} feature, is a beta release
            provided to you only for trial and evaluation. You cannot use this
            beta software release as part of any production environment. This
            beta software release is not intended for processing sensitive or
            personal data. There may be bugs or errors - we provide it “as is”,
            without any reps and warranties. We may never commercialize some beta
            software. Your use of the {feature_name} feature and any other
            software or material included in this beta release is subject to
            the Beta Software Terms (https://rasa.com/beta-terms/).
            Please make sure you read these terms and conditions prior to
            using this beta release.

            You need to explicitly enable the {feature_name} feature, before
            usage. Set the `{env_flag}=true` environment variable before
            running the command again.
            """
            )
        )


def ensure_beta_feature_is_enabled(feature_name: str, env_flag: str) -> None:
    """Checks if a beta feature is enabled. Raises an exception if not.

    If the feature is enabled, a usage warning is printed.

    Args:
        feature_name: The name of the feature.
        env_flag: The name of the environment variable that enables the feature.
    """
    if not get_bool_env_variable(env_flag, default_variable_value=False):
        raise BetaNotEnabledException(feature_name, env_flag)

    print_beta_usage_info(feature_name, env_flag)


def print_beta_usage_info(feature_name: str, env_flag: str) -> None:
    """Prints a usage warning about a beta feature.

    Args:
        feature_name: The name of the feature.
        env_flag: The name of the environment variable that enables the feature.
    """
    logger.warning(
        dedent(
            f"""\
        The {feature_name} feature is currently released in a beta version.

        The feature might change in the future 🔬. Your download and use of
        this beta release are subject to the Beta Software Terms
        (https://rasa.com/beta-terms/).
        It's not intended for production. Don't use it to process sensitive data.
        If you do, it's at your own risk. It's not mandatory to download
        or use this software release. We're looking forward to your feedback.

        If you want to disable this beta feature, set the environment variable
        `{env_flag}=false`.
    """
        )
    )

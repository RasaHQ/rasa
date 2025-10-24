import importlib.util
import sys

import structlog

structlogger = structlog.get_logger()


def check_tensorflow_installation() -> None:
    """Check if TensorFlow is installed without proper Rasa extras."""
    # Check if tensorflow is available in the environment
    tensorflow_available = importlib.util.find_spec("tensorflow") is not None

    if not tensorflow_available:
        return

    # Check if any TensorFlow-related extras were installed
    # We do this by checking for packages that are only installed with nlu/full extras
    tensorflow_extras_indicators = [
        "tensorflow_text",  # Only in nlu/full extras
        "tensorflow_hub",  # Only in nlu/full extras
        "tf_keras",  # Only in nlu/full extras
    ]

    extras_installed = any(
        importlib.util.find_spec(pkg) is not None
        for pkg in tensorflow_extras_indicators
    )

    if tensorflow_available and not extras_installed:
        structlogger.warning(
            "installation_utils.tensorflow_installation",
            warning=(
                "TensorFlow is installed but Rasa was not installed with TensorFlow "
                "support, i.e. additional packages required to use NLU components "
                "have not been installed. For the most reliable setup, delete your "
                "current virtual environment, create a new one, and install Rasa "
                "again. Please follow the instructions at "
                "https://rasa.com/docs/pro/installation/python"
            ),
        )


def check_tensorflow_integrity() -> None:
    """Check if TensorFlow installation is corrupted or incomplete."""
    # Only check if tensorflow is available
    if importlib.util.find_spec("tensorflow") is None:
        return

    try:
        # Try to import tensorflow - this will fail if installation is corrupted
        import tensorflow as tf

        # Try to access a basic TensorFlow function
        _ = tf.constant([1, 2, 3])
    except Exception:
        # Simplified error message for all TensorFlow corruption issues
        structlogger.error(
            "installation_utils.tensorflow_integrity",
            issue=(
                "TensorFlow is installed but appears to be corrupted or incomplete. "
                "For the most reliable setup, delete your current virtual "
                "environment, create a new one, and install Rasa again. "
                "Please follow the instructions at "
                "https://rasa.com/docs/pro/installation/python"
            ),
        )
        sys.exit(1)


def check_rasa_availability() -> None:
    """Check if Rasa is installed and importable."""
    if importlib.util.find_spec("rasa") is None:
        structlogger.error(
            "installation_utils.rasa_availability",
            issue=(
                "Rasa is not installed in this environment. "
                "Please follow the instructions at "
                "https://rasa.com/docs/pro/installation/python"
            ),
        )
        sys.exit(1)

    try:
        _ = importlib.import_module("rasa")
    except Exception as e:
        structlogger.error(
            "installation_utils.rasa_availability",
            issue=(
                f"Rasa is installed but cannot be imported: {e!s}."
                f"Please follow the instructions at "
                f"https://rasa.com/docs/pro/installation/python"
            ),
        )
        sys.exit(1)


def check_for_installation_issues() -> None:
    """Check for all potential installation issues.

    Returns:
        List of warning messages for detected issues.
    """
    # Check if Rasa is available first
    check_rasa_availability()

    # Check TensorFlow integrity first (more critical)
    check_tensorflow_integrity()

    # Check for orphaned TensorFlow
    check_tensorflow_installation()

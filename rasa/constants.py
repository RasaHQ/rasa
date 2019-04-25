import os

DEFAULT_ENDPOINTS_PATH = "endpoints.yml"
DEFAULT_CREDENTIALS_PATH = "credentials.yml"
DEFAULT_CONFIG_PATH = "config.yml"
DEFAULT_DOMAIN_PATH = "domain.yml"
DEFAULT_ACTIONS_PATH = "actions"
DEFAULT_MODELS_PATH = "models"
DEFAULT_DATA_PATH = "data"
DEFAULT_RESULTS_PATH = "results"
DEFAULT_REQUEST_TIMEOUT = 60 * 5  # 5 minutes

MINIMUM_COMPATIBLE_VERSION = "0.15.0a6"

GLOBAL_USER_CONFIG_PATH = os.path.expanduser("~/.config/rasa/config.yml")

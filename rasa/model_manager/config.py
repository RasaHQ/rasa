import os
import sys

from rasa.constants import RASA_REMOTE_STORAGE_ENV_VAR_NAME

DEFAULT_SERVER_BASE_WORKING_DIRECTORY = "working-data"

SERVER_BASE_WORKING_DIRECTORY = os.environ.get(
    "RASA_MODEL_SERVER_BASE_DIRECTORY", DEFAULT_SERVER_BASE_WORKING_DIRECTORY
)

SERVER_PORT = os.environ.get("RASA_MODEL_SERVER_PORT", 8000)

SERVER_BASE_URL = os.environ.get("RASA_MODEL_SERVER_BASE_URL", None)

# defaults to storing on the local hard drive
SERVER_MODEL_REMOTE_STORAGE = os.environ.get(RASA_REMOTE_STORAGE_ENV_VAR_NAME, None)

# The path to the python executable that is running this script
# we will use the same python to run training / bots
RASA_PYTHON_PATH = sys.executable

# the max limit for parallel training requests
DEFAULT_MAX_PARALLEL_TRAININGS = 10

MAX_PARALLEL_TRAININGS = os.getenv(
    "MAX_PARALLEL_TRAININGS", DEFAULT_MAX_PARALLEL_TRAININGS
)
# the max limit for parallel running bots
DEFAULT_MAX_PARALLEL_BOT_RUNS = 10

MAX_PARALLEL_BOT_RUNS = os.getenv(
    "MAX_PARALLEL_BOT_RUNS", DEFAULT_MAX_PARALLEL_BOT_RUNS
)

DEFAULT_SERVER_PATH_PREFIX = "talk"

DEFAULT_MIN_REQUIRED_DISCSPACE_MB = 1

MIN_REQUIRED_DISCSPACE_MB = int(
    os.getenv("MIN_REQUIRED_DISCSPACE_MB", DEFAULT_MIN_REQUIRED_DISCSPACE_MB)
)

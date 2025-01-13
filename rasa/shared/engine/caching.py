import os
from pathlib import Path

DEFAULT_CACHE_LOCATION = Path(".rasa", "cache")
DEFAULT_CACHE_NAME = "cache.db"
DEFAULT_CACHE_SIZE_MB = 1000

CACHE_LOCATION_ENV = "RASA_CACHE_DIRECTORY"
CACHE_DB_NAME_ENV = "RASA_CACHE_NAME"
CACHE_SIZE_ENV = "RASA_MAX_CACHE_SIZE"


def get_local_cache_location() -> Path:
    """Returns the location of the local cache."""
    return Path(os.environ.get(CACHE_LOCATION_ENV, DEFAULT_CACHE_LOCATION))


def get_max_cache_size() -> float:
    """Returns the maximum cache size."""
    return float(os.environ.get(CACHE_SIZE_ENV, DEFAULT_CACHE_SIZE_MB))


def get_cache_database_name() -> str:
    """Returns the database name in the cache."""
    return os.environ.get(CACHE_DB_NAME_ENV, DEFAULT_CACHE_NAME)

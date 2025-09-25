import importlib.util
import logging

logger = logging.getLogger(__name__)

# check if TensorFlow is available
TENSORFLOW_AVAILABLE = importlib.util.find_spec("tensorflow") is not None

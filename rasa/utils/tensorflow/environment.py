import logging
import os
from typing import Text, Dict
import typing

import rasa.shared.utils.io
from rasa.constants import (
    ENV_GPU_CONFIG,
    ENV_CPU_INTER_OP_CONFIG,
    ENV_CPU_INTRA_OP_CONFIG,
)
from rasa.utils.tensorflow.constants import TF_DETERMINISTIC_OPS
from rasa.shared.utils import io as shared_io_utils

if typing.TYPE_CHECKING:
    from tensorflow import config as tf_config

logger = logging.getLogger(__name__)


def _setup_gpu_environment() -> None:
    """Sets configuration for TensorFlow GPU environment based on env variable."""
    gpu_memory_config = os.getenv(ENV_GPU_CONFIG)

    if not gpu_memory_config:
        return

    # Import from tensorflow only if necessary (environment variable was set)
    from tensorflow import config as tf_config

    parsed_gpu_config = _parse_gpu_config(gpu_memory_config)
    physical_gpus = tf_config.list_physical_devices("GPU")

    # Logic taken from https://www.tensorflow.org/guide/gpu
    if physical_gpus:
        for gpu_id, gpu_id_memory in parsed_gpu_config.items():
            _allocate_gpu_memory(physical_gpus[gpu_id], gpu_id_memory)

    else:
        rasa.shared.utils.io.raise_warning(
            f"You have an environment variable '{ENV_GPU_CONFIG}' set but no GPUs were "
            f"detected to configure."
        )


def _allocate_gpu_memory(
    gpu_instance: "tf_config.PhysicalDevice", logical_memory: int
) -> None:
    """Create a new logical device for the requested amount of memory.

    Args:
        gpu_instance: PhysicalDevice instance of a GPU device.
        logical_memory: Absolute amount of memory to be allocated to the new logical
            device.
    """

    from tensorflow import config as tf_config

    try:
        tf_config.experimental.set_virtual_device_configuration(
            gpu_instance,
            [
                tf_config.experimental.VirtualDeviceConfiguration(
                    memory_limit=logical_memory
                )
            ],
        )

    except RuntimeError:
        # Helper explanation of where the error comes from
        raise RuntimeError(
            "Error while setting up tensorflow environment. "
            "Virtual devices must be set before GPUs have been initialized."
        )


def _parse_gpu_config(gpu_memory_config: Text) -> Dict[int, int]:
    """Parse GPU configuration variable from a string to a dict.

    Args:
        gpu_memory_config: String containing the configuration for GPU usage.

    Returns:
        Parsed configuration as a dictionary with GPU IDs as keys and requested memory
        as the value.
    """

    # gpu_config is of format "gpu_id_1:gpu_id_1_memory, gpu_id_2: gpu_id_2_memory"
    # Parse it and store in a dictionary
    parsed_gpu_config = {}

    try:
        for instance in gpu_memory_config.split(","):
            instance_gpu_id, instance_gpu_mem = instance.split(":")
            instance_gpu_id = int(instance_gpu_id)
            instance_gpu_mem = int(instance_gpu_mem)

            parsed_gpu_config[instance_gpu_id] = instance_gpu_mem
    except ValueError:
        # Helper explanation of where the error comes from
        raise ValueError(
            f"Error parsing GPU configuration. Please cross-check the format of "
            f"'{ENV_GPU_CONFIG}' at https://rasa.com/docs/rasa/tuning-your-model"
            f"#restricting-absolute-gpu-memory-available ."
        )

    return parsed_gpu_config


def _setup_cpu_environment() -> None:
    """Set configuration for the CPU environment based on environment variable."""
    inter_op_parallel_threads = os.getenv(ENV_CPU_INTER_OP_CONFIG)
    intra_op_parallel_threads = os.getenv(ENV_CPU_INTRA_OP_CONFIG)

    if not inter_op_parallel_threads and not intra_op_parallel_threads:
        return

    from tensorflow import config as tf_config

    if inter_op_parallel_threads:
        try:
            inter_op_parallel_threads = int(inter_op_parallel_threads.strip())
        except ValueError:
            raise ValueError(
                f"Error parsing the environment variable '{ENV_CPU_INTER_OP_CONFIG}'. "
                f"Please cross-check the value."
            )

        tf_config.threading.set_inter_op_parallelism_threads(inter_op_parallel_threads)

    if intra_op_parallel_threads:
        try:
            intra_op_parallel_threads = int(intra_op_parallel_threads.strip())
        except ValueError:
            raise ValueError(
                f"Error parsing the environment variable '{ENV_CPU_INTRA_OP_CONFIG}'. "
                f"Please cross-check the value."
            )

        tf_config.threading.set_intra_op_parallelism_threads(intra_op_parallel_threads)


def setup_tf_environment() -> None:
    """Setup CPU and GPU related environment settings for TensorFlow."""
    _setup_cpu_environment()
    _setup_gpu_environment()


def check_deterministic_ops() -> None:
    """Warn user if they have set TF_DETERMINISTIC_OPS."""
    if os.getenv(TF_DETERMINISTIC_OPS, False):
        shared_io_utils.raise_warning(
            f"You have set '{TF_DETERMINISTIC_OPS}' to 1. If you are "
            f"using one or more GPU(s) and use any of 'SparseFeaturizer', "
            f"'TEDPolicy', 'DIETClassifier', 'UnexpecTEDIntentPolicy', or "
            f"'ResponseSelector' training and testing will fail as there are no "
            f"deterministic GPU implementations of some underlying TF ops.",
            category=UserWarning,
        )

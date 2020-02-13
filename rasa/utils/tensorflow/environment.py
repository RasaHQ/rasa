import logging
import os
from typing import Text, Tuple
import warnings
from rasa.constants import (
    ENV_GPU_CONFIG,
    ENV_CPU_INTER_OP_CONFIG,
    ENV_CPU_INTRA_OP_CONFIG,
)
from tensorflow import config as tf_config

logger = logging.getLogger(__name__)


def setup_gpu_environment() -> None:
    """Set configuration for a GPU environment based on the environment variable set"""

    gpu_memory_config = os.getenv(ENV_GPU_CONFIG, None)
    if gpu_memory_config:
        parsed_gpu_config = parse_gpu_config(gpu_memory_config)
        physical_gpus = tf_config.list_physical_devices("GPU")

        # Logic taken from https://www.tensorflow.org/guide/gpu
        if physical_gpus:
            for gpu_id, gpu_id_memory in parsed_gpu_config.items():
                try:
                    tf_config.experimental.set_virtual_device_configuration(
                        physical_gpus[gpu_id],
                        [
                            tf_config.experimental.VirtualDeviceConfiguration(
                                memory_limit=gpu_id_memory
                            )
                        ],
                    )

                except RuntimeError:
                    # Add a helper explanation where the error comes from
                    raise RuntimeError(
                        "Error while setting up tensorflow environment. "
                        "Virtual devices must be set before GPUs have been initialized"
                    )

        else:
            warnings.warn(
                f"You have an environment variable '{ENV_GPU_CONFIG}' set but no GPUs were detected to configure"
            )


def parse_gpu_config(gpu_memory_config: Text):
    """Parse GPU configuration variable from a string to a dict"""

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
        # Add a helper explanation
        raise ValueError(
            f"Error parsing GPU configuration. Please cross-check the format of '{ENV_GPU_CONFIG}'"
        )

    return parsed_gpu_config


def setup_cpu_environment() -> Tuple[int, int]:
    """Set configuration for the CPU environment based on the environment variable set"""

    inter_op_parallel_threads = os.getenv(ENV_CPU_INTER_OP_CONFIG, None)
    intra_op_parallel_threads = os.getenv(ENV_CPU_INTRA_OP_CONFIG, None)

    if inter_op_parallel_threads:

        try:
            inter_op_parallel_threads = int(inter_op_parallel_threads.strip())
        except ValueError:
            raise ValueError(
                f"Error parsing the environment variable '{ENV_CPU_INTER_OP_CONFIG}'. Please "
                f"cross-check the value"
            )

        tf_config.threading.set_inter_op_parallelism_threads(inter_op_parallel_threads)

    if intra_op_parallel_threads:

        try:
            intra_op_parallel_threads = int(intra_op_parallel_threads.strip())
        except ValueError:
            raise ValueError(
                f"Error parsing the environment variable '{ENV_CPU_INTRA_OP_CONFIG}'. Please "
                f"cross-check the value"
            )

        tf_config.threading.set_intra_op_parallelism_threads(intra_op_parallel_threads)

    # Returning the actual values as a confirmation. Helps with tests too.
    return (
        tf_config.threading.get_inter_op_parallelism_threads(),
        tf_config.threading.get_intra_op_parallelism_threads(),
    )


def setup_tf_environment():

    setup_cpu_environment()
    setup_gpu_environment()

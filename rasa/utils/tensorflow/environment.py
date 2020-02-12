import logging
import os
from typing import Text

import tensorflow as tf
from rasa.constants import (
    ENV_GPU_CONFIG,
    ENV_CPU_INTER_OP_CONFIG,
    ENV_CPU_INTRA_OP_CONFIG,
)

logger = logging.getLogger(__name__)


def setup_gpu_environment(gpu_memory_config: Text) -> None:

    if gpu_memory_config:

        # Parse GPU config
        # gpu_config is of format "gpu_id_1:gpu_id_1_memory, gpu_id_2: gpu_id_2_memory"
        # Parse it and store in a dictionary
        parsed_gpu_config = {
            instance.split(":")[0].strip(): int(instance.split(":")[1].strip())
            for instance in gpu_memory_config.split(",")
        }

        physical_gpus = tf.config.list_physical_devices("GPU")

        # Logic taken from https://www.tensorflow.org/guide/gpu
        if physical_gpus:

            for gpu_id, gpu_id_memory in parsed_gpu_config.items():
                try:
                    tf.config.experimental.set_virtual_device_configuration(
                        physical_gpus[int(gpu_id)],
                        [
                            tf.config.experimental.VirtualDeviceConfiguration(
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
            logger.info(
                "You have an environment variable GPU_MEMORY_ALLOC set but no GPUs were detected to configure"
            )


def setup_cpu_environment(
    inter_op_parallel_threads: Text, intra_op_parallel_threads: Text
) -> None:

    if inter_op_parallel_threads:
        tf.config.threading.set_inter_op_parallelism_threads(
            int(inter_op_parallel_threads.strip())
        )

    if intra_op_parallel_threads:
        tf.config.threading.set_intra_op_parallelism_threads(
            int(intra_op_parallel_threads.strip())
        )


def setup_tf_environment():

    # Get all env variables
    gpu_memory_config = os.getenv(ENV_GPU_CONFIG, None)
    inter_op_parallel_threads = os.getenv(ENV_CPU_INTER_OP_CONFIG, None)
    intra_op_parallel_threads = os.getenv(ENV_CPU_INTRA_OP_CONFIG, None)

    setup_gpu_environment(gpu_memory_config)
    setup_cpu_environment(inter_op_parallel_threads, intra_op_parallel_threads)

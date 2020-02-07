import os
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


def setup_gpu_environment(gpu_memory_config):

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

                except RuntimeError as e:
                    # Virtual devices must be set before GPUs have been initialized
                    raise RuntimeError(
                        "Error while setting up tensorflow environment. "
                        "Virtual devices must be set before GPUs have been initialized"
                    )

        else:
            logger.info(
                "You have an environment variable GPU_MEMORY_ALLOC set but no GPUs were detected to configure"
            )


def setup_cpu_environment(inter_op_parallel_threads, intra_op_parallel_threads):

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
    gpu_memory_config = os.getenv("TF_GPU_MEMORY_ALLOC", None)
    inter_op_parallel_threads = os.getenv("TF_INTER_OP_PARALLELISM_THREADS", None)
    intra_op_parallel_threads = os.getenv("TF_INTRA_OP_PARALLELISM_THREADS", None)

    setup_gpu_environment(gpu_memory_config)
    setup_cpu_environment(inter_op_parallel_threads, intra_op_parallel_threads)

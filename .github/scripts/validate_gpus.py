import sys

import tensorflow as tf


def check_gpu_available():
    num_gpus = len(tf.config.list_physical_devices("GPU"))
    print(f"Num GPUs Available: {num_gpus}")
    if num_gpus <= 0:
        sys.exit(1)


if __name__ == "__main__":
    check_gpu_available()

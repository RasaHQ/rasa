import tensorflow as tf
from rasa.utils.tensorflow.environment import (
    setup_cpu_environment,
    setup_gpu_environment,
)


def test_tf_cpu_environment_setting():

    inter_op_threads = "2"
    intra_op_threads = "3"

    setup_cpu_environment(inter_op_threads, intra_op_threads)

    assert tf.config.threading.get_inter_op_parallelism_threads() == 2
    assert tf.config.threading.get_intra_op_parallelism_threads() == 3

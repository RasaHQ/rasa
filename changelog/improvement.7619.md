Bump TensorFlow version to 2.6.

Users can no longer set `TF_DETERMINISTIC_OPS=1` if they are using GPU(s) because a `tf.errors.UnimplementedError`
will be thrown by TensorFlow (read more [here](https://github.com/tensorflow/tensorflow/releases/tag/v2.6.0)).
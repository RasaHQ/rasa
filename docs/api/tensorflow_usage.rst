:desc: Find out how to configure your environment for efficient usage of TensorFlow inside Rasa Open Source.

.. _tensorflow_usage:

TensorFlow Configuration
========================

TensorFlow allows configuring options in the runtime environment via
`TF Config submodule <https://www.tensorflow.org/api_docs/python/tf/config>`_. Rasa Open Source supports a smaller subset of these
configuration options and makes appropriate calls to the ``tf.config`` submodule.
This smaller subset comprises of configurations that developers frequently use with Rasa Open Source.
All configuration options are specified using environment variables as shown in subsequent sections.

Optimizing CPU Performance
--------------------------

.. note::
    We recommend that you configure these options only if you are an advanced TensorFlow user and understand the 
    implementation of the machine learning components in your pipeline. These options affect how operations are carried 
    out under the hood in Tensorflow. Leaving them at their default values is fine.

Depending on the TensorFlow operations a NLU component or Core policy uses, you can leverage multi-core CPU
parallelism by tuning these options.

Parallelizing One Operation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set ``TF_INTRA_OP_PARALLELISM_THREADS`` as an environment variable to specify the maximum number of threads that can be used
to parallelize the execution of one operation. For example, operations like ``tf.matmul()`` and ``tf.reduce_sum`` can be executed
on multiple threads running in parallel. The default value for this variable is ``0`` which means TensorFlow would
allocate one thread per CPU core.

Parallelizing Multiple Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set ``TF_INTER_OP_PARALLELISM_THREADS`` as an environment variable to specify the maximum number of threads that can be used
to parallelize the execution of multiple **non-blocking** operations. These would include operations that do not have a
directed path between them in the TensorFlow graph. In other words, the computation of one operation does not affect the
computation of the other operation. The default value for this variable is ``0`` which means TensorFlow would allocate one thread per CPU core.

To understand more about how these two options differ from each other, refer to this
`stackoverflow thread <https://stackoverflow.com/questions/41233635/meaning-of-inter-op-parallelism-threads-and-intra-op-parallelism-threads/41233901#41233901>`_.


Optimizing GPU Performance
--------------------------

Limiting GPU Memory Growth
^^^^^^^^^^^^^^^^^^^^^^^^^^

TensorFlow by default blocks all the available GPU memory for the running process. This can be limiting if you are running
multiple TensorFlow processes and want to distribute memory across them. To prevent Rasa Open Source from blocking all
of the available GPU memory, set the environment variable ``TF_FORCE_GPU_ALLOW_GROWTH`` to ``True``.

Restricting Absolute GPU Memory Available
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You may want to limit the absolute amount of GPU memory that can be used by a Rasa Open Source process.

For example, say you have two visible GPUs(``GPU:0`` and ``GPU:1``) and you want to allocate 1024 MB from the first GPU
and 2048 MB from the second GPU. You can do this by setting the environment variable ``TF_GPU_MEMORY_ALLOC`` to ``"0:1024, 1:2048"``.

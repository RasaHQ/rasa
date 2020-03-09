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
    It is advisable to configure these options only if you are an advanced tensorflow user and understand
    how different operations are implemented under the hood.

These options are useful if you have implemented a custom component as part of your pipeline. Based on the tensorflow operations
that the component uses, you can leverage multi-core CPU parallelism by tuning these options.

Parallelizing One Operation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set ``TF_INTRA_OP_PARALLELISM_THREADS`` as an environment variable to specify the maximum number of threads that can be used
to parallelize the execution of one operation. For example, operations like ``tf.matmul()`` and ``tf.reduce_sum`` can be executed
on multiple threads running parallely. The default value for this variable is ``0`` which means TensorFlow should
pick an appropriate value depending on the system configuration.

Parallelizing Multiple Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set ``TF_INTER_OP_PARALLELISM_THREADS`` as an environment variable to specify the maximum number of threads that can be used
to parallelize the execution of multiple **non-blocking** operations. These would included operations which do not have a
directed path between them in the tensorflow graph. The default value for this variable is ``0`` which means TensorFlow
should pick an appropriate value depending on the system configuration.

To understand more about how these two options differ from each other, refer to this
`stackoverflow thread <https://stackoverflow.com/a/41233901/3001665>`_.


Optimizing GPU Performance
--------------------------

Limiting GPU Memory Growth
^^^^^^^^^^^^^^^^^^^^^^^^^^

TensorFlow by default blocks all the available GPU memory for the running process. This can be limiting if you are running
multiple TensorFlow processes and want to distribute memory across them. To prevent Rasa OS from blocking the
complete available GPU memory, set the environment variable ``TF_FORCE_GPU_ALLOW_GROWTH`` to ``True``.

Restricting Absolute GPU Memory Available
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Often, a developer wants to limit the absolute amount of GPU memory that can be used by a Rasa OS process.

For example, you may have two visible GPUs(``GPU:0`` and ``GPU:1``) and you want to allocate 1024 MB from the first GPU
and 2048 MB from the second GPU. You can do so by setting an environment variable as ``TF_GPU_MEMORY_ALLOC="0:1024, 1:2048"``.

Another scenario can be where you have access to 2 GPUs(``GPU:0`` and ``GPU:1``) but you would like to use only the second
GPU. ``TF_GPU_MEMORY_ALLOC="1:2048"`` would make 2048 MB of memory available from GPU 1.

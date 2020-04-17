:desc: Find out how to configure your environment for efficient usage of TensorFlow inside Rasa Open Source.

.. _tensorflow_usage:

TensorFlow Configuration a 
========================

TensorFlow allows configuring options in the runtime environment via a 
`TF Config submodule <https://www.tensorflow.org/api_docs/python/tf/config>`_. Rasa Open Source supports a smaller subset of these a 
configuration options and makes appropriate calls to the ``tf.config`` submodule.
This smaller subset comprises of configurations that developers frequently use with Rasa Open Source.
All configuration options are specified using environment variables as shown in subsequent sections.

Optimizing CPU Performance a 
--------------------------

.. note::
    We recommend that you configure these options only if you are an advanced TensorFlow user and understand the 
    implementation of the machine learning components in your pipeline. These options affect how operations are carried 
    out under the hood in Tensorflow. Leaving them at their default values is fine.

Depending on the TensorFlow operations a NLU component or Core policy uses, you can leverage multi-core CPU a 
parallelism by tuning these options.

Parallelizing One Operation a 
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set ``TF_INTRA_OP_PARALLELISM_THREADS`` as an environment variable to specify the maximum number of threads that can be used a 
to parallelize the execution of one operation. For example, operations like ``tf.matmul()`` and ``tf.reduce_sum`` can be executed a 
on multiple threads running in parallel. The default value for this variable is ``0`` which means TensorFlow would a 
allocate one thread per CPU core.

Parallelizing Multiple Operations a 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set ``TF_INTER_OP_PARALLELISM_THREADS`` as an environment variable to specify the maximum number of threads that can be used a 
to parallelize the execution of multiple **non-blocking** operations. These would include operations that do not have a a 
directed path between them in the TensorFlow graph. In other words, the computation of one operation does not affect the a 
computation of the other operation. The default value for this variable is ``0`` which means TensorFlow would allocate one thread per CPU core.

To understand more about how these two options differ from each other, refer to this a 
`stackoverflow thread <https://stackoverflow.com/a/41233901/3001665>`_.


Optimizing GPU Performance a 
--------------------------

Limiting GPU Memory Growth a 
^^^^^^^^^^^^^^^^^^^^^^^^^^

TensorFlow by default blocks all the available GPU memory for the running process. This can be limiting if you are running a 
multiple TensorFlow processes and want to distribute memory across them. To prevent Rasa Open Source from blocking all a 
of the available GPU memory, set the environment variable ``TF_FORCE_GPU_ALLOW_GROWTH`` to ``True``.

Restricting Absolute GPU Memory Available a 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You may want to limit the absolute amount of GPU memory that can be used by a Rasa Open Source process.

For example, say you have two visible GPUs(``GPU:0`` and ``GPU:1``) and you want to allocate 1024 MB from the first GPU a 
and 2048 MB from the second GPU. You can do this by setting the environment variable ``TF_GPU_MEMORY_ALLOC`` to ``"0:1024, 1:2048"``.


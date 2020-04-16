:desc: Information about changes between major versions of chatbot framework
       Rasa Core and how you can migrate from one version to another.

.. _migration-guide:

Migration Guide
===============
This page contains information about changes between major versions and
how you can migrate from one version to another.

.. _migration-to-rasa-1.0:

Rasa NLU 0.14.x and Rasa Core 0.13.x to Rasa 1.0
------------------------------------------------

General
~~~~~~~

- The scripts in ``rasa.core`` and ``rasa.nlu`` can no longer be executed. To train, test, run, ... an NLU or Core
  model, you should now use the command line interface ``rasa``. The functionality is, for the most part, the same as before.
  Some changes in commands reflect the combined training and running of NLU and Core models, but NLU and Core can still
  be trained and used individually. If you attempt to run one of the old scripts in ``rasa.core`` or ``rasa.nlu``,
  an error is thrown that points you to the command you
  should use instead. See all the new commands at :ref:`command-line-interface`.

- If you have written a custom output channel, all ``send_`` methods subclassed
  from the ``OutputChannel`` class need to take an additional ``**kwargs``
  argument. You can use these keyword args from your custom action code or the
  templates in your domain file to send any extra parameters used in your
  channel's send methods.

- If you were previously importing the ``Button`` or ``Element`` classes from
  ``rasa_core.dispatcher``, these are now to be imported from ``rasa_sdk.utils``.

Script parameters
~~~~~~~~~~~~~~~~~
- All script parameter names have been unified to follow the same schema.
  Any underscores (``_``) in arguments have been replaced with dashes (``-``).
  For example: ``--max_history`` has been changed to ``--max-history``. You can
  see all of the script parameters in the ``--help`` output of the commands
  in the :ref:`command-line-interface`.

- The ``--num_threads`` parameter was removed from the ``run`` command. The
  server will always run single-threaded, but will now run asynchronously. If you want to
  make use of multiple processes, feel free to check out the `Sanic server
  documentation <https://sanic.readthedocs.io/en/latest/sanic/deploying.html#running-via-gunicorn>`_.

- To avoid conflicts in script parameter names, connectors in the ``run`` command now need to be specified with
  ``--connector``, as ``-c`` is no longer supported. The maximum history in the ``rasa visualize`` command needs to be
  defined with ``--max-history``. Output paths and log files cannot be specified with ``-o`` anymore; ``--out`` and
  ``--log-file`` should be used. NLU data has been standarized to be ``--nlu`` and the name of
  any kind of data files or directory to be ``--data``.

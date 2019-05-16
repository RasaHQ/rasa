:desc: Information about changes between major versions of chatbot framework
       Rasa Core and how you can migrate from one version to another.

.. _migration:

Migration Guide
===============
This page contains information about changes between major versions and
how you can migrate from one version to another.

.. _migration-to-rasa-1.0:

0.14.x to Rasa 1.0

General
~~~~~~~

- The scripts in ``rasa.core`` and ``rasa.nlu`` can no longer be executed. To train, test, run, ... a rasa nlu or core
  model, you should now use the command line interface ``rasa``. The functionality is the same as before. If you run
  one of the old scripts in ``rasa.core`` or ``rasa.nlu`` an error is thrown that also points you to the command you
  should use instead.
  Mapping of old scripts to new commands:
  ``rasa.core.run`` -> ``rasa shell``
  ``rasa.core.server`` -> ``rasa run``
  ``rasa.core.test`` -> ``rasa test core``
  ``rasa.core.train`` -> ``rasa train core``
  ``rasa.core.visualize`` -> ``rasa visualize``
  ``rasa.nlu.convert`` -> ``rasa data``
  ``rasa.nlu.evaluate`` -> ``rasa test nlu``
  ``rasa.nlu.run`` -> ``rasa shell``
  ``rasa.nlu.server`` -> ``rasa run``
  ``rasa.nlu.test`` -> ``rasa test nlu``
  ``rasa.nlu.train`` -> ``rasa train nlu``

- If you have written a custom output channel, all ``send_`` methods subclassed
  from the ``OutputChannel`` class need to take an additional ``**kwargs``
  argument. You can use these keyword args from your custom action code or the
  templates in your domain file to send any extra parameters used in your
  channel's send methods.

- If you were importing the ``Button`` or ``Element`` classes from
  ``rasa_core.dispatcher``, these are now to be imported from ``rasa_sdk.utils``.

Script parameters
~~~~~~~~~~~~~~~~~
- All script parameter names have been unified to follow the same schema.
  Any underscores (``_``) in arguments have been replaced with dashes (``-``).
  Due to change the following argument names changed:
  ``--nlu_data`` -> ``--nlu-data``
  ``--dump_stories`` -> ``--dump-stories``
  ``--debug_plots`` -> ``--debug-plots``
  ``--max_history`` -> ``--max-history``
  ``--pre_load`` -> ``--pre-load``
  ``--max_training_processes`` -> ``--max-training-processes``
  ``--wait_time_between_pulls`` -> ``--wait-time-between-pulls``
  ``--response_log`` -> ``--response-log``
  ``--fail_on_prediction_errors`` -> ``--fail-on-prediction-errors``
  ``--max_stories`` -> ``--max-stories``
  ``--jwt_method`` -> ``--jwt-method``
  ``--jwt_secret`` -> ``--jwt-secret``
  ``--log_file`` -> ``--log-file``
  ``--enable_api`` -> ``--enable-api``
  ``--auth_token`` -> ``--auth-token``
  ``--skip_visualization`` -> ``--skip-visualization``
  ``--training_fraction`` -> ``--training-fraction``

- the ``--num_threads`` parameter got removed from the ``run`` command. The
  server will always run single threaded, but in an async way. If you want to
  make use of multiple processes, feel free to check out the sanic server
  documentation https://sanic.readthedocs.io/en/latest/sanic/deploying.html#running-via-gunicorn

- To avoid conflicts in the script parameter names, connectors in the ``run`` command now need to be specified with
  ``--connector``. ``-c`` is no longer supported. The maximum history in the ``rasa visualize`` command need to be
  defined with ``--max-history``. Output paths and log files cannot be specified with ``-o`` anymore. ``--out`` and
  ``--log-file`` should be used. Furthermore, we standardized the name of NLU data to be ``--nlu`` and the name of
  any kind of data files or directory to be ``--data``.

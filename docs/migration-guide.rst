:desc: Information about changes between major versions of chatbot framework
       Rasa Core and how you can migrate from one version to another.

.. _migration-guide:

Migration Guide
===============

.. edit-link::

This page contains information about changes between major versions and
how you can migrate from one version to another.

.. _migration-to-rasa-1.3:

Rasa 1.2 to Rasa 1.3
--------------------
.. warning::

  This is a release **breaking backwards compatibility**.
  It is not possible to load previously trained models. Please make sure to retrain a
  model before trying to use it with this improved version.

General
~~~~~~~
- Default parameters of ``EmbeddingIntentClassifier`` are changed. See :ref:`components` for details.
  Architecture implementation is changed as well, so **old trained models cannot be loaded**.
  Default parameters and architecture for ``EmbeddingPolicy`` are changed. See :ref:`policies` for details.
  It uses transformer instead of lstm. **Old trained models cannot be loaded**.
  They use ``inner`` similarity and ``softmax`` loss by default instead of
  ``cosine`` similarity and ``margin`` loss (can be set in config file).
  They use ``balanced`` batching strategy by default to counteract class imbalance problem.
  The meaning of ``evaluate_on_num_examples`` is changed. If it is non zero, random examples will be
  picked by stratified split and used as **hold out** validation set, so they will be excluded from training data.
  We suggest to set it to zero (default) if data set contains a lot of unique examples of dialogue turns.
  Removed ``label_tokenization_flag`` and ``label_split_symbol`` from component. Instead moved intent splitting to ``Tokenizer`` components via ``intent_tokenization_flag`` and ``intent_split_symbol`` flag.
- Default ``max_history`` for ``EmbeddingPolicy`` is ``None`` which means it'll use
  the ``FullDialogueTrackerFeaturizer``. We recommend to set ``max_history`` to
  some finite value in order to use ``MaxHistoryTrackerFeaturizer``
  for **faster training**. See :ref:`featurization` for details.
  We recommend to increase ``batch_size`` for ``MaxHistoryTrackerFeaturizer``
  (e.g. ``"batch_size": [32, 64]``)
- **Compare** mode of ``rasa train core`` allows the whole core config comparison.
  Therefore, we changed the naming of trained models. They are named by config file
  name instead of policy name. Old naming style will not be read correctly when
  creating **compare** plots (``rasa test core``). Please remove old trained models
  in comparison folder and retrain. Normal core training is unaffected.
- We updated the **evaluation metric** for our **NER**. We report the weighted precision and f1-score.
  So far we included ``no-entity`` in this report. However, as most of the tokens actually don't have
  an entity set, this will influence the weighted precision and f1-score quite a bit. From now on we
  exclude ``no-entity`` from the evaluation. The overall metrics now only include proper entities. You
  might see a drop in the performance scores when running the evaluation again.

.. _migration-to-rasa-1.0:

Rasa NLU 0.14.x and Rasa Core 0.13.x to Rasa 1.0
------------------------------------------------
.. warning::

  This is a release **breaking backwards compatibility**.
  It is not possible to load previously trained models. Please make sure to retrain a
  model before trying to use it with this improved version.

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

- Rasa NLU and Core previously used `separate configuration files 
  <https://legacy-docs.rasa.com/docs/nlu/0.15.1/migrations/?&_ga=2.218966814.608734414.1560704810-314462423.1543594887#id1>`_.
  These two files should be merged into a single file either named ``config.yml``, or passed via the ``--config`` parameter.

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

HTTP API
~~~~~~~~
- There are numerous HTTP API endpoint changes which can be found `here <http://rasa.com/docs/rasa/api/http-api/>`_.

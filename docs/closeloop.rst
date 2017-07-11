.. _section_closeloop:

Improving your models from feedback
===================================

When the rasa_nlu server is running, it keeps track of all the predictions it's made and saves these to a log file. 
By default log files are placed in ``logs/``. The files in this directory contain one json object per line.
You can fix any incorrect predictions and add them to your training set to improve your parser.
After adding these to your training data, but before retraining your model, it is strongly recommended that you use the
visualizer to spot any errors, see :ref:`Visualizing training data <visualizing-the-training-data>`.

Low confidence prediction logging
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You can choose to automatically log problematic predictions to be later processed in the Rasa NLU Trainer.
To use this logger, create a ``trainer_conf.json`` at the root of your Rasa installation folder:
.. code-block:: json

    {
      "source": "data/test/wit_converted_to_rasa.json", // the training file path
      "pending_file": "data/test/wit_converted_to_rasa-pending.json", // the file in which Rasa store the pending strings
      "threshold": 0.7 // the confidence limit, under that limit the string will be saved
    }

You will then be able to import those strings from the Rasa NLU Trainer using the "import pendings" button.
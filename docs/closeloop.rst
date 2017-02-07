.. _section_closeloop:

Closing the loop: improving your models from feedback
======================================================



When the rasa_nlu server is running, it keeps track of all the predictions it's made and saves these to a log file. 
By default this is called ``rasa_nlu_log.json``
You can fix any incorrect predictions and add them to your training set to improve your parser.
After adding these to your training data, but before retraining your model, it is strongly recommended that you use the
visualizer to spot any errors, see :ref:`Visualizing training data <visualizing-the-training-data>`.

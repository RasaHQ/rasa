.. _section_visualization:

Visualization
==================================


rasa NLU has a very simple visualizer to help you spot mistakes in your training data. 
To use it, run 

.. code-block:: bash

    $ python -m rasa_nlu.visualize path/to/data.json


and visit http://0.0.0.0:8080/ in your web browser. 

This is also helpful for spotting errors in your output, before retraining, see :ref:`section_closeloop`.


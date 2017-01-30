.. _section_visualization:

Visualization
==================================


rasa NLU has a very simple visualizer to help you spot mistakes in your training data. 
To use it, run 

.. code-block:: bash

    $ python -m rasa_nlu.visualize path/to/data.json


and visit http://0.0.0.0:8080/ in your web browser. 

This is also helpful for spotting errors in your output, before retraining, see :ref:`section_closeloop`.

GUI
-----------------------------------

An alternative GUI project to prepare training data is available on `golastmile/rasa-nlu-trainer <https://github.com/golastmile/rasa-nlu-trainer>`_.
To use it, you've to:

1. Install `Node and NPM <https://nodejs.org>`_
2. Run ``npm i -g rasa-nlu-trainer`` for the installation
3. Launch the GUI with ``rasa-nlu-trainer --source path/to/data.json``


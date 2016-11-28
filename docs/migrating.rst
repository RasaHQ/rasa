.. _section_migration:

Migrating an existing app from {wit,LUIS,api}.ai
====================================


Download your data from wit or LUIS. When you export your model from wit you will get a zipped directory. The file you need is `expressions.json`.
If you're exporting from LUIS you get a single json file, and that's the one you need. Create a config file (json format) like this one:

.. code-block:: json

     {
       "path" : "/path/to/models/",
       "data" : "expressions.json",
       "backend" : "mitie",
       "mitie_file":"/path/to/total_word_feature_extractor.dat"
     }

and then pass this file to the training script

.. code-block:: console

    python -m rasa_nlu.train -c config.json


you can also override any of the params in ``config.json`` with command line arguments. Run ``python -m rasa_nlu.train -h`` for details.


Running the server with your newly trained models
-----------------------------------------------

After training you will have a new dir containing your models, e.g. ``/path/to/models/model_XXXXXX``. 
Just pass this path to the ``rasa_nlu.server`` script

.. code-block:: console

    python -m rasa_nlu.server -d '/path/to/models/model_XXXXXX'

or otherwise set the ``server_model_dir`` flag in your config file.


Banana Peels
--------------------------

Just some specific things to watch out for for each of the services you might want to migrate from

wit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

it used to be that wit handled ``intents`` natively. Now they are slightly obscured. To create an ``intent`` in wit you have to create and ``entity`` which spans the entire text.


LUIS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

when you download your model, the entity locations are specified by the index of the tokens. This is pretty fragile because not every tokenizer will behave the same as LUIS's, so your entities may be incorrectly labelled. Run your training once and you'll get a copy of your training data in the ``model_XXXXX`` dir. Do any fixes required and use that to train. 

api.ai
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

api app exports generate multiple files rather than just one. so put them all in a directory (see ``data/restaurantBot``) and pass that path to the trainer. 


.. toctree::
   :maxdepth: 1

.. _section_multi:

Serving Multiple Apps
==================================

Depending on your choice of backend, rasa NLU can use quite a lot of memory. 
So if you are serving multiple models in production, you want to serve these
from the same process & avoid duplicating the memory load. 

If you're using a spaCy backend and your models are in the same language, you can
do this by replacing the ``server_model_dir`` config variable with a json object.

For example, if you have a restaurant bot and a hotel bot, your configuration might look like this:


.. code-block:: json

    {
      "server_model_dir": {
        "hotels" : "./model_XXXXXXX",
        "restaurants" : "./model_YYYYYYY"
      }
    }


You then pass an extra ``model`` parameter in your calls to ``/parse`` to specify which one to use:

.. code-block:: console

    $ curl 'localhost:5000/parse?q=hello&model=hotels'

or 

.. code-block:: console

    $ curl -XPOST localhost:5000/parse -d '{"q":"I am looking for Chinese food", "model": "restaurants"}'

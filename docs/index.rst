
Language Understanding with rasa NLU
====================================


rasa NLU is a tool for intent classification and entity extraction. 
You can think of rasa NLU as a set of high level APIs for building your own language parser using existing NLP and ML libraries.
The intended audience is mainly people developing bots. 
It can be used as a drop-in replacement for `wit <https://wit.ai>`_ , `LUIS <https://luis.ai>`_ , or `api.ai <https://api.ai>`_ but works as a local service rather than a web API. 

The setup process is designed to be as simple as possible. If you're currently using wit, LUIS, or api.ai, you just:
1. download your app data from wit or LUIS and feed it into rasa NLU
2. run rasa NLU on your machine and switch the URL of your wit/LUIS api calls to ``localhost:5000/parse``.

Reasons you might use this over one of the aforementioned services: 

- you don't have to hand over your data to FB/MSFT/GOOG
- you don't have to make a `https` call every time.
- you can tune models to work well on your particular use case.

These points are laid out in more detail in a `blog post <https://medium.com/lastmile-conversations/do-it-yourself-nlp-for-bot-developers-2e2da2817f3d>`_ .

rasa NLU is written in Python, but it you can use it from any language through the HTTP API. 
If your project *is* written in Python you can simply import the relevant classes.

rasa is a set of tools for building more advanced bots, developed by `LASTMILE <https://golastmile.com>`_ . This is the natural language understanding module, and the first component to be open sourced. 


The quickest quickstart in the west
------------------------------------


.. code-block:: console

    $ python setup.py install
    $ python -m rasa_nlu.server -e wit &
    $ curl 'http://localhost:5000/parse?q=hello'
    {"intent":"greet","entities":[]}


There you go! you just parsed some text. Important command line options for ``rasa_nlu.server`` are as follows:

- ``emulate``: which service to emulate, can be ``wit``, ``luis``, ``api`` or just leave blank for default mode. Emulating a service means that the HTTP api has the same request/response format, so if you're currently using (say) LUIS then you only have to swap the URL in your application's API calls.
- ``server_model_dir``: dir where your trained models are saved. If you leave this blank rasa_nlu will just use a naive keyword matcher.

run ``python -m rasa_nlu.server -h`` to see more details.


Contents:

.. toctree::
   :maxdepth: 1

   migrating
   http   
   closeloop
   troubleshoot
   roadmap
   license





Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


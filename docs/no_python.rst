.. _no_python:

But I don't code in python!
===========================


While python is the *lingua franca* of machine learning, we're aware
that most chatbots are built in javascript, and that many enterprises are 
more comfortable building & shipping applications in java, C#, etc. 

We've made every effort to make sure that you can **still** use Rasa Core
even if you don't want to use python. However, do consider that Rasa Core
is a *framework*, and doesn't fit into a REST API as easily as Rasa NLU does. 



Rasa Core with minimal Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can build a chatbot with Rasa Core by:

* defining a domain (:ref:`a yaml file <domain>`)
* writing / collecting stories (:ref:`markdown format <stories>`)
* running a script like ``python train_dm.py`` from the :ref:`command line <tutorial_babi>`

The only part where you need to write python is when you want to define custom actions. 
There's an excellent python library called :ref:`requests <http://docs.python-requests.org/en/master/>`, which makes HTTP programming painless.
If Rasa just needs to interact with your other services over HTTP, your actions will all look 
something like this:


.. code-block:: python

   from rasa_core.actions import Action
   import requests

   class ApiAction(Action):
       def run(self, tracker, dispatcher):
         data = requests.get(url).json
         return [SetSlot("api_result", data)]



Rasa Core with ZERO Python
^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are really constrained to not use any python, you can also use Rasa Core
through a :ref:`HTTP API <section_http>`.

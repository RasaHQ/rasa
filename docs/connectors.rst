.. _connectors:

Connecting to messaging & voice platforms
=========================================

Here's how to connect your conversational AI to the outside world.

Input channels are defined in the ``rasa_core.channels`` module.
Currently, there is an implementation for the command line as well as
connection to facebook.

.. _facebook_connector:

Facebook Messenger Setup
------------------------

Using run script
^^^^^^^^^^^^^^^^
If you want to connect to the facebook using the run script, e.g. using

.. code-block:: bash

  python -m rasa_core.run -d models/dialogue -u models/nlu/current \
      --port 5002 --connector facebook --credentials credentials.yml

you need to supply a ``credentials.yml`` with the following content:

.. literalinclude:: ../examples/moodbot/credentials.yml
   :linenos:


Directly using python
^^^^^^^^^^^^^^^^^^^^^

A ``FacebookInput`` instance provides a flask blueprint for creating
a webserver. This lets you separate the exact endpoints and implementation
from your webserver creation logic.

Code to create a Messenger-compatible webserver looks like this:


.. code-block:: python
    :linenos:

    from rasa_core.channels import HttpInputChannel
    from rasa_core.channels.facebook import FacebookInput
    from rasa_core.agent import Agent
    from rasa_core.interpreter import RegexInterpreter

    # load your trained agent
    agent = Agent.load("dialogue", interpreter=RegexInterpreter())

    input_channel = FacebookInput(
       fb_verify="YOUR_FB_VERIFY",  # you need tell facebook this token, to confirm your URL
       fb_secret="YOUR_FB_SECRET",  # your app secret
       fb_tokens={"YOUR_FB_PAGE_ID": "YOUR_FB_PAGE_TOKEN"},   # page ids + tokens you subscribed to
       debug_mode=True    # enable debug mode for underlying fb library
    )

    agent.handle_channel(HttpInputChannel(5004, "/app", input_channel))

The arguments for the ``HttpInputChannel`` are the port, the url prefix, and the input channel.
The default endpoint for receiving facebook messenger messages is ``/webhook``, so the example
above would listen for messages on ``/app/webhook``. This is the url you should add in the
facebook developer portal.

.. note::

   **How to get the FB credentials:** You need to set up a Facebook app and a page.

      1. To create the app go to: https://developers.facebook.com/ and click on *"Add a new app"*.
      2. go onto the dashboard for the app and under *Products*, click *Add Product* and *add Messenger*. Under the settings for Messenger, scroll down to *Token Generation* and click on the link to create a new page for your app.
      3. Go to the facebook page you just created and copy the *page id* from the url.

   For more detailed steps, visit the
   `messenger docs <https://developers.facebook.com/docs/graph-api/webhooks>`_.



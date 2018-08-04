:desc: How to Customize Your Rasa Core Configuration

.. _section_server:

Server Configuration
====================

.. note::

    Before you can use the server, you need to define a domain, create training
    data, and train a model. You can then use the trained model for remote code
    execution! See the :ref:`quickstart` for an introduction.



Overview
--------
The general idea is to run the actions within your code (arbitrary language),
instead of python. To do this, Rasa Core will start up a web server where you
need to pass the user messages to. Rasa Core on the other side will tell you
which actions you need to run. After running these actions, you need to notify
the framework that you executed them and tell the model about any update of the
internal dialogue state for that user. All of these interactions are done using
a HTTP REST interface.

You can also use a single, simpler endpoint called ``/respond``, which just returns
all of the messages your bot should send back to the user. In general, this only
works if all of your actions are simple utterances (messages sent to the user).
It can make use of custom actions, but then these *have* to be implemented in 
python and executed on the machine that runs the server. 

To activate the remote mode, include

.. code-block:: yaml

    action_factory: remote

within your ``domain.yml`` (you can find an example in
``examples/remote/concert_domain_remote.yml``).

.. note::

    If started as a HTTP server, Rasa Core will not handle output or input
    channels for you. That means you need to retrieve messages from the input
    channel (e.g. facebook messenger) and send messages to the user on your end.

    Hence, you also do not need to define any utterances in your domain yaml.
    Just list all the actions you need.

Running the server
------------------
You can run a simple http server that handles requests using your
models with

.. code-block:: bash

    $ python -m rasa_core.server -d examples/babi/models/policy/current -u examples/babi/models/nlu/current_py2 -o out.log

The different parameters are:

- ``-d``, path to the Rasa Core model.
- ``-u``, path to the Rasa NLU model.
- ``-o``, path to the log file.

.. _http_start_conversation:

Starting a conversation
-----------------------
To start a conversation, send a ``POST`` request to the ``/conversation/<sender_id>/parse`` endpoint.
``<sender_id>`` is the conversation id (e.g. ``default`` if you just have one
user, or the facebook user id or any other identifier).

.. code-block:: bash

    $ curl -XPOST localhost:5005/conversations/default/parse -d '{"query":"hello there"}'

The server will respond with the next action you should take:

.. code-block:: javascript

    {
      "next_action": "utter_ask_howcanhelp",
      "tracker": {
        "slots": {
          "info": null,
          "cuisine": null,
          "people": null,
          "matches": null,
          "price": null,
          "location": null
        },
        "sender_id": "default",
        "latest_message": {
          ...
        }
      }
    }

You now need to execute the action ``utter_ask_howcanhelp`` on your end. This
might include sending a message to the output channel (e.g. back to facebook).

After you finished running the mentioned action, you need to notify Rasa Core
about that:

.. code-block:: bash

    $ curl -XPOST http://localhost:5005/conversations/default/continue -d \
        '{"executed_action": "utter_ask_howcanhelp", "events": []}'

Here the API should respond with:

.. code-block:: javascript

    {
      "next_action":"action_listen",
      "tracker": {
        "slots": {
          "info": null,
          "cuisine": null,
          "people": null,
          "matches": null,
          "price": null,
          "location": null
        },
        "sender_id": "default",
        "latest_message": {
          ...
        }
      }
    }

This response tells you to wait for the next user message. You should not call
the continue endpoint after you received a response containing ``action_listen``
as the next action. Instead, wait for the next user message and call
``/conversations/default/parse`` again followed by subsequent
calls to ``/conversations/default/continue`` until you get ``action_listen``
again.

Events
------
Events allow you to modify the internal state of the dialogue. This information
will be used to predict the next action. E.g. you can set slots (to store
information about the user) or restart the conversation.

You can return multiple events as part of your query, e.g.:

.. code-block:: bash

    $ curl -XPOST http://localhost:5005/conversations/default/continue -d \
        '{"executed_action": "search_restaurants", "events": [{"event": "slot", "name": "cuisine", "value": "mexican"}, {"event": "slot", "name": "people", "value": 5}]}'

Here is a list of all available events you can append to the ``events`` array in
your call to ``/conversation/<sender_id>/continue``.

Set a slot
::::::::::

:name: ``slot``
:Examples: ``"events": [{"event": "slot", "name": "cuisine", "value": "mexican"}]``
:Description:
    Will set the value of the slot to the passed one. The value you set should
    be reasonable given the :ref:`slots type <slot_types>`.

Restart
:::::::

:name: ``restart``
:Examples: ``"events": [{"event": "restart"}]``
:Description:
    Restarts the conversation and resets all slots and past actions.

Reset Slots
:::::::::::

:name: ``reset_slots``
:Examples: ``"events": [{"event": "reset_slots"}]``
:Description:
    Resets all slots to their initial value.



Security Considerations
-----------------------

We recommend to not expose the Rasa Core server to the outside world but
rather connect to it from your backend over a private connection (e.g.
between docker containers).

Nevertheless, there is built in token authentication. If you specify a token
when starting the server, that token needs to be passed with every request:

.. code-block:: bash

    $ python -m rasa_core.server --auth_token thisismysecret -d examples/babi/models/policy/current -u examples/babi/models/nlu/current_py2 -o out.log

Your requests should pass the token, in our case ``thisismysecret``,
as a parameter:

.. code-block:: bash

    $ curl -XPOST localhost:5005/conversations/default/parse?token=thisismysecret -d '{"query":"hello there"}'

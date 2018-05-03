.. _section_http:

HTTP server
===========

.. note::

    Before you can use the server, you need to define a domain, create training
    data, and train a model. You can then use the trained model for remote code
    execution! See :ref:`tutorial_basics` for an introduction.


The HTTP api exists to make it easy for non-python projects to use Rasa Core.

Overview
--------
The general idea is to run the actions within your code (arbitrary language),
instead of python. To do this, Rasa Core will start up a web server where you
need to pass the user messages to. Rasa Core on the other side will tell you
which actions you need to run. After running these actions, you need to notify
the framework that you executed them and tell the model about any update of the
internal dialogue state for that user. All of these interactions are done using
a HTTP REST interface.

You can also use a single, simpler endpoint called `/respond`, which just returns
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

- ``-d``, which is the path to the Rasa Core model.
- ``-u``, which is the path to the Rasa NLU model.
- ``-o``, which is the path to the log file.

.. _http_start_conversation:

Starting a conversation
-----------------------
You need to do a ``POST`` to the ``/conversation/<sender_id>/parse`` endpoint.
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


Endpoints
---------

.. http:post:: /conversations/(str:sender_id)/parse

   Notify the dialogue engine that the user posted a new message. You must
   ``POST`` data in this format ``'{"query":"<your text to parse>"}'``,
   you can do this with

   **Example request**:

   .. sourcecode:: bash

      curl -XPOST localhost:5005/conversations/default/parse -d \
        '{"query":"hello there"}' | python -mjson.tool

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Vary: Accept
      Content-Type: text/javascript

      {
          "next_action": "utter_ask_howcanhelp",
          "tracker": {
              "latest_message": {
                  ...
              },
              "sender_id": "default",
              "slots": {
                  "cuisine": null,
                  "info": null,
                  "location": null,
                  "matches": null,
                  "people": null,
                  "price": null
              }
          }
      }

   :statuscode 200: no error


.. http:post:: /conversations/(str:sender_id)/continue

   Continue the prediction loop for the conversation with id `user_id`. Should
   be called until the endpoint returns ``action_listen`` as the next action.
   Between the calls to this endpoint, your code should execute the mentioned
   next action. If you receive ``action_listen`` as the next action, you should
   wait for the next user input.

   **Example request**:

   .. sourcecode:: bash

      curl -XPOST http://localhost:5005/conversations/default/continue -d \
        '{"executed_action": "utter_ask_howcanhelp", "events": []}' | python -mjson.tool

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Vary: Accept
      Content-Type: text/javascript

      {
          "next_action": "utter_ask_cuisine",
          "tracker": {
              "latest_message": {
                  ...
              },
              "sender_id": "default",
              "slots": {
                  "cuisine": null,
                  "info": null,
                  "location": null,
                  "matches": null,
                  "people": null,
                  "price": null
              }
          }
      }

   :statuscode 200: no error

.. http:post:: /conversations/(str:sender_id)/respond

   Notify the dialogue engine that the user posted a new message, and get
   a list of response messages the bot should send back.
   You must ``POST`` data in this format ``'{"query":"<your text to parse>"}'``,
   you can do this with

   **Example request**:

   .. sourcecode:: bash

      curl -XPOST localhost:5005/conversations/default/respond -d \
        '{"query":"hello there"}' | python -mjson.tool

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Vary: Accept
      Content-Type: text/javascript

      [
        {
          "text": "Hi! welcome to the pizzabot",
          "data": {"title": "order pizza", "payload": "/start_order"},
        }
      ]

   :statuscode 200: no error


.. http:get:: /conversations/(str:sender_id)/tracker

   Retrieves the current tracker state for the conversation with ``sender_id``.
   This includes the set slots as well as the latest message and all previous
   events.

   **Example request**:

   .. sourcecode:: bash

      curl http://localhost:5005/conversations/default/tracker | python -mjson.tool

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Vary: Accept
      Content-Type: text/javascript

      {
          "events": [
              {
                  "event": "action",
                  "name": "action_listen"
              },
              {
                  "event": "user",
                  "parse_data": {
                      "entities": [],
                      "intent": {
                          "confidence": 0.7561643619088745,
                          "name": "affirm"
                      },
                      "intent_ranking": [
                          ...
                      ],
                      "text": "hello there"
                  },
                  "text": "hello there"
              }
          ],
          "latest_message": {
              "entities": [],
              "intent": {
                  "confidence": 0.7561643619088745,
                  "name": "affirm"
              },
              "intent_ranking": [
                  ...
              ],
              "text": "hello there"
          },
          "paused": false,
          "sender_id": "default",
          "slots": {
              "cuisine": null,
              "info": null,
              "location": null,
              "matches": null,
              "people": null,
              "price": null
          }
      }

   :statuscode 200: no error

.. http:put:: /conversations/(str:sender_id)/tracker

   Replace the tracker state using events. Any existing tracker for
   ``sender_id`` will be discarded. A new tracker will be created and the
   passed events will be applied to create a new state.

   The format of the passed events is the same as for the ``/continue``
   endpoint.

   **Example request**:

   .. sourcecode:: bash

      curl -XPUT http://localhost:5005/conversations/default/tracker -d \
        '[{"event": "slot", "name": "cuisine", "value": "mexican"},{"event": "action", "name": "action_listen"}]' | python -mjson.tool

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Vary: Accept
      Content-Type: text/javascript

      {
          "events": [
              {
                  "event": "slot",
                  "name": "cuisine",
                  "value": "mexican"
              },
              {
                  "event": "action",
                  "name": "action_listen"
              }
          ],
          "latest_message": {
              "entities": [],
              "intent": {},
              "text": null
          },
          "paused": false,
          "sender_id": "default",
          "slots": {
              "cuisine": "mexican",
              "info": null,
              "location": null,
              "matches": null,
              "people": null,
              "price": null
          }
      }

   :statuscode 200: no error

.. http:post:: /conversations/(str:sender_id)/tracker/events

   Append the tracker state of the conversation with events. Any existing
   events will be kept and the new events will be appended, updating the
   existing state.

   The format of the passed events is the same as for the ``/continue``
   endpoint.

   **Example request**:

   .. sourcecode:: bash

      curl -XPOST http://localhost:5005/conversations/default/tracker/events -d \
        '[{"event": "slot", "name": "cuisine", "value": "mexican"},{"event": "action", "name": "action_listen"}]' | python -mjson.tool

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Vary: Accept
      Content-Type: text/javascript

      {
          "events": null,
          "latest_message": {
              "entities": [],
              "intent": {
                  "confidence": 0.7561643619088745,
                  "name": "affirm"
              },
              "intent_ranking": [
                  ...
              ],
              "text": "hello there"
          },
          "paused": false,
          "sender_id": "default",
          "slots": {
              "cuisine": "mexican",
              "info": null,
              "location": null,
              "matches": null,
              "people": null,
              "price": null
          }
      }

   :statuscode 200: no error


.. http:get:: /conversations

   List the sender ids of all the running conversations.

   **Example request**:

   .. sourcecode:: bash

      curl http://localhost:5005/conversations | python -mjson.tool

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Vary: Accept
      Content-Type: text/javascript

      ["default"]

   :statuscode 200: no error

.. http:get:: /version

   Version of Rasa Core that is currently running.

   **Example request**:

   .. sourcecode:: bash

      curl http://localhost:5005/version | python -mjson.tool

   **Example response**:

   .. sourcecode:: http

      HTTP/1.1 200 OK
      Vary: Accept
      Content-Type: text/javascript

      {
          "version" : "0.7.0"
      }

   :statuscode 200: no error


Security Considerations
-----------------------

We recommend to not expose the Rasa Core server to the outside world but
rather connect to it from your backend over a private connection (e.g.
between docker containers).

Nevertheless, there is build in token authentication. If you specify a token
when starting the server, that token needs to be passed with every request:

.. code-block:: bash

    $ python -m rasa_core.server --auth_token thisismysecret -d examples/babi/models/policy/current -u examples/babi/models/nlu/current_py2 -o out.log

Your requests should pass the token, in our case ``thisismysecret``,
as a parameter:

.. code-block:: bash

    $ curl -XPOST localhost:5005/conversations/default/parse?token=thisismysecret -d '{"query":"hello there"}'

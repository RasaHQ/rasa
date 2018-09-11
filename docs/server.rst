:desc: The Rasa Core REST API

.. _section_http:

HTTP API
========

.. note::

    Before you can use the server, you need to define a domain, create training
    data, and train a model. You can then use the trained model!
    See :ref:`quickstart` for an introduction.


The HTTP api exists to make it easy for python and non-python
projects to interact with Rasa Core. The API allows you to modify
the trackers (e.g. push or remote events).

.. note::

    If you are looking for documentation on how to run custom actions -
    head over to :ref:`customactions`.

.. contents::


Running the HTTP server
-----------------------

You can run a simple http server that handles requests using your
models with:

.. code-block:: bash

    $ python -m rasa_core.run \
        --enable_api
        -d models/dialogue \
        -u models/nlu/current \
        -o out.log

The different parameters are:

- ``--enable_api``, enables this additional API
- ``-d``, which is the path to the Rasa Core model.
- ``-u``, which is the path to the Rasa NLU model.
- ``-o``, which is the path to the log file.

.. note::

  If you are using custom actions - make sure to pass in the endpoint
  configuration for your action server as well using
  ``--endpoints endpoints.yml``.

Fetching models from a server
-----------------------------
You can also configure the http server to fetch models from another URL:

.. code-block:: bash

    $ python -m rasa_core.run \
        --enable_api \
        -d models/dialogue \
        -u models/nlu/current \
        --endpoints my_endpoints.yaml \
        -o out.log

The model server is specified in an ``EndpointConfig`` file
(``my_endpoints.yaml``), where you specify the server URL Rasa Core
regularly queries for zipped Rasa Core models:

.. code-block:: yaml

    model:
      url: http://my-server.com/models/default_core@latest

.. note::

    Your model server must provide zipped Rasa Core models, and have
    ``{"ETag": <model_hash_string>}`` as one of its headers. Core will
    only download a new model if this model hash changed.

Rasa Core sends requests to your model server with an ``If-None-Match``
header that contains the current model hash. If your model server can
provide a model with a different hash from the one you sent, it should send it
in as a zip file with an ``ETag`` header containing the new hash. If not, Rasa
Core expects an empty response with a ``204`` status code.

An example request Rasa Core might make to your model server looks like this:

.. code-block:: bash

      $ curl --header "If-None-Match: d41d8cd98f00b204e9800998ecf8427e" http://my-server.com/models/default_core@latest

Events
------
Events allow you to modify the internal state of the dialogue. This information
will be used to predict the next action. E.g. you can set slots (to store
information about the user) or restart the conversation.

You can return multiple events as part of your query, e.g.:

.. code-block:: bash

    $ curl -XPOST http://localhost:5005/conversations/default/tracker/events -d \
        '{"event": "slot", "name": "cuisine", "value": "mexican"}'


You can find a list of all events and their json representation
at :ref:`events`. You need to send these json formats to the endpoint to
log the event.


Security Considerations
-----------------------

We recommend to not expose the Rasa Core server to the outside world but
rather connect to it from your backend over a private connection (e.g.
between docker containers).

Nevertheless, there is built in token authentication. If you specify a token
when starting the server, that token needs to be passed with every request:

.. code-block:: bash

    $ python -m rasa_core.run \
        --enable_api \
        --auth_token thisismysecret \
        -d models/dialogue \
        -u models/nlu/current \
        -o out.log

Your requests should pass the token, in our case ``thisismysecret``,
as a parameter:

.. code-block:: bash

    $ curl -XGET localhost:5005/conversations/default/tracker?token=thisismysecret


Endpoints
---------

.. http:post:: /conversations/(str:sender_id)/respond

   .. note::

      This endpoint will be removed in the future. Rather consider using
      the ``RestInput`` channel. When added to core, it will provide you
      an endpoint at ``/webhooks/rest/webhook`` that returns the same
      output as this endpoint. The only difference is, that you need to send
      the message as ``{"message": }"<your text to parse>"}``.

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


.. include:: feedback.inc 

.. raw:: html
   :file: livechat.html

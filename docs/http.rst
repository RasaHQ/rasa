.. _section_http:

HTTP API
========


.. http:post:: /conversations/(str:sender_id)/parse
   :synopsis: Returns posts by the specified tag for the user

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


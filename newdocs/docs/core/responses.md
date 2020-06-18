# Responses

If you want your assistant to respond to user messages, you need to manage
these responses. In the training data for your bot,
your stories, you specify the actions your bot
should execute. These actions
can use responses to send messages back to the user.

There are three ways to manage these responses:


1. Responses are normally stored in your domain file, see here


2. Retrieval action responses are part of the training data, see here


3. You can also create a custom NLG service to generate responses, see here

## Including the responses in the domain

The default format is to include the responses in your domain file.
This file then contains references to all your custom actions,
available entities, slots and intents.

```
# all hashtags are comments :)
intents:
 - greet
 - default
 - goodbye
 - affirm
 - thank_you
 - change_bank_details
 - simple
 - hello
 - why
 - next_intent

entities:
 - name

slots:
  name:
    type: text

responses:
  utter_greet:
    - text: "hey there {name}!"  # {name} will be filled by slot (same name) or by custom action
  utter_channel:
    - text: "this is a default channel"
    - text: "you're talking to me on slack!"  # if you define channel-specific utterances, the bot will pick
      channel: "slack"                        # from those when talking on that specific channel
  utter_goodbye:
    - text: "goodbye 😢"   # multiple responses - bot will randomly pick one of them
    - text: "bye bye 😢"
  utter_default:   # utterance sent by action_default_fallback
    - text: "sorry, I didn't get that, can you rephrase it?"
```

In this example domain file, the section `responses` contains the
responses the assistant uses to send messages to the user.

**NOTE**: If you want to change the text, or any other part of the bots response,
you need to retrain the assistant before these changes will be picked up.

**NOTE**: Responses that are used in a story should be listed in the `stories`
section of the domain.yml file. In this example, the `utter_channel`
response is not used in a story so it is not listed in that section.

More details about the format of these responses can be found in the
documentation about the domain file format: Responses.

## Creating your own NLG service for bot responses

Retraining the bot just to change the text copy can be suboptimal for
some workflows. That’s why Core also allows you to outsource the
response generation and separate it from the dialogue learning.

The assistant will still learn to predict actions and to react to user input
based on past dialogues, but the responses it sends back to the user
are generated outside of Rasa Core.

If the assistant wants to send a message to the user, it will call an
external HTTP server with a `POST` request. To configure this endpoint,
you need to create an `endpoints.yml` and pass it either to the `run`
or `server` script. The content of the `endpoints.yml` should be

```
nlg:
  url: http://localhost:5055/nlg    # url of the nlg endpoint
  # you can also specify additional parameters, if you need them:
  # headers:
  #   my-custom-header: value
  # token: "my_authentication_token"    # will be passed as a get parameter
  # basic_auth:
  #   username: user
  #   password: pass
# example of redis external tracker store config
tracker_store:
  type: redis
  url: localhost
  port: 6379
  db: 0
  password: password
  record_exp: 30000
# example of mongoDB external tracker store config
#tracker_store:
  #type: mongod
  #url: mongodb://localhost:27017
  #db: rasa
  #user: username
  #password: password
```

Then pass the `enable-api` flag to the `rasa run` command when starting
the server:

```
$ rasa run \
   --enable-api \
   -m examples/babi/models \
   --log-file out.log \
   --endpoints endpoints.yml
```

The body of the `POST` request sent to the endpoint will look
like this:

```
{
  "tracker": {
    "latest_message": {
      "text": "/greet",
      "intent_ranking": [
        {
          "confidence": 1.0,
          "name": "greet"
        }
      ],
      "intent": {
        "confidence": 1.0,
        "name": "greet"
      },
      "entities": []
    },
    "sender_id": "22ae96a6-85cd-11e8-b1c3-f40f241f6547",
    "paused": false,
    "latest_event_time": 1531397673.293572,
    "slots": {
      "name": null
    },
    "events": [
      {
        "timestamp": 1531397673.291998,
        "event": "action",
        "name": "action_listen"
      },
      {
        "timestamp": 1531397673.293572,
        "parse_data": {
          "text": "/greet",
          "intent_ranking": [
            {
              "confidence": 1.0,
              "name": "greet"
            }
          ],
          "intent": {
            "confidence": 1.0,
            "name": "greet"
          },
          "entities": []
        },
        "event": "user",
        "text": "/greet"
      }
    ]
  },
  "arguments": {},
  "template": "utter_greet",
  "channel": {
    "name": "collector"
  }
}
```

The endpoint then needs to respond with the generated response:

```
{
    "text": "hey there",
    "buttons": [],
    "image": null,
    "elements": [],
    "attachments": []
}
```

Rasa will then use this response and sent it back to the user.

## Proactively Reaching Out to the User with External Events

You may want to proactively reach out to the user,
for example to display the output of a long running background operation
or notify the user of an external event.
To learn more, check out [reminderbot](https://github.com/RasaHQ/rasa/blob/master/examples/reminderbot/README.md) in
the Rasa examples directory or look into Reminders and External Events.

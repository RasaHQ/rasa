Prototype an Assistant
======================

.. edit-link::

Get started with Rasa Open source and learn how to create an assistant from scratch!

This page explains the basics of building an assistant with Rasa and shows the structure of a Rasa project.
You can test it out right here without installing anything. You can also install Rasa and follow along in your command line.


1. Define a basic user goal
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To prototype an assistant, start with defining a single user goal that your assistant will handle.

You can create a prototype using the user goal we have chosen “subscribing to a newsletter”, or you can come up
with your own. If you choose your own user goal, you can create a prototype that handles your goal by following
the instructions to edit each step accordingly as you go through the tutorial.

2. Create some NLU data
~~~~~~~~~~~~~~~~~~~~~~~

NLU data provides examples of what users might say to your assistant and what they mean by it.
Intent refers to what the user means with a specific message. Your assistant can only learn to recognize intents
for which it has seen examples, so we need to provide some data.

Add examples for your user-goal specific intents in the format shown below.
You can delete the ``subscribe`` and ``inform`` intents if you’re not using them;
you can also add or change examples for any of the other intents.

.. code-editor::
    :language: markdown
    :id: nlu
    :height: 550
    :tracking_endpoint: https://trainer-service.prototyping.rasa.com/startPrototyping

    ## intent: greet
    - Hi
    - Hey!
    - Hallo
    - Good day
    - Good morning

    ## intent: affirm
    - Yes
    - Yes please
    - Yes thank you
    - Yup - Yeah - Sure

    ## intent: deny
    - No
    - Nope
    - No thanks
    - Cancel
    - No please don’t

    ## intent: subscribe
    - I want to get the newsletter
    - Can you send me the newsletter?
    - Can you sign me up for the newsletter?

    ## intent: inform
    - My email is example@example.com
    - random@example.com
    - Please send it to anything@example.com
    - Email is something@example.com


3. Define a simple form
~~~~~~~~~~~~~~~~~~~~~~~

For most user goals, the bot will need to collect some information from the user to fulfill their request.
To do so, we define a form.

You can change the name of the form to reflect your user goal. Add to or replace the ``email`` item in the list
below to reflect the information the bot needs to collect for your user goal.
Leave the ``type`` field the same for any items you add.

.. code-editor::
    :language: yaml
    :id: form
    :height: 150
    :tracking_endpoint: https://trainer-service.prototyping.rasa.com/startPrototyping

    - newsletter_form:
        email:
        - type: from_text


4. Write some stories
~~~~~~~~~~~~~~~~~~~~~~~

Stories are example conversations of how your assistant should handle a user's intent in context.
The first stories you write should follow the happy path for your user goal.

A story contains one or more blocks of (user) intent and (bot) actions or responses.
The form you defined above is one kind of action; responses are just bot messages.
Give intuitive names to your responses starting with ``utter_`` for now; you’ll define what they return later.

Using the general template of the story we have shown you below, try to write a story or two that serve the
user goal you have chosen. If you’re using the user goal of subscribing to a newsletter, try adding a story
to account for the user saying "no" when asked if they want to subscribe to the newsletter.

.. code-editor::
    :language: markdown
    :id: stories
    :height: 250
    :tracking_endpoint: https://trainer-service.prototyping.rasa.com/startPrototyping

    ## happy path
    * greet
        - utter_greet
    * request_restaurant
        - restaurant_form
        - form{"name": "restaurant_form"}
        - form{"name": null}
        - utter_slots_values
    * thankyou
        - utter_noworries


5. Edit responses
~~~~~~~~~~~~~~~~~
To give your bot messages to respond to the user with, you need to define responses.
You can specify one or more text options for each response. If there are multiple, one of the options
will be chosen at random whenever that response is predicted.

You can add or change text for any of the responses below. If you’re using your own user goal,
replace the last three responses with the response you used in your stories above.

.. code-editor::
    :language: yaml
    :id: responses
    :height: 250
    :tracking_endpoint: https://trainer-service.prototyping.rasa.com/startPrototyping

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

.. note::

    Note: For this prototype, we have only defined responses, meaning the only thing the assistant does is
    respond with a predefined message. Custom actions, however, can be defined to do whatever you’d like.
    For example, for the user goal of subscribing to a newsletter, you could create a custom action that
    adds the user’s email to a database. You can see an example of this in Sara’s action code.

6. Train and run
~~~~~~~~~~~~~~~~

Rasa has a command line interface that allows you to train and run your bot from a terminal.
To train your bot on the NLU data, stories and responses you’ve just defined, run ``rasa train`` using
the button below:

.. train-button::
    :endpoint: https://trainer-service.prototyping.rasa.com/trainings
    :method: POST

7. What's next?
~~~~~~~~~~~~~~~

You can download this project and build on it to create a more advanced assistant.
In your downloaded project, you’ll notice several files that were configured for you that you didn’t edit on this page.
To learn more about configs, domains and actions, refer to the advanced tutorials.

.. download-button::

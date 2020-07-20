:desc: How to build simple FAQ and contextual assistants

.. _building-assistants:

Tutorial: Building Assistants
=============================

.. edit-link::

After following the basics of setting up an assistant in the `Rasa Tutorial <https://rasa.com/docs/rasa/user-guide/rasa-tutorial/>`_, we'll
now walk through building a basic FAQ chatbot and then build a bot that can handle
contextual conversations.

.. contents::
   :local:

.. _build-faq-assistant:

Building a simple FAQ assistant
-------------------------------

FAQ assistants are the simplest assistants to build and a good place to get started.
These assistants allow the user to ask a simple question and get a response. We’re going to
build a basic FAQ assistant using features of Rasa designed specifically for this type of assistant.

In this section we’re going to cover the following topics:

    - Responding to simple intents with the MemoizationPolicy
    - Handling FAQs using the ResponseSelector

We’re going to use content from `Sara <https://github.com/RasaHQ/rasa-demo>`_, the Rasa
assistant that, amongst other things, helps the user get started with the Rasa products.
You should first install Rasa using the `Step-by-step Installation Guide <https://rasa.com/docs/rasa/user-guide/installation/#step-by-step-installation-guide>`_
and then follow the `Rasa Tutorial <https://rasa.com/docs/rasa/user-guide/rasa-tutorial/>`_
to make sure you know the basics.

To prepare for this tutorial, we're going to create a new directory and start a
new Rasa project.

.. code-block:: bash

    mkdir rasa-assistant
    rasa init


Let's remove the default content from this bot, so that the ``nlu.md``, ``stories.md``
and ``domain.yml`` are empty.

Memoization Policy
^^^^^^^^^^^^^^^^^^

The MemoizationPolicy remembers examples from training stories for up to a ``max_history``
of turns. The number of "turns" includes messages the user sent, and actions the
assistant performed. For the purpose of a simple, context-less FAQ bot, we only need
to pay attention to the last message the user sent, and therefore we’ll set that to ``1``.

You can do this by editing your ``config.yml`` file as follows:

.. code-block:: yaml

  policies:
  - name: MemoizationPolicy
    max_history: 1
  - name: MappingPolicy

.. note::
   The MappingPolicy is there because it handles the logic of the ``/restart`` intent,
   which allows you to clear the conversation history and start fresh.

Now that we’ve defined our policies, we can add some stories for the ``goodbye``, ``thank`` and ``greet``
intents to the ``stories.md`` file:

.. code-block:: md

   ## greet
   * greet
     - utter_greet

   ## thank
   * thank
     - utter_noworries

   ## goodbye
   * bye
     - utter_bye

We’ll also need to add the intents, actions and templates to our ``domain.yml`` file in the following sections:

.. code-block:: md

   intents:
     - greet
     - bye
     - thank

   actions:
     - utter_greet
     - utter_noworries
     - utter_bye

   templates:
     utter_noworries:
       - text: No worries!
     utter_greet:
       - text: Hi
     utter_bye:
       - text: Bye!

Finally, we’ll copy over some NLU data from Sara into our ``nlu.md``
(more can be found `here <https://github.com/RasaHQ/rasa-demo/blob/master/data/nlu/nlu.md>`__):

.. code-block:: md

   ## intent:greet
   - Hi
   - Hey
   - Hi bot
   - Hey bot
   - Hello
   - Good morning
   - hi again
   - hi folks

   ## intent:bye
   - goodbye
   - goodnight
   - good bye
   - good night
   - see ya
   - toodle-oo
   - bye bye
   - gotta go
   - farewell

   ## intent:thank
   - Thanks
   - Thank you
   - Thank you so much
   - Thanks bot
   - Thanks for that
   - cheers

You can now train a first model and test the bot, by running the following commands:

.. code-block:: bash

   rasa train
   rasa shell

This bot should now be able to reply to the intents we defined consistently, and in any order.

While it's good to test the bot interactively, we should also add end to end test cases that
can later be included as part of our CI/CD system. `End to end stories <https://rasa.com/docs/rasa/user-guide/evaluating-models/#end-to-end-evaluation>`_
include NLU data, so that both components of Rasa can be tested.  Create a file called
``test_stories.md`` in the root directory with some test cases:

.. code-block:: md

   ## greet + goodbye
   * greet: Hi!
     - utter_greet
   * bye: Bye
     - utter_bye

   ## greet + thanks
   * greet: Hello there
     - utter_greet
   * thank: thanks a bunch
     - utter_noworries

   ## greet + thanks + goodbye
   * greet: Hey
     - utter_greet
   * thank: thank you
     - utter_noworries
   * bye: bye bye
     - utter_bye

To test our model against the test file, run the command:

.. code-block:: bash

   rasa test --e2e --stories test_stories.md

The test command will produce a directory named ``results``. It should contain a file
called ``failed_stories.md``, where any test cases that failed will be printed. It will
also specify whether it was an NLU or Core prediction that went wrong.  As part of a
CI/CD pipeline, the test option ``--fail-on-prediction-errors`` can be used to throw
an exception that and stop the pipeline.

Response Selectors
^^^^^^^^^^^^^^^^^^

The :ref:`response-selector` NLU component is designed to make it easier to handle dialogue
elements like :ref:`small-talk` and FAQ messages in a simple manner. By using the ResponseSelector,
you only need one story to handle all FAQs, instead of adding new stories every time you
want to increase your bot's scope.

People often ask Sara different questions surrounding the Rasa products, so let’s
start with three intents: ``ask_channels``, ``ask_languages``, and ``ask_rasax``.
We’re going to copy over some NLU data from the `Sara training data <https://github.com/RasaHQ/rasa-demo/blob/master/data/nlu/nlu.md>`_
into our ``nlu.md``. It’s important that these intents have an ``faq/`` prefix, so they’re
recognised as the faq intent by the ResponseSelector:

.. code-block:: md

   ## intent: faq/ask_channels
   - What channels of communication does rasa support?
   - what channels do you support?
   - what chat channels does rasa uses
   - channels supported by Rasa
   - which messaging channels does rasa support?

   ## intent: faq/ask_languages
   - what language does rasa support?
   - which language do you support?
   - which languages supports rasa
   - can I use rasa also for another laguage?
   - languages supported

   ## intent: faq/ask_rasax
   - I want information about rasa x
   - i want to learn more about Rasa X
   - what is rasa x?
   - Can you tell me about rasa x?
   - Tell me about rasa x
   - tell me what is rasa x

Next, we’ll need to define the responses associated with these FAQs in a new file called ``responses.md`` in the ``data/`` directory:

.. code-block:: md

   ## ask channels
   * faq/ask_channels
     - We have a comprehensive list of [supported connectors](https://rasa.com/docs/core/connectors/), but if
       you don't see the one you're looking for, you can always create a custom connector by following
       [this guide](https://rasa.com/docs/rasa/user-guide/connectors/custom-connectors/).

   ## ask languages
   * faq/ask_languages
     - You can use Rasa to build assistants in any language you want!

   ## ask rasa x
   * faq/ask_rasax
    - Rasa X is a tool to learn from real conversations and improve your assistant. Read more [here](https://rasa.com/docs/rasa-x/)

To use the Response Selector we need to add it to the end of the expanded `supervised_embeddings <https://rasa.com/docs/rasa/nlu/choosing-a-pipeline/#section-supervised-embeddings-pipeline>`_
NLU pipeline in our ``config.yml``:

.. code-block:: yaml

   pipeline:
   - name: "WhitespaceTokenizer"
   - name: "RegexFeaturizer"
   - name: "CRFEntityExtractor"
   - name: "EntitySynonymMapper"
   - name: "CountVectorsFeaturizer"
   - name: "CountVectorsFeaturizer"
     analyzer: "char_wb"
     min_ngram: 1
     max_ngram: 4
   - name: "EmbeddingIntentClassifier"
   - name: "ResponseSelector"

Now that we’ve defined the NLU side, we need to make Core aware of these changes. Open your ``domain.yml`` file and add the ``faq`` intent:

.. code-block:: yaml

   intents:
     - greet
     - bye
     - thank
     - faq

We’ll also need to add a `retrieval action <https://rasa.com/docs/rasa/core/retrieval-actions/>`_,
which takes care of sending the response predicted from the ResponseSelector back to the user,
to the list of actions. These actions always have to start with the ``respond_`` prefix:

.. code-block:: yaml

   actions:
     - utter_greet
     - utter_noworries
     - utter_bye
     - respond_faq

Next we’ll write a story so that Core knows which action to predict:

.. code-block:: md

   ## Some question from FAQ
   * faq
       - respond_faq

This prediction is handled by the MemoizationPolicy, as we described earlier.

After all of the changes are done, train a new model and test the modified FAQs:

.. code-block:: bash

   rasa train
   rasa shell

At this stage it makes sense to add a few test cases to your ``test_stories.md`` file again:

.. code-block:: md

   ## ask channels
   * faq: What messaging channels does Rasa support?
     - respond_faq

   ## ask languages
   * faq: Which languages can I build assistants in?
     - respond_faq

   ## ask rasa x
   * faq: What’s Rasa X?
     - respond_faq

You can read more in this `blog post <https://blog.rasa.com/response-retrieval-models/>`_ and the
`Retrieval Actions <https://rasa.com/docs/rasa/core/retrieval-actions/>`_ page.

Using the features we described in this tutorial, you can easily build a context-less assistant.
When you’re ready to enhance your assistant with context, check out :ref:`build-contextual-assistant`.

.. _build-contextual-assistant:

Building contextual assistants
------------------------------

Whether you’ve just created an FAQ bot or are starting from scratch, the next step is to expand
your bot to handle contextual conversations.

In this tutorial we’re going to cover a variety of topics:

    - Handling business logic
    - Handling unexpected user input
    - Failing gracefully
    - More complex contextual conversations

Please make sure you’ve got all the data from the :ref:`build-faq-assistant` section before starting this part.
You will need to make some adjustments to your configuration file, since we now need to pay attention to context:

.. code-block:: yaml

   policies:
   - name: MemoizationPolicy
   - name: MappingPolicy

We removed the ``max_history: 1`` configuration. The default is ``5``,
meaning Core will pay attention to the past 5 turns when making a prediction
(see explanation of `max history <https://rasa.com/docs/rasa/core/policies/#max-history>`_).

Business logic
^^^^^^^^^^^^^^

A lot of conversational assistants have user goals that involve collecting a bunch of information
from the user before being able to do something for them. This is called slot filling. For
example, in the banking industry you may have a user goal of transferring money, where you
need to collect information about which account to transfer from, whom to transfer to and the
amount to transfer. This type of behaviour can and should be handled in a rule based way, as it
is clear how this information should be collected.

For this type of use case, we can use Forms and our FormPolicy. The `FormPolicy <https://rasa.com/docs/rasa/core/policies/#form-policy>`_
works by predicting the form as the next action until all information is gathered from the user.

As an example, we will build out the SalesForm from Sara. The user wants to contact
our sales team, and for this we need to gather the following pieces of information:

    - Their job
    - Their bot use case
    - Their name
    - Their email
    - Their budget
    - Their company

We will start by defining the ``SalesForm`` as a new class in the file called ``actions.py``.
The first method we need to define is the name, which like in a regular Action
returns the name that will be used in our stories:

.. code-block:: python

   from rasa_sdk.forms import FormAction

   class SalesForm(FormAction):
       """Collects sales information and adds it to the spreadsheet"""

       def name(self):
           return "sales_form"

Next we have to define the ``required_slots`` method which specifies which pieces of information to
ask for, i.e. which slots to fill.

.. code-block:: python

       @staticmethod
       def required_slots(tracker):
           return [
               "job_function",
               "use_case",
               "budget",
               "person_name",
               "company",
               "business_email",
               ]

Note: you can customise the required slots function not to be static. E.g. if the ``job_function`` is a
developer, you could add a ``required_slot`` about the users experience level with Rasa

Once you’ve done that, you’ll need to specify how the bot should ask for this information. This
is done by specifying ``utter_ask_{slotname}`` templates in your domain file. For the above
we’ll need to specify the following:

.. code-block:: yaml

   utter_ask_business_email:
     - text: What's your business email?
   utter_ask_company:
     - text: What company do you work for?
   utter_ask_budget:
     - text: "What's your annual budget for conversational AI? 💸"
   utter_ask_job_function:
     - text: "What's your job? 🕴"
   utter_ask_person_name:
     - text: What's your name?
   utter_ask_use_case:
     - text: What's your use case?

We’ll also need to define all these slots in our domain:

.. code-block:: yaml

   slots:
     company:
       type: unfeaturized
     job_function:
       type: unfeaturized
     person_name:
       type: unfeaturized
     budget:
       type: unfeaturized
     business_email:
       type: unfeaturized
     use_case:
       type: unfeaturized

Going back to our Form definition, we need to define the ``submit`` method as well,
which will do something with the information the user has provided once the form is complete:

.. code-block:: python

   def submit(
           self,
           dispatcher: CollectingDispatcher,
           tracker: Tracker,
           domain: Dict[Text, Any],
       ) -> List[Dict]:

       dispatcher.utter_message("Thanks for getting in touch, we’ll contact you soon")
       return []

In this case, we only tell the user that we’ll be in touch with them, however
usually you would send this information to an API or a database. See the `rasa-demo <https://github.com/RasaHQ/rasa-demo/blob/master/demo/actions.py#L69>`_
for an example of how to store this information in a spreadsheet.

We’ll need to add the form we just created to a new section in the domain file:

.. code-block:: yaml

   forms:
     - sales_form

We also need to create an intent to activate the form, as well as an intent for providing all the
information the form asks the user for. For the form activation intent, we can create an
intent called ``contact_sales``. Add the following training data to your nlu file:

.. code-block:: md

   ## intent:contact_sales
   - I wanna talk to your sales people.
   - I want to talk to your sales people
   - I want to speak with sales
   - Sales
   - Please schedule a sales call
   - Please connect me to someone from sales
   - I want to get in touch with your sales guys
   - I would like to talk to someone from your sales team
   - sales please

You can view the full intent `here <https://github.com/RasaHQ/rasa-demo/blob/master/data/nlu/nlu.md#intentcontact_sales>`__)

We will also create an intent called ``inform`` which covers any sort of information the user
provides to the bot. *The reason we put all this under one intent, is because there is no
real intent behind providing information, only the entity is important.* Add the following
data to your NLU file:

.. code-block:: md

   ## intent:inform
   - [100k](budget)
   - [100k](budget)
   - [240k/year](budget)
   - [150,000 USD](budget)
   - I work for [Rasa](company)
   - The name of the company is [ACME](company)
   - company: [Rasa Technologies](company)
   - it's a small company from the US, the name is [Hooli](company)
   - it's a tech company, [Rasa](company)
   - [ACME](company)
   - [Rasa Technologies](company)
   - [maxmeier@firma.de](business_email)
   - [bot-fan@bots.com](business_email)
   - [maxmeier@firma.de](business_email)
   - [bot-fan@bots.com](business_email)
   - [my email is email@rasa.com](business_email)
   - [engineer](job_function)
   - [brand manager](job_function)
   - [marketing](job_function)
   - [sales manager](job_function)
   - [growth manager](job_function)
   - [CTO](job_function)
   - [CEO](job_function)
   - [COO](job_function)
   - [John Doe](person_name)
   - [Jane Doe](person_name)
   - [Max Mustermann](person_name)
   - [Max Meier](person_name)
   - We plan to build a [sales bot](use_case) to increase our sales by 500%.
   - we plan to build a [sales bot](use_case) to increase our revenue by 100%.
   - a [insurance tool](use_case) that consults potential customers on the best life insurance to choose.
   - we're building a [conversational assistant](use_case) for our employees to book meeting rooms.

.. note::
    Entities like ``business_email`` and ``budget`` would usually be handled by pretrained entity extractors
    (e.g. :ref:`DucklingHTTPExtractor` or :ref:`SpacyEntityExtractor`), but for this tutorial
    we want to avoid any additional setup.

The intents and entities will need to be added to your domain as well:

.. code-block:: yaml

   intents:
     - greet
     - bye
     - thank
     - faq
     - contact_sales
     - inform

   entities:
     - company
     - job_function
     - person_name
     - budget
     - business_email
     - use_case

A story for a form is very simple, as all the slot collection form happens inside the form, and
therefore doesn’t need to be covered in your stories.

.. code-block:: md

   ## sales form
   * contact_sales
       - sales_form
       - form{"name": "sales_form"}
       - form{"name": null}

As a final step, let’s add the FormPolicy to our config file:

.. code-block:: yaml

   policies:
     - name: MemoizationPolicy
     - name: KerasPolicy
     - name: MappingPolicy
     - name: FormPolicy

At this point, you already have a working form, so let’s try it out. Make sure to uncomment the
``action_endpoint`` in your ``endpoints.yml`` to make Rasa aware of the action server that will run our form:

.. code-block:: yaml

   action_endpoint:
    url: "http://localhost:5055/webhook"

Then start the action server in a new terminal window:
.. code-block:: bash

    rasa run actions

Then you can retrain and talk to your bot:

.. code-block:: bash

   rasa train
   rasa shell

This simple form will work out of the box, however you will likely want to add a bit
more capability to handle different situations. One example of this is validating
slots, to make sure the user provided information correctly (read more about it
`here <https://rasa.com/docs/rasa/core/forms/#validating-user-input>`__).

Another example is that you may want to fill slots from things other than entities
of the same name. E.g. for the "use case" situation in our Form, we would expect
the user to type a full sentence and not something that you could necessarily
extract as an entity. In this case we can make use of the ``slot_mappings`` method,
where you can describe what your entities should be extracted from. Here we can
use the ``from_text`` method to extract the users whole message:

.. code-block:: python

    def slot_mappings(self) -> Dict[Text: Union[Dict, List[Dict]]]:
        # type: () -> Dict[Text: Union[Dict, List[Dict]]]
        """A dictionary to map required slots to
        - an extracted entity
        - intent: value pairs
        - a whole message
        or a list of them, where a first match will be picked"""
        return {"use_case": self.from_text(intent="inform")}

Now our bot will extract the full user message when asking for the use case slot,
and we don’t need to use the ``use_case`` entity defined before.

All of the methods within a form can be customised to handle different branches in your
business logic. Read more about this `here <https://rasa.com/docs/rasa/core/forms/#>`_.
However, you should make sure not to handle any unhappy paths inside the form. These
should be handled by writing regular stories, so your model can learn this behaviour.

Handling unexpected user input
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All expected user inputs should be handled by the form we defined above, i.e. if the
user provides the information the bot asks for. However, in real situations, the user
will often behave differently. In this section we’ll go through various forms of
"interjections" and how to handle them within Rasa.

The decision to handle these types of user input should always come from reviewing
real conversations. You should first build part of your assistant, test it with real users
(whether that's your end user, or your colleague) and then add what's missing. You shouldn't
try to implement every possible edge case that you think might happen, because in the end
your users may never actually behave in that way. `Rasa X <https://rasa.com/docs/rasa-x/installation-and-setup/docker-compose-script/>`__
is a tool that can help you review conversations and make these types of decisions.

Generic interjections
"""""""""""""""""""""

If you have generic interjections that should always have the same single response no
matter the context, you can use the :ref:`mapping-policy` to handle these. It will always
predict the same action for an intent, and when combined with a forgetting mechanism,
you don’t need to write any stories either.

The greet intent is a good example where we will always give the same response and
yet we don’t want the intent to affect the dialogue history. To do this, the response
must be an action that returns the ``UserUtteranceReverted()`` event to remove the
interaction from the dialogue history.

First, open the ``domain.yml`` and modify the greet intent and add ``action_greet`` as shown here:

.. code-block:: yaml

   intents:
     - greet: {triggers: action_greet}
     - bye
     - thank
     - faq
     - contact_sales
     - inform

   Actions:
     - utter_greet
     - utter_noworries
     - utter_bye
     - respond_faq
     - action_greet

Remove any stories using the "greet" intent if you have them.

Next, we need to define ``action_greet``. Add the following action to your ``actions.py`` file:

.. code-block:: python

   from rasa_sdk import Action
   from rasa_sdk.events import UserUtteranceReverted

   class ActionGreetUser(Action):
   """Revertible mapped action for utter_greet"""

   def name(self):
       return "action_greet"

   def run(self, dispatcher, tracker, domain):
       dispatcher.utter_template("utter_greet", tracker)
       return [UserUtteranceReverted()]

To test the modified intents, we need to re-start our action server:

.. code-block:: bash

   rasa run actions

Then we can retrain the model, and try out our additions:

.. code-block:: bash

   rasa train
   rasa shell

To handle FAQs defined with retrieval actions, you can add a simple story that will be handled by the MemoizationPolicy:

.. code-block:: md

   ## just sales, continue
   * contact_sales
       - sales_form
       - form{"name": "sales_form"}
   * faq
       - respond_faq
       - sales_form
       - form{"name": null}

This will break out of the form and deal with the users FAQ question, and then return back to the original task.
If you find it difficult to write stories in this format, you can always use `Interactive Learning <https://rasa.com/docs/rasa/core/interactive-learning/>`_
to help you create them.

As always, make sure to add an end to end test case to your `test_stories.md` file.

Contextual questions
""""""""""""""""""""

You can also handle `contextual questions <https://rasa.com/docs/rasa/dialogue-elements/completing-tasks/#contextual-questions)>`_,
like the user asking the question "Why do you need to know that". The user could ask this based on a certain slot
the bot has requested, and the response should differ for each slot.

To handle this, we need to make the ``requested_slot`` featurized, and assign it the categorical type:

.. code-block:: yaml

   slots:
     requested_slot:
       type: categorical
       values:
         - business_email
         - company
         - person_name
         - use_case
         - budget
         - job_function

This means that Core will pay attention to the value of the slot when making a prediction
(read more about other `featurized slots <https://rasa.com/docs/rasa/api/core-featurization/>`_), whereas
unfeaturized slots are only used for storing information. The stories for this should look as follows:

.. code-block:: md

   ## explain email
   * contact_sales
       - sales_form
       - form{"name": "sales_form"}
       - slot{"requested_slot": "business_email"}
   * explain
       - utter_explain_why_email
       - sales_form
       - form{"name": null}

   ## explain email
   * contact_sales
       - sales_form
       - form{"name": "sales_form"}
       - slot{"requested_slot": "budget"}
   * explain
       - utter_explain_why_budget
       - sales_form
       - form{"name": null}

We’ll need to add the intent and utterances we just added to our domain:

.. code-block:: yaml

   intents:
   - greet: {triggers: action_greet_user}
   - bye
   - thank
   - faq
   - explain

   actions:
   - utter_explain_why_budget
   - utter_explain_why_email

   templates:
     utter_explain_why_budget:
     - text: We need to know your budget to recommend a subscription
     utter_explain_why_email:
     - text: We need your email so we can contact you

Finally, we’ll need to add some NLU data for the explain intent:

.. code-block:: md

   ## intent:explain
   - why
   - why is that
   - why do you need it
   - why do you need to know that?
   - could you explain why you need it?

Then you can retrain your bot and test it again:

.. code-block:: bash

   rasa train
   rasa shell

.. note::
    You will need to add a story for each of the values of the ``requested_slot`` slot
    for the bot to handle every case of "Why do you need to know that"

Don’t forget to add a few end to end stories to your ``test_stories.md`` for testing as well.

Failing gracefully
^^^^^^^^^^^^^^^^^^

Even if you design your bot perfectly, users will inevitably say things to your
assistant that you did not anticipate. In these cases, your assistant will fail,
and it’s important you ensure it does so gracefully.

Fallback policy
"""""""""""""""

One of the most common failures is low NLU confidence, which is handled very nicely with
the TwoStageFallbackPolicy. You can enable it by adding the following to your configuration file,

.. code-block:: yaml

   policies:
     - name: TwoStageFallbackPolicy
       nlu_threshold: 0.8

and adding the ``out_of_scope`` intent to your domain file:

.. code-block:: yaml

   intents:
   - out_of_scope

When the nlu confidence falls below the defined threshold, the bot will prompt the user to
rephrase their message. If the bot isn’t able to get their message three times, there
will be a final action where the bot can e.g. hand off to a human.

To try this out, retrain your model and send a message like "order me a pizza" to your bot:

.. code-block:: bash

   rasa train
   rasa shell

There are also a bunch of ways in which you can customise this policy. In Sara, our demo bot,
we’ve customised it to suggest intents to the user within a certain confidence range to make
it easier for the user to give the bot the information it needs.

This is done by customising the action ``ActionDefaultAskAffirmation`` as shown in the `Sara rasa-demo action server <https://github.com/RasaHQ/rasa-demo/blob/master/demo/actions.py#L443>`_
We define some intent mappings to make it more intuitive to the user what an intent means.

.. image:: /_static/images/intent_mappings.png
   :width: 240
   :alt: Intent Mappings
   :align: center

Out of scope intent
"""""""""""""""""""

It is good practice to also handle questions you know your users may ask, but you don’t necessarily have a skill
implemented yet.

You can define an ``out_of_scope`` intent to handle generic out of scope requests, like "I’m hungry" and have
the bot respond with a default message like "Sorry, I can’t handle that request":

.. code-block:: md

   * out_of_scope
     utter_out_of_scope

We’ll need to add NLU data for the `out_of_scope` intent as well:

.. code-block:: md

   ## intent:out_of_scope
   - I want to order food
   - What is 2 + 2?
   - Who’s the US President?
   - I need a job

And finally we’ll add a template to our domain file:

.. code-block:: yaml

   actions:
   - utter_out_of_scope

   templates:
     utter_out_of_scope:
     - text: Sorry, I can’t handle that request.

We can now re-train, and test this addition

.. code-block:: bash

   rasa train
   rasa shell

Going one step further, if you observe your users asking for certain things, that you’ll
want to turn into a user goal in future, you can handle these as separate intents, to let
the user know you’ve understood their message, but don’t have a solution quite yet. E.g.,
let’s say the user asks "I want to apply for a job at Rasa", we can then reply with
"I understand you’re looking for a job, but I’m afraid I can’t handle that skill yet."

.. code-block:: md

   * ask_job
     utter_job_not_handled

More complex contextual conversations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Not every user goal you define will fall under the category of business logic. For the
other cases you will need to use stories and context to help the user achieve their goal.

If we take the example of the "getting started" skill from Sara, we want to give them
different information based on whether they’ve built an AI assistant before and are
migrating from a different tool etc. This can be done quite simply with stories and
the concept of `max history <https://rasa.com/docs/rasa/core/policies/#max-history>`_.

.. code-block:: md
  :emphasize-lines: 4,5,6,7,8,24,25,26,27,28

   ## new to rasa + built a bot before
   * how_to_get_started
     - utter_getstarted
     - utter_first_bot_with_rasa
   * affirm
     - action_set_onboarding
     - slot{"onboarding": true}
     - utter_built_bot_before
   * affirm
     - utter_ask_migration
   * deny
     - utter_explain_rasa_components
     - utter_rasa_components_details
     - utter_ask_explain_nlucorex
   * affirm
     - utter_explain_nlu
     - utter_explain_core
     - utter_explain_x
     - utter_direct_to_step2

   ## not new to rasa + core
   * how_to_get_started
     - utter_getstarted
     - utter_first_bot_with_rasa
   * deny
     - action_set_onboarding
     - slot{"onboarding": false}
     - utter_ask_which_product
   * how_to_get_started{"product": "core"}
     - utter_explain_core
     - utter_anything_else


The above example mostly leverages intents to guide the flow, however you can also
guide the flow with entities and slots. For example, if the user gives you the
information that they’re new to Rasa at the beginning, you may want to skip this
question by storing this information in a slot.

.. code-block:: md

   * how_to_get_started{"user_type": "new"}
     - slot{"user_type":"new"}
     - action_set_onboarding
     - slot{"onboarding": true}
     - utter_getstarted_new
     - utter_built_bot_before

For this to work, keep in mind that the slot has to be featurized in your domain
file. This time we can use the ``text`` slot type, as we only care about whether the
`slot was set or not <https://rasa.com/docs/rasa/core/slots/>`_.

AugmentedMemoizationPolicy
""""""""""""""""""""""""""

To make your bot more robust to interjections, you can replace the MemoizationPolicy
with the AugmentedMemoizationPolicy. It works the same way as the MemoizationPolicy,
but if no exact match is found it additionally has a mechanism that forgets a certain
amount of steps in the conversation history to find a match in your stories (read more
`here <https://rasa.com/docs/rasa/core/policies/#augmented-memoization-policy>`__)

Using ML to generalise
""""""""""""""""""""""

Aside from the more rule-based policies we described above, Core also has some ML
policies you can use. These come in as an additional layer in your policy configuration,
and only jump in if the user follows a path that you have not anticipated. **It is important
to understand that using these policies does not mean letting go of control over your
assistant.** If a rule based policy is able to make a prediction, that prediction will
always have a higher priority (read more `here <https://rasa.com/docs/rasa/core/policies/#action-selection>`__) and predict the next action. The
ML based policies give your assistant the chance not to fail, whereas if they are not
used your assistant will definitely fail, like in state machine based dialogue systems.

These types of unexpected user behaviors are something our `EmbeddingPolicy <https://blog.rasa.com/attention-dialogue-and-learning-reusable-patterns/>`_ deals with
very well. It can learn to bring the user back on track after some
interjections during the main user goal the user is trying to complete. For example,
in the conversation below (extracted from a conversation on `Rasa X <https://rasa.com/docs/rasa-x/user-guide/review-conversations/>`__):

.. code-block:: md

   ## Story from conversation with a2baab6c83054bfaa8d598459c659d2a on November 28th 2019
   * greet
     - action_greet_user
     - slot{"shown_privacy":true}
   * ask_whoisit
     - action_chitchat
   * ask_whatspossible
     - action_chitchat
   * telljoke
     - action_chitchat
   * how_to_get_started{"product":"x"}
     - slot{"product":"x"}
     - utter_explain_x
     - utter_also_explain_nlucore
   * affirm
     - utter_explain_nlu
     - utter_explain_core
     - utter_direct_to_step2

Here we can see the user has completed a few chitchat tasks first, and then ultimately
asks how they can get started with Rasa X. The EmbeddingPolicy correctly predicts that
Rasa X should be explained to the user, and then also takes them down the getting started
path, without asking all the qualifying questions first.

Since the ML policy generalized well in this situation, it makes sense to add this story
to your training data to continuously improve your bot and help the ML generalize even
better in future. `Rasa X <https://rasa.com/docs/rasa-x/>`_ is a tool that can help
you improve your bot and make it more contextual.

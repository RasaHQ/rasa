:desc: How to build simple FAQ and contextual assistants a
 a
.. _building-assistants: a
 a
Tutorial: Building Assistants a
============================= a
 a
.. edit-link:: a
 a
After following the basics of setting up an assistant in the `Rasa Tutorial <https://rasa.com/docs/rasa/user-guide/rasa-tutorial/>`_, we'll a
now walk through building a basic FAQ chatbot and then build a bot that can handle a
contextual conversations. a
 a
.. contents:: a
   :local: a
 a
.. _build-faq-assistant: a
 a
Building a simple FAQ assistant a
------------------------------- a
 a
FAQ assistants are the simplest assistants to build and a good place to get started. a
These assistants allow the user to ask a simple question and get a response. We’re going to a
build a basic FAQ assistant using features of Rasa designed specifically for this type of assistant. a
 a
In this section we’re going to cover the following topics: a
 a
    - `Responding to simple intents <respond-with-memoization-policy>`_ with the MemoizationPolicy a
    - `Handling FAQs <faqs-response-selector>`_ using the ResponseSelector a
 a
 a
We’re going to use content from `Sara <https://github.com/RasaHQ/rasa-demo>`_, the Rasa a
assistant that, amongst other things, helps the user get started with the Rasa products. a
You should first install Rasa using the `Step-by-step Installation Guide <https://rasa.com/docs/rasa/user-guide/installation/#step-by-step-installation-guide>`_ a
and then follow the `Rasa Tutorial <https://rasa.com/docs/rasa/user-guide/rasa-tutorial/>`_ a
to make sure you know the basics. a
 a
To prepare for this tutorial, we're going to create a new directory and start a a
new Rasa project. a
 a
.. code-block:: bash a
 a
    mkdir rasa-assistant a
    rasa init a
 a
 a
Let's remove the default content from this bot, so that the ``nlu.md``, ``stories.md`` a
and ``domain.yml`` files are empty. a
 a
.. _respond-with-memoization-policy: a
 a
Memoization Policy a
^^^^^^^^^^^^^^^^^^ a
 a
The MemoizationPolicy remembers examples from training stories for up to a ``max_history`` a
of turns. The number of "turns" includes messages the user sent, and actions the a
assistant performed. For the purpose of a simple, context-less FAQ bot, we only need a
to pay attention to the last message the user sent, and therefore we’ll set that to ``1``. a
 a
You can do this by editing your ``config.yml`` file as follows (you can remove ``TEDPolicy`` for now): a
 a
.. code-block:: yaml a
 a
  policies: a
  - name: MemoizationPolicy a
    max_history: 1 a
  - name: MappingPolicy a
 a
.. note:: a
   The MappingPolicy is there because it handles the logic of the ``/restart`` intent, a
   which allows you to clear the conversation history and start fresh. a
 a
Now that we’ve defined our policies, we can add some stories for the ``goodbye``, ``thank`` and ``greet`` a
intents to the ``stories.md`` file: a
 a
.. code-block:: md a
 a
   ## greet a
   * greet a
     - utter_greet a
 a
   ## thank a
   * thank a
     - utter_noworries a
 a
   ## goodbye a
   * bye a
     - utter_bye a
 a
We’ll also need to add the intents, actions and responses to our ``domain.yml`` file in the following sections: a
 a
.. code-block:: md a
 a
   intents: a
     - greet a
     - bye a
     - thank a
 a
   responses: a
     utter_noworries: a
       - text: No worries! a
     utter_greet: a
       - text: Hi a
     utter_bye: a
       - text: Bye! a
 a
Finally, we’ll copy over some NLU data from Sara into our ``nlu.md`` file a
(more can be found `here <https://github.com/RasaHQ/rasa-demo/blob/master/data/nlu/nlu.md>`__): a
 a
.. code-block:: md a
 a
   ## intent:greet a
   - Hi a
   - Hey a
   - Hi bot a
   - Hey bot a
   - Hello a
   - Good morning a
   - hi again a
   - hi folks a
 a
   ## intent:bye a
   - goodbye a
   - goodnight a
   - good bye a
   - good night a
   - see ya a
   - toodle-oo a
   - bye bye a
   - gotta go a
   - farewell a
 a
   ## intent:thank a
   - Thanks a
   - Thank you a
   - Thank you so much a
   - Thanks bot a
   - Thanks for that a
   - cheers a
 a
You can now train a first model and test the bot, by running the following commands: a
 a
.. code-block:: bash a
 a
   rasa train a
   rasa shell a
 a
This bot should now be able to reply to the intents we defined consistently, and in any order. a
 a
For example: a
 a
.. image:: /_static/images/memoization_policy_convo.png a
   :alt: Memoization Policy Conversation a
   :align: center a
 a
 a
While it's good to test the bot interactively, we should also add end to end test cases that a
can later be included as part of a :ref:`CI/CD system <setting-up-ci-cd>`. End-to-end :ref:`test conversations <end-to-end-testing>` a
include NLU data, so that both components of Rasa can be tested. The file a
``tests/conversation_tests.md`` contains example test conversations. Delete all the test conversations and replace a
them with some test conversations for your assistant so far: a
 a
.. code-block:: md a
 a
   ## greet + goodbye a
   * greet: Hi! a
     - utter_greet a
   * bye: Bye a
     - utter_bye a
 a
   ## greet + thanks a
   * greet: Hello there a
     - utter_greet a
   * thank: thanks a bunch a
     - utter_noworries a
 a
   ## greet + thanks + goodbye a
   * greet: Hey a
     - utter_greet a
   * thank: thank you a
     - utter_noworries a
   * bye: bye bye a
     - utter_bye a
 a
To test our model against the test file, run the command: a
 a
.. code-block:: bash a
 a
   rasa test --stories tests/conversation_tests.md a
 a
The test command will produce a directory named ``results``. It should contain a file a
called ``failed_stories.md``, where any test cases that failed will be printed. It will a
also specify whether it was an NLU or Core prediction that went wrong.  As part of a a
CI/CD pipeline, the test option ``--fail-on-prediction-errors`` can be used to throw a
an exception that stops the pipeline. a
 a
.. _faqs-response-selector: a
 a
Response Selectors a
^^^^^^^^^^^^^^^^^^ a
 a
The :ref:`response-selector` NLU component is designed to make it easier to handle dialogue a
elements like :ref:`small-talk` and FAQ messages in a simple manner. By using the ResponseSelector, a
you only need one story to handle all FAQs, instead of adding new stories every time you a
want to increase your bot's scope. a
 a
People often ask Sara different questions surrounding the Rasa products, so let’s a
start with three intents: ``ask_channels``, ``ask_languages``, and ``ask_rasax``. a
We’re going to copy over some NLU data from the `Sara training data <https://github.com/RasaHQ/rasa-demo/blob/master/data/nlu/nlu.md>`_ a
into our ``nlu.md``. It’s important that these intents have an ``faq/`` prefix, so they’re a
recognised as the faq intent by the ResponseSelector: a
 a
.. code-block:: md a
 a
   ## intent: faq/ask_channels a
   - What channels of communication does rasa support? a
   - what channels do you support? a
   - what chat channels does rasa uses a
   - channels supported by Rasa a
   - which messaging channels does rasa support? a
 a
   ## intent: faq/ask_languages a
   - what language does rasa support? a
   - which language do you support? a
   - which languages supports rasa a
   - can I use rasa also for another laguage? a
   - languages supported a
 a
   ## intent: faq/ask_rasax a
   - I want information about rasa x a
   - i want to learn more about Rasa X a
   - what is rasa x? a
   - Can you tell me about rasa x? a
   - Tell me about rasa x a
   - tell me what is rasa x a
 a
Next, we’ll need to define the responses associated with these FAQs in a new file called ``responses.md`` in the ``data/`` directory: a
 a
.. code-block:: md a
 a
   ## ask channels a
   * faq/ask_channels a
     - We have a comprehensive list of [supported connectors](https://rasa.com/docs/core/connectors/), but if a
       you don't see the one you're looking for, you can always create a custom connector by following a
       [this guide](https://rasa.com/docs/rasa/user-guide/connectors/custom-connectors/). a
 a
   ## ask languages a
   * faq/ask_languages a
     - You can use Rasa to build assistants in any language you want! a
 a
   ## ask rasa x a
   * faq/ask_rasax a
    - Rasa X is a tool to learn from real conversations and improve your assistant. Read more [here](https://rasa.com/docs/rasa-x/) a
 a
The ResponseSelector should already be at the end of the NLU pipeline in our ``config.yml``: a
 a
.. code-block:: yaml a
 a
    language: en a
    pipeline: a
      - name: WhitespaceTokenizer a
      - name: RegexFeaturizer a
      - name: LexicalSyntacticFeaturizer a
      - name: CountVectorsFeaturizer a
      - name: CountVectorsFeaturizer a
        analyzer: "char_wb" a
        min_ngram: 1 a
        max_ngram: 4 a
      - name: DIETClassifier a
        epochs: 100 a
      - name: EntitySynonymMapper a
      - name: ResponseSelector a
        epochs: 100 a
 a
Now that we’ve defined the NLU side, we need to make Core aware of these changes. Open your ``domain.yml`` file and add the ``faq`` intent: a
 a
.. code-block:: yaml a
 a
   intents: a
     - greet a
     - bye a
     - thank a
     - faq a
 a
We’ll also need to add a `retrieval action <https://rasa.com/docs/rasa/core/retrieval-actions/>`_, a
which takes care of sending the response predicted from the ResponseSelector back to the user, a
to the list of actions. These actions always have to start with the ``respond_`` prefix: a
 a
.. code-block:: yaml a
 a
   actions: a
     - respond_faq a
 a
Next we’ll write a story so that Core knows which action to predict: a
 a
.. code-block:: md a
 a
   ## Some question from FAQ a
   * faq a
       - respond_faq a
 a
This prediction is handled by the MemoizationPolicy, as we described earlier. a
 a
After all of the changes are done, train a new model and test the modified FAQs: a
 a
.. code-block:: bash a
 a
   rasa train a
   rasa shell a
 a
At this stage it makes sense to add a few test cases to your ``test_stories.md`` file again: a
 a
.. code-block:: md a
 a
   ## ask channels a
   * faq: What messaging channels does Rasa support? a
     - respond_faq a
 a
   ## ask languages a
   * faq: Which languages can I build assistants in? a
     - respond_faq a
 a
   ## ask rasa x a
   * faq: What’s Rasa X? a
     - respond_faq a
 a
You can read more in this `blog post <https://blog.rasa.com/response-retrieval-models/>`_ and the a
`Retrieval Actions <https://rasa.com/docs/rasa/core/retrieval-actions/>`_ page. a
 a
Using the features we described in this tutorial, you can easily build a context-less assistant. a
When you’re ready to enhance your assistant with context, check out :ref:`tutorial-contextual-assistants`. a
 a
 a
.. note:: a
    Here's a minimal checklist of files we modified to build a basic FAQ assistant: a
 a
      - ``data/nlu.md``: Add NLU training data for ``faq/`` intents a
      - ``data/responses.md``: Add responses associated with ``faq/`` intents a
      - ``config.yml``: Add ``ReponseSelector`` in your NLU pipeline a
      - ``domain.yml``: Add a retrieval action ``respond_faq`` and intent ``faq`` a
      - ``data/stories.md``: Add a simple story for FAQs a
      - ``test_stories.md``: Add E2E test stories for your FAQs a
 a
 a
.. _tutorial-contextual-assistants: a
 a
Building a contextual assistant a
------------------------------- a
 a
Whether you’ve just created an FAQ bot or are starting from scratch, the next step is to expand a
your bot to handle contextual conversations. a
 a
In this tutorial we’re going to cover a variety of topics: a
 a
    - :ref:`handling-business-logic` a
    - :ref:`handling-unexpected-user-input` a
    - :ref:`failing-gracefully` a
    - :ref:`more-complex-contextual-conversations` a
 a
Please make sure you’ve got all the data from the :ref:`build-faq-assistant` section before starting this part. a
You will need to make some adjustments to your configuration file, since we now need to pay attention to context: a
 a
.. code-block:: yaml a
 a
   policies: a
   - name: MemoizationPolicy a
   - name: MappingPolicy a
 a
We removed the ``max_history: 1`` configuration. The default is ``5``, a
meaning Core will pay attention to the past 5 turns when making a prediction a
(see explanation of `max history <https://rasa.com/docs/rasa/core/policies/#max-history>`_). a
 a
.. _handling-business-logic: a
 a
Handling business logic a
^^^^^^^^^^^^^^^^^^^^^^^ a
 a
A lot of conversational assistants have user goals that involve collecting a bunch of information a
from the user before being able to do something for them. This is called slot filling. For a
example, in the banking industry you may have a user goal of transferring money, where you a
need to collect information about which account to transfer from, whom to transfer to and the a
amount to transfer. This type of behaviour can and should be handled in a rule based way, as it a
is clear how this information should be collected. a
 a
For this type of use case, we can use Forms and our FormPolicy. The `FormPolicy <https://rasa.com/docs/rasa/core/policies/#form-policy>`_ a
works by predicting the form as the next action until all information is gathered from the user. a
 a
As an example, we will build out the SalesForm from Sara. The user wants to contact a
our sales team, and for this we need to gather the following pieces of information: a
 a
    - Their job a
    - Their bot use case a
    - Their name a
    - Their email a
    - Their budget a
    - Their company a
 a
We will start by defining the ``SalesForm`` as a new class in the file called ``actions.py``. a
The first method we need to define is the name, which like in a regular Action a
returns the name that will be used in our stories: a
 a
.. code-block:: python a
 a
   from rasa_sdk.forms import FormAction a
 a
   class SalesForm(FormAction): a
       """Collects sales information and adds it to the spreadsheet""" a
 a
       def name(self): a
           return "sales_form" a
 a
Next we have to define the ``required_slots`` method which specifies which pieces of information to a
ask for, i.e. which slots to fill. a
 a
.. code-block:: python a
 a
       @staticmethod a
       def required_slots(tracker): a
           return [ a
               "job_function", a
               "use_case", a
               "budget", a
               "person_name", a
               "company", a
               "business_email", a
               ] a
 a
Note: you can customise the required slots function not to be static. E.g. if the ``job_function`` is a a
developer, you could add a ``required_slot`` about the users experience level with Rasa a
 a
Once you’ve done that, you’ll need to specify how the bot should ask for this information. This a
is done by specifying ``utter_ask_{slotname}`` responses in your ``domain.yml`` file. For the above a
we’ll need to specify the following: a
 a
.. code-block:: yaml a
 a
   utter_ask_business_email: a
     - text: What's your business email? a
   utter_ask_company: a
     - text: What company do you work for? a
   utter_ask_budget: a
     - text: "What's your annual budget for conversational AI? 💸" a
   utter_ask_job_function: a
     - text: "What's your job? 🕴" a
   utter_ask_person_name: a
     - text: What's your name? a
   utter_ask_use_case: a
     - text: What's your use case? a
 a
We’ll also need to define all these slots in our ``domain.yml`` file: a
 a
.. code-block:: yaml a
 a
   slots: a
     company: a
       type: unfeaturized a
     job_function: a
       type: unfeaturized a
     person_name: a
       type: unfeaturized a
     budget: a
       type: unfeaturized a
     business_email: a
       type: unfeaturized a
     use_case: a
       type: unfeaturized a
 a
Going back to our Form definition, we need to define the ``submit`` method as well, a
which will do something with the information the user has provided once the form is complete: a
 a
.. code-block:: python a
 a
   def submit( a
           self, a
           dispatcher: CollectingDispatcher, a
           tracker: Tracker, a
           domain: Dict[Text, Any], a
       ) -> List[Dict]: a
 a
       dispatcher.utter_message("Thanks for getting in touch, we’ll contact you soon") a
       return [] a
 a
In this case, we only tell the user that we’ll be in touch with them, however a
usually you would send this information to an API or a database. See the `rasa-demo <https://github.com/RasaHQ/rasa-demo/blob/master/actions/actions.py#L149>`_ a
for an example of how to store this information in a spreadsheet. a
 a
We’ll need to add the form we just created to a new section in our ``domain.yml`` file: a
 a
.. code-block:: yaml a
 a
   forms: a
     - sales_form a
 a
We also need to create an intent to activate the form, as well as an intent for providing all the a
information the form asks the user for. For the form activation intent, we can create an a
intent called ``contact_sales``. Add the following training data to your nlu file: a
 a
.. code-block:: md a
 a
   ## intent:contact_sales a
   - I wanna talk to your sales people. a
   - I want to talk to your sales people a
   - I want to speak with sales a
   - Sales a
   - Please schedule a sales call a
   - Please connect me to someone from sales a
   - I want to get in touch with your sales guys a
   - I would like to talk to someone from your sales team a
   - sales please a
 a
You can view the full intent `here <https://github.com/RasaHQ/rasa-demo/blob/master/data/nlu/nlu.md#intentcontact_sales>`__) a
 a
We will also create an intent called ``inform`` which covers any sort of information the user a
provides to the bot. *The reason we put all this under one intent, is because there is no a
real intent behind providing information, only the entity is important.* Add the following a
data to your NLU file: a
 a
.. code-block:: md a
 a
   ## intent:inform a
   - [100k](budget) a
   - [100k](budget) a
   - [240k/year](budget) a
   - [150,000 USD](budget) a
   - I work for [Rasa](company) a
   - The name of the company is [ACME](company) a
   - company: [Rasa Technologies](company) a
   - it's a small company from the US, the name is [Hooli](company) a
   - it's a tech company, [Rasa](company) a
   - [ACME](company) a
   - [Rasa Technologies](company) a
   - [maxmeier@firma.de](business_email) a
   - [bot-fan@bots.com](business_email) a
   - [maxmeier@firma.de](business_email) a
   - [bot-fan@bots.com](business_email) a
   - [my email is email@rasa.com](business_email) a
   - [engineer](job_function) a
   - [brand manager](job_function) a
   - [marketing](job_function) a
   - [sales manager](job_function) a
   - [growth manager](job_function) a
   - [CTO](job_function) a
   - [CEO](job_function) a
   - [COO](job_function) a
   - [John Doe](person_name) a
   - [Jane Doe](person_name) a
   - [Max Mustermann](person_name) a
   - [Max Meier](person_name) a
   - We plan to build a [sales bot](use_case) to increase our sales by 500%. a
   - we plan to build a [sales bot](use_case) to increase our revenue by 100%. a
   - a [insurance tool](use_case) that consults potential customers on the best life insurance to choose. a
   - we're building a [conversational assistant](use_case) for our employees to book meeting rooms. a
 a
.. note:: a
    Entities like ``business_email`` and ``budget`` would usually be handled by pretrained entity extractors a
    (e.g. :ref:`DucklingHTTPExtractor` or :ref:`SpacyEntityExtractor`), but for this tutorial a
    we want to avoid any additional setup. a
 a
The intents and entities will need to be added to your ``domain.yml`` file as well: a
 a
.. code-block:: yaml a
 a
   intents: a
     - greet a
     - bye a
     - thank a
     - faq a
     - contact_sales a
     - inform a
 a
   entities: a
     - company a
     - job_function a
     - person_name a
     - budget a
     - business_email a
     - use_case a
 a
A story for a form is very simple, as all the slot collection form happens inside the form, and a
therefore doesn’t need to be covered in your stories. You just need to write a single story showing when the form should be activated. For the sales form, add this story a
to your ``stories.md`` file: a
 a
 a
.. code-block:: md a
 a
   ## sales form a
   * contact_sales a
       - sales_form                   <!--Run the sales_form action--> a
       - form{"name": "sales_form"}   <!--Activate the form--> a
       - form{"name": null}           <!--Deactivate the form--> a
 a
 a
 a
As a final step, let’s add the FormPolicy to our config file: a
 a
.. code-block:: yaml a
 a
   policies: a
     - name: MemoizationPolicy a
     - name: TEDPolicy a
     - name: MappingPolicy a
     - name: FormPolicy a
 a
At this point, you already have a working form, so let’s try it out. Make sure to uncomment the a
``action_endpoint`` in your ``endpoints.yml`` to make Rasa aware of the action server that will run our form: a
 a
.. code-block:: yaml a
 a
   action_endpoint: a
    url: "http://localhost:5055/webhook" a
 a
Then start the action server in a new terminal window: a
 a
.. code-block:: bash a
 a
    rasa run actions a
 a
Then you can retrain and talk to your bot: a
 a
.. code-block:: bash a
 a
   rasa train a
   rasa shell a
 a
This simple form will work out of the box, however you will likely want to add a bit a
more capability to handle different situations. One example of this is validating a
slots, to make sure the user provided information correctly (read more about it a
`here <https://rasa.com/docs/rasa/core/forms/#validating-user-input>`__). a
 a
Another example is that you may want to fill slots from things other than entities a
of the same name. E.g. for the "use case" situation in our Form, we would expect a
the user to type a full sentence and not something that you could necessarily a
extract as an entity. In this case we can make use of the ``slot_mappings`` method, a
where you can describe what your entities should be extracted from. Here we can a
use the ``from_text`` method to extract the users whole message: a
 a
.. code-block:: python a
 a
    def slot_mappings(self) -> Dict[Text, Union[Dict, List[Dict[Text, Any]]]]: a
        """A dictionary to map required slots to a
        - an extracted entity a
        - intent: value pairs a
        - a whole message a
        or a list of them, where a first match will be picked""" a
        return {"use_case": self.from_text(intent="inform")} a
 a
Now our bot will extract the full user message when asking for the use case slot, a
and we don’t need to use the ``use_case`` entity defined before. a
 a
All of the methods within a form can be customised to handle different branches in your a
business logic. Read more about this `here <https://rasa.com/docs/rasa/core/forms/#>`_. a
However, you should make sure not to handle any unhappy paths inside the form. These a
should be handled by writing regular stories, so your model can learn this behaviour. a
 a
 a
.. note:: a
    Here's a minimal checklist of files we modified to handle business logic using a form action: a
 a
      - ``actions.py``: Define the form action, including the ``required_slots``, ``slot_mappings`` and ``submit`` methods a
      - ``data/nlu.md``: a
          - Add examples for an intent to activate the form a
          - Add examples for an ``inform`` intent to fill the form a
      - ``domain.yml``: a
          - Add all slots required by the form a
          - Add ``utter_ask_{slot}`` responses for all required slots a
          - Add your form action to the ``forms`` section a
          - Add all intents and entities from your NLU training data a
      - ``data/stories.md``: Add a story for the form a
      - ``config.yml``: a
          - Add the ``FormPolicy`` to your policies a
          - Add entity extractors to your pipeline a
      - ``endpoints.yml``: Define the ``action_endpoint`` a
 a
 a
.. _handling-unexpected-user-input: a
 a
Handling unexpected user input a
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ a
 a
All expected user inputs should be handled by the form we defined above, i.e. if the a
user provides the information the bot asks for. However, in real situations, the user a
will often behave differently. In this section we’ll go through various forms of a
"interjections" and how to handle them within Rasa. a
 a
The decision to handle these types of user input should always come from reviewing a
real conversations. You should first build part of your assistant, test it with real users a
(whether that's your end user, or your colleague) and then add what's missing. You shouldn't a
try to implement every possible edge case that you think might happen, because in the end a
your users may never actually behave in that way. `Rasa X <https://rasa.com/docs/rasa-x/installation-and-setup/docker-compose-script/>`__ a
is a tool that can help you review conversations and make these types of decisions. a
 a
Generic interjections a
""""""""""""""""""""" a
 a
If you have generic interjections that should always have the same single response no a
matter the context, you can use the :ref:`mapping-policy` to handle these. It will always a
predict the same action for an intent, and when combined with a forgetting mechanism, a
you don’t need to write any stories either. a
 a
For example, let's say you see users having conversations like the following one with a
your assistant, where they write a greeting in the middle of a conversation - a
maybe because they were gone for a few minutes: a
 a
.. image:: /_static/images/greet_interjection.png a
   :width: 240 a
   :alt: Greeting Interjection a
   :align: center a
 a
The greet intent is a good example where we will always give the same response and a
yet we don’t want the intent to affect the dialogue history. To do this, the response a
must be an action that returns the ``UserUtteranceReverted()`` event to remove the a
interaction from the dialogue history. a
 a
First, open the ``domain.yml`` file and modify the greet intent and add a new block ```actions``` in a
the file, next, add the ``action_greet`` as shown here: a
 a
.. code-block:: yaml a
 a
   intents: a
     - greet: {triggers: action_greet} a
     - bye a
     - thank a
     - faq a
     - contact_sales a
     - inform a
 a
   actions: a
     - action_greet a
 a
Remove any stories using the "greet" intent if you have them. a
 a
Next, we need to define ``action_greet``. Add the following action to your ``actions.py`` file: a
 a
.. code-block:: python a
 a
   from rasa_sdk import Action a
   from rasa_sdk.events import UserUtteranceReverted a
 a
   class ActionGreetUser(Action): a
   """Revertible mapped action for utter_greet""" a
 a
   def name(self): a
       return "action_greet" a
 a
   def run(self, dispatcher, tracker, domain): a
       dispatcher.utter_template("utter_greet", tracker) a
       return [UserUtteranceReverted()] a
 a
To test the modified intents, we need to re-start our action server: a
 a
.. code-block:: bash a
 a
   rasa run actions a
 a
Then we can retrain the model, and try out our additions: a
 a
.. code-block:: bash a
 a
   rasa train a
   rasa shell a
 a
FAQs are another kind of generic interjections that should always get the same response. a
For example, a user might ask a related FAQ in the middle of filling a form: a
 a
.. image:: /_static/images/generic_interjection.png a
   :width: 240 a
   :alt: Generic Interjections a
   :align: center a
 a
To handle FAQs defined with retrieval actions, you can add a simple story that will be handled by the MemoizationPolicy: a
 a
.. code-block:: md a
 a
   ## just sales, continue a
   * contact_sales a
       - sales_form a
       - form{"name": "sales_form"} a
   * faq a
       - respond_faq a
       - sales_form a
       - form{"name": null} a
 a
This will break out of the form and deal with the users FAQ question, and then return back to the original task. a
For example: a
 a
.. image:: /_static/images/generic_interjection_handled.png a
   :width: 240 a
   :alt: Generic Interjection Handled a
   :align: center a
 a
If you find it difficult to write stories in this format, you can always use `Interactive Learning <https://rasa.com/docs/rasa/core/interactive-learning/>`_ a
to help you create them. a
 a
As always, make sure to add an end to end test case to your `test_stories.md` file. a
 a
Contextual questions a
"""""""""""""""""""" a
 a
You can also handle `contextual questions <https://rasa.com/docs/rasa/dialogue-elements/completing-tasks/#contextual-questions)>`_, a
like the user asking the question "Why do you need to know that". The user could ask this based on a certain slot a
the bot has requested, and the response should differ for each slot. For example: a
 a
.. image:: /_static/images/contextual_interjection.png a
   :width: 240 a
   :alt: Contextual Interjection a
   :align: center a
 a
To handle this, we need to make the ``requested_slot`` featurized, and assign it the categorical type: a
 a
.. code-block:: yaml a
 a
   slots: a
     requested_slot: a
       type: categorical a
       values: a
         - business_email a
         - company a
         - person_name a
         - use_case a
         - budget a
         - job_function a
 a
This means that Core will pay attention to the value of the slot when making a prediction a
(read more about other `featurized slots <https://rasa.com/docs/rasa/api/core-featurization/>`_), whereas a
unfeaturized slots are only used for storing information. The stories for this should look as follows: a
 a
.. code-block:: md a
 a
   ## explain email a
   * contact_sales a
       - sales_form a
       - form{"name": "sales_form"} a
       - slot{"requested_slot": "business_email"} a
   * explain a
       - utter_explain_why_email a
       - sales_form a
       - form{"name": null} a
 a
   ## explain budget a
   * contact_sales a
       - sales_form a
       - form{"name": "sales_form"} a
       - slot{"requested_slot": "budget"} a
   * explain a
       - utter_explain_why_budget a
       - sales_form a
       - form{"name": null} a
 a
We’ll need to add the intent and utterances we just added to our ``domain.yml`` file: a
 a
.. code-block:: yaml a
 a
   intents: a
   - greet: {triggers: action_greet_user} a
   - bye a
   - thank a
   - faq a
   - explain a
 a
   responses: a
     utter_explain_why_budget: a
     - text: We need to know your budget to recommend a subscription a
     utter_explain_why_email: a
     - text: We need your email so we can contact you a
 a
Finally, we’ll need to add some NLU data for the explain intent: a
 a
.. code-block:: md a
 a
   ## intent:explain a
   - why a
   - why is that a
   - why do you need it a
   - why do you need to know that? a
   - could you explain why you need it? a
 a
Then you can retrain your bot and test it again: a
 a
.. code-block:: bash a
 a
   rasa train a
   rasa shell a
 a
.. note:: a
    You will need to add a story for each of the values of the ``requested_slot`` slot a
    for the bot to handle every case of "Why do you need to know that" a
 a
Don’t forget to add a few end to end stories to your ``test_stories.md`` for testing as well. a
 a
 a
.. note:: a
    Here's a minimal checklist of  of files we modified to handle unexpected user input: a
 a
      - ``actions.py``: Define ``action_greet`` a
      - ``data/nlu.md``: Add training data for an ``explain`` intent a
      - ``domain.yml``: a
          - Map intent ``greet`` to  ``action_greet_user`` a
          - Make ``requested_slot`` a categorical slots with all required slots as values a
          - Add the ``explain`` intent a
          - Add responses for contextual question interruptions a
      - ``data/stories.md``: a
          - Remove stories using mapped intents if you have them a
          - Add stories with FAQ & contextual interruptions in the middle of filling a form a
 a
 a
.. _failing-gracefully: a
 a
Failing gracefully a
^^^^^^^^^^^^^^^^^^ a
 a
Even if you design your bot perfectly, users will inevitably say things to your a
assistant that you did not anticipate. In these cases, your assistant will fail, a
and it’s important you ensure it does so gracefully. a
 a
Fallback policy a
""""""""""""""" a
 a
One of the most common failures is low NLU confidence, which is handled very nicely with a
the TwoStageFallbackPolicy. You can enable it by adding the following to your configuration file, a
 a
.. code-block:: yaml a
 a
   policies: a
     - name: TwoStageFallbackPolicy a
       nlu_threshold: 0.8 a
 a
and adding the ``out_of_scope`` intent to your ``domain.yml`` file: a
 a
.. code-block:: yaml a
 a
   intents: a
   - out_of_scope a
 a
When the nlu confidence falls below the defined threshold, the bot will prompt the user to a
rephrase their message. If the bot isn’t able to get their message three times, there a
will be a final action where the bot can e.g. hand off to a human. a
 a
To try this out, retrain your model and send a message like "order me a pizza" to your bot: a
 a
.. code-block:: bash a
 a
   rasa train a
   rasa shell a
 a
There are also a bunch of ways in which you can customise this policy. In Sara, our demo bot, a
we’ve customised it to suggest intents to the user within a certain confidence range to make a
it easier for the user to give the bot the information it needs. a
 a
This is done by customising the action ``ActionDefaultAskAffirmation`` as shown in the a
`Sara rasa-demo action server <https://github.com/RasaHQ/rasa-demo/blob/master/actions/actions.py#L443>`_ a
We define some intent mappings to make it more intuitive to the user what an intent means. a
 a
.. image:: /_static/images/intent_mappings.png a
   :width: 240 a
   :alt: Intent Mappings a
   :align: center a
 a
Out of scope intent a
""""""""""""""""""" a
 a
It is good practice to also handle questions you know your users may ask, but for which you haven't necessarily implemented a user goal yet. a
 a
You can define an ``out_of_scope`` intent to handle generic out of scope requests, like "I’m hungry" and have a
the bot respond with a default message like "Sorry, I can’t handle that request": a
 a
.. code-block:: md a
 a
   * out_of_scope a
     utter_out_of_scope a
 a
We’ll need to add NLU data for the ``out_of_scope`` intent as well: a
 a
.. code-block:: md a
 a
   ## intent:out_of_scope a
   - I want to order food a
   - What is 2 + 2? a
   - Who’s the US President? a
   - I need a job a
 a
And finally we’ll add a response to our ``domain.yml`` file: a
 a
.. code-block:: yaml a
 a
   responses: a
     utter_out_of_scope: a
     - text: Sorry, I can’t handle that request. a
 a
We can now re-train, and test this addition a
 a
.. code-block:: bash a
 a
   rasa train a
   rasa shell a
 a
Going one step further, if you observe your users asking for certain things, that you’ll a
want to turn into a user goal in future, you can handle these as separate intents, to let a
the user know you’ve understood their message, but don’t have a solution quite yet. E.g., a
let’s say the user asks "I want to apply for a job at Rasa", we can then reply with a
"I understand you’re looking for a job, but I’m afraid I can’t handle that skill yet." a
 a
.. code-block:: md a
 a
   * ask_job a
     utter_job_not_handled a
 a
.. note:: a
    Here's a minimal checklist of files we modified to help our assistant fail gracefully: a
 a
      - ``data/nlu.md``: a
          - Add training data for the ``out_of_scope`` intent & any specific out of scope intents that you want to handle seperately a
      - ``data/stories.md``: a
          - Add stories for any specific out of scope intents a
      - ``domain.yml``: a
          - Add the ``out_of_scope`` intent & any specific out of scope intents a
          - Add an ``utter_out_of_scope`` response & responses for any specific out of scope intents a
      - ``actions.py``: a
          - Customise ``ActionDefaultAskAffirmation`` to suggest intents for the user to choose from a
      - ``config.yml``: a
          - Add the TwoStageFallbackPolicy to the ``policies`` section a
 a
 a
.. _more-complex-contextual-conversations: a
 a
More complex contextual conversations a
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ a
 a
Not every user goal you define will fall under the category of business logic. For the a
other cases you will need to use stories and context to help the user achieve their goal. a
 a
If we take the example of the "getting started" skill from Sara, we want to give them a
different information based on whether they’ve built an AI assistant before and are a
migrating from a different tool etc. This can be done quite simply with stories and a
the concept of `max history <https://rasa.com/docs/rasa/core/policies/#max-history>`_. a
 a
.. code-block:: md a
  :emphasize-lines: 4,5,6,7,8,24,25,26,27,28 a
 a
   ## new to rasa + built a bot before a
   * how_to_get_started a
     - utter_getstarted a
     - utter_first_bot_with_rasa a
   * affirm a
     - action_set_onboarding a
     - slot{"onboarding": true} a
     - utter_built_bot_before a
   * affirm a
     - utter_ask_migration a
   * deny a
     - utter_explain_rasa_components a
     - utter_rasa_components_details a
     - utter_ask_explain_nlucorex a
   * affirm a
     - utter_explain_nlu a
     - utter_explain_core a
     - utter_explain_x a
     - utter_direct_to_step2 a
 a
   ## not new to rasa + core a
   * how_to_get_started a
     - utter_getstarted a
     - utter_first_bot_with_rasa a
   * deny a
     - action_set_onboarding a
     - slot{"onboarding": false} a
     - utter_ask_which_product a
   * how_to_get_started{"product": "core"} a
     - utter_explain_core a
     - utter_anything_else a
 a
 a
The above example mostly leverages intents to guide the flow, however you can also a
guide the flow with entities and slots. For example, if the user gives you the a
information that they’re new to Rasa at the beginning, you may want to skip this a
question by storing this information in a slot. a
 a
.. code-block:: md a
 a
   * how_to_get_started{"user_type": "new"} a
     - slot{"user_type":"new"} a
     - action_set_onboarding a
     - slot{"onboarding": true} a
     - utter_getstarted_new a
     - utter_built_bot_before a
 a
For this to work, keep in mind that the slot has to be featurized in your ``domain.yml`` a
file. This time we can use the ``text`` slot type, as we only care about whether the a
`slot was set or not <https://rasa.com/docs/rasa/core/slots/>`_. a
 a
AugmentedMemoizationPolicy a
"""""""""""""""""""""""""" a
 a
To make your bot more robust to interjections, you can replace the MemoizationPolicy a
with the AugmentedMemoizationPolicy. It works the same way as the MemoizationPolicy, a
but if no exact match is found it additionally has a mechanism that forgets a certain a
amount of steps in the conversation history to find a match in your stories (read more a
`here <https://rasa.com/docs/rasa/core/policies/#augmented-memoization-policy>`__) a
 a
Using ML to generalise a
"""""""""""""""""""""" a
 a
Aside from the more rule-based policies we described above, Core also has some ML a
policies you can use. These come in as an additional layer in your policy configuration, a
and only jump in if the user follows a path that you have not anticipated. **It is important a
to understand that using these policies does not mean letting go of control over your a
assistant.** If a rule based policy is able to make a prediction, that prediction will a
always have a higher priority (read more `here <https://rasa.com/docs/rasa/core/policies/#action-selection>`__) and predict the next action. The a
ML based policies give your assistant the chance not to fail, whereas if they are not a
used your assistant will definitely fail, like in state machine based dialogue systems. a
 a
These types of unexpected user behaviors are something our `TEDPolicy <https://blog.rasa.com/unpacking-the-ted-policy-in-rasa-open-source/>`_ deals with a
very well. It can learn to bring the user back on track after some a
interjections during the main user goal the user is trying to complete. For example, a
in the conversation below (extracted from a conversation on `Rasa X <https://rasa.com/docs/rasa-x/user-guide/review-conversations/>`__): a
 a
.. code-block:: md a
 a
   ## Story from conversation with a2baab6c83054bfaa8d598459c659d2a on November 28th 2019 a
   * greet a
     - action_greet_user a
     - slot{"shown_privacy":true} a
   * ask_whoisit a
     - action_chitchat a
   * ask_whatspossible a
     - action_chitchat a
   * telljoke a
     - action_chitchat a
   * how_to_get_started{"product":"x"} a
     - slot{"product":"x"} a
     - utter_explain_x a
     - utter_also_explain_nlucore a
   * affirm a
     - utter_explain_nlu a
     - utter_explain_core a
     - utter_direct_to_step2 a
 a
Here we can see the user has completed a few chitchat tasks first, and then ultimately a
asks how they can get started with Rasa X. The TEDPolicy correctly predicts that a
Rasa X should be explained to the user, and then also takes them down the getting started a
path, without asking all the qualifying questions first. a
 a
Since the ML policy generalized well in this situation, it makes sense to add this story a
to your training data to continuously improve your bot and help the ML generalize even a
better in future. `Rasa X <https://rasa.com/docs/rasa-x/>`_ is a tool that can help a
you improve your bot and make it more contextual. a
 a
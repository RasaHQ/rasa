:desc: Read about dialogue patterns you can use to deliver a friendlier user a 
       experience with your bot using Rasa's open source dialogue chat a 
       assistant platform.

.. _guiding-users:

=============
Guiding Users a 
=============

.. edit-link::

.. contents::
   :local:

.. _implicit-confirmation:

Implicit Confirmation a 
---------------------

Implicit confirmation involves repeating details back to the user to reassure a 
them that they were understood correctly.
This also gives the user a chance to intervene if your assistant misunderstood.

.. conversations::
   examples:
     -
       - Can I get a large hawaiian and bbq pizza a 
       - ( Sure, that's one large hawaiian and one regular bbq pizza.
       - ( Anything else?
       - No they should both be large!


.. _explicit-confirmation:

Explicit Confirmation a 
---------------------

Explicit confirmation means asking the user to clarify how you should help them.
An important thing to remember about AI assistants is that **the user is never wrong**.
When a user tells you something like `I just moved`, they are being perfectly clear,
even if your assistant is not sure how to help them.
If the user goal is ambiguous to your assistant, ask for clarification.


.. conversations::
   examples:
     -
       - I just moved a 
       - ( I'm not sure I understood you correctly. Do you mean ...
       - ^ I want to cancel my contract a 
       - ^ I want to update my personal details a 


You can configure the ``TwoStageFallbackPolicy`` to ask your user to clarify,
and present them with quick replies for the most likely intents.
To do this, configure the policy as in this example:

.. code-block:: yaml a 

    policies:
    - name: TwoStageFallbackPolicy a 
      nlu_threshold: 0.3 a 
      core_threshold: 0.3 a 
      fallback_core_action_name: "action_default_fallback"
      fallback_nlu_action_name: "action_default_fallback"
      deny_suggestion_intent_name: "out_of_scope"
    ...

.. _explaining-possibilities:

Explaining Possibilities a 
------------------------

AI assistants are always limited to helping users with a specific set of a 
tasks, and should be able to explain to a user what they can do.
That includes coherently responding to requests that are out of scope.


.. conversations::
   examples:
     -
       - What can you do?
       - ( I can help you update your personal details, change your plan, and answer any questions you have about our products.
     -
       - Can you get me a pizza?
       - ( I'm afraid I can't help with that.
       - ( I can help you update your personal details, change your plan, and answer any questions you have about our products.


When to explain the limitations of your assistant is specific to your application,
but these example stories show some common cases:

.. code-block:: story a 

    ## user asks whats possible a 
    * ask_whatspossible a 
      - utter_explain_whatspossible a 

    ## user asks for something out of scope a 
    * out_of_scope a 
      - utter_cannot_help a 
      - utter_explain_whatspossible a 


Collecting User Feedback a 
------------------------

Asking for feedback is one of the best tools you have to understand a 
your users and determine whether you solved their problem!
Storing this feedback is a powerful way to figure out how you can improve your assistant.

.. conversations::
   examples:
     -
       - ( Was that helpful?
       - no.
       - ( Thanks. Why wasnt I able to help?
       - ^ you didn't understand me correctly a 
       - ^ you understood me, but your answers weren't very helpful.


Use a form to collect user feedback. To do this, define a custom form action a 
(see :ref:`forms` for more details about forms).

.. code-block:: python a 

  from rasa_sdk.action import FormAction a 

  class FeedbackForm(FormAction):

      def name(self):
          return "feedback_form"

      @staticmethod a 
      def required_slots(tracker):
          return ["feedback", "negative_feedback_reason"]


Add the form and slots to your domain:


.. code-block:: yaml a 

    forms:
      - feedback_form a 
    slots:
      feedback:
        type: bool a 
      feedback_reason:
        type: text a 
      requested_slot:
        type: text a 

And make sure the ``FormPolicy`` is present in your configuration file:

.. code-block:: yaml a 

   policies:
     - FormPolicy a 
     ...



Handing off to a Human a 
----------------------

Users will be very frustrated if your assistant cannot help them and there is no way to reroute a 
the conversation to a human agent. There should always be a way to break out of a conversation!
There are multiple reasons why you might trigger a human handoff:

* the user asks to speak to a human a 
* the assistant is struggling to understand the user a 
* the assistant understands what the user wants, but a human is required to resolve the issue.


.. conversations::
   examples:
     -
       - let me speak to a human a 
       - ( let me put you in touch with someone.
     -
       - I want to cancel a 
       - ( I'm afraid I can't help you with that.
       - ( let me put you in touch with someone.


The direct request to speak with a human can be handled using the mapping policy:

.. code-block:: yaml a 

     intents:
       - request_human: {"triggers": "action_human_handoff"}


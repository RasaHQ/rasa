:desc: Read about dialogue patterns you can use to deliver a friendlier user a
       experience with your bot using Rasa's open source dialogue chat a
       assistant platform. a
 a
.. _guiding-users: a
 a
============= a
Guiding Users a
============= a
 a
.. edit-link:: a
 a
.. contents:: a
   :local: a
 a
.. _implicit-confirmation: a
 a
Implicit Confirmation a
--------------------- a
 a
Implicit confirmation involves repeating details back to the user to reassure a
them that they were understood correctly. a
This also gives the user a chance to intervene if your assistant misunderstood. a
 a
.. conversations:: a
   examples: a
     - a
       - Can I get a large hawaiian and bbq pizza a
       - ( Sure, that's one large hawaiian and one regular bbq pizza. a
       - ( Anything else? a
       - No they should both be large! a
 a
 a
.. _explicit-confirmation: a
 a
Explicit Confirmation a
--------------------- a
 a
Explicit confirmation means asking the user to clarify how you should help them. a
An important thing to remember about AI assistants is that **the user is never wrong**. a
When a user tells you something like `I just moved`, they are being perfectly clear, a
even if your assistant is not sure how to help them. a
If the user goal is ambiguous to your assistant, ask for clarification. a
 a
 a
.. conversations:: a
   examples: a
     - a
       - I just moved a
       - ( I'm not sure I understood you correctly. Do you mean ... a
       - ^ I want to cancel my contract a
       - ^ I want to update my personal details a
 a
 a
You can configure the ``TwoStageFallbackPolicy`` to ask your user to clarify, a
and present them with quick replies for the most likely intents. a
To do this, configure the policy as in this example: a
 a
.. code-block:: yaml a
 a
    policies: a
    - name: TwoStageFallbackPolicy a
      nlu_threshold: 0.3 a
      core_threshold: 0.3 a
      fallback_core_action_name: "action_default_fallback" a
      fallback_nlu_action_name: "action_default_fallback" a
      deny_suggestion_intent_name: "out_of_scope" a
    ... a
 a
.. _explaining-possibilities: a
 a
Explaining Possibilities a
------------------------ a
 a
AI assistants are always limited to helping users with a specific set of a
tasks, and should be able to explain to a user what they can do. a
That includes coherently responding to requests that are out of scope. a
 a
 a
.. conversations:: a
   examples: a
     - a
       - What can you do? a
       - ( I can help you update your personal details, change your plan, and answer any questions you have about our products. a
     - a
       - Can you get me a pizza? a
       - ( I'm afraid I can't help with that. a
       - ( I can help you update your personal details, change your plan, and answer any questions you have about our products. a
 a
 a
When to explain the limitations of your assistant is specific to your application, a
but these example stories show some common cases: a
 a
.. code-block:: story a
 a
    ## user asks whats possible a
    * ask_whatspossible a
      - utter_explain_whatspossible a
 a
    ## user asks for something out of scope a
    * out_of_scope a
      - utter_cannot_help a
      - utter_explain_whatspossible a
 a
 a
Collecting User Feedback a
------------------------ a
 a
Asking for feedback is one of the best tools you have to understand a
your users and determine whether you solved their problem! a
Storing this feedback is a powerful way to figure out how you can improve your assistant. a
 a
.. conversations:: a
   examples: a
     - a
       - ( Was that helpful? a
       - no. a
       - ( Thanks. Why wasnt I able to help? a
       - ^ you didn't understand me correctly a
       - ^ you understood me, but your answers weren't very helpful. a
 a
 a
Use a form to collect user feedback. To do this, define a custom form action a
(see :ref:`forms` for more details about forms). a
 a
.. code-block:: python a
 a
  from rasa_sdk.action import FormAction a
 a
  class FeedbackForm(FormAction): a
 a
      def name(self): a
          return "feedback_form" a
 a
      @staticmethod a
      def required_slots(tracker): a
          return ["feedback", "negative_feedback_reason"] a
 a
 a
Add the form and slots to your domain: a
 a
 a
.. code-block:: yaml a
 a
    forms: a
      - feedback_form a
    slots: a
      feedback: a
        type: bool a
      feedback_reason: a
        type: text a
      requested_slot: a
        type: text a
 a
And make sure the ``FormPolicy`` is present in your configuration file: a
 a
.. code-block:: yaml a
 a
   policies: a
     - FormPolicy a
     ... a
 a
 a
 a
Handing off to a Human a
---------------------- a
 a
Users will be very frustrated if your assistant cannot help them and there is no way to reroute a
the conversation to a human agent. There should always be a way to break out of a conversation! a
There are multiple reasons why you might trigger a human handoff: a
 a
* the user asks to speak to a human a
* the assistant is struggling to understand the user a
* the assistant understands what the user wants, but a human is required to resolve the issue. a
 a
 a
.. conversations:: a
   examples: a
     - a
       - let me speak to a human a
       - ( let me put you in touch with someone. a
     - a
       - I want to cancel a
       - ( I'm afraid I can't help you with that. a
       - ( let me put you in touch with someone. a
 a
 a
The direct request to speak with a human can be handled using the mapping policy: a
 a
.. code-block:: yaml a
 a
     intents: a
       - request_human: {"triggers": "action_human_handoff"} a
 a
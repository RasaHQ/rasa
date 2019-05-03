.. _elements-completing-tasks:

================
Completing Tasks
================

.. contents::
   :local:

.. _simple-questions:

Simple Questions
----------------

Simple questions, or FAQs, should receive the same answer
no matter what happened previously in the conversation.
Users will often ask a basic set of questions 
and your assistant should answer them reliably.

.. conversations:: 
   :path: _static/elements/completing-tasks/simple-questions.yml

Just like greetings and goodbyes, you can use the mapping policy to achieve this.
See :ref:`greetings` .

Business Logic
--------------

.. note::
   There is an in-depth tutorial `here <https://blog.rasa.com/building-contextual-assistants-with-rasa-formaction/>`_ about how to use Rasa Forms for slot filling and business logic.

Your AI assistant will often have to follow some pre-defined business logic.
To figure out how to help users, your assistant will often have to ask a few questions.
The answers you get will impact the rest of the conversation; for example, some products might
be limited to users in a certain country or above a certain age. It is good practice to
implement that logic inside a form, separating it from the learned behaviour. A single form
can cover all the happy paths (e.g. all the ways that a user can provide the required information).
You can read more about forms in `this tutorial <https://blog.rasa.com/building-contextual-assistants-with-rasa-formaction/>`_

.. conversations:: 
   :path: _static/elements/completing-tasks/business-logic.yml

See :ref:`conditional-logic` for details on how to use forms to implement business logic.

Contextual Questions
--------------------

Unlike answers to FAQs, correct responses to contextual questions depend on the conversation history.
These include questions which refer to something earlier in the conversation and are ambiguous
on their own.
Real users will often ask questions like "which is better?" and "why?".
It is frustrating for users if your assistant doesn't understand this,
and can only answer full questions like "which of your savings accounts has a better interest rate?"
Understanding contextual questions is a key difference between `level 2 and level 3 assistants <https://www.oreilly.com/ideas/the-next-generation-of-ai-assistants-in-enterprise>`_ .


.. conversations:: 
   :path: _static/elements/completing-tasks/contextual-questions.yml


Unhappy Paths
-------------

When your assistant asks a user for information, you will often get responses other
than the information you asked for. For example, the user might refuse to provide this information,
they might correct something they said earlier, or interrupt with chitchat.
It is important that your assistant can handle these edge cases. There
are so many things a user might say other than provide you the information you asked for,
and a simple interruption shouldn't throw off your assistant completely.
This is a key reason for building an assistant that can learn from real data.

The best way to collect training data for unhappy paths is to use 
:ref:`section_interactive_learning_forms`

.. conversations:: 
   :path: _static/elements/completing-tasks/unhappy-paths.yml


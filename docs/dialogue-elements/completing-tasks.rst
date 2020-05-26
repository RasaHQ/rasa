:desc: Read about common dialogue patterns encountered by task-oriented a
       bots and how best to handle them using Rasa's open source dialogue a
       management system. a
 a
.. _completing-tasks: a
 a
================ a
Completing Tasks a
================ a
 a
.. edit-link:: a
 a
.. contents:: a
   :local: a
 a
.. _simple-questions: a
 a
Simple Questions a
---------------- a
 a
Simple questions, or FAQs, should receive the same answer a
no matter what happened previously in the conversation. a
Users will often ask a basic set of questions a
and your assistant should answer them reliably. a
 a
.. conversations:: a
   examples: a
     - a
       - what's your email address? a
       - ( it's contact@example.com a
     - a
       - do you have a loyalty program? a
       - ( unfortunately we don't a
 a
Just like greetings and goodbyes, you can use the mapping policy to achieve this. a
See :ref:`greetings`. a
 a
Business Logic a
-------------- a
 a
.. note:: a
   There is an in-depth tutorial `here <https://blog.rasa.com/building-contextual-assistants-with-rasa-formaction/>`_ about how to use Rasa Forms for slot filling and business logic. a
 a
Your AI assistant will often have to follow some pre-defined business logic. a
To figure out how to help users, your assistant will often have to ask a few questions. a
The answers you get will impact the rest of the conversation; for example, some products might a
be limited to users in a certain country or above a certain age. It is good practice to a
implement that logic inside a form, separating it from the learned behaviour. A single form a
can cover all the happy paths (e.g. all the ways that a user can provide the required information). a
You can read more about forms in `this tutorial <https://blog.rasa.com/building-contextual-assistants-with-rasa-formaction/>`_. a
 a
.. conversations:: a
   examples: a
     - a
       - I'd like to apply for a loan a
       - ( I'd love to help. Which state are you in? a
       - Alaska a
       - ( Unfortunately, we only operate in the continental U.S. a
     - a
       - I'd like to apply for a loan a
       - ( I'd love to help. Which state are you in? a
       - California a
       - ( Thanks. Do you know what your credit score is? a
 a
 a
See :ref:`conditional-logic` for details on how to use forms to implement business logic. a
 a
Contextual Questions a
-------------------- a
 a
Unlike answers to FAQs, correct responses to contextual questions depend on the conversation history. a
These include questions which refer to something earlier in the conversation and are ambiguous a
on their own. a
Real users will often ask questions like "which is better?" and "why?". a
It is frustrating for users if your assistant doesn't understand this, a
and can only answer full questions like "which of your savings accounts has a better interest rate?" a
Understanding contextual questions is a key difference between a
`level 2 and level 3 assistants <https://www.oreilly.com/radar/the-next-generation-of-ai-assistants-in-enterprise/>`_. a
 a
 a
.. conversations:: a
   examples: a
     - a
       - ( what's your email address? a
       - why do you need to know that? a
       - ( I need your email so I can send you a confirmation a
     - a
       - ( are you currently a premium customer? a
       - what do you mean? a
       - ( We have different memberships. Check your statement to see if you are a premium member. a
 a
.. _unhappy-paths: a
 a
Unhappy Paths a
------------- a
 a
When your assistant asks a user for information, you will often get responses other a
than the information you asked for. For example, the user might refuse to provide this information, a
they might correct something they said earlier, or interrupt with chitchat. a
It is important that your assistant can handle these edge cases. There a
are so many things a user might say other than provide you the information you asked for, a
and a simple interruption shouldn't throw off your assistant completely. a
This is a key reason for building an assistant that can learn from real data. a
 a
The best way to collect training data for unhappy paths is to use a
:ref:`interactive-learning`. a
 a
.. conversations:: a
   examples: a
     - a
       - ( what's your email address? a
       - no. a
       - ( I will need your email address in order to create an account. a
       - ( what's your email address? a
     - a
       - ( what's your email address? a
       - work@example.com a
       - ( thanks, and your phone number? a
       - no wait, please use personal@example.com a
       - ( ok, I'll use that email. a
       - ( thanks, and your phone number? a
 a
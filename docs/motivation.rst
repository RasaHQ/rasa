.. _motivation:

Motivation
==========

Where Rasa Core fits in
-----------------------


Rasa Core takes in structured input: **intents** and **entities**,
**button clicks**, etc., and decides which **action** your bot should run next.
If you want your system to handle free text, you need to also use Rasa NLU or
another NLU tool.


.. image:: _static/images/rasa_system.png


Guiding Principles
------------------

The main idea behind Rasa Core is that thinking of conversations as a flowchart doesn't scale.
It's very hard to reason about *all possible conversations* explicitly, but it's
very easy to tell, mid-conversation, if a response is right or wrong.

The Wrong Way™
--------------
The typical way to implement conversation flows is to use a state machine. 
For example, you might need to collect some data from a user to fulfill their order, and manually
take them through the states ``BROWSING``, ``CHECKING_OUT``, ``ADDING_PAYMENT``, ``ORDER_COMPLETE``, etc.
The complexity comes in when users stray from the "happy path" and you need to add code for handling
questions, clarifications, comparisons, rejections, etc. Manually inserting these edge cases into your
state machine is a big pain.

A typical 'simple' bot might have 5-10 states and a couple of hundred rules governing its behavior.
When your bot doesn't behave as you want it to, it can be very tricky to figure out what went wrong.

Similarly, when you want to add some new functionality, you end up clashing with rules you wrote earlier,
and it gets harder and harder to make progress.

Our view is that taking flowcharts literally and implementing them as a state machine is not a good idea,
but that **flowcharts are still useful for describing happy paths and for
visualising dialogues**.

The Rasa Way
------------

Rather than writing a bunch of ``if/else`` statements, a Rasa bot learns from real conversations. 
A probabilistic model chooses which action to take, and this can be trained using 
supervised, reinforcement, or interactive learning.

The advantages of this approach are that:

 - debugging is easier
 - your bot can be more flexible
 - your bot can improve from experience without writing more code
 - you can add new functionality to your bot without debugging hundreds of rules.


Where to Start
--------------

After going through the :ref:`installation`, most users should start with
:ref:`tutorial_basics`. However, if you already have a bunch of conversations
you'd like to use as a training set, check the :ref:`tutorial_supervised`.

Questions
---------

*Why python?*

    Because of its ecosystem of machine learning tools.
    Head over to :ref:`no_python` for details.

*Is this only for ML experts?*

    You can use Rasa if you don't know anything about machine learning, but if
    you do it's easy to experiment.


*How much training data do I need?*

    You can bootstrap from zero training data by using interactive learning.
    Try the tutorials!

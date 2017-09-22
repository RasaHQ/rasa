.. _tour:

A Quick Tour (no ML)
====================


.. note:: 
   In this tutorial we will build a demo app with a rule-based policy. 
   This is just to show what each of the classes do, and how they fit together.

Here we show how to combine the relevant classes in an application. 
This might be easier to follow if you also look at :ref:`plumbing`.


Define a Domain
^^^^^^^^^^^^^^^

The first thing we need is a ``Domain`` instance. 
The domain specifies the universe of possibilities in which the bot operates. 
You can define a domain as a python class or as a yaml file. 
Below is the restaurant domain from the examples folder.


A Domain defines the following things:


+---------------+------------------------------------------------------------------------------------------------------+
| ``intents``   | things you expect users to say. See Rasa NLU for details.                                            |
+---------------+------------------------------------------------------------------------------------------------------+
| ``entities``  | pieces of info you want to extract from messages. See Rasa NLU for details.                          |
+---------------+------------------------------------------------------------------------------------------------------+
| ``actions``   | things your bot can do and say                                                                       |
+---------------+------------------------------------------------------------------------------------------------------+
| ``slots``     | pieces of info you want to keep track of during a conversation. usually some overlap with entities.  |
+---------------+------------------------------------------------------------------------------------------------------+
| ``templates`` | template strings for the things your bot can say                                                     |
+---------------+------------------------------------------------------------------------------------------------------+




Here is an example of a restaurant domain from the examples folder:


.. literalinclude:: ../examples/restaurant_domain.yml


**How does this fit together?**
Rasa takes the ``intent`` and ``entities`` as input, and returns the ``action`` that should be taken
next.
If the action is just to say something to the user, Rasa will look for a matching template in the domain, fill in
any variables, and respond.
There is one special action, ``ActionListen`` which means to stop taking further actions until the user says something else.
For more actions which do more than just sent a message, you define them as python classes and include them in a file.

You can instantiate your ``Domain`` like this:

.. testcode::

   from rasa_core.domain import TemplateDomain

   domain = TemplateDomain.load("examples/restaurant_domain.yml")



Define an interpreter
^^^^^^^^^^^^^^^^^^^^^

Instead of using Rasa NLU, we will use a dummy interpreter which is helpful for testing. The `RegexInterpreter`'s `parse` method takes string of the format `"_intent[entity1=value1,entity2=value2]"` and returns this same information in the canonical dict format we would get from Rasa NLU.


.. doctest::

   >>> from rasa_core.interpreter import RegexInterpreter
   >>> interpreter = RegexInterpreter()
   >>> result = interpreter.parse("_greet[name=rasa]")
   >>> pp.pprint(result)
   {   u'entities': [   {   u'end': 16,
                            u'entity': u'name',
                            u'start': 12,
                            u'value': u'rasa'}],
       u'intent': {   u'confidence': 1.0, u'name': u'greet'},
       u'intent_ranking': [{   u'confidence': 1.0, u'name': u'greet'}],
       u'text': u'_greet[name=rasa]'}


Define a Policy
^^^^^^^^^^^^^^^

We'll create a really simple, deterministic policy. Again, this tutorial is just to show how the pieces fit together. To create ML-based policies see the other tutorials.

If the user greets the bot, we respond with a greeting, and then listen (i.e. wait for them to say something again). For goodbye we respond with a good bye message and then listen, for any other intent we respond with the default message and then listen. 


.. literalinclude:: ../examples/hello_world/run.py
   :pyobject: SimplePolicy


Put the pieces together
^^^^^^^^^^^^^^^^^^^^^^^

Now we're going to glue some pieces together to create an actual bot. 
We instantiate the policy, and an ``Agent`` instance, which owns an ``interpreter``, a ``policy``, and a ``domain``.

We will pass messages directly to the bot, but this is just for
this is just for demonstration purposes. You can look at how to
build a command line bot and a facebook bot by checking out the examples folder.

.. testsetup::

   from examples.hello_world.run import SimplePolicy, HelloInterpreter

.. testcode::

   from rasa_core.agent import Agent
   from rasa_core.domain import TemplateDomain
   from rasa_core.tracker_store import InMemoryTrackerStore

   default_domain = TemplateDomain.load("examples/default_domain.yml")
   agent = Agent(
      default_domain,
      policies=[SimplePolicy()],
      interpreter=HelloInterpreter(),
      tracker_store=InMemoryTrackerStore(default_domain))


We can then try sending it a message:

.. doctest::

   >>> agent.handle_message("_greet")
   [u'hey there!']

And there we have it! A minimal bot containing all the important pieces of Rasa Core.

If you want to handle input from the command line (or a different input channel) you need handle
that channel instead of handling messages directly, e.g.:

.. code-block:: python

   from rasa_core.channels.console import ConsoleInputChannel
   agent.handle_channel(ConsoleInputChannel())

In this case messages will be retrieved from the command line because we specified
the ``ConsoleInputChannel``. Responses are printed to the command line as well. You
can find a complete example on how to load an agent and chat with it on the
command line in the following python example: ``examples/concerts/run.py``.

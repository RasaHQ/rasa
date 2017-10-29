.. _stories:

Training Data Format
====================

Here's an example from the bAbi data:

.. code-block:: md

   ## story_07715946
   * _greet
      - action_ask_howcanhelp
   * _inform[location=rome,price=cheap]
      - action_on_it
      - action_ask_cuisine
   * _inform[cuisine=spanish]
      - action_ask_numpeople
   * _inform[people=six]
      - action_ack_dosearch


This is what we call a **story**. A story starts with a name preceeded by two
hashes ``## story_03248462``, this is arbitrary but can be used for debugging.


A story starts with a name preceded by two hashes, e.g. ``## story_07715946``.
The choice of name is arbitrary but can be helpful for debugging. The body of
a story is a sequence of events, separated by newlines. An event such as
``* _inform[location=rome,price=cheap]`` is a user utterance, annotated as a
dialogue act.
The general format is ``* _intent[entities]``, where entities are key-value
pairs separated by commas.

The end of a story is denoted by a newline, and then a new story starts again with ``##``.

.. _story-visualization:

Visualization of story training data
------------------------------------
Sometimes it is helpful to get an overview of the conversational paths that are described within a story file. To make
debugging easier and to ease discussions about bot flows, you can visualize the content of a story file. For this to
work, you need to install graphviz. These are the instructions to do that on OSX, for other systems the instructions
might be slightly different:

.. code-block:: bash

   brew install graphviz
   pip install pygraphviz --install-option="--include-path=/usr/include/graphviz" --install-option="--library-path=/usr/lib/graphviz/"

As soon as this is installed you can visualize stories like this:

..  code-block:: bash

   python -m rasa_core.visualize -d examples/concerts/data/stories.md -o graph.png

This will run through the stories in ``examples/concerts/data/stories.md`` and create a graph stored in the
output image ``graph.png``.

.. image:: _static/images/concert_stories.png

You can also call the visualization directly from your code using:

.. testcode::

   from rasa_core.training_utils.dsl import StoryFileReader
   from rasa_core.domain import TemplateDomain
   from rasa_core.training_utils.visualization import visualize_stories
   from rasa_nlu.converters import load_data

   domain = TemplateDomain.load("examples/restaurant_domain.yml")
   stories = StoryFileReader.read_from_file("examples/babi/data/babi_task5_dev_rasa_even_smaller.md",
                                                domain)

   training_data = load_data("examples/babi/data/babi_dialog_nlu.json")
   visualize_stories(stories, "graph.png", training_data=training_data)

Which will create the same image as the above python script call. The shown graph is still very simple, but the graphs can get quite complex.

If you want to replace the intent & entity messages in the box with real world messages
(if you want to replace the messages from the stories file, which
usually look like ``_greet`` with real messages e.g. ``hello``), you can pass in a Rasa
NLU training data instance to replace them with messages from your training data, as we
did in the above example with the ``training_data`` variable.

.. _evaluation:

Evaluating Models
=================


.. _story-visualization:

Visualization of Stories
------------------------
Sometimes it is helpful to get an overview of the conversational paths that
are described within a story file. To make debugging easier and to ease
discussions about bot flows, you can visualize the content of a story file.

.. note::
   For this to
   work, you need to **install graphviz**. These are the instructions to do that
   on OSX, for other systems the instructions might be slightly different:

   .. code-block:: bash

      brew install graphviz
      pip install pygraphviz --install-option="--include-path=/usr/include/graphviz" \
        --install-option="--library-path=/usr/lib/graphviz/"

As soon as this is installed you can visualize stories like this:

..  code-block:: bash

   cd examples/concertbot/
   python -m rasa_core.visualize -d concert_domain.yml -s data/stories.md -o graph.png

This will run through the stories of the ``concertbot`` example in
``data/stories.md`` and create a graph stored in the
output image ``graph.png``.

.. image:: _static/images/concert_stories.png

We can also run the visualisation directly from code. For this example, we can
create a ``visualize.py`` in ``examples/concertbot`` with the following code:

.. literalinclude:: ../examples/concertbot/visualize.py

Which will create the same image as the above python script call. The shown
graph is still very simple, but the graphs can get quite complex.

If you want to replace the messages from the stories file, which usually look
like ``greet`` with real messages e.g. ``Hello``, you can pass in a Rasa
NLU training data instance to replace them with messages from your training
data.

.. note::

   The story visualization needs to load your domain. If you have
   any custom actions written in python make sure they are part of the python
   path, and can be loaded by the visualization script using the module path
   given for the action in the domain (e.g. ``actions.ActionSearchVenues``).


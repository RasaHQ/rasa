Contributing
============

Contributions are very much encouraged! Please create an issue before doing any work to avoid disappointment.

We created a tag that should get you started quickly if you are searching for
`interesting topics to get started <https://github.com/RasaHQ/rasa_nlu/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22>`_.


Python Conventions
^^^^^^^^^^^^^^^^^^

Python code should follow the pep-8 spec.

Python 2 and 3 Cross Compatibility
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To ensure cross compatibility between Python 2 and 3 we prioritize Python 3 conventions.
Keep in mind that:

- all string literals are unicode strings
- division generates floating point numbers. Use ``//`` for truncated division
- some built-ins, e.g. ``map`` and ``filter`` return iterators in Python 3. If you want to make use of them import the Python 3 version of them from ``builtins``. Otherwise use list comprehensions, which work uniformly across versions
- use ``io.open`` instead of the builtin ``open`` when working with files
- The following imports from ``__future__`` are mandatory in every python file: ``unicode_literals``, ``print_function``, ``division``, and ``absolute_import``

Please refer to this `cheat sheet <http://python-future.org/compatible_idioms.html#>`_ to learn how to write different constructs compatible with Python 2 and 3.

Code of conduct
^^^^^^^^^^^^^^^

rasa NLU adheres to the `Contributor Covenant Code of Conduct <http://contributor-covenant.org/version/1/4/>`_.
By participating, you are expected to uphold this code.

Documentation
^^^^^^^^^^^^^
Everything should be properly documented. To locally test the documentation you need to install

.. code-block:: bash

    brew install sphinx
    pip install sphinx_rtd_theme

After that, you can compile and view the documentation using:

.. code-block:: bash

    cd docs
    make html
    cd _build/html
    python -m SimpleHTTPServer 8000 .
    # python 3: python -m http.server

The documentation will be running on http://localhost:8000/.

Code snippets that are part of the documentation can be tested using

.. code-block:: bash

    make doctest

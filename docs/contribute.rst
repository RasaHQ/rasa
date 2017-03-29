Contributing
============

Contributions are very much encouraged! Please create an issue before doing any work to avoid disappointment.

We created a tag that should get you started quickly if you are searching for
`interesting topics to get started <https://github.com/golastmile/rasa_nlu/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22>`_.


Python Conventions
^^^^^^^^^^^^^^^^^^

Python code should follow the pep-8 spec. 

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

The documentation will be running on http://localhost:8000/.

Code snippets that are part of the documentation can be tested using

.. code-block:: bash

    make doctest

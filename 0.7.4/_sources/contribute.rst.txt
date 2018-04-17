Contributing
============

Contributions are very much encouraged!
Please create an issue before doing any work to avoid disappointment.


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
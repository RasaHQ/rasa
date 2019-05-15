:desc: Add new endpoints to the configuration file of Rasa NLU to connect
       your API's to integrate with open source NLU.

.. _section_endpoint_configuration:

Endpoint Configuration
======================

To connect NLU to other endpoints, you can specify an endpoint configuration
within a `YAML <https://en.wikipedia.org/wiki/YAML>`_ file.
Then run Rasa NLU with the flag
``--endpoints <path to endpoint configuration.yml``.

For example:

.. code-block:: bash

    rasa run \
        --path <working directory of the server> \
        --endpoints <path to endpoint configuration>.yml

.. note::

    You can use environment variables within configuration files
    by specifying them with ``${name of environment variable}``.
    These placeholders are then replaced by the value of the environment
    variable.

Model Server
------------

To use models from a model server, add this to your endpoint configuration:

.. code-block:: yaml

    model:
        url: <path to your model>
        token: <authentication token>   # [optional]
        token_name: <name of the token  # [optional] (default: token)

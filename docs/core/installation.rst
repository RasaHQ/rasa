:desc: Join our community by installing open source Rasa Core for machine
       learning based dialogue management to develop on premise contextual
       assistants and chatbots. 

.. _installation:

Installation
============

.. contents::

Install Rasa Core to get started with the Rasa stack.

.. note::

    You can also get started without installing anything by going
    to the :ref:`quickstart`


Install Rasa Core
-----------------

Stable (Recommended)
~~~~~~~~~~~~~~~~~~~~

The recommended way to install Rasa Core is using pip which will install the latest stable release of Rasa Core:

.. copyable::

    pip install -U rasa

Unless you've already got numpy & scipy installed, we highly recommend
that you install and use `Anaconda <https://www.anaconda.com/download/>`_.

.. note::

    If you want to run custom action code, please also take a look at
    :ref:`customactions`. You'll need to install Rasa Core to train and
    use the model and ``rasa_core_sdk`` to develop your custom action code.


Latest (Most recent github)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you want to use the bleeding edge version of Rasa use github + setup.py:

.. code-block:: bash

    git clone https://github.com/RasaHQ/rasa.git
    cd rasa
    pip install -r requirements.txt
    pip install -e .

Development (github & development dependencies)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to change the Rasa Core code and want to run the tests or
build the documentation, you need to install the development dependencies:

.. code-block:: bash

    pip install -r requirements-dev.txt
    pip install -e .


Add Natural Language Understanding
----------------------------------

We use Rasa NLU for intent classification & entity extraction. To get it, run:

.. code-block:: bash

    pip install rasa[tensorflow]

Full instructions can be found
`in the NLU documentation <https://rasa.com/docs/nlu/installation/>`_.

You can also use other NLU services like wit.ai, dialogflow, or LUIS.
In fact, you don't need to use NLU at all, if your messaging app uses buttons
rather than free text.


Build your first Rasa assistant!
--------------------------------
After following the quickstart and installing Rasa Core, the next step is to
build your first Rasa assistant yourself! To get you started, we have prepared a
Rasa Stack starter-pack which has all the files you need to build your first custom
chatbot. On top of that, the starter-pack includes a training data set ready
for you to use.

Click the link below to get the Rasa Stack starter-pack:

`Rasa Stack starter-pack <https://github.com/RasaHQ/starter-pack-rasa-stack>`_

Let us know how you are getting on! If you have any questions about the starter-pack or
using Rasa Stack in general, post your questions on `Rasa Community Forum <https://forum.rasa.com>`_!


Using Docker Compose
--------------------
Rasa provides all components as official Docker images which are continuously updated.
To quickly run Rasa Core with other components,
you can use the provided `docker compose <https://docs.docker.com/compose/overview/>`_ file.
This is useful for a quick local setup or if you want to host the Rasa components on cloud services.


Compose File Example
~~~~~~~~~~~~~~~~~~~~
.. literalinclude:: ../../docker/docker-compose.yml


.. note::

    If you do not require components like `nlu <https://rasa.com/docs/nlu/>`_ or duckling,
    you can simply remove them from your docker compose file.

Running it
~~~~~~~~~~
To run all components locally, execute :code:`docker-compose up`.
You can then interact with your chat bot using the :ref:`section_http`.
For example:

    .. code-block:: bash

        curl -XPOST \
            --header 'content-type: application/json' \
            --data '{"message": "Hi Bot"}' \
            http://localhost:5005/webhooks/rest/webhook

To run commands inside a specific container, use ``docker-compose run <container name>``.
For example to train the core model:

    .. code-block:: bash

         docker-compose run rasa train

Volume Explanation
~~~~~~~~~~~~~~~~~~
- **./rasa-app-data/models/current/dialogue**: This directory contains the trained Rasa Core models.
  You can also move previously trained models to this directory to load them within the Docker container.
- **./rasa-app-data/config**: This directory is for the configuration of the endpoints and of the
  different :ref:`connectors` you can use Rasa Core with.

  - To connect other components with Rasa Core this directory should contain a file ``endpoints.yml``,
    which specifies how to reach these components.
    For the shown docker-compose example the file should look like this:

        .. code-block:: yaml

            action_endpoint:
                url: 'http://action_server:5055/webhook'
            nlu:
                url: 'http://rasa_nlu:5000'

  - If you use connectors to :ref:`connectors`
    you have to configure the required credentials for these in a file `credentials.yml`.
    Use the provided credentials by adding ``--credentials <path to your credentials file>``
    to the run command of Rasa Core.

- **./rasa-app-data/project**: This directory contains your Rasa project and may be used to train a model.
- **./rasa-app-data/models/**: This directory contains the nlu project and its trained models.
  You can also move previously trained models to this directory to load them within the Docker container.

.. note::

    You can also use custom directory structures or port mappings.
    But don't forget to reflect this changes in the docker compose file and in your endpoint configuration.

.. include:: feedback.inc

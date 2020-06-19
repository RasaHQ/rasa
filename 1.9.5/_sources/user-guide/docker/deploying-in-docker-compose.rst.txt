:desc: Use Docker Compose to deploy a Rasa Open Source assistant

.. _deploying-rasa-in-docker-compose:

Deploying a Rasa Open Source Assistant in Docker Compose
========================================================

If you would like to deploy your assistant without Rasa X, you can do so by deploying it in Docker Compose.
To deploy Rasa X and your assistant together, see the :ref:`recommended-deployment-methods`.

.. contents::
   :local:
   :depth: 1


Installing Docker
~~~~~~~~~~~~~~~~~

If you're not sure if you have Docker installed, you can check by running:

  .. code-block:: bash

    docker -v && docker-compose -v
    # Docker version 18.09.2, build 6247962
    # docker-compose version 1.23.2, build 1110ad01

If Docker is installed on your machine, the output should show you your installed
versions of Docker and Docker Compose. If the command doesn't work, you'll have to
install Docker.
See `Docker Installation <https://docs.docker.com/install/>`_ for details.


.. _docker-compose-configuring-channels:

Configuring Channels
~~~~~~~~~~~~~~~~~~~~

To run your AI assistant in production, don't forget to configure your required
:ref:`messaging-and-voice-channels` in ``credentials.yml``. For example, to add a
REST channel, uncomment this section in the ``credentials.yml``:

  .. code-block:: yaml

    rest:
      # you don't need to provide anything here - this channel doesn't
      # require any credentials

The REST channel will open your bot up to incoming requests at the ``/webhooks/rest/webhook`` endpoint.


Using Docker Compose to Run Multiple Services
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Docker Compose provides an easy way to run multiple containers together without
having to run multiple commands or configure networks. This is essential when you
want to deploy an assistant that also has an action server.

.. contents::
   :local:
   :depth: 2

Start by creating a file called ``docker-compose.yml``:

      .. code-block:: bash

        touch docker-compose.yml

Add the following content to the file:

      .. parsed-literal::

        version: '3.0'
        services:
          rasa:
            image: rasa/rasa:\ |release|-full
            ports:
              - 5005:5005
            volumes:
              - ./:/app
            command:
              - run

The file starts with the version of the Docker Compose specification that you
want to use.
Each container is declared as a ``service`` within the ``docker-compose.yml``.
The first service is the ``rasa`` service, which runs your Rasa server.

To add the action server, add the image of your action server code. To learn how to deploy
an action server image, see :ref:`building-an-action-server-image`.

   .. parsed-literal::

      version: '3.0'
      services:
        rasa:
          image: rasa/rasa:\ |release|-full
          ports:
            - 5005:5005
          volumes:
            - ./:/app
          command:
            - run
        app:
          image: <your action server image>
          expose: 5055

The ``expose: 5005`` is what allows the ``rasa`` service to reach the ``app`` service on that port.
To instruct the ``rasa`` service to send its action requests to that endpoint, add it to your ``endpoints.yml``:

      .. code-block:: yaml

        action_endpoint:
          url: http://app:5055/webhook

To run the services configured in your ``docker-compose.yml`` execute:

   .. code-block:: bash

       docker-compose up

You should then be able to interact with your bot via requests to port 5005, on the webhook endpoint that
corresponds to a :ref:`configured channel <docker-compose-configuring-channels>`:

   .. code-block:: bash

     curl -XPOST http://localhost:5005/webhooks/rest/webhook \
       -H "Content-type: application/json" \
       -d '{"sender": "test", "message": "hello"}'

.. _building-an-action-server-image:

Building an Action Server Image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you build an image that includes your action code and store it in a container registry, you can run it locally
or as part of your deployment, without having to move code between servers.
In addition, you can add any additional dependencies of systems or Python libraries
that are part of your action code but not included in the base ``rasa/rasa-sdk`` image.

This documentation assumes you are pushing your images to `DockerHub <https://hub.docker.com/>`_.
DockerHub will let you host multiple public repositories and
one private repository for free. Be sure to first `create an account <https://hub.docker.com/signup/>`_
and `create a repository <https://hub.docker.com/signup/>`_ to store your images. You could also push images to
a different Docker registry, such as `Google Container Registry <https://cloud.google.com/container-registry>`_,
`Amazon Elastic Container Registry <https://aws.amazon.com/ecr/>`_, or
`Azure Container Registry <https://azure.microsoft.com/en-us/services/container-registry/>`_.

To create your image:

  #. Move your actions code to a folder ``actions`` in your project directory.
     Make sure to also add an empty ``actions/__init__.py`` file:

      .. code-block:: bash

          mkdir actions
          mv actions.py actions/actions.py
          touch actions/__init__.py  # the init file indicates actions.py is a python module
          
     The ``rasa/rasa-sdk`` image will automatically look for the actions in ``actions/actions.py``.

  #. If your actions have any extra dependencies, create a list of them in a file,
     ``actions/requirements-actions.txt``.
  #. Create a file named ``Dockerfile`` in your project directory,
     in which you'll extend the official SDK image, copy over your code, and add any custom dependencies (if necessary).
     For example:

      .. parsed-literal::

         # Extend the official Rasa SDK image
         FROM rasa/rasa-sdk:\ |version|.0

         # Use subdirectory as working directory
         WORKDIR /app

         # Copy any additional custom requirements
         COPY actions/requirements-actions.txt ./

         # Change back to root user to install dependencies
         USER root

         # Install extra requirements for actions code, if necessary (otherwise comment this out)
         RUN pip install -r requirements-actions.txt

         # Copy actions folder to working directory
         COPY ./actions /app/actions

         # By best practices, don't run the code with root user
         USER 1001

You can then build the image via the following command:

      .. code-block:: bash

        docker build . -t <account_username>/<repository_name>:<custom_image_tag>

The ``<custom_image_tag>`` should reference how this image will be different from others. For
example, you could version or date your tags, as well as create different tags that have different code for production
and development servers. You should create a new tag any time you update your code and want to re-deploy it.

If you are using Docker Compose locally, you can use this image directly in your
``docker-compose.yml``:

      .. code-block:: yaml

         version: '3.0'
         services:
           app:
             image: <account_username>/<repository_name>:<custom_image_tag>

If you're building this image to make it available from another server,
for example a Rasa X or Rasa Enterprise deployment, you should push the image to a cloud repository.
You can push the image to DockerHub via:

      .. code-block:: bash

        docker login --username <account_username> --password <account_password>
        docker push <account_username>/<repository_name>:<custom_image_tag>

To authenticate and push images to a different container registry, please refer to the documentation of
your chosen container registry.

Then, reference the new image tag in your ``docker-compose.override.yml``:

      .. code-block:: yaml

         version: '3.0'
         services:
           app:
             image: <account_username>/<repository_name>:<custom_image_tag>

Configuring a Tracker Store
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, all conversations are saved in memory. This means that all
conversations are lost as soon as you restart the Rasa server.
If you want to persist your conversations, you can use a different
:ref:`Tracker Store <tracker-stores>`.

To add a tracker store to a Docker Compose deployment, you need to add a new
service to your ``docker-compose.yml`` and modify the ``endpoints.yml`` to add
the new tracker store, pointing to your new service. More information about how
to do so can be found in the tracker store documentation:

  - :ref:`sql-tracker-store`
  - :ref:`redis-tracker-store`
  - :ref:`mongo-tracker-store`
  - :ref:`custom-tracker-store`

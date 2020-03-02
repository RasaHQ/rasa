.. _deploying-rasa-in-docker-compose:

Deploying a Rasa Assistant in Docker-Compose
============================================

.. contents::
   :local:
   :depth: 2

.. _running-the-rasa-server:


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


Running the Rasa Server
~~~~~~~~~~~~~~~~~~~~~~~

To run your AI assistant in production, configure your required
:ref:`messaging-and-voice-channels` in ``credentials.yml``. If this file does not
exist, create it using:

.. code-block:: bash

  touch credentials.yml

Then edit it according to your connected channels.
After, run the trained model with:

.. code-block:: bash

  docker run \
    -v $(pwd)/models:/app/models \
    rasa/rasa:latest-full \
    run

Command Description:

  - ``-v $(pwd)/models:/app/models``: Mounts the directory with the trained Rasa model
    in the container
  - ``rasa/rasa:latest-full``: Use the Rasa image with the tag ``latest-full``
  - ``run``: Executes the ``rasa run`` command. For more information see
    :ref:`command-line-interface`.


Using Docker Compose to Run Multiple Services
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To run Rasa together with other services, such as a server for custom actions, it is
recommend to use `Docker Compose <https://docs.docker.com/compose/>`_.
Docker Compose provides an easy way to run multiple containers together without
having to run multiple commands.

.. contents::
   :local:
   :depth: 2

Start by creating a file called ``docker-compose.yml``:

.. code-block:: bash

  touch docker-compose.yml

Add the following content to the file:

.. code-block:: yaml

  version: '3.0'
  services:
    rasa:
      image: rasa/rasa:latest-full
      ports:
        - 5005:5005
      volumes:
        - ./:/app
      command:
        - run


The file starts with the version of the Docker Compose specification that you
want to use.
Each container is declared as a ``service`` within the docker-compose file.
The first service is the ``rasa`` service.

The command is similar to the ``docker run`` command.
The ``ports`` part defines a port mapping between the container and your host
system. In this case it makes ``5005`` of the ``rasa`` service available on
port ``5005`` of your host.
This is the port of the :ref:`REST Channel <rest_channels>` interface of Rasa.

.. note::

    Since Docker Compose starts a set of Docker containers, it is no longer
    possible to connect to the command line of a single container after executing the
    ``run`` command.

To run the services configured in your ``docker-compose.yml`` execute:

.. code-block:: bash

    docker-compose up




.. _building-an-action-server-image:

Deploying an Action Server Image
********************************

If you build an image that includes your action code and store it in a container registry, you can run it locally
or as part of your Rasa X or Rasa Enterprise deployment, without having to move code between servers.
In addition, you can add any additional dependencies of systems or Python libraries
that are part of your action code but not included in the base ``rasa/rasa-sdk`` image.

This documentation assumes you are pushing your images to `DockerHub <https://hub.docker.com/>`_.
DockerHub will let you host multiple public repositories and
one private repository for free. Be sure to first `create an account <https://hub.docker.com/signup/>`_
and `create a repository <https://hub.docker.com/signup/>`_ to store your images. You could also push images to
a different Docker registry, such as `Google Container Registry <https://cloud.google.com/container-registry>`_,
`Amazon Elastic Container Registry <https://aws.amazon.com/ecr/>`_, or
`Azure Container Registry <https://azure.microsoft.com/en-us/services/container-registry/>`_.

To create your image, first create a list of your custom actions requirements in a file,
``actions/requirements-actions.txt``. Then create a file named ``Dockerfile`` in your project directory,
in which you'll extend the official SDK image, copy over your code, and add any custom dependencies. For example:

.. code-block:: docker

   # Extend the official Rasa SDK image
   FROM rasa/rasa-sdk:latest

   # Use subdirectory as working directory
   WORKDIR /app

   # Copy any additional custom requirements
   COPY actions/requirements-actions.txt ./

   # Install extra requirements for actions code
   RUN pip install -r requirements-actions.txt

   # Copy actions code to working directory
   COPY ./actions /app/actions


You can then build the image via the following command:

.. code-block:: bash

  docker build . -t <account_username>/<repository_name>:<custom_image_tag>

The ``<custom_image_tag>`` should reference how this image will be different from others. For
example, you could version or date your tags, as well as create different tags that have different code for production
and development servers. You should create a new tag any time you update your code and want to re-deploy it.

If you are using docker-compose locally, you can use this image directly in your
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

Adding a Custom Tracker Store
*****************************

By default, all conversations are saved in memory. This means that all
conversations are lost as soon as you restart the Rasa server.
If you want to persist your conversations, you can use a different
:ref:`Tracker Store <tracker-stores>`.

.. contents::
   :local:
   :depth: 2

Using PostgreSQL as Tracker Store
#################################

Start by adding PostgreSQL to your docker-compose file:

.. code-block:: yaml

  postgres:
    image: postgres:latest

Then add PostgreSQL to the ``tracker_store`` section of your endpoint
configuration ``config/endpoints.yml``:

.. code-block:: yaml

  tracker_store:
    type: sql
    dialect: "postgresql"
    url: postgres
    db: rasa

Using MongoDB as Tracker Store
##############################

Start by adding MongoDB to your docker-compose file. The following example
adds the MongoDB as well as a UI (you can skip this), which will be available
at ``localhost:8081``. Username and password for the MongoDB instance are
specified as ``rasa`` and ``example``.

.. code-block:: yaml

  mongo:
    image: mongo
    environment:
      MONGO_INITDB_ROOT_USERNAME: rasa
      MONGO_INITDB_ROOT_PASSWORD: example
  mongo-express:
    image: mongo-express
    ports:
      - 8081:8081
    environment:
      ME_CONFIG_MONGODB_ADMINUSERNAME: rasa
      ME_CONFIG_MONGODB_ADMINPASSWORD: example

Then add the MongoDB to the ``tracker_store`` section of your endpoints
configuration ``endpoints.yml``:

.. code-block:: yaml

  tracker_store:
    type: mongod
    url: mongodb://mongo:27017
    username: rasa
    password: example

Then start all components with ``docker-compose up``.

Using Redis as Tracker Store
##############################

Start by adding Redis to your docker-compose file:

.. code-block:: yaml

  redis:
    image: redis:latest

Then add Redis to the ``tracker_store`` section of your endpoint
configuration ``endpoints.yml``:

.. code-block:: yaml

  tracker_store:
    type: redis
    url: redis

Using a Custom Tracker Store Implementation
###########################################

If you have a custom implementation of a tracker store you have two options
to add this store to Rasa:

  - extending the Rasa image
  - mounting it as volume

Then add the required configuration to your endpoint configuration
``endpoints.yml`` as it is described in :ref:`tracker-stores`.
If you want the tracker store component (e.g. a certain database) to be part
of your Docker Compose file, add a corresponding service and configuration
there.

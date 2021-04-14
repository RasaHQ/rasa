:desc: How to deploy your Rasa Assistant with Docker Compose or Kubernetes/Openshift

.. _deploying-your-rasa-assistant:

Deploying Your Rasa Assistant
=============================

.. edit-link::

This page explains when and how to deploy an assistant built with Rasa.
It will allow you to make your assistant available to users and set you up with a production-ready environment.

.. contents::
   :local:
   :depth: 2


When to Deploy Your Assistant
-----------------------------

The best time to deploy your assistant and make it available to test users is once it can handle the most
important happy paths or is what we call a `minimum viable assistant <https://rasa.com/docs/rasa/glossary>`_.

The recommended deployment methods described below make it easy to share your assistant
with test users via the `share your assistant feature in
Rasa X <https://rasa.com/docs/rasa-x/user-guide/share-assistant/#share-your-bot>`_.
Then, when you’re ready to make your assistant available via one or more :ref:`messaging-and-voice-channels`,
you can easily add them to your existing deployment set up.

.. _recommended-deployment-methods:

Recommended Deployment Methods
------------------------------

The recommended way to deploy an assistant is using either the Server Quick-Install or Helm Chart
options we support. Both deploy Rasa X and your assistant. They are the easiest ways to deploy your assistant,
allow you to use Rasa X to view conversations and turn them into training data, and are production-ready.
For more details on deployment methods see the `Rasa X Installation Guide <https://rasa.com/docs/rasa-x/installation-and-setup/installation-guide/>`_.

Server Quick-Install
~~~~~~~~~~~~~~~~~~~~

The Server Quick-Install script is the easiest way to deploy Rasa X and your assistant. It installs a Kubernetes
cluster on your machine with sensible defaults, getting you up and running in one command.

    - Default: Make sure you meet the `OS Requirements <https://rasa.com/docs/rasa-x/installation-and-setup/install/quick-install-script/#hardware-os-requirements>`_,
      then run:

      .. copyable::

         curl -s get-rasa-x.rasa.com | sudo bash

    - Custom: See `Customizing the Script <https://rasa.com/docs/rasa-x/installation-and-setup/customize/#server-quick-install>`_
      and the `Server Quick-Install docs <https://rasa.com/docs/rasa-x/installation-and-setup/install/quick-install-script>`_ docs.

Helm Chart
~~~~~~~~~~

For assistants that will receive a lot of user traffic, setting up a Kubernetes or Openshift deployment via
our Helm charts is the best option. This provides a scalable architecture that is also straightforward to deploy.
However, you can also customize the Helm charts if you have specific requirements.

    - Default: Read the `Helm Chart Installation <https://rasa.com/docs/rasa-x/installation-and-setup/install/helm-chart/>`_ docs.
    - Custom: Read the above, as well as the `Advanced Configuration <https://rasa.com/docs/rasa-x/installation-and-setup/customize/#helm-chart>`_
      documentation, and customize the `open source Helm charts <https://github.com/RasaHQ/rasa-x-helm>`_ to your needs.

.. _rasa-only-deployment:

Alternative Deployment Methods
------------------------------

Docker Compose
~~~~~~~~~~~~~~

You can also run Rasa X in a Docker Compose setup, without the cluster environment. We have an install script
for doing so, as well as manual instructions for any custom setups.

    - Default: Read the `Docker Compose Install Script <https://rasa.com/docs/rasa-x/installation-and-setup/install/docker-compose/#docker-compose-install-script>`_ docs or watch the `Masterclass Video <https://www.youtube.com/watch?v=IUYdwy8HPVc>`_ on deploying Rasa X.
    - Custom: Read the `Docker Compose Manual Install <https://rasa.com/docs/rasa-x/installation-and-setup/install/docker-compose/#docker-compose-manual-install>`_ documentation for full customization options.

Rasa Open Source Only Deployment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is also possible to deploy a Rasa assistant without Rasa X using Docker Compose. To do so, you can build your
Rasa Assistant locally or in Docker. Then you can deploy your model in Docker Compose.

.. toctree::
   :titlesonly:
   :maxdepth: 1

   Building a Rasa Assistant Locally <rasa-tutorial>
   docker/building-in-docker
   docker/deploying-in-docker-compose


Deploying Your Action Server
----------------------------

.. _building-an-action-server-image:

Building an Action Server Image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you build an image that includes your action code and store it in a container registry, you can run it
as part of your deployment, without having to move code between servers.
In addition, you can add any additional dependencies of systems or Python libraries
that are part of your action code but not included in the base ``rasa/rasa-sdk`` image.

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
         FROM rasa/rasa-sdk:|rasa_sdk_version|

         # Use subdirectory as working directory
         WORKDIR /app

         # Copy any additional custom requirements, if necessary (uncomment next line)
         # COPY actions/requirements-actions.txt ./

         # Change back to root user to install dependencies
         USER root

         # Install extra requirements for actions code, if necessary (uncomment next line)
         # RUN pip install -r requirements-actions.txt

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


Using your Custom Action Server Image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you're building this image to make it available from another server,
for example a Rasa X or Rasa Enterprise deployment, you should push the image to a cloud repository.

This documentation assumes you are pushing your images to `DockerHub <https://hub.docker.com/>`_.
DockerHub will let you host multiple public repositories and
one private repository for free. Be sure to first `create an account <https://hub.docker.com/signup/>`_
and `create a repository <https://hub.docker.com/signup/>`_ to store your images. You could also push images to
a different Docker registry, such as `Google Container Registry <https://cloud.google.com/container-registry>`_,
`Amazon Elastic Container Registry <https://aws.amazon.com/ecr/>`_, or
`Azure Container Registry <https://azure.microsoft.com/en-us/services/container-registry/>`_.

You can push the image to DockerHub via:

      .. code-block:: bash

        docker login --username <account_username> --password <account_password>
        docker push <account_username>/<repository_name>:<custom_image_tag>

To authenticate and push images to a different container registry, please refer to the documentation of
your chosen container registry.

How you reference the custom action image will depend on your deployment. Pick the relevant documentation for
your deployment:

    - `Server Quick-Install <https://rasa.com/docs/rasa-x/installation-and-setup/customize/#quick-install-script-customizing>`_
    - `Helm Chart <https://rasa.com/docs/rasa-x/installation-and-setup/customize/#adding-a-custom-action-server>`_
    - `Docker Compose <https://rasa.com/docs/rasa-x/installation-and-setup/customize/#connecting-a-custom-action-server>`_
    - :ref:`Rasa Open Source Only <running-multiple-services>`

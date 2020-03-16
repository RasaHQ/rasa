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

.. raw:: html

    The best time to deploy your assistant and make it available to test users is once it can handle the most
    important happy paths or is what we call a <a style="text-decoration: none"
    href="https://rasa.com/docs/rasa/glossary">minimum viable assistant</a>.

The recommended deployment methods described below make it easy to share your assistant
with test users via the `share your assistant feature in
Rasa X <https://rasa.com/docs/rasa-x/user-guide/enable-workflows#conversations-with-test-users>`_.
Then, when you’re ready to make your assistant available via one or more :ref:`messaging-and-voice-channels`,
you can easily add them to your existing deployment set up.

.. _recommended-deployment-methods:

Recommended Deployment Methods
------------------------------

The recommended way to deploy an assistant is using either the One-Line Deployment or Kubernetes/Openshift
options we support. Both deploy Rasa X and your assistant. They are the easiest ways to deploy your assistant,
allow you to use Rasa X to view conversations and turn them into training data, and are production-ready.

One-Line Deploy Script
~~~~~~~~~~~~~~~~~~~~~~

The one-line deployment script is the easiest way to deploy Rasa X and your assistant. It installs a Kubernetes
cluster on your machine with sensible defaults, getting you up and running in one command.

    - Default: Make sure you meet the `OS Requirements <https://rasa.com/docs/rasa-x/installation-and-setup/one-line-deploy-script/#hardware-os-requirements>`_,
      then run:

      .. copyable::

         curl -s get-rasa-x.rasa.com | sudo bash

    - Custom: See `Customizing the Script <https://rasa.com/docs/rasa-x/installation-and-setup/one-line-deploy-script/#customizing-the-script>`_
      in the `One-Line Deploy Script <https://rasa.com/docs/rasa-x/installation-and-setup/one-line-deploy-script/#customizing-the-script>`_ docs.

Kubernetes/Openshift
~~~~~~~~~~~~~~~~~~~~

For assistants that will receive a lot of user traffic, setting up a Kubernetes or Openshift deployment via
our helm charts is the best option. This provides a scalable architecture that is also straightforward to deploy.
However, you can also customize the Helm charts if you have specific requirements.

    - Default: Read the `Deploying in Openshift or Kubernetes <https://rasa.com/docs/rasa-x/installation-and-setup/openshift-kubernetes/>`_ docs.
    - Custom: Read the above, as well as the `Advanced Configuration <https://rasa.com/docs/rasa-x/installation-and-setup/openshift-kubernetes/#advanced-configuration>`_
      documentation, and customize the `open source Helm charts <https://github.com/RasaHQ/rasa-x-helm>`_ to your needs.

.. _rasa-only-deployment:

Alternative Deployment
----------------------

Docker-Compose
~~~~~~~~~~~~~~

You can also run Rasa X in a Docker-Compose setup, without the cluster environment. We have a quick install script
for doing so, as well as manual instructions for any custom setups.

    - Default: Watch the `Masterclass Video <https://www.youtube.com/watch?v=IUYdwy8HPVc>`_ on deploying Rasa X or read the `Docker-Compose Quick Install <https://rasa.com/docs/rasa-x/installation-and-setup/docker-compose-script/>`_ docs.
    - Custom: Read the docs `Docker-Compose Manual Install <https://rasa.com/docs/rasa-x/installation-and-setup/docker-compose-manual/>`_ documentation for full customization options.

Rasa-Only Deployment
~~~~~~~~~~~~~~~~~~~~

It is also possible to deploy a Rasa assistant using Docker-Compose without Rasa X. To do so, you can build your
Rasa Assistant locally or in Docker. Then you can deploy your model in Docker-Compose.

.. toctree::
   :titlesonly:
   :maxdepth: 1

   Building a Rasa Assistant Locally <rasa-tutorial>
   docker/building-in-docker
   docker/deploying-in-docker-compose

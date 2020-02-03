:desc: How to deploy your Rasa Assistant with Docker or Kubernetes

.. _deploying-your-rasa-assistant:

Deploying your Rasa Assistant
===============================

.. edit-link::

This page explains when and how to deploy an assistant built with Rasa. 
It will allow you to make your assistant available to users and set you up with a production-ready environment.
If you haven't used Rasa before, we'd recommend that you start with the :ref:`rasa-tutorial`.

.. contents::
   :local:


When to deploy your assistant
--------------------------------

The best time to deploy your assistant and make it available to test users is once it can handle the most important happy paths, or is what we call a "minimum viable assistant".

The recommended deployment methods described below make it easy to share your assistant with test users via the `share your bot feature in Rasa X <../../rasa-x/docs/user-guide/enable-workflows#conversations-with-test-users>`_, even before selecting which channel you want to use. Then, when you’re ready to make your assistant available via :ref:`messaging-and-voice-channels`, you can easily add them to your existing deployment set up.

.. _recommended-deployment-methods:

Recommended Deployment Methods
------------------------------

The recommended way to deploy an assistant is using either the Docker-Compose or Kubernetes options we support. Both deploy Rasa X and your assistant. They are the easiest ways to deploy your assistant, allow you to use Rasa X to view conversations and turn them into training data, and are production-ready.

Kubernetes
~~~~~~~~~~

Kubernetes is the best option if you plan to serve many users. It's straightforward to deploy if you use the helm charts we provide. However, you can also customize the Helm charts if you have specific requirements.

    - Default: Read the docs `here <../../rasa-x/docs/installation-and-setup/openshift-kubernetes/>`_.
    - Custom: Read the docs `here <../../rasa-x/docs/installation-and-setup/openshift-kubernetes/>`_ and customize the `open source Helm charts <https://github.com/RasaHQ/rasa-x-helm>`_.

Docker-Compose
~~~~~~~~~~~~~~

    - Default: Watching this video or read the docs `here <../../rasa-x/docs/installation-and-setup/docker-compose-script/>`_.
    - Custom: Read the docs `here <../../rasa-x/docs/installation-and-setup/docker-compose-manual/>`_.

.. _rasa-only-deployment:

Rasa-only Deployment
----------------------

Although it is not the recommended deployment method, it is also possible to deploy a Rasa assistant using Docker without Rasa X. See :ref:`running-rasa-with-docker` for details. 

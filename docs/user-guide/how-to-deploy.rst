:desc: How to deploy your Rasa Assistant with Docker Compose or Kubernetes/Openshift a
 a
.. _deploying-your-rasa-assistant: a
 a
Deploying Your Rasa Assistant a
============================= a
 a
.. edit-link:: a
 a
This page explains when and how to deploy an assistant built with Rasa. a
It will allow you to make your assistant available to users and set you up with a production-ready environment. a
 a
.. contents:: a
   :local: a
   :depth: 2 a
 a
 a
When to Deploy Your Assistant a
----------------------------- a
 a
The best time to deploy your assistant and make it available to test users is once it can handle the most a
important happy paths or is what we call a `minimum viable assistant <https://rasa.com/docs/rasa/glossary>`_. a
 a
The recommended deployment methods described below make it easy to share your assistant a
with test users via the `share your assistant feature in a
Rasa X <https://rasa.com/docs/rasa-x/user-guide/enable-workflows#conversations-with-test-users>`_. a
Then, when you’re ready to make your assistant available via one or more :ref:`messaging-and-voice-channels`, a
you can easily add them to your existing deployment set up. a
 a
.. _recommended-deployment-methods: a
 a
Recommended Deployment Methods a
------------------------------ a
 a
The recommended way to deploy an assistant is using either the One-Line Deployment or Kubernetes/Openshift a
options we support. Both deploy Rasa X and your assistant. They are the easiest ways to deploy your assistant, a
allow you to use Rasa X to view conversations and turn them into training data, and are production-ready. a
 a
One-Line Deploy Script a
~~~~~~~~~~~~~~~~~~~~~~ a
 a
The one-line deployment script is the easiest way to deploy Rasa X and your assistant. It installs a Kubernetes a
cluster on your machine with sensible defaults, getting you up and running in one command. a
 a
    - Default: Make sure you meet the `OS Requirements <https://rasa.com/docs/rasa-x/installation-and-setup/one-line-deploy-script/#hardware-os-requirements>`_, a
      then run: a
 a
      .. copyable:: a
 a
         curl -s get-rasa-x.rasa.com | sudo bash a
 a
    - Custom: See `Customizing the Script <https://rasa.com/docs/rasa-x/installation-and-setup/one-line-deploy-script/#customizing-the-script>`_ a
      in the `One-Line Deploy Script <https://rasa.com/docs/rasa-x/installation-and-setup/one-line-deploy-script/#customizing-the-script>`_ docs. a
 a
Kubernetes/Openshift a
~~~~~~~~~~~~~~~~~~~~ a
 a
For assistants that will receive a lot of user traffic, setting up a Kubernetes or Openshift deployment via a
our helm charts is the best option. This provides a scalable architecture that is also straightforward to deploy. a
However, you can also customize the Helm charts if you have specific requirements. a
 a
    - Default: Read the `Deploying in Openshift or Kubernetes <https://rasa.com/docs/rasa-x/installation-and-setup/openshift-kubernetes/>`_ docs. a
    - Custom: Read the above, as well as the `Advanced Configuration <https://rasa.com/docs/rasa-x/installation-and-setup/openshift-kubernetes/#advanced-configuration>`_ a
      documentation, and customize the `open source Helm charts <https://github.com/RasaHQ/rasa-x-helm>`_ to your needs. a
 a
.. _rasa-only-deployment: a
 a
Alternative Deployment Methods a
------------------------------ a
 a
Docker Compose a
~~~~~~~~~~~~~~ a
 a
You can also run Rasa X in a Docker Compose setup, without the cluster environment. We have a quick install script a
for doing so, as well as manual instructions for any custom setups. a
 a
    - Default: Read the `Docker Compose Quick Install <https://rasa.com/docs/rasa-x/installation-and-setup/docker-compose-script/>`_ docs or watch the `Masterclass Video <https://www.youtube.com/watch?v=IUYdwy8HPVc>`_ on deploying Rasa X. a
    - Custom: Read the docs `Docker Compose Manual Install <https://rasa.com/docs/rasa-x/installation-and-setup/docker-compose-manual/>`_ documentation for full customization options. a
 a
Rasa Open Source Only Deployment a
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ a
 a
It is also possible to deploy a Rasa assistant without Rasa X using Docker Compose. To do so, you can build your a
Rasa Assistant locally or in Docker. Then you can deploy your model in Docker Compose. a
 a
.. toctree:: a
   :titlesonly: a
   :maxdepth: 1 a
 a
   Building a Rasa Assistant Locally <rasa-tutorial> a
   docker/building-in-docker a
   docker/deploying-in-docker-compose a
 a
 a
Deploying Your Action Server a
---------------------------- a
 a
.. _building-an-action-server-image: a
 a
Building an Action Server Image a
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ a
 a
If you build an image that includes your action code and store it in a container registry, you can run it a
as part of your deployment, without having to move code between servers. a
In addition, you can add any additional dependencies of systems or Python libraries a
that are part of your action code but not included in the base ``rasa/rasa-sdk`` image. a
 a
To create your image: a
 a
  #. Move your actions code to a folder ``actions`` in your project directory. a
     Make sure to also add an empty ``actions/__init__.py`` file: a
 a
      .. code-block:: bash a
 a
          mkdir actions a
          mv actions.py actions/actions.py a
          touch actions/__init__.py  # the init file indicates actions.py is a python module a
 a
     The ``rasa/rasa-sdk`` image will automatically look for the actions in ``actions/actions.py``. a
 a
  #. If your actions have any extra dependencies, create a list of them in a file, a
     ``actions/requirements-actions.txt``. a
 a
  #. Create a file named ``Dockerfile`` in your project directory, a
     in which you'll extend the official SDK image, copy over your code, and add any custom dependencies (if necessary). a
     For example: a
 a
      .. parsed-literal:: a
 a
         # Extend the official Rasa SDK image a
         FROM rasa/rasa-sdk:\ |version|.0 a
 a
         # Use subdirectory as working directory a
         WORKDIR /app a
 a
         # Copy any additional custom requirements, if necessary (uncomment next line) a
         # COPY actions/requirements-actions.txt ./ a
 a
         # Change back to root user to install dependencies a
         USER root a
 a
         # Install extra requirements for actions code, if necessary (uncomment next line) a
         # RUN pip install -r requirements-actions.txt a
 a
         # Copy actions folder to working directory a
         COPY ./actions /app/actions a
 a
         # By best practices, don't run the code with root user a
         USER 1001 a
 a
You can then build the image via the following command: a
 a
      .. code-block:: bash a
 a
        docker build . -t <account_username>/<repository_name>:<custom_image_tag> a
 a
The ``<custom_image_tag>`` should reference how this image will be different from others. For a
example, you could version or date your tags, as well as create different tags that have different code for production a
and development servers. You should create a new tag any time you update your code and want to re-deploy it. a
 a
 a
Using your Custom Action Server Image a
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ a
 a
If you're building this image to make it available from another server, a
for example a Rasa X or Rasa Enterprise deployment, you should push the image to a cloud repository. a
 a
This documentation assumes you are pushing your images to `DockerHub <https://hub.docker.com/>`_. a
DockerHub will let you host multiple public repositories and a
one private repository for free. Be sure to first `create an account <https://hub.docker.com/signup/>`_ a
and `create a repository <https://hub.docker.com/signup/>`_ to store your images. You could also push images to a
a different Docker registry, such as `Google Container Registry <https://cloud.google.com/container-registry>`_, a
`Amazon Elastic Container Registry <https://aws.amazon.com/ecr/>`_, or a
`Azure Container Registry <https://azure.microsoft.com/en-us/services/container-registry/>`_. a
 a
You can push the image to DockerHub via: a
 a
      .. code-block:: bash a
 a
        docker login --username <account_username> --password <account_password> a
        docker push <account_username>/<repository_name>:<custom_image_tag> a
 a
To authenticate and push images to a different container registry, please refer to the documentation of a
your chosen container registry. a
 a
How you reference the custom action image will depend on your deployment. Pick the relevant documentation for a
your deployment: a
 a
    - `One-Line Deployment <https://rasa.com/docs/rasa-x/installation-and-setup/one-line-deploy-script/#customizing-the-script>`_ a
    - `Kubernetes or Openshift <https://rasa.com/docs/rasa-x/installation-and-setup/openshift-kubernetes/#adding-a-custom-action-server>`_ a
    - `Docker Compose <https://rasa.com/docs/rasa-x/installation-and-setup/docker-compose-script/#connect-a-custom-action-server>`_ a
    - :ref:`Rasa Open Source Only <running-multiple-services>` a
 a
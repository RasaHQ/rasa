:desc: Setup Rasa NLU with Docker in your own infrastructure for local
       intent recognition and entity recognition. 

.. _section_docker:

Running in Docker
=================

.. contents::

Images
------

`Rasa NLU docker images <https://hub.docker.com/r/rasa/rasa_nlu/tags/>`_ are provided for different backends:

- ``spacy``: If you use the :ref:`section_pretrained_embeddings_spacy_pipeline` pipeline
- ``tensorflow``: If you use the :ref:`section_supervised_embeddings_pipeline`
  pipeline
- ``mitie``: If you use the :ref:`section_mitie_pipeline` pipeline
- ``bare``: If you want to take a base image and enhance it with your custom
  dependencies
- ``full`` (default): If you use components from different pre-defined pipelines
  and want to have everything included in the image.

.. note::

    For the ``tensorflow`` and the ``full`` image a x86_64 CPU with AVX support
    is required.


Training NLU
------------

To train a NLU model you need to mount two directories into the Docker container:

- a directory containing your project which in turn includes your NLU
  configuration and your NLU training data
- a directory which will contain the trained NLU model

.. code-block:: shell

    docker run \
        -v <project_directory>:/app/project \
        -v <model_output_directory>:/app/model \
        rasa/rasa_nlu:latest \
        run \
            python -m rasa.nlu.train \
                -c /app/project/<nlu configuration>.yml \
                -d /app/project/<nlu data> \
                -o /app/model

Running NLU with Rasa Core
--------------------------

See this `guide <https://rasa.com/docs/core/docker_walkthrough/>`_ which
describes how to set up all Rasa components as Docker containers and how to
connect them.


Running NLU as Standalone Server
--------------------------------

To run NLU as server you have to

- mount a directory with the trained NLU models
- expose a port

.. code-block:: bash

    docker run \
        -p 5000:5000 \
        -v <directory with nlu models>:/app/projects \
        rasa/rasa_nlu:latest \
        start \
            --path /app/projects
            --port 5000

You can then send requests to your NLU server as it is described in the
:ref:`section_http`, e.g. if it is running on the localhost:

.. code-block:: bash

    curl --request GET \
         --url 'http://localhost:5000/parse?q=Hello%20world!'


.. include:: feedback.inc
